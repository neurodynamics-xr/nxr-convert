"""Tests for nxr_convert.atlas (ported from nsp, D135) phases 2–3: canonical frames, vector reductions, joint cells and transport (synthetic)."""
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from nxr_convert.atlas import joint, rollup
from nxr_convert.atlas.ladder import vertex_areas
from nxr_convert.atlas.trees import subject_tree
from nxr_convert.atlas.vectors import Frames, band_tensor, vector_stats

from atlas_meshes import icosphere, two_hemispheres


def _have_nxr_compute():
    try:
        from nxr_convert.atlas.frames import nxr_compute_dir
        nxr_compute_dir()
        return shutil.which("node") is not None
    except FileNotFoundError:
        return False


needs_nxr = pytest.mark.skipif(not _have_nxr_compute(), reason="nxr-compute Node binding not available")


def analytic_frames(p):
    """Meridian frames on a sphere centred at the origin of each hemisphere block."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    north = np.array([0, 0, 1.0]) - u[:, 2:3] * u
    nn = np.linalg.norm(north, axis=1, keepdims=True)
    sing = nn[:, 0] < 1e-9
    e1 = north / np.maximum(nn, 1e-300)
    e1[sing] = np.nan
    return Frames(e1, np.cross(u, e1), u, sing)


# ── frames ────────────────────────────────────────────────────────────────────────────────

@needs_nxr
def test_frames_on_a_sphere_are_the_meridians():
    from nxr_convert.atlas.frames import hemisphere_frames, meridian_angle
    p, f = icosphere(4, 1.0)
    N, S = int(np.argmax(p[:, 2])), int(np.argmin(p[:, 2]))
    for domain in ("sphere", "cortex"):
        fr = hemisphere_frames(p, f, p, N, S, domain=domain)
        ang = meridian_angle(fr, p, f, p)
        assert np.nanmedian(ang) < 1.0
        ok = ~fr.singular
        assert np.allclose(np.cross(fr.e1[ok], fr.e2[ok]), fr.normal[ok], atol=1e-6)   # right-handed
        assert fr.singular[N] and fr.singular[S]


# ── closed-form vector reductions ─────────────────────────────────────────────────────────

def test_vector_stats_roll_up_and_recover_a_constant_field():
    pos, faces, s = two_hemispheres(3)
    blocks = np.r_[pos[: len(pos) // 2] + [0.05, 0, 0], pos[len(pos) // 2:] - [0.05, 0, 0]]
    fr = analytic_frames(blocks)
    t = subject_tree(pos, faces, s, depth=4)
    area = vertex_areas(pos, faces)
    v = 2.0 * np.nan_to_num(fr.e1) - 1.0 * np.nan_to_num(fr.e2)                        # (2, −1) in the frame
    st = vector_stats(t, area, fr, v)
    for level in (-1, 0, 2, 4):
        w = rollup.space(st["w"], 4, level)
        assert np.allclose(rollup.space(st["c1"], 4, level) / w, 2.0)
        assert np.allclose(rollup.space(st["c2"], 4, level) / w, -1.0)
    assert np.isclose(rollup.space(st["w"], 4, -1)[0], area[~fr.singular].sum())


def test_band_tensor_trace_is_the_total_power():
    pos, faces, s = two_hemispheres(2)
    blocks = np.r_[pos[: len(pos) // 2] + [0.05, 0, 0], pos[len(pos) // 2:] - [0.05, 0, 0]]
    fr = analytic_frames(blocks)
    t = subject_tree(pos, faces, s, depth=3)
    area = vertex_areas(pos, faces)
    rng = np.random.default_rng(3)
    V = len(pos)
    K = rng.standard_normal((3 * V, 5)).astype(np.float32)
    data = rng.standard_normal((5, 1200)).astype(np.float32)
    r = band_tensor(t, area, fr, K, data, 200.0, [(8.0, 16.0)], frame_s=0.5, chunk_frames=2)
    T = r["tensor"]
    ok_leaves = np.ones(t.n_leaves, bool)
    trace = T["t11"] + T["t22"] + T["tnn"]
    # leaves without a singular vertex: the frame trace equals the world-axis total
    sing_leaf = np.unique(t.codes[fr.singular])
    ok_leaves[sing_leaf] = False
    assert np.allclose(trace[:, ok_leaves], T["total"][:, ok_leaves], rtol=1e-4)
    assert r["samples"].sum() == 1200


# ── joint cells and transport ─────────────────────────────────────────────────────────────

def test_antisymmetric_connection_goes_there_and_back():
    rng = np.random.default_rng(4)
    om = rng.uniform(-np.pi, np.pi, (6, 6))
    a = joint.antisymmetric(om)
    assert np.allclose(np.angle(np.exp(1j * (a + a.T))), 0, atol=1e-12)


def test_transport_and_drift_along_a_path():
    level = 2                                                   # nodes 0..3 left, 4..7 right
    om = np.zeros((8, 8))
    om[0, 1], om[1, 0] = 0.3, -0.3
    om[1, 2], om[2, 1] = 0.2, -0.2
    path = np.array([0, 1, 2, 2, 5])
    phi = joint.transported_angle(path, om, level)
    assert np.allclose(phi[:4], [0, 0.3, 0.5, 0.5]) and np.isnan(phi[4])        # the hop to the right breaks it
    psi = np.array([0.1, 0.4, 0.6, 0.6, 0.0])                   # orientation turns exactly with transport
    d = joint.orientation_drift(path, psi, om, level)
    assert np.allclose(d[:3], 0, atol=1e-12) and np.isnan(d[3])
    psi2 = psi.copy()
    psi2[2] += np.pi                                            # an axis is the same axis turned by π
    assert np.allclose(joint.orientation_drift(path, psi2, om, level)[:3], 0, atol=1e-12)


def test_cell_density_on_joint_cells():
    D = 3
    power = np.ones((2, 2 * 2 ** D, 10))
    w = np.full(2 * 2 ** D, 0.5)
    samples = np.full(10, 4)
    dens = joint.cell_density(power, w, samples, D, level=1, time_level=2)
    assert dens.shape == (2, 4, 3)
    assert np.allclose(dens, 1 / (0.5 * 4))


def test_levi_civita_tensor_rotation_commutes_with_frame_sums():
    from nxr_convert.atlas.scalars import analytic_bands
    from nxr_convert.atlas.vectors import LeviCivita
    pos, faces, s = two_hemispheres(2)
    blocks = np.r_[pos[: len(pos) // 2] + [0.05, 0, 0], pos[len(pos) // 2:] - [0.05, 0, 0]]
    fr = analytic_frames(blocks)
    t = subject_tree(pos, faces, s, depth=3)
    area = vertex_areas(pos, faces)
    rng = np.random.default_rng(5)
    V = len(pos)
    rho = rng.uniform(-np.pi, np.pi, V)
    lc = LeviCivita((2,), {2: np.zeros(8, int)}, {2: rho}, {2: np.zeros((8, 8))})
    K = rng.standard_normal((3 * V, 4)).astype(np.float32)
    data = rng.standard_normal((4, 800)).astype(np.float32)
    r = band_tensor(t, area, fr, K, data, 200.0, [(8.0, 16.0)], frame_s=0.5, chunk_frames=3, lc=lc, lc_levels=(2,))
    # direct: rotate every sample, then sum
    (_, A), = list(analytic_bands(data, 200.0, [(8.0, 16.0)]))
    J = (K.astype(np.complex64) @ A).reshape(V, 3, -1)
    e1, e2 = np.nan_to_num(fr.e1), np.nan_to_num(fr.e2)
    J1, J2 = np.einsum("vk,vkt->vt", e1, J), np.einsum("vk,vkt->vt", e2, J)
    c, s_ = np.cos(rho)[:, None], np.sin(rho)[:, None]
    R1, R2 = c * J1 + s_ * J2, -s_ * J1 + c * J2
    w = np.where(~fr.singular, area, 0.0)
    node = t.at(2)
    direct = np.zeros((8, 8))
    t11 = (np.abs(R1) ** 2).reshape(V, 8, 100).sum(-1) * w[:, None]
    np.add.at(direct, node, t11)
    assert np.allclose(r["lc"][2]["t11"][0], direct, rtol=1e-4)
