"""Tests for nxr_convert.atlas (ported from nsp, D135) — the dyadic bookkeeping trees, roll-ups and scalar reductions (synthetic data only)."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from nxr_convert.atlas import hcp, rollup, scalars
from nxr_convert.atlas.ladder import surface_ladder, v8_sort, vertex_areas
from nxr_convert.atlas.trees import Template, Tree, subject_tree

from atlas_meshes import icosphere, two_hemispheres

FIX = Path(__file__).parent / "fixtures"


# ── the ladder ────────────────────────────────────────────────────────────────────────────

def test_ladder_is_nested_equal_area_and_arithmetic():
    p, f = icosphere(3)
    lad = surface_ladder(p, f)
    assert lad.finest >= 6
    fine = lad.labels[-1]
    for l, lab in enumerate(lad.labels):
        assert np.array_equal(lab, fine >> (lad.finest - l))            # THE LEVELS ARE ARITHMETIC
        areas = lad.tile_areas(l)
        assert len(areas) == 2 ** l and (areas > 0).all()
    cv = np.std(lad.tile_areas(4)) / np.mean(lad.tile_areas(4))
    assert cv < 0.05                                                     # equal area


def test_vertex_areas_total_the_surface():
    p, f = icosphere(4, radius=1.0)
    assert abs(vertex_areas(p, f).sum() - 4 * np.pi) / (4 * np.pi) < 0.01


def test_v8_sort_matches_node():
    fx = json.loads((FIX / "atlas_v8_sort.json").read_text())
    for c in fx["cases"]:
        k = [np.nan if v is None else v for v in c["keys"]]
        assert v8_sort(list(range(len(k))), lambda x, y: k[x] - k[y]) == c["order"]


# ── trees and roll-up ────────────────────────────────────────────────────────────────────

def test_subject_tree_codes_carry_the_hemisphere():
    pos, faces, s = two_hemispheres(3)
    t = subject_tree(pos, faces, s, ["unassigned", "Cortex L", "Cortex R"], depth=4)
    n = len(pos) // 2
    assert t.codes[:n].max() < 2 ** 4 <= t.codes[n:].min()
    assert np.array_equal(t.at(0), np.r_[np.zeros(n), np.ones(n)])
    assert np.array_equal(t.coarsen(2).codes, t.codes >> 2)


def test_group_tree_assigns_through_the_sphere():
    p, f = icosphere(3)
    lad = surface_ladder(p, f)
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    tpl = Template("synthetic", lad.finest, {"lh": u, "rh": u}, {"lh": lad.labels[-1], "rh": lad.labels[-1]})
    _, _, s = two_hemispheres(3)
    sphere = np.vstack([u, u]) * 0.1                                     # the subject's sphere = the template's
    t = tpl.tree_for(sphere, s, depth=5)
    n = len(u)
    assert np.array_equal(t.codes[:n], lad.labels[5])
    assert np.array_equal(t.codes[n:], 2 ** 5 + lad.labels[5])


def test_rollup_space_and_time_commute_and_are_exact():
    rng = np.random.default_rng(0)
    D, F = 5, 37
    x = rng.random((3, 2 * 2 ** D, F))
    a = rollup.time(rollup.space(x, D, 2, axis=1), 3, axis=2)
    b = rollup.space(rollup.time(x, 3, axis=2), D, 2, axis=1)
    assert np.allclose(a, b)
    assert np.isclose(rollup.space(x, D, -1, axis=1).sum(), x.sum())
    mx = rollup.time(x, 2, op="max", axis=2)
    assert np.allclose(mx[..., 0], x[..., :4].max(-1)) and np.allclose(mx[..., -1], x[..., 36:].max(-1))


# ── scalar reductions ───────────────────────────────────────────────────────────────────

def test_map_stats_roll_up_to_the_area_weighted_mean():
    pos, faces, s = two_hemispheres(3)
    t = subject_tree(pos, faces, s, depth=4)
    area = vertex_areas(pos, faces)
    x = pos[:, 2] * 10 + 1
    st = scalars.map_stats(t, area, x)
    whole = rollup.space(st["s1"], 4, -1)[0] / rollup.space(st["w"], 4, -1)[0]
    assert np.isclose(whole, np.sum(area * x) / area.sum())
    assert np.isclose(rollup.space(st["max"], 4, -1, op="max")[0], x.max())


def test_octave_bank_partitions_power():
    f = np.linspace(0.1, 200, 5000)
    bands = scalars.octave_bands(1.0, 6)
    g2 = (scalars.band_gains(f, bands) ** 2).sum(0)
    inside = (f > 2 ** 0.25) & (f < 64 / 2 ** 0.25)
    assert np.allclose(g2[inside], 1.0)
    assert (g2 <= 1 + 1e-12).all()


def test_band_power_sums_the_cortical_current():
    pos, faces, s = two_hemispheres(2)
    t = subject_tree(pos, faces, s, depth=3)
    area = vertex_areas(pos, faces)
    rng = np.random.default_rng(1)
    sfreq, n = 200.0, 2000
    data = rng.standard_normal((6, n)).astype(np.float32)
    K = rng.standard_normal((len(pos), 6)).astype(np.float32)
    bands = [(8.0, 16.0)]
    r = scalars.band_power(t, area, K, data, sfreq, bands, frame_s=0.5, chunk_frames=3)
    (_, A), = list(scalars.analytic_bands(data, sfreq, bands))
    J2 = np.abs(K.astype(np.complex64) @ A) ** 2
    direct = (area[:, None] * J2).sum(0)
    assert np.allclose(r["power"][0].sum(0), direct.reshape(-1, 100).sum(1), rtol=1e-4)
    assert r["samples"].sum() == n


def test_connectome_rolls_up_by_shifting_both_ends():
    rng = np.random.default_rng(2)
    D = 6
    codes = rng.integers(0, 2 * 2 ** D, size=(500, 2))
    codes[:20, 1] = -1                                                   # off-cortex ends drop out
    fine = hcp.connectome(codes, D, D).toarray()
    coarse = hcp.connectome(codes, D, 3).toarray()
    assert np.allclose(rollup.space(rollup.space(fine, D, 3, axis=0), D, 3, axis=1), coarse)
    assert np.isclose(coarse.sum() / 2, 480)


def test_bad_samples_carry_no_power_and_no_count():
    pos, faces, s = two_hemispheres(2)
    t = subject_tree(pos, faces, s, depth=3)
    area = vertex_areas(pos, faces)
    rng = np.random.default_rng(6)
    data = rng.standard_normal((6, 2000)).astype(np.float32)
    K = rng.standard_normal((len(pos), 6)).astype(np.float32)
    good = np.ones(2000, bool)
    good[100:200] = False                                   # the whole of frame 1 (frames of 100 samples)
    good[450:470] = False                                   # part of frame 4
    r = scalars.band_power(t, area, K, data, 200.0, [(8.0, 16.0)], frame_s=0.5, chunk_frames=3, good=good)
    assert r["samples"][1] == 0 and r["samples"][4] == 80 and r["samples"].sum() == 2000 - 120
    assert np.all(r["power"][0][:, 1] == 0)
    (_, A), = list(scalars.analytic_bands(data, 200.0, [(8.0, 16.0)]))
    J2 = np.abs(K.astype(np.complex64) @ A) ** 2 * good[None, :]
    direct = (area[:, None] * J2).sum(0).reshape(-1, 100).sum(1)
    assert np.allclose(r["power"][0].sum(0), direct, rtol=1e-4)


# ── tower frames: tiles that are not a whole number of samples ─────────────────────────────

def test_band_power_on_tower_tiles_counts_samples_per_tile():
    from nxr_convert.atlas.tower import Placement
    pos, faces, s = two_hemispheres(2)
    t = subject_tree(pos, faces, s, depth=3)
    area = vertex_areas(pos, faces)
    rng = np.random.default_rng(7)
    sfreq, n = 1200.0, 6000
    data = rng.standard_normal((6, n)).astype(np.float32)
    K = rng.standard_normal((len(pos), 6)).astype(np.float32)
    good = np.ones(n, bool)
    good[1000:1100] = False
    codes = Placement("r", sfreq, n, start_s=49440.3).sample_codes(-18)       # 395.5 samples per tile, day-aligned
    bands = [(8.0, 16.0)]
    (_, A), = list(scalars.analytic_bands(data, sfreq, bands))
    J2 = np.abs(K.astype(np.complex64) @ A) ** 2 * good[None, :]
    per_sample = (area[:, None] * J2).sum(0)
    f = codes - codes[0]
    direct = np.bincount(f, weights=per_sample)
    for cf in (1, 3, 16):                                               # chunk edges cut tiles: the sums must not see them
        r = scalars.band_power(t, area, K, data, sfreq, bands, chunk_frames=cf, good=good, codes=codes)
        assert np.allclose(r["power"][0].sum(0), direct, rtol=1e-4)
        assert np.array_equal(r["samples"], np.bincount(f, weights=good).astype(int))
        assert r["power"].shape[-1] == f[-1] + 1
        env_max = np.full(f[-1] + 1, -np.inf)
        np.maximum.at(env_max, f, np.sqrt(J2).max(0))
        assert np.allclose(r["envmax"][0].max(0), env_max, rtol=1e-4)


def test_rollup_time_aligns_to_the_tower():
    rng = np.random.default_rng(8)
    x = rng.random((2, 13))
    code0 = 4 * 17 + 3                                                   # frame 0 is the last frame of its level-2 tile
    y = rollup.time(x, 2, axis=1, code0=code0)
    assert y.shape[1] == ((code0 + 12) >> 2) - (code0 >> 2) + 1
    assert np.allclose(y[:, 0], x[:, 0]) and np.allclose(y[:, 1], x[:, 1:5].sum(1))
    assert np.allclose(y.sum(1), x.sum(1))
    assert np.allclose(rollup.time(rollup.time(x, 1, axis=1, code0=code0), 1, axis=1, code0=code0 >> 1), y)
