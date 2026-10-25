"""0.2.6: a Hodge-Dirac basis PER SURFACE (``hodge --surface``), the interpolation operator between two surfaces (``interp``),
the FreeSurfer pial writer (``surface --fs-dir``) and Brainstorm's secondary surface. All on synthetic stores."""
import hashlib
import json
import struct

import numpy as np
import pytest
import scipy.sparse as sp
import zarr
from scipy.io import savemat

from eeg_fixture import _surface
from nxr_convert.freesurfer import read_annot, read_surf
from nxr_convert.hodge import add_hodge
from nxr_convert.interp import add_interpolation
from nxr_convert.surface import add_surface, export_freesurfer, export_surface
from test_hodge import _synthetic
from test_subject import _scouts

OCT_V, OCT_F = _surface()
OCT_F0 = OCT_F.astype(np.int64) - 1


def _octa(sub, name, shift, primary=False, **kw):
    return add_surface(sub, name=name, folder="sources", vertices=OCT_V * 0.05 + shift, faces=OCT_F0, primary=primary, **kw)["id"]


def _row(sub, table, path):
    return sub.find(table, path)


# ---- hodge --surface -----------------------------------------------------------------------------------------------------

def _two(sub, tmp_path, target):
    """A primary 'a' and a secondary 'b'; an nsp output (6 glued vertices) solved on ``target``."""
    a, b = _octa(sub, "a", 0.0, primary=True), _octa(sub, "b", 1.0)
    d = tmp_path / "nsp"
    d.mkdir(exist_ok=True)
    _synthetic(d)
    np.save(d / "s_glued_V.npy", OCT_V * 0.05 + (0.0 if target == "a" else 1.0))
    return a, b, d


def test_hodge_primary_legacy_and_secondary_suffixed(sub, tmp_path):
    a, b, d = _two(sub, tmp_path, "a")
    add_hodge(sub, d, sub="s")                                              # the primary: today's paths
    op = _row(sub, "operator", "hodgeDirac_eigenvectors")
    assert op["to_manifold_id"] == a and _row(sub, "manifold", "eigen/hodgeDirac") and _row(sub, "field", "eigen/hodgeDirac_kind")
    assert _row(sub, "selection", "hodgeDirac_support")["manifold_id"] == a
    assert json.loads(sub.db.one("SELECT transition_json FROM coordinate_system WHERE name = 'eigenvalue'")["transition_json"] or "null") is None
    np.save(d / "s_glued_V.npy", OCT_V * 0.05 + 1.0)
    info = add_hodge(sub, d, sub="s", surface="b")                          # the secondary: suffixed, bound to b
    assert info["axis"] == "eigen/hodgeDirac__b"
    op2 = _row(sub, "operator", "hodgeDirac__b_eigenvectors")
    assert op2["to_manifold_id"] == b and op["id"] != op2["id"] and op["to_manifold_id"] == a        # both coexist
    ax = _row(sub, "manifold", "eigen/hodgeDirac__b")
    assert ax["derived_from_id"] == b and _row(sub, "field", "eigen/hodgeDirac__b_kind")["manifold_id"] == ax["id"]
    assert _row(sub, "selection", "hodgeDirac__b_support")["manifold_id"] == b
    cs = sub.db.value("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", ax["geometry_id"])
    assert cs == "eigenvalue"
    assert sub.db.value("SELECT count(*) FROM space WHERE subject_id = ? AND name = 'spectrum:hodgeDirac__b'", sub.id) == 1
    Ea = zarr.open_array(str(sub.store / "hodgeDirac_eigenvectors"), mode="r")[:]
    Eb = zarr.open_array(str(sub.store / "hodgeDirac__b_eigenvectors"), mode="r")[:]
    assert Ea.shape == (6, 4) and np.array_equal(Ea, Eb)


def test_hodge_replace_touches_one_surface_only(sub, tmp_path):
    a, b, d = _two(sub, tmp_path, "a")
    add_hodge(sub, d, sub="s")
    np.save(d / "s_glued_V.npy", OCT_V * 0.05 + 1.0)
    add_hodge(sub, d, sub="s", surface="b")
    before = _row(sub, "operator", "hodgeDirac_eigenvectors")["id"]
    with pytest.raises(FileExistsError):
        add_hodge(sub, d, sub="s", surface="b")
    add_hodge(sub, d, sub="s", surface="b", replace=True)
    assert _row(sub, "operator", "hodgeDirac_eigenvectors")["id"] == before
    assert sub.db.value("SELECT count(*) FROM operator WHERE subject_id = ? AND name LIKE 'hodgeDirac%'", sub.id) == 2


def test_hodge_wrong_surface_names_the_available(sub, tmp_path):
    _, _, d = _two(sub, tmp_path, "a")
    with pytest.raises(KeyError, match=r"'zz'.*a, b"):
        add_hodge(sub, d, sub="s", surface="zz")


# ---- FreeSurfer files ------------------------------------------------------------------------------------------------------

def _write_surf(path, v, f):
    path.write_bytes(b"\xff\xff\xfe" + b"created by test\n\n" + struct.pack(">ii", len(v), len(f)) +
                     v.astype(">f4").tobytes() + f.astype(">i4").tobytes())


def _write_annot(path, labels, table):
    """version -2 colour table; ``table`` = [(name, r, g, b)], ``labels`` the entry index per vertex (-1 none)."""
    ann = lambda r, g, b: r + (g << 8) + (b << 16)
    out = struct.pack(">i", len(labels))
    for i, k in enumerate(labels):
        out += struct.pack(">ii", i, ann(*table[k][1:]) if k >= 0 else 0)
    out += struct.pack(">iii", 1, -2, len(table)) + struct.pack(">i", 5) + b"test\0" + struct.pack(">i", len(table))
    for i, (n, r, g, b) in enumerate(table):
        out += struct.pack(">ii", i, len(n) + 1) + n.encode() + b"\0" + struct.pack(">iiii", r, g, b, 0)
    path.write_bytes(out)


def _fsdir(tmp_path):
    fs = tmp_path / "fs"
    (fs / "surf").mkdir(parents=True)
    (fs / "label").mkdir()
    for k, h in enumerate(("lh", "rh")):
        _write_surf(fs / "surf" / f"{h}.pial", OCT_V * 50 + 100 * k, OCT_F0)
        _write_surf(fs / "surf" / f"{h}.sphere.reg", OCT_V * 100, OCT_F0)
        _write_annot(fs / "label" / f"{h}.aparc.annot", [0, 0, 0, 1, 1, -1], [("bankssts", 25, 100, 40), ("insula", 255, 192, 32)])
    return fs


def test_freesurfer_readers_round_trip(tmp_path):
    fs = _fsdir(tmp_path)
    v, f = read_surf(fs / "surf" / "lh.pial")
    np.testing.assert_allclose(v, OCT_V * 50, atol=1e-4)
    assert (f == OCT_F0).all()
    idx, names, rgb = read_annot(fs / "label" / "lh.aparc.annot")
    assert idx.tolist() == [0, 0, 0, 1, 1, -1] and names == ["bankssts", "insula"] and rgb.tolist() == [[25, 100, 40], [255, 192, 32]]


def test_freesurfer_pial_surface(sub, tmp_path):
    _octa(sub, "a", 0.0, primary=True)
    fs = _fsdir(tmp_path)
    info = export_freesurfer(sub, fs)
    assert info["n_vertices"] == 12 and info["hemispheres"] == 2 and info["registration"] and info["atlases"] == 1
    m = _row(sub, "manifold", "sources/cortex_pial")
    assert m["is_primary"] == 0 and _row(sub, "manifold", "sources/a")["is_primary"] == 1
    xyz = zarr.open_array(str(sub.store / "sources/cortex_pial/vertices"), mode="r")[:]
    np.testing.assert_allclose(xyz[6:], (OCT_V * 50 + 100) / 1000, atol=1e-6)                       # rh after lh, metres
    h1, h2 = _row(sub, "manifold", "sources/cortex_pial_1"), _row(sub, "manifold", "sources/cortex_pial_2")
    assert (h1["label"], h1["from_index"], h1["n_vertices"]) == ("left", 0, 6) and (h2["label"], h2["from_index"]) == ("right", 6)
    sph = zarr.open_array(str(sub.store / "sources/cortex_pial_sphere/vertices"), mode="r")[:]
    np.testing.assert_allclose(np.linalg.norm(sph, axis=1), 0.1, atol=1e-6)
    cs = sub.db.value("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", m["geometry_id"])
    assert cs == "freesurfer-tkr"
    assert _row(sub, "selection", "sources/cortex_pial_atlases/Desikan-Killiany")
    prod = json.loads(m["producer_json"])
    sha = lambda p: hashlib.sha256((fs / "surf" / p).read_bytes()).hexdigest()
    assert (prod["sha256_lh"], prod["sha256_rh"], prod["nV_lh"], prod["nV_rh"]) == (sha("lh.pial"), sha("rh.pial"), 6, 6)
    with pytest.raises(ValueError, match="already occupies"):
        export_freesurfer(sub, fs)


def test_freesurfer_sphere_count_refused(sub, tmp_path):
    fs = _fsdir(tmp_path)
    _write_surf(fs / "surf" / "rh.sphere.reg", OCT_V[:5] * 100, np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="sphere has 5"):
        export_freesurfer(sub, fs)


def test_freesurfer_scs_affine_is_fitted(sub, tmp_path):
    fs = _fsdir(tmp_path)
    v = np.concatenate([OCT_V * 50, OCT_V * 50 + 100]) / 1000
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    savemat(tmp_path / "t.mat", {"Vertices": v @ R.T + [0.01, 0.02, 0.03], "Faces": OCT_F})
    info = export_freesurfer(sub, fs, scs_from=tmp_path / "t.mat")
    assert info["scs_rms_m"] < 1e-9 and _row(sub, "operator", "cortex_pial_scs")["kind"] == "transition"
    A = zarr.open_array(str(sub.store / "cortex_pial_scs"), mode="r")[:]
    np.testing.assert_allclose(A[:, :3], R, atol=1e-6)
    savemat(tmp_path / "bad.mat", {"Vertices": v + np.random.default_rng(0).normal(0, 1e-3, v.shape), "Faces": OCT_F})
    with pytest.raises(ValueError, match="no affine"):
        export_freesurfer(sub, fs, name="other", scs_from=tmp_path / "bad.mat")


def test_hodge_sha_must_match_the_surface(sub, tmp_path):
    fs = _fsdir(tmp_path)
    export_freesurfer(sub, fs)
    sha = lambda p: hashlib.sha256((fs / "surf" / p).read_bytes()).hexdigest()
    d = tmp_path / "nsp"
    d.mkdir()
    _synthetic(d)
    np.save(d / "s_glued_V.npy", np.asarray(zarr.open_array(str(sub.store / "sources/cortex_pial/vertices"), mode="r")[:6]))
    meta = lambda lh: json.dump({"hodge": {"nV": 6, "nE": 9, "nF": 5}, "source": {"sha256_lh": lh, "sha256_rh": sha("rh.pial")}},
                                open(d / "s_hodge.json", "w"))
    meta("0" * 64)
    with pytest.raises(ValueError, match="different lh"):
        add_hodge(sub, d, sub="s", surface="cortex_pial")
    meta(sha("lh.pial"))
    assert add_hodge(sub, d, sub="s", surface="cortex_pial")["surface"] == "cortex_pial"


# ---- Brainstorm's secondary surface (unchanged writer) -----------------------------------------------------------------

def test_bst_secondary_surface_has_sphere_and_hemispheres_and_is_not_primary(sub, tmp_path):
    _octa(sub, "a", 0.0, primary=True)
    atlas = np.array([("Structures", _scouts([([1, 2, 3], "Cortex L", [1, 0, 0]), ([4, 5, 6], "Cortex R", [0, 0, 1])]))],
                     dtype=[("Name", "O"), ("Scouts", "O")])
    savemat(tmp_path / "tess_cortex_pial_high.mat", {"Vertices": OCT_V, "Faces": OCT_F, "Comment": "high", "Atlas": atlas,
                                                    "Reg": {"Sphere": {"Vertices": OCT_V * 100}}})
    info = export_surface(sub, tmp_path / "tess_cortex_pial_high.mat", primary=False)
    assert info["registration"] and info["hemispheres"] == 2
    assert _row(sub, "manifold", "sources/cortex_pial_high_sphere") and _row(sub, "manifold", "sources/cortex_pial_high_2")["label"] == "right"
    assert _row(sub, "manifold", "sources/cortex_pial_high")["is_primary"] == 0 and _row(sub, "manifold", "sources/a")["is_primary"] == 1


# ---- interp ------------------------------------------------------------------------------------------------------------------

def _icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        mid, nf = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                v.append((v[a] + v[b]) / np.linalg.norm(v[a] + v[b]))
                mid[k] = len(v) - 1
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, np.int64)


def _two_hemis(sub, name, level, sphere=True, primary=False):
    """Two hemispheres, each an icosphere (apart in space, on the SAME registration sphere)."""
    v, f = _icosphere(level)
    n = len(v)
    return add_surface(sub, name=name, folder="sources", vertices=np.concatenate([v * 0.05 - 0.1, v * 0.05 + 0.1]),
                       faces=np.concatenate([f, f + n]), sphere=np.concatenate([v, v]) * 0.1 if sphere else None,
                       hemis=[("left", 0, n), ("right", n, n)], primary=primary)


def _read_csr(sub, name, shape):
    z = lambda p: zarr.open_array(str(sub.store / name / p), mode="r")[:]
    return sp.csr_matrix((z("data"), z("indices"), z("indptr")), shape=shape)


def test_interp_weights_gathers_and_hemispheres(sub):
    _two_hemis(sub, "lo", 1, primary=True)
    _two_hemis(sub, "hi", 2)
    nlo, nhi = 2 * 42, 2 * 162
    info = add_interpolation(sub, "lo", "hi")                              # hi <- lo: hi vertices in lo triangles
    op = _row(sub, "operator", "hi_from_lo")
    assert (op["kind"], op["layout"], op["n_rows"], op["n_cols"]) == ("interpolation", "csr", nhi, nlo)
    assert op["from_manifold_id"] == _row(sub, "manifold", "sources/lo")["id"] and op["to_manifold_id"] == _row(sub, "manifold", "sources/hi")["id"]
    M = _read_csr(sub, "hi_from_lo", (nhi, nlo))
    assert info["max_row_sum_error"] < 1e-12 and M.data.min() >= 0 and np.abs(np.asarray(M.sum(1)).ravel() - 1).max() < 1e-12
    assert info["exact_gather_share"] == pytest.approx(2 * 42 / nhi)          # lo's vertices are hi's
    assert np.diff(M.indptr).max() == 3 and (np.diff(M.indptr) == 1).sum() == 84
    assert info["roundtrip_rms_over_sd"] < 0.15                              # a smooth field survives
    for r in range(nhi):                                                      # no row mixes the hemispheres
        c = M.indices[M.indptr[r]:M.indptr[r + 1]]
        assert (c < 42).all() if r < 162 else (c >= 42).all()
    with pytest.raises(FileExistsError):
        add_interpolation(sub, "lo", "hi")
    add_interpolation(sub, "lo", "hi", replace=True)
    # the other way every row is an exact gather
    assert add_interpolation(sub, "hi", "lo")["exact_gather_share"] == 1.0


def test_interp_refuses_without_a_sphere_or_hemispheres(sub):
    _two_hemis(sub, "lo", 1, primary=True)
    _two_hemis(sub, "bare", 1, sphere=False)
    with pytest.raises(ValueError, match="no registration sphere"):
        add_interpolation(sub, "lo", "bare")
    _octa(sub, "plain", 3.0, sphere=OCT_V * 0.1)
    with pytest.raises(ValueError, match="hemisphere"):
        add_interpolation(sub, "lo", "plain")
    with pytest.raises(KeyError, match="lo"):
        add_interpolation(sub, "lo", "nope")


def test_interp_splits_a_surface_without_components(sub):
    """A schema-48 surface has no hemisphere components: its two mesh pieces are the hemispheres, left first (+y in scs)."""
    v, f = _icosphere(1)
    n = len(v)
    _two_hemis(sub, "lo", 1, primary=True)
    for name, ys in (("bare", (0.1, -0.1)), ("flipped", (-0.1, 0.1))):
        add_surface(sub, name=name, folder="sources", vertices=np.concatenate([v * 0.05 + [0, ys[0], 0], v * 0.05 + [0, ys[1], 0]]),
                    faces=np.concatenate([f, f + n]), sphere=np.concatenate([v, v]) * 0.1)
    info = add_interpolation(sub, "bare", "lo")
    M = _read_csr(sub, "lo_from_bare", (2 * n, 2 * n))
    assert info["exact_gather_share"] == 1.0 and (M.indices[:n] < n).all() and (M.indices[n:] >= n).all()
    with pytest.raises(ValueError, match="lies to the right"):
        add_interpolation(sub, "flipped", "lo")


def test_hodge_streams_the_split_layout_in_blocks(sub, tmp_path, monkeypatch):
    """The split layout is written MODE_BLOCK modes at a time and equals the whole-package result (up to each mode's sign)."""
    import nxr_convert.hodge as H
    _octa(sub, "a", 0.0, primary=True)
    _octa(sub, "b", 1.0)
    out = {}
    for tag, split, blk in (("old", False, 64), ("new", True, 1)):
        d = tmp_path / tag
        d.mkdir()
        _synthetic(d, split=split)
        np.save(d / "s_glued_V.npy", OCT_V * 0.05)
        monkeypatch.setattr(H, "MODE_BLOCK", blk)
        add_hodge(sub, d, sub="s", replace=True)
        out[tag] = zarr.open_array(str(sub.store / "hodgeDirac_eigenvectors"), mode="r")[:]
    s = np.sign((out["old"] * out["new"]).sum(0))
    np.testing.assert_allclose(out["new"] * s, out["old"], atol=1e-6)
