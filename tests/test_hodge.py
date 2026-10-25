"""``nxr-convert hodge`` on a real subject: nsp's rheome-hodge-dirac outputs → the package in the subject, read back.

Needs NXR_TEST_HODGE_NSP (the nsp output folder), NXR_TEST_HODGE_DATASET (a SCRATCH dataset folder holding the subject; it
is written) and NXR_TEST_HODGE_SUBJECT; skips without them. Runs the command TWICE (the second with --replace)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from nxr_convert.cli import main

NSP, DS, SUB = (os.environ.get(k) for k in ("NXR_TEST_HODGE_NSP", "NXR_TEST_HODGE_DATASET", "NXR_TEST_HODGE_SUBJECT"))
needs_real = pytest.mark.skipif(not (NSP and DS and SUB), reason="no NXR_TEST_HODGE_* (a real nsp output and a scratch dataset)")


@needs_real
def test_hodge_package_round_trip(capsys):
    from nxr_convert.crud import Subject, open_dataset
    nsp_sub = os.environ.get("NXR_TEST_HODGE_NSP_SUBJECT", SUB)
    args = ["hodge", "--dataset", DS, "--subject", SUB, "--from", NSP, "--nsp-subject", nsp_sub, "--replace"]
    assert main(args) == 0 and main(args) == 0                    # re-adding replaces, never duplicates
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["replaced"] and out["position_err_m"] <= 1e-6 and out["n_exact"] + out["n_coexact"] == out["n_modes"]

    ld = lambda n: np.load(Path(NSP) / f"{nsp_sub}_{n}.npy")
    meta = json.loads((Path(NSP) / f"{nsp_sub}_hodge.json").read_text())["hodge"]
    nV, nE = meta["nV"], meta["nE"]
    mu_all, X, keep, Ed = ld("hodge_mu").ravel(), ld("hodge_modes").astype(np.float64), ld("glued_keep").ravel(), ld("idt_E")

    with open_dataset(DS) as ds:
        s = Subject.open(ds, SUB)
        sid, store = s.row, s.store
        rows = {t: ds.db.all(f"SELECT * FROM {t} WHERE subject_id = ? AND name LIKE 'hodgeDirac%'", sid["id"])
                for t in ("manifold", "field", "operator", "selection")}
    assert [len(rows[t]) for t in ("manifold", "field", "operator", "selection")] == [1, 1, 1, 1]
    op = rows["operator"][0]
    assert op["kind"] == "eigenvectors" and op["from_manifold_id"] == rows["manifold"][0]["id"] and op["status"] == "complete"
    mu = zarr.open_array(str(store / "eigen/hodgeDirac"), mode="r")[:]
    kind = zarr.open_array(str(store / "eigen/hodgeDirac_kind"), mode="r")[:]
    E = zarr.open_array(str(store / "hodgeDirac_eigenvectors"), mode="r")[:]
    assert E.shape == (op["n_rows"], op["n_cols"]) and E.shape[1] == mu.size == rows["manifold"][0]["n_vertices"]
    assert np.all(np.diff(mu) >= 0) and np.allclose(mu, np.sort(mu_all[mu_all > 1]))
    off = np.setdiff1d(np.arange(E.shape[0]), keep)
    assert not E[off].any()                                        # zero off the glued surface

    # INDEPENDENT: an exact mode's potential A gives the solve's 1-form, (A[head] − A[tail]) / μ = its 1-form part
    pos = np.where(mu_all > 1)[0][np.argsort(mu_all[mu_all > 1], kind="stable")]
    for k in np.where(kind == 0)[0]:
        A = E[keep, k].astype(np.float64)
        one = X[pos[k], nV:nV + nE]
        dA = (A[Ed[:, 1]] - A[Ed[:, 0]]) / mu[k]
        assert np.abs(dA - one).max() <= 1e-3 * np.abs(one).max(), k
    # a coexact mode's 0-form part is zero, and its potential is not
    for k in np.where(kind == 2)[0]:
        assert np.abs(X[pos[k], :nV]).max() < 1e-6 * np.abs(X[pos[k]]).max() and np.abs(E[:, k]).max() > 0


# ---- package(): the OLD full-mode layout and nsp's SPLIT layout read alike -------------------------------------------------

def _synthetic(d: Path, sub: str = "s", split: bool = False, split_norm: float = 1.0):
    """A tiny complex (nV 6, nE 9, nF 5) with 2 exact + 2 coexact modes, written in either layout.
    ``split_norm`` is Σψ²⋆ of the split arrays: 1 (ψ) or 0.5 (already ψ/√2)."""
    rng = np.random.default_rng(3)
    nV, nE, nF = 6, 9, 5
    M = rng.uniform(0.5, 2, nV + nE + nF)
    iF = rng.integers(0, nV, (nF, 3))
    unit = lambda x, w: x / np.sqrt((x ** 2 * w).sum())
    psi = np.array([unit(rng.normal(size=nV), M[:nV]) for _ in range(3)])               # row 0: the constant-like zero mode
    chi = np.array([unit(rng.normal(size=nF), M[nV + nE:]) for _ in range(3)])
    le, lc = np.array([0, 4.0, 25.0]), np.array([0, 9.0, 16.0])
    mu_pos = np.sqrt(np.r_[le, lc])
    json.dump({"hodge": {"nV": nV, "nE": nE, "nF": nF}}, open(d / f"{sub}_hodge.json", "w"))
    for n, a in (("hodge_mass", M), ("glued_keep", np.arange(nV)), ("idt_F", iF), ("glued_V", rng.normal(size=(nV, 3)))):
        np.save(d / f"{sub}_{n}.npy", a)
    if split:
        f = np.sqrt(split_norm)
        np.save(d / f"{sub}_hodge_exact_modes.npy", (psi * f).astype(np.float32))
        np.save(d / f"{sub}_hodge_coexact_modes.npy", (chi * f).astype(np.float32))
        np.save(d / f"{sub}_hodge_exact_lambda.npy", le[:, None])
        np.save(d / f"{sub}_hodge_coexact_lambda.npy", lc[:, None])
    else:                                                    # every mode a full form, weight ½ on its 0- or 2-form
        rows, mus = [], []
        for sgn in (-1, 1):
            for m, x in zip(mu_pos, np.r_[np.c_[psi, np.zeros((3, nE + nF))], np.c_[np.zeros((3, nV + nE)), chi]]):
                v = x / np.sqrt(2)
                v[nV:nV + nE] = sgn * rng.normal(size=nE) if m else 0
                rows.append(v), mus.append(sgn * m)
        np.save(d / f"{sub}_hodge_modes.npy", np.array(rows, np.float32))
        mus = np.array(mus)
        np.save(d / f"{sub}_hodge_mu.npy", mus[:, None])
    if split:
        np.save(d / f"{sub}_hodge_mu.npy", np.r_[-mu_pos, mu_pos][:, None])
    return d


def _same_up_to_sign(a, b):
    s = np.sign((a * b).sum(1))
    return np.abs(a - s[:, None] * b).max(1) / np.abs(b).max(1)


@pytest.mark.parametrize("norm", [1.0, 0.5])
def test_package_split_equals_old(tmp_path, norm):
    from nxr_convert.hodge import package
    (tmp_path / "o").mkdir(), (tmp_path / "n").mkdir()
    old = package(_synthetic(tmp_path / "o"), "s")
    new = package(_synthetic(tmp_path / "n", split=True, split_norm=norm), "s")
    assert old["mu"].size == new["mu"].size == 4
    np.testing.assert_allclose(new["mu"], old["mu"], rtol=1e-6)
    assert (new["kind"] == old["kind"]).all() and new["mixing"] == 0.0
    assert _same_up_to_sign(new["potentials"].astype(np.float64), old["potentials"]).max() < 1e-6
    assert (new["keep"] == old["keep"]).all()


def test_package_split_refuses(tmp_path):
    from nxr_convert.hodge import package
    d = _synthetic(tmp_path, split=True, split_norm=0.8)                   # neither ψ nor ψ/√2
    with pytest.raises(ValueError, match="neither 1"):
        package(d, "s")
    d = _synthetic(tmp_path, split=True)
    np.save(d / "s_hodge_mu.npy", np.array([[-5.0], [5.0]]))              # not the merged ladders' √λ
    with pytest.raises(ValueError, match="hodge_mu"):
        package(d, "s")
    d = _synthetic(tmp_path, split=True)
    np.save(d / "s_hodge_exact_lambda.npy", np.zeros((2, 1)))              # sizes disagree
    with pytest.raises(ValueError, match="sizes"):
        package(d, "s")


# the split fsaverage outputs and the stored references (a zarr + hodge.json/.f32 folder); skips without them
FSA, TPL = (Path(os.environ[k]) if os.environ.get(k) else None for k in ("NXR_TEST_HODGE_SPLIT", "NXR_TEST_HODGE_REF"))


@pytest.mark.skipif(not (FSA and TPL and FSA.exists() and TPL.exists()), reason="no NXR_TEST_HODGE_SPLIT / NXR_TEST_HODGE_REF")
def test_package_split_parity_fsaverage(tmp_path):
    """The first 64 modes of nsp's split fsaverage outputs against BOTH stored references (templates-nxr's zarr and the
    mode-major hodge.f32); read from a 65-row copy (zero mode + 64) of each ladder, so the 5 GB files are never loaded."""
    from nxr_convert.hodge import package
    n, K = 65, 64
    for f in FSA.glob("fsaverage_*"):
        if not any(t in f.name for t in ("modes", "lambda", "_hodge_mu")):
            (tmp_path / f.name).symlink_to(f)
    for a in ("exact", "coexact"):
        np.save(tmp_path / f"fsaverage_hodge_{a}_modes.npy", np.asarray(np.load(FSA / f"fsaverage_hodge_{a}_modes.npy", mmap_mode="r")[:n]))
        np.save(tmp_path / f"fsaverage_hodge_{a}_lambda.npy", np.load(FSA / f"fsaverage_hodge_{a}_lambda.npy")[:n])
    m = np.sqrt(np.r_[np.load(tmp_path / "fsaverage_hodge_exact_lambda.npy").ravel(), np.load(tmp_path / "fsaverage_hodge_coexact_lambda.npy").ravel()])
    np.save(tmp_path / "fsaverage_hodge_mu.npy", np.r_[m, -m])
    pk = package(tmp_path, "fsaverage")
    P, keep = pk["potentials"][:K].astype(np.float64), pk["keep"]
    z = zarr.open_array(str(TPL / "fsaverage.nxr.zarr/hodgeDirac_eigenvectors"), mode="r")
    ref_zarr = np.asarray(z[:, :K]).T[:, keep].astype(np.float64)
    mu_z = zarr.open_array(str(TPL / "fsaverage.nxr.zarr/eigen/hodgeDirac"), mode="r")[:K]
    kind_z = zarr.open_array(str(TPL / "fsaverage.nxr.zarr/eigen/hodgeDirac_kind"), mode="r")[:K]
    j = json.loads((TPL / "fsaverage.eigenmodes/hodge.json").read_text())
    ref_f32 = np.fromfile(TPL / "fsaverage.eigenmodes/hodge.f32", dtype=np.float32, count=K * j["V"]).reshape(K, -1).astype(np.float64)
    for ref, mu, kind in ((ref_zarr, mu_z, kind_z), (ref_f32, np.array(j["mu"][:K]), np.array(j["kind"][:K]))):
        assert _same_up_to_sign(P, ref).max() < 1e-5
        np.testing.assert_allclose(pk["mu"][:K], mu, rtol=1e-5)
        assert (pk["kind"][:K] == kind).all()
