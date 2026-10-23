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
pytestmark = pytest.mark.skipif(not (NSP and DS and SUB), reason="no NXR_TEST_HODGE_* (a real nsp output and a scratch dataset)")


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
