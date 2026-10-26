"""`nxr-convert carry-inverse` — an inverse kernel carried from another store of the same subject: its channels by name, the cortex checked,
written as a conversion writes one, its method record kept with where it came from."""
import json

import numpy as np
import zarr

from nxr_convert.cli import main
from nxr_convert.crud import open_dataset
from test_subject import RUN1, RUN2, S, dataset, protocol  # noqa: F401  (the fixture: a Brainstorm subject with an MN kernel a session)

K = "MN_EEG_KERNEL_000000_0000"


def _kernel(folder, name):
    """the kernel's row, its matrix and its channels' indices"""
    with open_dataset(folder, lock=False) as ds:
        k = ds.db.one("SELECT * FROM operator WHERE name = ?", name)
        sel = ds.db.read("selection", k["from_selection_id"])
    store = folder / f"{S}.nxr.zarr"
    return k, zarr.open_array(str(store / k["path"]))[...], zarr.open_array(str(store / sel["array_path"]))[...]


def test_a_kernel_is_carried_from_an_earlier_import(protocol, tmp_path):
    a, b = dataset(tmp_path / "a"), dataset(tmp_path / "b")
    for d in (a, b):
        assert main(["subject", str(protocol), "--subject", S, "--dataset", str(d)]) == 0
    carry = ["carry-inverse", "--dataset", str(b), "--subject", S, "--from", str(a), "--kernel", K]
    assert main(carry + ["--session", RUN1, "--name", "MN_carried"]) == 0
    src, W0, c0 = _kernel(a, K)
    new, W1, c1 = _kernel(b, "MN_carried")
    assert new["kind"] == "inverse kernel" and new["session"] == RUN1 and new["params_json"] == src["params_json"]
    assert json.loads(new["producer_json"])["carried_from"] == {"dataset": str(a), "subject": S, "operator": src["id"], "name": K}
    assert (new["n_rows"], new["n_cols"], new["to_value_type"]) == (src["n_rows"], src["n_cols"], src["to_value_type"])
    np.testing.assert_array_equal(W1, W0)                     # the matrix
    np.testing.assert_array_equal(c1, c0)                     # the same channels (the same set here, so the same indices)
    assert main(carry + ["--session", RUN1, "--name", "MN_carried"]) == 1     # present: refused
    assert main(carry + ["--session", RUN2, "--name", "MN_run2"]) == 1        # run 2 has 6 of its 8 channels: refused, by name
