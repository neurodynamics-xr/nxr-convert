"""Several sessions in one subject store (register D121) and relinking a moved raw
link (D122).

Brainstorm reuses a file stem in every condition — the channel file, the head model
and the kernels are named the same in each — and every one of those nodes used to be
written at the same path, so the second session silently overwrote the first: its
kernels, its head model, and (with a different channel count) the channel manifold
the first session's recording points at. Found converting a research-cohort subject:
three conditions, noise 295 channels, rest runs 328.
"""
import json
import shutil
from pathlib import Path

import pytest

from eeg_fixture import make_eeg_protocol
from nxr_convert.cli import main
from nxr_convert.crud import open_dataset
from nxr_convert.db import text_parity
from nxr_convert.naming import claim_node_name, session_label

NOISE = "sub-S01_ses-02_task-noise_meg"
RUN1 = "sub-S01_ses-02_task-rest_run-01_meg"
RUN2 = "sub-S01_ses-02_task-rest_run-02_meg"


def _dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "store" / "d"
    if not (folder / "dataset.sqlite").exists():
        assert main(["dataset", "create", str(folder.parent), "d"]) == 0
    return folder


def _convert(root: Path, cond: str, folder: Path) -> int:
    return main(["convert", str(root), "--subject", "S01", "--condition", cond, "--dataset", str(folder)])


def _row(folder: Path, table: str, path: str) -> dict | None:
    with open_dataset(folder, lock=False) as ds:                           # a reader: no writer lock (#26)
        s = ds.subject("S01")
        return ds.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND path = ?", s["id"], path) if s else None


def _protocol(tmp_path: Path, conds: dict[str, int]) -> Path:
    root = tmp_path / "proto"
    for cond, n_chan in conds.items():
        make_eeg_protocol(root, subject="S01", condition=cond, n_chan=n_chan)
    return root


def test_session_label_is_the_bids_part_that_differs():
    assert session_label(RUN2) == "ses-02_task-rest_run-02"
    assert session_label(NOISE) == "ses-02_task-noise"
    assert session_label("PET") == "PET"


def test_a_second_session_never_overwrites_the_first(tmp_path):
    root = _protocol(tmp_path, {NOISE: 6, RUN1: 8})
    folder = _dataset(tmp_path)
    assert _convert(root, NOISE, folder) == 0
    assert _convert(root, RUN1, folder) == 0
    # the noise run keeps the bare stems; the rest run's differ, so they are qualified
    noise_chan = _row(folder, "manifold", "timeseries/channel")
    run_chan = _row(folder, "manifold", "timeseries/channel__ses-02_task-rest_run-01")
    assert (noise_chan["n_vertices"], noise_chan["session"]) == (6, NOISE)
    assert (run_chan["n_vertices"], run_chan["session"]) == (8, RUN1)
    # each recording is on ITS channels (the first factor of its product)
    with open_dataset(folder) as ds:
        for rec, chan in ((NOISE, noise_chan), (RUN1, run_chan)):
            f = _row(folder, "field", f"timeseries/{rec}")
            g = ds.db.read("manifold", f["manifold_id"])["geometry_id"]
            assert ds.db.value("SELECT manifold_id FROM geometry_dimension WHERE geometry_id = ? AND ordinal = 0", g) == chan["id"]
    # each session's operators are its own, and point at its own channels; the kernel INVERTS its leadfield (D106)
    k1 = _row(folder, "operator", "MN_EEG_KERNEL_000000_0000")
    k2 = _row(folder, "operator", "MN_EEG_KERNEL_000000_0000__ses-02_task-rest_run-01")
    assert (k1["session"], k2["session"]) == (NOISE, RUN1)
    assert k1["from_manifold_id"] == noise_chan["id"] and k2["from_manifold_id"] == run_chan["id"]
    h2 = _row(folder, "operator", "headmodel_surf_openmeeg__ses-02_task-rest_run-01")
    assert k2["derived_from_id"] == h2["id"] and h2["to_manifold_id"] == run_chan["id"]


def test_identical_sources_are_shared_not_duplicated(tmp_path):
    root = _protocol(tmp_path, {RUN1: 8, RUN2: 8})     # byte-identical channel files and kernels
    folder = _dataset(tmp_path)
    assert _convert(root, RUN1, folder) == 0
    assert _convert(root, RUN2, folder) == 0
    # the channel file and head model are byte-identical: shared
    assert _row(folder, "manifold", "timeseries/channel__ses-02_task-rest_run-02") is None
    assert _row(folder, "operator", "headmodel_surf_openmeeg__ses-02_task-rest_run-02") is None
    # the kernel is NOT: Brainstorm writes its own condition into HeadModelFile, so it is that session's operator
    k2 = _row(folder, "operator", "MN_EEG_KERNEL_000000_0000__ses-02_task-rest_run-02")
    assert k2["session"] == RUN2
    assert k2["derived_from_id"] == _row(folder, "operator", "headmodel_surf_openmeeg")["id"]
    chan = _row(folder, "manifold", "timeseries/channel")
    assert chan["session"] == RUN1                  # the node stays the first session's


def test_a_session_that_fails_leaves_the_others_whole(tmp_path):
    root = _protocol(tmp_path, {RUN1: 8, RUN2: 6})
    folder = _dataset(tmp_path)
    assert _convert(root, RUN1, folder) == 0
    (root / "data" / "S01" / f"@raw{RUN2}" / f"{RUN2}.bst").unlink()
    assert _convert(root, RUN2, folder) != 0
    with open_dataset(folder) as ds:
        s = ds.subject("S01")
        assert ds.db.value("SELECT count(*) FROM field WHERE subject_id = ? AND session = ?", s["id"], RUN2) == 0
        assert ds.db.value("SELECT count(*) FROM manifold WHERE subject_id = ? AND session = ?", s["id"], RUN2) == 0
        assert not (folder / "S01.nxr.zarr" / "timeseries" / "channel__ses-02_task-rest_run-02").exists()
        for node in ("timeseries/channel", f"timeseries/{RUN1}"):
            assert (folder / "S01.nxr.zarr" / node / "zarr.json").exists(), node
        assert text_parity(ds.db, ds.root)["mismatched"] == 0


def test_claim_is_deterministic_and_numbers_a_third_source(sub):
    for path, sha in (("x", "a"), ("x__run-02", "b")):
        sub.manifold(name=path, path=path, type="points", n_vertices=1, topology={"kind": "none", "rank": 0},
                     producer_json=json.dumps({"source_sha1": sha}))
    assert claim_node_name(sub, "", "x", "task-y_run-02", "a") == ("x", True)
    assert claim_node_name(sub, "", "x", "run-02", "b") == ("x__run-02", True)
    assert claim_node_name(sub, "", "x", "run-02", "c") == ("x__run-02_2", False)


def test_a_moved_raw_link_is_relinked_beside_itself(tmp_path, capsys):
    """The link records its .bst by absolute path; after the protocol moves, that path
    is gone. Brainstorm relinks to the file beside the link — so does the converter."""
    root = _protocol(tmp_path, {RUN1: 8})
    moved = tmp_path / "moved"
    shutil.move(str(root), str(moved))             # the recorded absolute path now dangles
    folder = _dataset(tmp_path)
    assert _convert(moved, RUN1, folder) == 0
    out = capsys.readouterr()
    assert '"stage": "relinked"' in (out.out + out.err)
    assert str(moved) in _row(folder, "field", f"timeseries/{RUN1}")["producer_json"]


def test_a_raw_link_whose_binary_is_nowhere_says_where_it_looked(tmp_path, capsys):
    root = _protocol(tmp_path, {RUN1: 8})
    (root / "data" / "S01" / f"@raw{RUN1}" / f"{RUN1}.bst").unlink()
    folder = _dataset(tmp_path)
    assert _convert(root, RUN1, folder) != 0
    out = capsys.readouterr()
    assert "neither at the recorded path" in (out.out + out.err)
    assert _row(folder, "field", f"timeseries/{RUN1}") is None
