"""``nxr-convert subject`` — the whole Brainstorm subject as ONE composition into its dataset (D121–D126, D141–D143):
sessions, volumes, a PET cortical map, fibres, their ends on the cortex and the connectome Brainstorm assigned — every
node a row of the dataset's database, every zarr.json the view's text.

The protocol is the EEG fixture grown into what a real research-cohort subject holds: two
sessions whose files share stems, an atlas on the cortex, a T1 + uint16 atlas + float
PET volume, a map-only ``PET`` condition, and a fibres file with an assignment.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.io import savemat

from eeg_fixture import _surface, make_eeg_protocol
from nxr_convert.cli import main
from nxr_convert.crud import open_dataset
from nxr_convert.db import text_parity
from test_volumes import SHAPE, _labels, _write

S = "S01"
RUN1 = "sub-S01_task-rest_run-01"
RUN2 = "sub-S01_task-rest_run-02"


def _scouts(rows):
    return np.array([(np.asarray(v, float), lab, np.asarray(c, float)) for v, lab, c in rows],
                    dtype=[("Vertices", "O"), ("Label", "O"), ("Color", "O")])


@pytest.fixture
def protocol(tmp_path):
    root = tmp_path / "proto"
    make_eeg_protocol(root, subject=S, condition=RUN1, n_chan=8)
    make_eeg_protocol(root, subject=S, condition=RUN2, n_chan=6)
    anat = root / "anat" / S
    v, f = _surface()
    # the cortex the kernels (and the fibres) sit on, with a 2-region atlas
    atlas = np.array([("Desikan-Killiany", _scouts([([1, 2, 3], "frontal L", [1, 0, 0]),
                                                    ([4, 5, 6], "occipital R", [0, 0, 1])]))],
                     dtype=[("Name", "O"), ("Scouts", "O")])
    savemat(anat / "tess_cortex_mid_low.mat", {"Vertices": v, "Faces": f, "Comment": "mid_6V", "Atlas": atlas})
    # volumes: T1, a uint16 atlas, a float PET SUVR
    _write(anat / "subjectimage_MRI_T1.mat", np.zeros(SHAPE, np.uint8), comment="MRI T1")
    dk = np.zeros(SHAPE, np.uint16)
    dk[1, 1, 1] = 1035
    _write(anat / "subjectimage_Desikan-Killiany_volatlas.mat", dk,
           labels=_labels([(0, "Unknown", [0, 0, 0]), (1035, "insula L", [255, 192, 32])]))
    # named only by its creation time, as a scripted Brainstorm save is — its Comment is its name
    _write(anat / "subjectimage_260928_1516.mat", np.full(SHAPE, 1.25, np.float32), comment="PET trc_suvr")
    # a map-only PET condition: a static SUVR on the cortex (Brainstorm's two identical columns)
    pet = root / "data" / S / "PET"
    pet.mkdir(parents=True)
    amp = np.arange(6, dtype=float)[:, None].repeat(2, axis=1)
    # the REAL file's shape: no DataType field (a research cohort's projected PET has none), an
    # empty ImagingKernel, two identical time columns
    savemat(pet / "results_surface_PET_trc_suvr_260928.mat", {
        "ImagingKernel": np.zeros((0, 0)), "ImageGridAmp": amp, "Time": [0.0, 1.0], "Comment": "PET trc SUVR",
        "SurfaceFile": f"{S}/tess_cortex_mid_low.mat", "HeadModelType": "surface", "nComponents": 1.0})
    # fibres: 3 fibres x 4 points; ends placed on known octahedron vertices
    starts, ends = v[[0, 0, 3]], v[[4, 1, 5]]
    pts = np.stack([np.linspace(a, b, 4) for a, b in zip(starts, ends)])
    fib_scouts = np.array([(f"{S}_Desikan-Killiany", np.array([[1, 2], [1, 1], [0, 2]], float))],
                          dtype=[("ConnectFile", "O"), ("Assignment", "O")])
    savemat(anat / "tess_fibers_streamlines.mat", {"Comment": "fibers_4Pt_3Fib", "Points": pts,
                                                   "Colors": np.zeros((3, 4, 3), np.uint8), "Scouts": fib_scouts})
    return root


def dataset(tmp_path):
    root = tmp_path / "store"
    assert main(["dataset", "create", str(root), "cohort", "--source-tool", "brainstorm"]) == 0
    return root / "cohort"


def row(db, sub, table, path):
    return db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND path = ?", sub["id"], path)


def test_the_whole_subject_converts_into_one_composition(protocol, tmp_path):
    folder = dataset(tmp_path)
    assert main(["subject", str(protocol), "--subject", S, "--dataset", str(folder)]) == 0
    with open_dataset(folder) as ds:
        db = ds.db
        sub = ds.subject(S)
        assert sub["status"] == "complete" and sub["path"] == f"{S}.nxr.zarr"
        assert db.value("SELECT count(*) FROM manifold WHERE status <> 'complete'") == 0
        store = folder / sub["path"]
        # two sessions, their own channels (8 vs 6) and kernels
        assert row(db, sub, "field", f"timeseries/{RUN1}")["session"] == RUN1
        assert row(db, sub, "manifold", "timeseries/channel")["n_vertices"] == 8
        assert row(db, sub, "manifold", "timeseries/channel__task-rest_run-02")["n_vertices"] == 6
        k2 = row(db, sub, "operator", "MN_EEG_KERNEL_000000_0000__task-rest_run-02")
        assert k2["kind"] == "inverse kernel" and json.loads(k2["params_json"])["inverse"]["measure"] == "MNE"
        # the recording is on the PRODUCT channels × its time Line (D37), its chunk grid and envelope rows beside it
        rec = row(db, sub, "field", f"timeseries/{RUN1}")
        prod = db.read("manifold", rec["manifold_id"])
        assert prod["type"] == "product" and prod["path"] is None
        env = db.one("SELECT * FROM selection WHERE of_field_id = ? AND name LIKE '%_envelope'", rec["id"])
        assert [m["measure"] for m in db.all("SELECT measure FROM selection_measurement WHERE selection_id = ? ORDER BY component", env["id"])] == ["min", "max"]
        assert db.one("SELECT 1 FROM selection WHERE of_field_id = ? AND name LIKE '%_chunks'", rec["id"])
        # volumes: T1, atlas as codes (its dictionary keeps Brainstorm's value), PET as a field on the T1 volume
        dk = row(db, sub, "selection", "mri/Desikan-Killiany")
        code = int(zarr.open_array(str(store / "mri/Desikan-Killiany"))[1, 1, 1])
        e = db.one("SELECT * FROM dictionary_entry WHERE dictionary_id = ? AND code = ?", dk["dictionary_id"], code)
        # ONE app-wide 'Desikan-Killiany' (the cortex's atlas made it first): the volume's labels joined it, by name
        assert e["name"] == "insula L" and json.loads(e["attributes_json"]) == {"source_id": 1035}
        assert row(db, sub, "field", "mri/PET_trc_suvr")["unit"] == "SUVR"
        assert not (store / "mri" / "260928_1516").exists()
        # the PET cortical map: one value per vertex, on its surface, in the PET session
        m = row(db, sub, "field", "sources/surface_PET_trc_suvr_260928")
        assert m["manifold_id"] == row(db, sub, "manifold", "sources/cortex_mid_low")["id"]
        assert (m["session"], m["unit"]) == ("PET", "SUVR")
        np.testing.assert_array_equal(zarr.open_array(str(store / m["path"]))[...], np.arange(6, dtype=np.float32))
        # fibres: a curve manifold (a path topology of 3 components), points fibre-major
        fib = row(db, sub, "manifold", "sources/fibers_streamlines")
        top = db.read("topology", fib["topology_id"])
        assert (fib["type"], top["kind"], top["n_components"], fib["n_vertices"]) == ("curve", "path", 3, 12)
        assert zarr.open_array(str(store / "sources/fibers_streamlines/points")).shape == (12, 3)
        # the ends operator: fibre 0 runs vertex 0 -> 4, fibre 1 0 -> 1, fibre 2 3 -> 5
        op = row(db, sub, "operator", "fibers_streamlines_ends")
        assert (op["layout"], op["n_rows"], op["n_cols"], op["nnz"]) == ("csr", 6, 12, 6)
        indptr = zarr.open_array(str(store / "fibers_streamlines_ends/indptr"))[...]
        indices = zarr.open_array(str(store / "fibers_streamlines_ends/indices"))[...]
        per_vertex = {r: sorted(indices[indptr[r]:indptr[r + 1]].tolist()) for r in range(6)}
        assert per_vertex == {0: [0, 4], 1: [7], 2: [], 3: [8], 4: [3], 5: [11]}
        # the connectome = the hand tally on the PRODUCT regions × regions
        C = row(db, sub, "field", "sources/fibers_streamlines_connectome_Desikan-Killiany")
        np.testing.assert_array_equal(zarr.open_array(str(store / C["path"]))[...], [[0, 0, 1], [0, 1, 1], [1, 1, 0]])
        reg = row(db, sub, "manifold", "sources/cortex_mid_low_Desikan-Killiany_regions")
        assert reg["n_vertices"] == 3 and reg["partition_id"] == row(db, sub, "selection", "sources/cortex_mid_low_atlases/Desikan-Killiany")["id"]
        assert [d["manifold_id"] for d in db.all("SELECT manifold_id FROM geometry_dimension WHERE geometry_id = ?",
                                                 db.read("manifold", C["manifold_id"])["geometry_id"])] == [reg["id"], reg["id"]]
        # the store is the image of the rows
        assert text_parity(db, ds.root)["mismatched"] == 0
    cat = json.loads((folder.parent / "catalog.json").read_text())
    assert cat["datasets"][0]["subjects"] == 1


def test_a_failure_leaves_no_subject(protocol, tmp_path):
    folder = dataset(tmp_path)
    (protocol / "anat" / S / "tess_fibers_streamlines.mat").write_bytes(b"not a mat file")
    assert main(["subject", str(protocol), "--subject", S, "--dataset", str(folder)]) == 1
    assert not (folder / f"{S}.nxr.zarr").exists()
    with open_dataset(folder) as ds:
        assert ds.subject(S) is None and ds.db.value("SELECT count(*) FROM manifold") == 0


def test_an_existing_subject_is_refused(protocol, tmp_path):
    folder = dataset(tmp_path)
    assert main(["subject", str(protocol), "--subject", S, "--dataset", str(folder)]) == 0
    assert main(["subject", str(protocol), "--subject", S, "--dataset", str(folder)]) == 1


def test_a_subject_and_a_node_are_deleted_by_the_procedure(protocol, tmp_path):
    folder = dataset(tmp_path)
    assert main(["subject", str(protocol), "--subject", S, "--dataset", str(folder)]) == 0
    assert main(["remove", "--dataset", str(folder), "--subject", S, "--node", "sources/fibers_streamlines_connectome_Desikan-Killiany"]) == 0
    assert not (folder / f"{S}.nxr.zarr" / "sources/fibers_streamlines_connectome_Desikan-Killiany").exists()
    assert main(["remove", "--dataset", str(folder), "--subject", S]) == 0
    assert not (folder / f"{S}.nxr.zarr").exists()
    with open_dataset(folder) as ds:
        assert ds.db.value("SELECT count(*) FROM subject") == 0 and ds.db.value("SELECT count(*) FROM field") == 0
