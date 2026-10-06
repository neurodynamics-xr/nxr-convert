"""Atlases of any integer width, written as CODES (register D123), and float maps as
Fields on the anatomical volume (D124).

Found on a research-cohort subject: DKT, Desikan-Killiany and Destrieux come as uint16
(label values up to ~15000) and were skipped ("only uint8 is supported"); the PET
mean/SUVR volumes are float32 and were skipped too. And even ASEG's voxels were
Brainstorm label VALUES while the database numbers a dictionary 0..N-1 — two
readers disagreeing about the same voxel.
"""
import numpy as np
import pytest
import zarr
from scipy.io import savemat

from conftest import row_at
from nxr_convert.mri import import_subjectimages, read_subjectimage

from test_mri import LPA, NAS, ORIGIN, R_OK, RPA, T_OK

SHAPE = (8, 8, 8)


def _scs():
    return {"R": R_OK, "T": T_OK, "NAS": NAS, "LPA": LPA, "RPA": RPA, "Origin": ORIGIN}


def _labels(rows):
    cell = np.empty((len(rows), 3), dtype=object)
    for i, (v, name, rgb) in enumerate(rows):
        cell[i, 0] = float(v)
        cell[i, 1] = name
        cell[i, 2] = np.asarray(rgb, dtype=float)
    return cell


def _write(path, cube, *, labels=None, comment=""):
    savemat(path, {"Cube": cube, "Voxsize": [1.0, 1.0, 1.0], "SCS": _scs(), "Comment": comment,
                   "Labels": labels if labels is not None else np.zeros((0, 0))})


@pytest.fixture
def anat(tmp_path):
    d = tmp_path / "anat"
    d.mkdir()
    _write(d / "subjectimage_MRI_T1.mat", np.zeros(SHAPE, np.uint8), comment="MRI T1")
    return d


def test_a_uint16_atlas_is_read_as_codes_with_its_source_values_kept(anat):
    cube = np.zeros(SHAPE, np.uint16)
    cube[0, 0, 0] = 11101          # Destrieux-sized label values
    cube[1, 1, 1] = 12101
    cube[2, 2, 2] = 777            # a value the table does not name
    p = anat / "subjectimage_Destrieux_volatlas.mat"
    _write(p, cube, labels=_labels([(0, "Unknown", [0, 0, 0]),
                                    (11101, "G_and_S_frontomargin L", [23, 220, 60]),
                                    (12101, "G_and_S_frontomargin R", [23, 220, 60])]))
    img = read_subjectimage(p)
    assert img.kind == "atlas" and img.cube.dtype == np.uint8
    lab = img.labels
    assert lab["names"][-1] == "unlabelled 777" and lab["unlabelled"] == [777]
    assert lab["ids"].tolist() == [0, 1, 2, 3]
    assert lab["source_ids"].tolist() == [0, 11101, 12101, 777]
    # voxel code k IS table row k, and row k's source value is what Brainstorm stored
    for ijk, value in (((0, 0, 0), 11101), ((1, 1, 1), 12101), ((2, 2, 2), 777), ((3, 3, 3), 0)):
        assert lab["source_ids"][img.cube[ijk]] == value


def test_a_large_atlas_widens_to_uint16(anat):
    n = 300
    cube = (np.arange(np.prod(SHAPE)) % n).astype(np.uint16).reshape(SHAPE)
    p = anat / "subjectimage_Big_volatlas.mat"
    _write(p, cube, labels=_labels([(v, f"r{v}", [1, 2, 3]) for v in range(n)]))
    img = read_subjectimage(p)
    assert img.cube.dtype == np.uint16 and int(img.cube.max()) == n - 1


def test_the_rows_carry_codes_and_the_dictionary_keeps_brainstorms_values(anat, sub):
    cube = np.zeros(SHAPE, np.uint16)
    cube[4, 4, 4] = 1035
    _write(anat / "subjectimage_Desikan-Killiany_volatlas.mat", cube,
           labels=_labels([(0, "Unknown", [0, 0, 0]), (1035, "insula L", [255, 192, 32])]))
    done = {r["name"]: r for r in import_subjectimages(sub, anat)}
    sub.complete()
    assert set(done) == {"T1", "Desikan-Killiany"}
    sel = row_at(sub, "selection", "mri/Desikan-Killiany")
    assert sel["type"] == "set" and int(zarr.open_array(str(sub.at("mri/Desikan-Killiany")))[4, 4, 4]) == 1
    assert zarr.open_array(str(sub.at("mri/Desikan-Killiany"))).dtype == np.uint8
    e = sub.db.all("SELECT code, attributes_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", sel["dictionary_id"])
    assert [(x["code"], __import__("json").loads(x["attributes_json"])["source_id"]) for x in e] == [(0, 0), (1, 1035)]


def test_a_float_pet_volume_is_a_field_on_the_T1s_volume(anat, sub):
    pet = np.linspace(0, 2.5, np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    _write(anat / "subjectimage_PET_18FNAV4694_suvr_260928.mat", pet, comment="PET 18FNAV4694_suvr")
    done = {r["name"]: r for r in import_subjectimages(sub, anat)}
    sub.complete()
    assert done["PET_18FNAV4694_suvr_260928"]["kind"] == "scalar"
    f = row_at(sub, "field", "mri/PET_18FNAV4694_suvr_260928")
    vol = row_at(sub, "manifold", "mri/T1_volume")
    assert (f["manifold_id"], f["data_type"], f["unit"], f["value_type"]) == (vol["id"], "float32", "SUVR", "scalar")
    np.testing.assert_allclose(zarr.open_array(str(sub.at(f["path"])))[...], pet)
    # ONE volume: the map is not given a second manifold, nor transitions
    assert sub.db.value("SELECT count(*) FROM manifold WHERE type = 'volume'") == 1
    assert sub.db.value("SELECT count(*) FROM operator") == 1          # the T1's scs only (no ncs in the fixture)


def test_a_map_on_another_grid_is_reported_not_written(anat, sub):
    _write(anat / "subjectimage_PET_x.mat", np.ones((4, 4, 4), np.float32), comment="PET x")
    seen = []
    done = import_subjectimages(sub, anat, progress=seen.append)
    assert [r["name"] for r in done] == ["T1"]
    assert any(r["stage"] == "skipped" and "grid" in r["reason"] for r in seen)


def test_an_int16_intensity_volume_is_still_refused(anat):
    p = anat / "subjectimage_CT.mat"
    _write(p, np.zeros(SHAPE, np.int16))
    with pytest.raises(ValueError, match="int16"):
        read_subjectimage(p)
