"""The MRI importer — and above all, the gate that refuses a mirrored brain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from conftest import row_at
from nxr_convert.mri import (
    Mri, mri_name, to_scs, verify_scs, write_mri, find_subjectimages,
)

VOX = np.array([1.0, 1.0, 1.0])

# The OMEGA T1's own SCS, at FULL PRECISION, read back out of the store.
#
# It was typed to 8 decimals first, and the third row was reconstructed rather
# than read — which left a matrix that was not orthonormal (R·Rᵀ − I ≈ 1e-5
# against the real 3.3e-16) and made every tight geometric assertion fail for a
# reason that had nothing to do with the code under test. A fixture standing in
# for real data has to BE the real data.
R_OK = np.array([
    [0.019208185928019046, 0.99430713773067, 0.10480630444394416],
    [-0.9994839433121204, 0.01639661322217421, 0.02762242089492196],
    [0.025746701820008933, -0.10528279504593563, 0.9941089680777995],
])
T_OK = np.array([-125.77019365743884, 121.56174120969419, -73.6804796099093])
NAS = np.array([127.80636667578327, 218.54912887930732, 93.95282945719043])
LPA = np.array([50.727717761122406, 117.71801660858351, 85.27042077331473])
RPA = np.array([200.8959923640762, 112.88941055236529, 80.869789583697])
ORIGIN = np.array([125.8118550625993, 115.30371358047441, 83.07010517850586])


def scs_ok() -> dict:
    return {"R": R_OK, "T": T_OK, "NAS": NAS, "LPA": LPA, "RPA": RPA, "Origin": ORIGIN}


class TestConvention:
    """The convention, pinned by the numbers it was measured from."""

    def test_origin_maps_to_the_scs_origin(self):
        assert np.linalg.norm(to_scs(ORIGIN, scs_ok(), VOX)[0]) < 1e-2

    def test_nas_is_anterior_lpa_is_left_rpa_is_right(self):
        nas, lpa, rpa = (to_scs(p, scs_ok(), VOX)[0] for p in (NAS, LPA, RPA))
        assert nas[0] > 100          # +x anterior
        assert lpa[1] > 70           # +y LEFT
        assert rpa[1] < -70          # -y right

    def test_the_fiducials_define_the_z_zero_plane(self):
        for p in (NAS, LPA, RPA):
            assert abs(to_scs(p, scs_ok(), VOX)[0][2]) < 1e-2


class TestVerifyScs:
    def test_a_correct_registration_passes(self):
        verify_scs(scs_ok(), VOX)

    def test_A_MIRRORED_VOLUME_IS_REFUSED(self):
        """THE gate. Flipping y swaps left and right and changes nothing else —
        the render stays a perfectly plausible brain, so nothing downstream and
        no reviewer can catch it."""
        scs = scs_ok()
        flip = np.diag([1.0, -1.0, 1.0])
        scs["R"] = flip @ scs["R"]
        scs["T"] = flip @ scs["T"]
        with pytest.raises(ValueError, match="LEFT-RIGHT ORIENTATION IS WRONG"):
            verify_scs(scs, VOX)

    def test_a_swapped_lpa_rpa_is_refused(self):
        scs = scs_ok()
        scs["LPA"], scs["RPA"] = scs["RPA"], scs["LPA"]
        with pytest.raises(ValueError, match="LEFT-RIGHT"):
            verify_scs(scs, VOX)

    def test_missing_fiducials_are_fatal(self):
        scs = scs_ok()
        del scs["LPA"]
        with pytest.raises(ValueError, match="missing"):
            verify_scs(scs, VOX)

    def test_a_missing_origin_is_NOT_fatal(self):
        """Brainstorm omits `Origin` on a volume atlas sharing the T1's grid. It
        is derivable, and the gate that matters needs only the fiducials."""
        scs = scs_ok()
        del scs["Origin"]
        verify_scs(scs, VOX)

    def test_a_wrong_origin_is_fatal_when_present(self):
        scs = scs_ok()
        scs["Origin"] = scs["Origin"] + 5
        with pytest.raises(ValueError, match="origin"):
            verify_scs(scs, VOX)

    def test_a_tilted_fiducial_plane_is_refused(self):
        scs = scs_ok()
        scs["NAS"] = scs["NAS"] + np.array([0.0, 0.0, 4.0])
        with pytest.raises(ValueError, match="z=0 plane"):
            verify_scs(scs, VOX)


class TestName:
    def test_a_timestamp_only_file_is_named_by_its_comment(self):
        assert mri_name("subjectimage_260928_1516.mat", "PET 18FNAV4694_suvr") == "PET_18FNAV4694_suvr"
        assert mri_name("subjectimage_MRI_T1.mat", "anything") == "T1"          # a real stem wins
        assert mri_name("subjectimage_260928_1516.mat") == "260928_1516"        # no comment: the stem

    @pytest.mark.parametrize("filename,expected", [
        ("subjectimage_MRI_T1.mat", "T1"),
        ("subjectimage_T1.mat", "T1"),
        ("subjectimage_ASEG_volatlas.mat", "ASEG"),
        ("subjectimage_Desikan-Killiany_volatlas.mat", "Desikan-Killiany"),
        ("subjectimage_tissues_simnibs4.mat", "tissues_simnibs4"),
    ])
    def test_derives_from_the_filename(self, filename, expected):
        # From the FILENAME, not the Comment: a re-import must overwrite the node
        # it wrote last time, and a Comment is editable in the Brainstorm GUI.
        assert mri_name(filename) == expected


def _t1(shape=(8, 8, 8), scs=None):
    return Mri(name="T1", kind="anatomical", cube=np.zeros(shape, np.uint8),
               voxsize=VOX, scs=scs or scs_ok(), ncs={}, comment="MRI T1", source="t1.mat")


class TestWrite:
    def test_the_T1_is_a_field_on_ONE_volume_and_the_frames_are_operators(self, sub):
        assert write_mri(sub, _t1()) == "mri/T1"
        sub.complete()
        vol = row_at(sub, "manifold", "mri/T1_volume")
        dims = sub.db.all("SELECT name, n_vertices, spacing, unit FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", vol["geometry_id"])
        assert (vol["type"], [d["n_vertices"] for d in dims], {d["unit"] for d in dims}, [d["name"] for d in dims]) == \
            ("volume", [8, 8, 8], {"mm"}, ["i", "j", "k"])
        t1 = row_at(sub, "field", "mri/T1")
        assert (t1["manifold_id"], t1["data_type"], t1["comment"], t1["kind"]) == (vol["id"], "uint8", "MRI T1", "T1-weighted")
        # voxel index -> SCS mm as ONE 3x4 affine: R scaled by the voxel size, then T — a TRANSITION between two charts
        a = zarr.open_array(str(sub.at("T1_scs")))[...]
        np.testing.assert_allclose(a[:, :3], R_OK * VOX[None, :])
        op = row_at(sub, "operator", "T1_scs")
        assert (op["kind"], op["from_manifold_id"]) == ("transition", vol["id"])
        cs = {r["id"]: r["name"] for r in sub.db.all("SELECT id, name FROM coordinate_system")}
        assert (cs[op["from_coordinate_system_id"]], cs[op["to_coordinate_system_id"]]) == ("voxel:T1_volume", "scs")
        assert json.loads(op["producer_json"])["nas"] == scs_ok()["NAS"].tolist()

    def test_an_atlas_is_a_partition_ON_the_T1s_volume_read_with_its_dictionary(self, sub):
        write_mri(sub, _t1())
        cube = np.zeros((8, 8, 8), np.uint8)
        cube[1:] = 1
        img = Mri(name="ASEG", kind="atlas", cube=cube, voxsize=VOX, scs=scs_ok(), ncs={}, comment="c", source="f.mat",
                  labels={"ids": np.array([0, 1], np.int32), "source_ids": np.array([0, 3], np.int32), "names": ["Unknown", "Cortex L"],
                          "colors": np.array([[0, 0, 0], [205, 62, 78]], np.uint8)})
        assert write_mri(sub, img) == "mri/ASEG"
        sub.complete()
        np.testing.assert_array_equal(zarr.open_array(str(sub.at("mri/ASEG")))[...], cube)
        a = row_at(sub, "selection", "mri/ASEG")
        assert (a["type"], a["manifold_id"], a["array_path"]) == ("set", row_at(sub, "manifold", "mri/T1_volume")["id"], "mri/ASEG")
        e = sub.db.all("SELECT code, name, color_json, attributes_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", a["dictionary_id"])
        assert [(x["code"], x["name"], json.loads(x["attributes_json"])["source_id"]) for x in e] == [(0, "Unknown", 0), (1, "Cortex L", 3)]
        assert sub.db.value("SELECT scope FROM dictionary WHERE id = ?", a["dictionary_id"]) == "app"
        assert row_at(sub, "manifold", "mri/ASEG_volume") is None, "one Volume (decision 2)"

    def test_an_atlas_before_the_T1_is_refused(self, sub):
        img = Mri(name="ASEG", kind="atlas", cube=np.zeros((4, 4, 4), np.uint8), voxsize=VOX,
                  scs=scs_ok(), ncs={}, comment="", source="f.mat",
                  labels={"ids": np.array([0], np.int32), "names": ["Unknown"], "colors": np.zeros((1, 3), np.uint8)})
        with pytest.raises(ValueError, match="import the T1 first"):
            write_mri(sub, img)

    def test_an_atlas_on_another_grid_is_refused(self, sub):
        write_mri(sub, _t1((8, 8, 8)))
        img = Mri(name="ASEG", kind="atlas", cube=np.zeros((4, 4, 4), np.uint8), voxsize=VOX,
                  scs=scs_ok(), ncs={}, comment="", source="f.mat",
                  labels={"ids": np.array([0], np.int32), "names": ["Unknown"], "colors": np.zeros((1, 3), np.uint8)})
        with pytest.raises(ValueError, match="one Volume"):
            write_mri(sub, img)

    def test_chunks_are_cubic_not_slabs(self, sub):
        """A slab chunking along axis 0 makes a SAGITTAL slice read the whole
        volume. Cubic chunks make any orthogonal slice touch one plane of them."""
        write_mri(sub, _t1((200, 200, 200)))
        assert zarr.open_array(str(sub.at("mri/T1"))).chunks == (64, 64, 64)
        assert json.loads(row_at(sub, "field", "mri/T1")["chunk_shape_json"]) == [64, 64, 64]

    def test_EVERY_CHUNK_IS_ON_DISK_including_the_empty_ones(self, sub):
        """A cube of mostly air must still write all 27 of its chunks — a missing
        chunk reaches the renderer as a console 404 it can neither predict nor suppress."""
        cube = np.zeros((192, 192, 192), np.uint8)
        cube[0, 0, 0] = 255
        write_mri(sub, Mri(name="T1", kind="anatomical", cube=cube, voxsize=VOX, scs=scs_ok(), ncs={}, comment="", source="f.mat"))
        c = sub.at("mri/T1") / "c"
        written = {p for p in c.rglob("*") if p.is_file()}
        assert len(written) == 27, f"3x3x3 grid, got {len(written)} chunk files"
        np.testing.assert_array_equal(zarr.open_array(str(sub.at("mri/T1")))[...], cube)


class TestDiscovery:
    def test_anatomical_volumes_come_before_atlases(self, tmp_path):
        for n in ["subjectimage_ASEG_volatlas.mat", "subjectimage_MRI_T1.mat",
                  "._subjectimage_MRI_T1.mat"]:
            (tmp_path / n).write_bytes(b"")
        # AppleDouble sidecars are not subject images, and the T1 leads.
        assert [p.name for p in find_subjectimages(tmp_path)] == [
            "subjectimage_MRI_T1.mat", "subjectimage_ASEG_volatlas.mat"]
