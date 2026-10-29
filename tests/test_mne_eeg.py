"""``nxr-convert mne-eeg`` on a synthetic package: a tiny template (two icospheres as the white surfaces, MNE's ico-2 source
space on them), a few minutes of EEG with stage and artefact annotations, three random kernels and a band-power file."""
import json

import numpy as np
import pytest
import zarr

mne = pytest.importorskip("mne")

from atlas_meshes import icosphere  # noqa: E402
from test_surfaces import _write_annot, _write_surf  # noqa: E402

from nxr_convert.cli import main  # noqa: E402
from nxr_convert.crud import Subject, open_dataset  # noqa: E402
from nxr_convert.mne_eeg import package_files, recording_name  # noqa: E402

FS, SECONDS = 100.0, 240.0
NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]
SHIFT = np.array([0.0, 0.01, -0.02])                     # the head → MRI transform: a translation


def _template(root):
    fs = root / "fsavg"
    for d in ("surf", "label", "bem"):
        (fs / d).mkdir(parents=True)
    v, f = icosphere(3, radius=100.0)                    # mm, as FreeSurfer writes
    for k, h in enumerate(("lh", "rh")):
        white = v * 0.6 + [(-1) ** (k + 1) * 35.0, 0, 0]
        _write_surf(fs / "surf" / f"{h}.white", white, f)
        for s in ("sphere", "sphere.reg"):
            _write_surf(fs / "surf" / f"{h}.{s}", v, f)
        _write_annot(fs / "label" / f"{h}.aparc.annot", (v[:, 2] > 0).astype(int), [("lower", 25, 100, 40), ("upper", 255, 192, 32)])
    src = mne.setup_source_space(fs.name, spacing="ico2", subjects_dir=fs.parent, add_dist=False, verbose="ERROR")
    mne.write_source_spaces(fs / "bem" / "fsaverage-ico-5-src.fif", src, verbose="ERROR")
    return fs, src


def _package(root, src, *, bads=("Oz",)):
    out = root / "pkg"
    out.mkdir()
    rng = np.random.default_rng(0)
    n = int(FS * SECONDS)
    x = (rng.standard_normal((len(NAMES), n)) * 1e-5).astype(np.float32)
    info = mne.create_info(NAMES, FS, "eeg")
    raw = mne.io.RawArray(x.astype(np.float64), info, verbose="ERROR")
    pos = {c: np.array([np.cos(k), np.sin(k), 0.5]) * 0.09 for k, c in enumerate(NAMES)}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"))
    raw.info["bads"] = list(bads)
    stages = ["W", "N1", "N2", "N2", "N3", "R", "R", "W"]
    on = [30.0 * k for k in range(len(stages))]
    ann = mne.Annotations(on + [60.0], [30.0] * len(stages) + [30.0], [f"stage/{s}" for s in stages] + ["BAD_artifact"],
                          ch_names=[()] * len(stages) + [("Fz", "C3")])
    raw.set_annotations(ann)
    raw.set_eeg_reference("average", projection=True, verbose="ERROR").apply_proj(verbose="ERROR")
    raw.save(out / "sub-01_task-sleep_desc-preproc_eeg.fif", fmt="single", verbose="ERROR")
    trans = mne.transforms.Transform("head", "mri", np.r_[np.c_[np.eye(3), SHIFT], [[0, 0, 0, 1]]])
    mne.write_trans(out / "sub-01_eeg-trans.fif", trans)
    good = [c for c in NAMES if c not in bads]
    verts = [s["vertno"].tolist() for s in src]
    n_src = sum(len(v) for v in verts)
    K = {}
    for m in ("MNE", "dSPM", "eLORETA"):
        K[m] = rng.standard_normal((n_src, len(good))).astype(np.float32)
        np.save(out / f"sub-01_kernel-{m}_ori-normal.npy", K[m])
    (out / "sub-01_kernel.json").write_text(json.dumps({
        "channels": good, "bads": list(bads), "vertices": {"lh": verts[0], "rh": verts[1]}, "orientation": "normal", "lambda2": 1 / 9,
        "methods": ["MNE", "dSPM", "eLORETA"], "units": {"MNE": "A·m per V"}, "inverse": {"loose": 1.0, "depth": 0.8},
        "noise_covariance": "ad hoc diagonal"}))
    sp = rng.random((5, 2, n_src)).astype(np.float32)
    sp[1] = np.nan                                                                     # N1: too little clean signal
    np.savez_compressed(out / "sub-01_stage_bandpower.npz", source_power=sp, source_relative=sp / np.nansum(sp, axis=1, keepdims=True),
                        sensor_power=rng.random((5, 2, len(good))).astype(np.float32), stages=np.array(["W", "N1", "N2", "N3", "R"]),
                        bands=np.array(["delta", "alpha"]), band_edges=np.array([[0.5, 4.0], [8.0, 12.0]]), channels=np.array(good))
    (out / "sub-01_mne_eeg.json").write_text(json.dumps({"mne": mne.__version__, "preprocess": {"reference": "average (applied)"}}))
    return out, raw, K, sp, pos, good


@pytest.fixture
def converted(tmp_path):
    fs, src = _template(tmp_path)
    pkg, raw, K, sp, pos, good = _package(tmp_path, src)
    assert main(["dataset", "create", str(tmp_path / "store"), "eeg"]) == 0
    assert main(["mne-eeg", "--dataset", str(tmp_path / "store" / "eeg"), "--subject", "sub-01", "--from", str(pkg), "--fsaverage", str(fs)]) == 0
    ds = open_dataset(tmp_path / "store" / "eeg", lock=False)
    yield {"ds": ds, "sub": Subject.open(ds, "sub-01"), "src": src, "raw": raw, "K": K, "sp": sp, "pos": pos, "good": good,
           "pkg": pkg, "fs": fs, "store": tmp_path / "store" / "eeg"}
    ds.close()


def read(sub, path):
    return np.asarray(zarr.open_array(str(sub.at(path)), mode="r")[...])


def test_the_cortex_is_the_source_space_on_the_white_surface(converted):
    sub, src = converted["sub"], converted["src"]
    cx = sub.find("manifold", "sources/cortex_white_ico5")
    n = sum(s["nuse"] for s in src)
    assert cx["is_primary"] == 1 and cx["n_vertices"] == n and cx["n_faces"] == sum(len(s["use_tris"]) for s in src)
    white = np.concatenate([s["rr"][s["vertno"]] for s in src])                       # MNE's own read of the same surfaces, m
    assert np.allclose(read(sub, "sources/cortex_white_ico5/vertices"), white, atol=1e-6)
    assert sub.find("manifold", "sources/cortex_white_ico5_sphere") is not None
    assert sub.find("selection", "sources/cortex_white_ico5_atlases/Desikan-Killiany") is not None


def test_the_kernels_map_the_good_channels_onto_that_cortex(converted):
    sub, db = converted["sub"], converted["sub"].db
    cx = sub.find("manifold", "sources/cortex_white_ico5")
    ch = sub.find("manifold", "timeseries/channel_eeg")
    for m, K in converted["K"].items():
        op = sub.find("operator", f"kernel-{m}_ori-normal")
        assert op["kind"] == "inverse kernel" and op["to_manifold_id"] == cx["id"] and op["from_manifold_id"] == ch["id"]
        assert np.array_equal(read(sub, op["path"]), K)
        assert np.array_equal(read(sub, f"kernel-{m}_ori-normal_channels"), [NAMES.index(c) for c in converted["good"]])
        assert json.loads(op["params_json"])["inverse"]["measure"] == m
        assert json.loads(op["producer_json"])["method"] == m
    assert db.value("SELECT count(*) FROM operator WHERE subject_id = ? AND kind = 'noise covariance'", sub.id) == 0   # no -inv.fif


def test_the_recording_its_channels_and_its_events(converted):
    sub, raw = converted["sub"], converted["raw"]
    rec = "timeseries/sub-01_task-sleep_desc-preproc_eeg"
    assert np.array_equal(read(sub, rec), raw.get_data().astype(np.float32))
    pos = read(sub, "timeseries/channel_eeg")
    assert np.allclose(pos, [converted["pos"][c] + SHIFT for c in NAMES], atol=1e-6)        # head → MRI
    flags = sub.find("selection", f"{rec}_flags")
    assert flags is not None
    ev = sub.find("selection", f"{rec}_events")
    assert ev["n_members"] == 9
    assert list(json.loads(ev["params_json"])["channels"].values()) == [[NAMES.index("Fz"), NAMES.index("C3")]]   # the mark's channels
    bad = sub.db.all("SELECT start, stop FROM selection_span WHERE selection_id = ?", sub.find("selection", f"{rec}_bad_segments")["id"])
    assert [(b["start"], b["stop"]) for b in bad] == [(60.0, 90.0)]


def test_band_power_per_stage_on_cortex_and_bands(converted):
    sub, sp = converted["sub"], converted["sp"]
    names = {r["name"] for r in sub.db.all("SELECT name FROM field WHERE subject_id = ? AND path LIKE 'frequency/%'", sub.id)}
    assert names == {f"{p}_{s}{r}" for s in ("W", "N2", "N3", "R") for p, r in (("bandpower", ""), ("bandpower", "_relative"), ("sensorpower", ""))}
    assert np.allclose(read(sub, "frequency/bandpower_N2"), sp[2].T)
    sensor = read(sub, "frequency/sensorpower_W")
    assert sensor.shape == (len(NAMES), 2) and np.isnan(sensor[NAMES.index("Oz")]).all()


def test_a_session_already_imported_is_refused(converted, capsys):
    converted["ds"].close()
    capsys.readouterr()
    assert main(["mne-eeg", "--dataset", str(converted["store"]), "--subject", "sub-01", "--from", str(converted["pkg"]),
                 "--fsaverage", str(converted["fs"])]) != 0
    assert "already exists" in capsys.readouterr().out
    converted["ds"] = open_dataset(converted["store"], lock=False)


def test_the_eeg_file_is_the_first_part(tmp_path):
    for n in ("sub-01_task-sleep_desc-preproc_eeg.fif", "sub-01_task-sleep_desc-preproc_eeg-1.fif", "sub-01_kernel.json"):
        (tmp_path / n).write_text("")
    assert package_files(tmp_path, "sub-01")["eeg"].name == "sub-01_task-sleep_desc-preproc_eeg.fif"
    for n in ("sub-01_task-sleep_desc-preproc_eeg.fif", "sub-01_task-sleep_desc-preproc_eeg-1.fif"):
        (tmp_path / n).unlink()
    for n in ("sub-01_task-sleep_split-01_desc-preproc_eeg.fif", "sub-01_task-sleep_split-02_desc-preproc_eeg.fif"):
        (tmp_path / n).write_text("")
    f = package_files(tmp_path, "sub-01")["eeg"]
    assert f.name.startswith("sub-01_task-sleep_split-01") and recording_name(f) == "sub-01_task-sleep_desc-preproc_eeg"
