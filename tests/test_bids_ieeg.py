"""``nxr-convert bids-ieeg`` on a synthetic BIDS-iEEG subject: two BrainVision runs (float32 multiplexed, int16 vectorized),
channels with a bad one and one without a contact, MNI electrodes in mm, events — and a template of two icosphere
hemispheres with a two-region parcellation, written as GIFTI."""
import json

import numpy as np
import pytest
import zarr

from atlas_meshes import icosphere

from nxr_convert.bids_ieeg import read_block, read_vhdr
from nxr_convert.cli import main
from nxr_convert.crud import Subject, open_dataset

SPACE = "MNI152NLin2009aSym"
NAMES = ["LA1", "LA2", "LA3", "RB1", "RB2", "ECG"]
R_MM = 60.0                                               # the template hemispheres' radius
CENTRES = {"L": np.array([-40.0, 0, 0]), "R": np.array([40.0, 0, 0])}


def _vhdr(path, names, fs, fmt, orientation, res, unit="µV", comment=True):
    lines = ["Brain Vision Data Exchange Header File Version 1.0", "; a comment line", "[Common Infos]", "Codepage=UTF-8",
             f"DataFile={path.with_suffix('.eeg').name}", f"MarkerFile={path.with_suffix('.vmrk').name}", "DataFormat=BINARY",
             f"DataOrientation={orientation}", f"NumberOfChannels={len(names)}", f"SamplingInterval={1e6 / fs:g}", "",
             "[Binary Infos]", f"BinaryFormat={fmt}", "", "[Channel Infos]"]
    lines += [f"Ch{k + 1}={n.replace(',', chr(92) + '1')},,{res:g},{unit}" for k, n in enumerate(names)]
    if comment:
        lines += ["", "[Comment]", "Ch1=not a channel"]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def _tsv(path, rows):
    keys = list(rows[0])
    path.write_text("\n".join(["\t".join(keys)] + ["\t".join(str(r[k]) for k in keys) for r in rows]) + "\n")


def _subject(bids, rng):
    d = bids / "sub-01" / "ieeg"
    d.mkdir(parents=True)
    contacts = {"LA1": CENTRES["L"] + [0, R_MM + 1, 0], "LA2": CENTRES["L"] + [0, R_MM + 3, 0], "LA3": CENTRES["L"] + [0, R_MM + 20, 0],
                "RB1": CENTRES["R"] + [0, 0, R_MM - 2], "RB2": CENTRES["R"] + [0, 0, -R_MM + 1]}
    _tsv(bids / "sub-01" / f"sub-01_space-{SPACE}_electrodes.tsv",
         [{"name": n, "x": p[0], "y": p[1], "z": p[2], "size": 0.8, "group": n[:2]} for n, p in contacts.items()])
    (bids / "sub-01" / f"sub-01_space-{SPACE}_coordsystem.json").write_text(json.dumps({"iEEGCoordinateSystem": SPACE, "iEEGCoordinateUnits": "mm"}))
    data = {}
    for stem, fs, fmt, orient in (("sub-01_task-rest_run-01", 200.0, "IEEE_FLOAT_32", "MULTIPLEXED"),
                                  ("sub-01_task-sleep_acq-N2_run-01", 512.0, "INT_16", "VECTORIZED")):
        n = int(fs * 10)
        if fmt == "INT_16":
            raw = rng.integers(-2000, 2000, (len(NAMES), n)).astype("<i2")
            x, res = raw.astype(np.float64) * 0.1e-6, 0.1
            (raw if orient == "VECTORIZED" else raw.T).tofile(d / f"{stem}_ieeg.eeg")
        else:
            raw = (rng.standard_normal((len(NAMES), n)) * 50).astype("<f4")
            x, res = raw.astype(np.float64) * 1e-6, 1.0
            raw.T.tofile(d / f"{stem}_ieeg.eeg")
        _vhdr(d / f"{stem}_ieeg.vhdr", NAMES, fs, fmt, orient, res)
        _tsv(d / f"{stem}_channels.tsv", [{"name": c, "type": "ECG" if c == "ECG" else "SEEG", "units": "µV",
                                           "status": "bad" if c == "LA2" else "good"} for c in NAMES])
        _tsv(d / f"{stem}_events.tsv", [{"onset": 1.0, "duration": 2.0, "trial_type": "spindle"}, {"onset": 5.5, "duration": "n/a", "trial_type": "IED"},
                                        {"onset": 6.0, "duration": 0, "trial_type": "IED"}])
        (d / f"{stem}_ieeg.json").write_text(json.dumps({"SamplingFrequency": fs, "SEEGChannelCount": 5, "ECOGChannelCount": 0,
                                                         "PowerLineFrequency": 50}))
        data[stem] = x.astype(np.float32)
    return data, contacts


def _template(root):
    nib = pytest.importorskip("nibabel")
    from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
    d = root / "tpl"
    d.mkdir()
    v, f = icosphere(3, radius=R_MM)
    for h in ("L", "R"):
        for name, scale in (("midthickness", 1.0), ("inflated", 1.3)):
            img = GiftiImage(darrays=[GiftiDataArray((v * scale + CENTRES[h]).astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
                                      GiftiDataArray(f.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")])
            nib.save(img, d / f"tpl-{SPACE}_hemi-{h}_{name}.surf.gii")
        lt = GiftiLabelTable()
        for key, nm, rgba in ((0, "???", (0, 0, 0, 0)), (1, "lower", (0.2, 0.4, 0.6, 1)), (2, "upper", (0.8, 0.1, 0.1, 1))):
            lab = GiftiLabel(key, *rgba)
            lab.label = nm
            lt.labels.append(lab)
        keys = np.where(v[:, 2] > 0, 2, 1).astype(np.int32)
        nib.save(GiftiImage(darrays=[GiftiDataArray(keys, intent="NIFTI_INTENT_LABEL")], labeltable=lt),
                 d / f"tpl-{SPACE}_hemi-{h}_atlas-Demo2_dseg.label.gii")
    _tsv(d / f"tpl-{SPACE}_atlas-Demo2_dseg.tsv", [{"index": 1, "name": "Lower region"}, {"index": 2, "name": "Upper region"}])
    return d


@pytest.fixture
def converted(tmp_path):
    rng = np.random.default_rng(1)
    data, contacts = _subject(tmp_path / "bids", rng)
    tpl = _template(tmp_path)
    assert main(["dataset", "create", str(tmp_path / "store"), "ieeg"]) == 0
    store = tmp_path / "store" / "ieeg"
    assert main(["bids-ieeg", "--dataset", str(store), "--subject", "sub-01", "--bids", str(tmp_path / "bids"), "--template-surfaces", str(tpl)]) == 0
    ds = open_dataset(store, lock=False)
    yield {"ds": ds, "sub": Subject.open(ds, "sub-01"), "data": data, "contacts": contacts, "store": store, "bids": tmp_path / "bids"}
    ds.close()


def read(sub, path):
    return np.asarray(zarr.open_array(str(sub.at(path)), mode="r")[...])


def test_the_header_is_read_as_brainvision_writes_it(tmp_path):
    _vhdr(tmp_path / "x_ieeg.vhdr", ["A,1", "B"], 512.0, "INT_16", "MULTIPLEXED", 0.5)
    np.array([[1, 2], [3, 4], [5, 6]], dtype="<i2").tofile(tmp_path / "x_ieeg.eeg")
    h = read_vhdr(tmp_path / "x_ieeg.vhdr")
    assert h["names"] == ["A,1", "B"] and h["sfreq"] == pytest.approx(512.0, rel=1e-5) and h["n_samples"] == 3 and h["units"] == ["µV", "µV"]
    assert np.allclose(read_block(h, 1, 2), np.array([[3, 5], [4, 6]]) * 0.5e-6)


def test_the_template_is_the_dataset_s_once(converted):
    ds, sub = converted["ds"], converted["sub"]
    t = ds.template()
    assert t["name"] == f"tpl-{SPACE}" and sub.row["template_id"] == t["id"]
    tpl = Subject.open(ds, t["name"])
    mid = tpl.find("manifold", "sources/midthickness")
    assert mid["is_primary"] == 1 and tpl.find("manifold", "sources/inflated")["is_primary"] == 0
    atlas = tpl.find("selection", "sources/midthickness_atlases/Demo2")
    levels = [r["name"] for r in ds.db.all("SELECT name FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", atlas["dictionary_id"])]
    assert levels == ["unassigned", "Lower region L", "Upper region L", "Lower region R", "Upper region R"]


def test_each_run_a_session_its_recording_in_volts(converted):
    sub = converted["sub"]
    for stem, x in converted["data"].items():
        f = sub.find("field", f"timeseries/{stem}_ieeg")
        assert f["unit"] == "V" and f["session"] == stem.removeprefix("sub-01_")
        spacing = sub.db.value("SELECT d.spacing FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.path = ?",
                               f"timeseries/{stem}_ieeg_time")
        assert spacing == 1 / (512.0 if "N2" in stem else 200.0)                 # the sidecar's rate, not the header's rounded interval
        assert np.allclose(read(sub, f["path"]), x, rtol=1e-6, atol=0)
        ev = sub.db.all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal",
                        sub.find("selection", f"timeseries/{stem}_ieeg_events")["id"])
        assert [(e["start"], e["stop"]) for e in ev] == [(1.0, 3.0), (5.5, 5.5), (6.0, 6.0)]
    # the two runs' channels.tsv are byte-identical, so are their electrodes: one channel manifold
    assert sub.db.value("SELECT count(*) FROM manifold WHERE subject_id = ? AND path LIKE 'timeseries/channel_ieeg%' AND type = 'points'", sub.id) == 1


def test_the_channels_at_their_contacts(converted):
    sub, contacts = converted["sub"], converted["contacts"]
    pos = read(sub, "timeseries/channel_ieeg")
    for k, n in enumerate(NAMES):
        if n in contacts:
            assert np.allclose(pos[k], contacts[n] / 1000.0, atol=1e-7)
        else:
            assert np.isnan(pos[k]).all()


def test_the_contacts_on_the_template_cortex(converted):
    sub, ds = converted["sub"], converted["ds"]
    tpl = Subject.open(ds, ds.template()["name"])
    op = sub.find("operator", "channel_ieeg_to_midthickness")
    assert op["to_manifold_id"] == tpl.find("manifold", "sources/midthickness")["id"] and op["layout"] == "csr"
    on = [NAMES.index(n) for n in ("LA1", "LA2", "RB1", "RB2")]                         # within 5 mm; LA3 is 20 mm out, ECG has no contact
    assert op["nnz"] == len(on) and sorted(read(sub, "channel_ieeg_to_midthickness/indices").tolist()) == on
    dist = read(sub, "timeseries/channel_ieeg_cortex_distance")
    assert np.isnan(dist[NAMES.index("ECG")]) and 0.015 < dist[NAMES.index("LA3")] < 0.021
    assert read(sub, "timeseries/channel_ieeg_cortex_vertex")[NAMES.index("ECG")] == -1

    def labels(member):
        s = sub.find("selection", f"timeseries/channel_ieeg_cortex/{member}")
        names = {r["code"]: r["name"] for r in ds.db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", s["dictionary_id"])}
        return [names[int(c)] for c in read(sub, s["path"])]
    assert labels("on_cortex") == ["on cortex", "on cortex", "off cortex", "on cortex", "on cortex", "no position"]
    assert labels("Demo2") == ["Lower region L", "Lower region L", "unassigned", "Upper region R", "Lower region R", "unassigned"]


def test_a_run_already_there_is_skipped(converted, capsys):
    converted["ds"].close()
    capsys.readouterr()
    assert main(["bids-ieeg", "--dataset", str(converted["store"]), "--subject", "sub-01", "--bids", str(converted["bids"])]) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [o["stage"] for o in out].count("skip-run") == 2 and out[-1]["runs"] == []
    converted["ds"] = open_dataset(converted["store"], lock=False)
