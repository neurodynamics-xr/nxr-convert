"""`resample_rows`, and `nxr-convert resample` — a STORED recording resampled beside it: same channels and session, its own Time Line, chunk grid and envelope."""
import json

import numpy as np
import pytest
import zarr
from scipy.signal import resample_poly

from nxr_convert.cli import main
from nxr_convert.crud import open_dataset
from nxr_convert.timeseries import resample_rows

from test_qc import FS, T, ds_with_recording  # noqa: F401  (the fixture: a 600 Hz, 120 s recording, 13 channels, a 10 Hz rhythm)

REC = "sub-01_ses-01_task-rest_run-01_meg"


def test_resample_keeps_below_nyquist_removes_above():
    fs, to, n = 2400.0, 300.0, 24000
    t = np.arange(n) / fs
    x = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 600 * t)]).astype(np.float32)
    y = resample_rows(lambda a, b: x[a:b], 2, n, fs, to, step=1)
    assert y.shape == (2, -(-n * 300 // 2400)) and y.dtype == np.float32
    mid = slice(len(t[::8]) // 4, -len(t[::8]) // 4)             # away from the filter edges
    assert abs(np.abs(y[0, mid]).max() - 1) < 0.01
    assert 20 * np.log10(np.abs(y[1, mid]).max() / 1) < -40


def test_resample_writes_a_recording_beside_the_stored_one(ds_with_recording, capsys):
    folder, _ = ds_with_recording
    assert main(["resample", "--dataset", str(folder), "--subject", "sub-01", "--recording", REC, "--to", "300"]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["name"] == f"{REC}_300Hz" and res["sfreq"] == 300 and res["n_samples"] == int(T * 300) and res["n_channels"] == 13
    with open_dataset(folder) as ds:
        db = ds.db
        old = db.one("SELECT * FROM field WHERE name = ?", REC)
        new = db.one("SELECT * FROM field WHERE name = ?", f"{REC}_300Hz")
        assert new["kind"] == "recording" and new["status"] == "complete" and new["session"] == old["session"]
        assert json.loads(new["producer_json"])["resampled_from"]["field"] == old["id"]
        fac = lambda f: [r["manifold_id"] for r in db.all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE p.id = ? ORDER BY d.ordinal", f["manifold_id"])]
        (c0, t_old), (c1, t_new) = fac(old), fac(new)
        assert c0 == c1 and t_old != t_new                                                     # the same channels, a Time Line of its own
        dim = db.one("SELECT d.* FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.id = ?", t_new)
        assert dim["spacing"] == pytest.approx(1 / 300) and dim["n_vertices"] == int(T * 300)
        names = {s["name"] for s in db.all("SELECT name FROM selection WHERE of_field_id = ?", new["id"])}
        assert {f"{REC}_300Hz_chunks", f"{REC}_300Hz_envelope"} <= names
        x = np.asarray(zarr.open_array(str(folder / "sub-01.nxr.zarr" / old["path"]), mode="r")[:], np.float64)
        y = np.asarray(zarr.open_array(str(folder / "sub-01.nxr.zarr" / new["path"]), mode="r")[:])
    ref = resample_poly(x, 1, 2, axis=1).astype(np.float32)
    assert y.shape == ref.shape and np.max(np.abs(y - ref)) <= 1e-6 * np.max(np.abs(ref))   # the rows are resample_poly's
    p = np.abs(np.fft.rfft(y[0])) ** 2
    assert np.fft.rfftfreq(y.shape[1], 1 / 300)[np.argmax(p[1:]) + 1] == pytest.approx(10, abs=.05)   # the 10 Hz rhythm kept
    assert main(["resample", "--dataset", str(folder), "--subject", "sub-01", "--recording", REC, "--to", "300"]) == 1   # present: refused


def test_badsegments_go_onto_the_recording_and_its_resampled_copy(ds_with_recording, capsys, tmp_path):
    folder, _ = ds_with_recording
    assert main(["resample", "--dataset", str(folder), "--subject", "sub-01", "--recording", REC, "--to", "300"]) == 0
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"subject": "sub-01", "settings": {"ignored": 1}, "recordings": [{"name": REC, "sfreq": FS, "segments": [
        {"label": "bad_1-7Hz", "start_s": [1.0, 10.0], "stop_s": [2.5, 10.5]}, {"label": "bad_40-240Hz", "start_s": [20.0], "stop_s": [21.0]},
        {"label": "blink", "start_s": [3.0], "stop_s": [3.2]}]}]}))
    args = ["badsegments", "--dataset", str(folder), "--subject", "sub-01", "--from", str(src)]
    assert main(args) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert [(r["recording"], r["bad_segments"], r["seconds"]) for r in res["recordings"]] == [(REC, 3, 3.0), (f"{REC}_300Hz", 3, 3.0)]
    with open_dataset(folder) as ds:
        for name in (REC, f"{REC}_300Hz"):
            sel = ds.db.one("SELECT s.* FROM selection s JOIN field f ON f.id = s.of_field_id WHERE f.name = ? AND s.name LIKE '%\\_bad\\_segments' ESCAPE '\\'", name)
            spans = ds.db.all("SELECT start, stop, code FROM selection_span WHERE selection_id = ? ORDER BY start", sel["id"])
            assert sel["type"] == "spans" and [(r["start"], r["stop"]) for r in spans] == [(1.0, 2.5), (10.0, 10.5), (20.0, 21.0)]   # the blink is not bad
            assert spans[0]["code"] == spans[1]["code"] != spans[2]["code"]                      # a code per label
    assert main(args) == 1                                                                       # present: refused
    assert main(args + ["--replace"]) == 0
