"""``nxr-convert qc`` on a synthetic recording: 1/f noise with a 10 Hz rhythm and 60 Hz line noise, one flat MEG channel, one EEG channel."""
import io
import json
import tarfile

import numpy as np
import pytest

from nxr_convert.cli import main
from nxr_convert.crud import Subject, create_dataset, open_dataset
from nxr_convert.qc import measures, psd
from nxr_convert.sensors import export_sensors
from nxr_convert.timeseries import export_timeseries

FS, T = 600.0, 120.0


def signal(n_meg=12, seed=0):
    rng = np.random.default_rng(seed)
    n = int(FS * T)
    f = np.fft.rfftfreq(n, 1 / FS)
    x = np.fft.irfft((rng.standard_normal((n_meg + 1, f.size)) + 1j * rng.standard_normal((n_meg + 1, f.size))) / np.maximum(f, 1) ** .5, n)
    t = np.arange(n) / FS
    x += 3 * np.std(x) * np.sin(2 * np.pi * 10 * t) + np.std(x) * np.sin(2 * np.pi * 60 * t)
    x *= 1e-13
    x[3] = 1e-20 * rng.standard_normal(n)                                  # a flat MEG channel
    return x.astype(np.float32), t


@pytest.fixture
def ds_with_recording(tmp_path):
    ds = create_dataset(tmp_path / "store", "ds")
    sub = Subject.create(ds, "sub-01", source_format="brainstorm", immediate=True)
    x, t = signal()
    chans = [{"Name": f"MLC{i:02d}", "Type": "MEG", "Loc": np.zeros((3, 1)), "Group": ""} for i in range(12)] + [
        {"Name": "EEG001", "Type": "EEG", "Loc": np.zeros((3, 1)), "Group": ""}]
    s = export_sensors(sub, {"Channel": chans}, np.ones(13), name="channel_x", session="ses-01_task-rest_run-01")
    export_timeseries(sub, "sub-01_ses-01_task-rest_run-01_meg", {"F": x, "Time": t}, session="ses-01_task-rest_run-01",
                      channels_id=s["id"], channel_names=[c["Name"] for c in chans])
    ds.close()
    q = tmp_path / "sub-01.tar.gz"                                         # meg-qc's archive, its one JSON
    body = json.dumps({"sessions": [{"session": "ses-01", "amp_p2p_pT": 1500.0, "kept_frac": .88, "n_bad_channels": 0}]}).encode()
    with tarfile.open(q, "w:gz") as tf:
        info = tarfile.TarInfo("./sub-01_megqc.json"); info.size = len(body); tf.addfile(info, io.BytesIO(body))
    return tmp_path / "store" / "ds", q


def test_measures_read_the_spectrum():
    x, _ = signal()
    f, P = psd(x.astype(np.float64), FS)
    assert f[1] - f[0] == pytest.approx(.25) and f[-1] == FS / 2
    m = measures(f, np.median(P[[0, 1, 2, 4]], 0))
    assert m["alpha_peak_hz"][0] == pytest.approx(10, abs=.25) and m["line_residual_db"][0] > 10
    assert -1.5 < m["aperiodic_slope"][0] < -.5


def test_qc_writes_the_psd_its_measures_and_the_flag(ds_with_recording, capsys):
    folder, q = ds_with_recording
    assert main(["qc", "--dataset", str(folder), "--subject", "sub-01", "--megqc", str(q)]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["recordings"][0]
    assert res["meg"] == 12 and res["flagged_channels"] == 1 and res["qc_flag"] == "fail" and res["megqc"]   # line residual > 10 dB
    assert main(["qc", "--dataset", str(folder), "--subject", "sub-01"]) == 1                         # present: refused
    assert main(["qc", "--dataset", str(folder), "--subject", "sub-01", "--megqc", str(q), "--replace"]) == 0
    with open_dataset(folder) as ds:
        db = ds.db
        f = db.all("SELECT * FROM field WHERE kind = 'psd'")
        assert len(f) == 1 and f[0]["label"] == "preprocessed" and f[0]["function"] == "welch" and f[0]["status"] == "complete"
        ms = db.all("SELECT * FROM selection_measurement WHERE of_field_id = ?", f[0]["id"])
        per = {(m["code"], m["measure"]): m["value"] for m in ms}
        assert per[(None, "qc_flag")] == 2 and per[(3, "channel_status")] == 1 and (12, "channel_status") not in per   # EEG not measured
        assert per[(None, "alpha_peak_hz")] == pytest.approx(10, abs=.25)
        mq = db.all("SELECT measure, value FROM selection_measurement WHERE computed_by = 'meg-qc'")
        assert sorted(m["measure"] for m in mq) == ["megqc_amp_p2p_pT", "megqc_kept_frac", "megqc_n_bad_channels"]   # once, after --replace
        assert db.value("SELECT count(*) FROM dictionary WHERE name IN ('measure', 'qc flag', 'channel status') AND scope = 'dataset'") == 3
