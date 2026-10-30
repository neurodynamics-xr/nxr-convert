"""``nxr-convert qc`` — SENSOR QC of a subject's recordings, as rows of the existing tables (schema 49, no DDL change; spec:
Product 2026-10-24 "Sensor PSD + QC", the 0.2.7 brief 2026-10-25).

Per recording (every run, task-noise included):
  frequency/<rec>_psd_preprocessed   Field   the sensor PSD of the store's own signal: Welch, 4 s Hann windows, 50 % overlap,
                                             one-sided density (T²/Hz), every channel, on channels × <rec>_freq (0 … fs/2, 0.25 Hz)
  measurements on the recording's channel ``_names`` selection, of the PSD field: per channel (code = channel index) and per
  recording (code null, the median over the good channels of one sensor type: magnetometers where the system has both
  magnetometers and gradiometers, as Elekta/MEGIN does): line_residual_db · muscle_power · muscle_ratio_db ·
  alpha_peak_hz · alpha_peak_power · aperiodic_slope · channel_status · excluded_fraction · qc_flag (the v1 rule, stated in
  the ``qc flag`` dictionary)
  meg-qc's session metrics (A1–A4, I1–I2, H1, K1) as measurements of the rest recordings (of the recording field).
MEG channels are those typed MEG, MEG MAG or MEG GRAD (not MEG REF); flat / noisy is judged within each type, as magnetometers
(T) and gradiometers (T/m) are not on one scale, and a gradiometer's power measurements carry (T/m)². The line frequency is
``--line-freq`` (default 60 Hz; 50 Hz mains elsewhere).
Not here: the ``clean`` stage (SSP + exclusion), the PSD figure, the gallery, and the ``<rec>_qc_rejected`` spans — bst-timefreq-gr
records how many 4 s epochs it rejected, not which, and its thresholds do not reproduce its counts on the store's (pre-SSP)
sensors (one rest run: 51 epochs over its peak-to-peak threshold against its 6, measured 2026-10-25), so the epochs are not
guessed: ``excluded_fraction`` is 0 with ``"excluded": "none marked"`` until the pipeline exports its rejected epochs.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.signal import welch

from .crud import Subject, jdump, lattice_counts, remove_node
from .db import now_utc

BY = "nxr-convert qc"
WIN_S, OVERLAP = 4.0, 0.5
MEASURES = {   # name: (unit, reduction, description)
    "line_residual_db": ("dB", "max", "power at L·k ± 0.5 Hz over its ± 2–5 Hz neighbours, the max over k = 1..5 (L the line "
                                      "frequency, 60 Hz unless stated)"),
    "muscle_power": ("T²", "sum", "65–115 Hz band power"),
    "muscle_ratio_db": ("dB", "mean", "65–115 Hz power over 1–45 Hz power"),
    "alpha_peak_hz": ("Hz", "mean", "the max of the 1/f-detrended spectrum in 7–13 Hz; null without ≥ 1 dB prominence"),
    "alpha_peak_power": ("T²/Hz", "mean", "the PSD at the alpha peak"),
    "aperiodic_slope": ("1", "mean", "log-log slope of the PSD over 2–40 Hz, 7–13 Hz excluded"),
    "channel_status": ("1", "max", "0 ok · 1 flat · 2 noisy (median power < 1e-3 / > 10 × the median of the channels of its type)"),
    "excluded_fraction": ("1", "mean", "the share of the recording in rejected epochs"),
    "qc_flag": ("1", "max", "0 pass · 1 check · 2 fail (the qc flag dictionary's rule)"),
}
RULE = {0: ("pass", "otherwise"),
        1: ("check", "no alpha peak (rest runs), muscle_ratio_db > −10, aperiodic_slope outside [−3, −0.5], > 3 % MEG channels flat or "
                     "noisy, or line_residual_db > 3"),
        2: ("fail", "excluded_fraction > 0.5, > 10 % MEG channels flat or noisy, or line_residual_db > 10")}
MEG_TYPES = ("MEG", "MEG MAG", "MEG GRAD")     # Brainstorm's channel types; MEG REF (CTF references) is not measured
MEGQC = ("amp_p2p_pT", "lf_rest_vs_noise_dB", "lf_front_vs_back_dB", "ref_rest_vs_noise_dB", "n_bad_channels", "rest_s", "head_disp_mm",
         "kept_frac")


# ── the computations (arrays in, numbers out) ──────────────────────────────────────────────────────────────────────────

def psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(WIN_S * fs))
    return welch(x, fs=fs, window="hann", nperseg=n, noverlap=int(n * OVERLAP), detrend="constant", scaling="density", axis=-1)


def _band(f, P, lo, hi):
    m = (f >= lo) & (f <= hi)
    return P[..., m].sum(-1) * (f[1] - f[0])


def measures(f: np.ndarray, P: np.ndarray, line_freq: float = 60.0) -> dict[str, np.ndarray]:
    """per row of P [n × f]: the spectrum measures (alpha null → nan)"""
    P = np.maximum(np.atleast_2d(P), np.finfo(float).tiny)
    out: dict[str, np.ndarray] = {}
    lr = np.full(len(P), -np.inf)
    for k in range(1, 6):
        c = line_freq * k
        if c + 5 > f[-1]:
            break
        on = (np.abs(f - c) <= .5)
        nb = (np.abs(f - c) >= 2) & (np.abs(f - c) <= 5)
        lr = np.maximum(lr, 10 * np.log10(P[:, on].mean(1) / P[:, nb].mean(1)))
    out["line_residual_db"] = lr
    mus = _band(f, P, 65, 115)
    out["muscle_power"] = mus
    out["muscle_ratio_db"] = 10 * np.log10(mus / _band(f, P, 1, 45))
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 13))
    lf = np.log10(f[fit])
    slope, icpt = np.polyfit(lf, np.log10(P[:, fit]).T, 1)
    out["aperiodic_slope"] = slope
    a = (f >= 7) & (f <= 13)
    det = 10 * (np.log10(P[:, a]) - (slope[:, None] * np.log10(f[a])[None] + icpt[:, None]))     # dB above the 1/f fit
    k = det.argmax(1)
    ok = det[np.arange(len(P)), k] >= 1.0
    out["alpha_peak_hz"] = np.where(ok, f[a][k], np.nan)
    out["alpha_peak_power"] = np.where(ok, P[:, a][np.arange(len(P)), k], np.nan)
    return out


def channel_status(f: np.ndarray, P: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """per row of P: 0 ok · 1 flat · 2 noisy, against the median of the rows of the same group (sensor type)"""
    b = (f >= 1) & (f <= 100)
    med = np.median(P[:, b], 1)
    groups = np.zeros(len(P), int) if groups is None else np.asarray(groups)
    ref = np.empty_like(med)
    for g in np.unique(groups):
        ref[groups == g] = np.median(med[groups == g])
    return np.where(med < 1e-3 * ref, 1, np.where(med > 10 * ref, 2, 0))


def qc_flag(rec: dict[str, float], bad_share: float, rest: bool) -> int:
    if rec["excluded_fraction"] > .5 or bad_share > .10 or rec["line_residual_db"] > 10:
        return 2
    if ((rest and not np.isfinite(rec["alpha_peak_hz"])) or rec["muscle_ratio_db"] > -10
            or not -3 <= rec["aperiodic_slope"] <= -.5 or bad_share > .03 or rec["line_residual_db"] > 3):
        return 1
    return 0


def read_megqc(tar: str | Path) -> dict[str, dict[str, Any]]:
    """meg-qc's <sub>.tar.gz → {session: {metric: value}}"""
    with tarfile.open(tar) as t:
        m = next(x for x in t.getmembers() if x.name.endswith("_megqc.json"))
        d = json.load(t.extractfile(m))
    return {s["session"]: {k: s.get(k) for k in MEGQC} for s in d.get("sessions", [])}


# ── the rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _factors(db, manifold_id: str) -> list[str]:
    return [r["manifold_id"] for r in db.all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id "
                                             "WHERE p.id = ? ORDER BY d.ordinal", manifold_id)]


def _meg(sub: Subject, channels_id: str, n: int) -> tuple[np.ndarray, list[str]]:
    """each channel's MEG sensor type, by the channel labelling's ``type`` partition (D111): an index into the returned type
    names, −1 for a channel that is not MEG — every channel one type ("MEG") when there is no partition"""
    t = sub.db.one("SELECT s.path, s.dictionary_id FROM selection s JOIN selection l ON l.id = s.member_of_id WHERE s.name = 'type' "
                   "AND s.manifold_id = ? AND l.name LIKE '%\\_labels' ESCAPE '\\'", channels_id)
    if t is None:
        return np.zeros(n, int), ["MEG"]
    codes = np.asarray(zarr.open_array(str(sub.at(t["path"])), mode="r")[:])
    g, names = np.full(n, -1), []
    for e in sub.db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", t["dictionary_id"]):
        if e["name"] in MEG_TYPES and (codes == e["code"]).any():
            g[codes == e["code"]] = len(names)
            names.append(e["name"])
    return g, names


def _dictionaries(sub: Subject) -> None:
    sub.dictionary("measure", scope="dataset", entries=[{"code": k, "name": n, "ordinal": k, "measurement_unit": u,
                                                          "attributes_json": jdump({"reduction": r, "description": d})}
                                                         for k, (n, (u, r, d)) in enumerate(MEASURES.items())])
    sub.dictionary("qc flag", scope="dataset", entries=[{"code": k, "name": n, "ordinal": k, "attributes_json": jdump({"rule": t, "version": 1})}
                                                         for k, (n, t) in RULE.items()])
    sub.dictionary("channel status", ["ok", "flat", "noisy"], scope="dataset")


def add_qc(sub: Subject, *, megqc: str | Path | None = None, replace: bool = False, block: int = 16, line_freq: float = 60.0,
           progress=None) -> dict:
    db, out, stamp = sub.db, [], now_utc()
    mq = read_megqc(megqc) if megqc else {}
    _dictionaries(sub)
    recs = db.all("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND status = 'complete' ORDER BY name", sub.id)
    for rec in recs:
        ch_id, t_id = _factors(db, rec["manifold_id"])
        dim = db.one("SELECT d.* FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.id = ?", t_id)
        fs = 1.0 / dim["spacing"]
        names = db.one("SELECT id FROM selection WHERE manifold_id = ? AND name LIKE '%\\_names' ESCAPE '\\' AND member_of_id IS NULL", ch_id)
        name = rec["path"].rsplit("/", 1)[-1]
        fpath = f"frequency/{name}_psd_preprocessed"
        have = db.one("SELECT id FROM field WHERE subject_id = ? AND path = ?", sub.id, fpath)
        if have and not replace:
            raise FileExistsError(f"{sub.row['name']}: {fpath} exists (--replace to recompute)")
        if have:
            remove_node(db, sub.root, "field", have["id"])
        a = zarr.open_array(str(sub.at(rec["path"])), mode="r")
        n_ch = a.shape[0]
        P = None
        for c0 in range(0, n_ch, block):
            f, p = psd(np.asarray(a[c0:c0 + block], np.float64), fs)
            P = np.zeros((n_ch, len(f)), np.float32) if P is None else P
            P[c0:c0 + block] = p
        groups, types = _meg(sub, ch_id, n_ch)
        meg = groups >= 0
        if not meg.any():
            raise ValueError(f"{name}: no MEG channels (types {', '.join(MEG_TYPES)})")
        ref = types.index("MEG MAG") if "MEG MAG" in types else 0     # the recording's measures: one sensor type, never pooled
        lpath = f"frequency/{name}_freq"
        line = db.value("SELECT id FROM manifold WHERE subject_id = ? AND path = ?", sub.id, lpath) or sub.manifold(
            name=f"{name}_freq", path=lpath, type="line", n_vertices=len(f), session=rec["session"], topology={"kind": "lattice", "rank": 1},
            **lattice_counts([len(f)]),
            geometry={"form": "closed_form", "cs_key": "hz", "dimensions": [{"name": "frequency", "n_vertices": len(f),
                      "spacing": float(f[1] - f[0]), "origin": 0.0, "unit": "Hz"}]})
        unit = f"{rec['unit']}²/Hz" if rec.get("unit") else None
        fid = sub.field(name=f"{name}_psd_preprocessed", path=fpath, kind="psd", label="preprocessed", function="welch", manifold_id=sub.product([ch_id, line], session=rec["session"]),
                        data=P, derived_from_id=rec["id"], session=rec["session"], unit=unit,
                        description="the sensor PSD of the stored signal (notch + high-pass as stored; no SSP, no exclusion)",
                        producer_json=jdump({"stage": "preprocessed", "window_s": WIN_S, "overlap": OVERLAP, "window": "hann", "scaling": "density",
                                             "excluded": "none marked", "ssp": None, "by": BY, "line_freq_hz": line_freq,
                                             "meg_types": types, "recording_measures_from": types[ref]}))
        m = measures(f, P, line_freq)
        st = channel_status(f, P[meg], groups[meg])
        bad = float((st > 0).mean())
        good = (groups[meg] == ref) & (st == 0)
        rm = {k: float(v[0]) for k, v in measures(f, np.median(P[meg][good if good.any() else groups[meg] == ref], 0), line_freq).items()}
        rm.update(excluded_fraction=0.0, channel_status=float(st.max()))
        rest = "task-rest" in name
        rm["qc_flag"] = float(qc_flag(rm, bad, rest))
        rows = []
        idx = np.where(meg)[0]
        for k, v in m.items():
            rows += [(int(i), k, float(v[i])) for i in idx if np.isfinite(v[i])]
        rows += [(int(i), "channel_status", float(s)) for i, s in zip(idx, st)]
        rows += [(None, k, v) for k, v in rm.items() if np.isfinite(v)]
        unit_of = {k: u for k, (u, _, _) in MEASURES.items()}
        grad = {int(i) for i in np.where(groups == (types.index("MEG GRAD") if "MEG GRAD" in types else -2))[0]}

        def _unit(code, k):
            u = unit_of[k] if unit_of[k] != "T²/Hz" else unit
            return "(T/m)²" + u[2:] if u and code in grad and u.startswith("T²") else u     # a gradiometer's power: (T/m)²
        with db.tx():
            if names is None:
                raise KeyError(f"{name}: no channel _names selection on its channels")
            # the PSD's own measurements went with it (cascade); meg-qc's are of the recording
            db.execute("DELETE FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND computed_by = 'meg-qc'", names["id"], rec["id"])
            for code, k, v in rows:
                db.insert("selection_measurement", {"selection_id": names["id"], "code": code, "of_field_id": fid, "measure": k, "value": v,
                                                    "unit": _unit(code, k), "computed_utc": stamp, "computed_by": BY})
            ses = next((v for k, v in mq.items() if f"_{k}_" in f"_{name}_"), None) if rest else None     # meg-qc keys a session 'ses-02'

            for k, v in (ses or {}).items():
                if v is not None:
                    db.insert("selection_measurement", {"selection_id": names["id"], "code": None, "of_field_id": rec["id"], "measure": f"megqc_{k}",
                                                        "value": float(v), "computed_utc": stamp, "computed_by": "meg-qc"})
        out.append({"recording": name, "psd": fpath, "channels": int(n_ch), "meg": int(meg.sum()), "flagged_channels": int((st > 0).sum()),
                    "alpha_peak_hz": rm["alpha_peak_hz"] if np.isfinite(rm["alpha_peak_hz"]) else None, "qc_flag": RULE[int(rm["qc_flag"])][0], "megqc": bool(ses)})
        if progress:
            progress({"stage": "qc", **out[-1]})
    return {"subject": sub.row["name"], "recordings": out}
