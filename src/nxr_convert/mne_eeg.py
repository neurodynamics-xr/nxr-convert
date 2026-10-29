"""MNE-Python's EEG SOURCE PACKAGE (nsp's ``mne-eeg-source``) → a subject, as entities (D141). Needs ``nxr-convert[mne]``.

The package, per subject (``<p>`` its file prefix): the preprocessed EEG ``<p>_task-<t>_desc-preproc_eeg.fif`` (float32, the
average reference applied, ``info['bads']``, the montage in MNE's head frame; MNE may split it), the head → MRI transform
``<p>_eeg-trans.fif``, the inverse operator ``<p>_eeg-inv.fif``, the cortex-normal kernels ``<p>_kernel-<M>_ori-normal.npy``
[sources × good channels] with ``<p>_kernel.json`` (their channel order, vertices, settings), the per-stage band power
``<p>_stage_bandpower.npz`` and the provenance ``<p>_mne_eeg.json``. The template is FreeSurfer's fsaverage as MNE ships it.

    sources/cortex_white_ico5   Manifold  THE CORTEX THE KERNELS' ROWS INDEX — fsaverage's white surface at the vertices of MNE's ico-5
                                          source space (``bem/fsaverage-ico-5-src.fif``: its ``vertno``, its ``use_tris``), lh then rh,
                                          metres in FreeSurfer's surface RAS (MNE's MRI frame); the primary, with its hemispheres, its
                                          sphere and FreeSurfer's parcellations at those vertices (``surface.add_surface``)
    timeseries/channel_eeg      Manifold  the EEG channels at their electrodes, head → MRI by the transform: the cortex's frame
    timeseries/<recording>      Field     the preprocessed EEG [channels × samples], chunked as every recording is
                                          (``timeseries._recording``); its ANNOTATIONS are its events (``stage/<label>`` a span per
                                          scored epoch; a ``BAD_*`` mark a bad segment, the channels it names kept on the event)
    kernel-<M>_ori-normal       Operator  inverse kernel [V × good channels] for MNE · dSPM · eLORETA (``inverse.export_inverse``), its
                                          channels and the noise covariance it was computed with
    frequency/bandpower_<stage>[_relative]   Field  source power [V × bands] on cortex × bands (``timefreq``'s shape), a scored stage each
    frequency/sensorpower_<stage>            Field  sensor power [channels × bands] on channels × bands (a bad channel NaN)

NO SOURCE TIME SERIES are stored (Diellor, 2026-10-28): a source is a kernel row times the EEG, computed where it is read. The
forward (``-fwd.fif``) and the inverse operator's other parts are not stored; the transform rides in the channels' producer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .grid import BLOCK
from .naming import claim_node_name, sanitize_node_name, session_label, source_fingerprint

CORTEX = "cortex_white_ico5"
CHANNELS = "channel_eeg"
# what each method's kernel is, in Brainstorm's words (``inverse.inverse_method_of`` reads them) and the units it maps V to
METHODS = {"MNE": ("amplitude", "A.m"), "dSPM": ("dspm2018", "dSPM"), "eLORETA": ("eloreta", "A.m")}


def _mne():
    try:
        import mne
    except ImportError as e:                                      # the core never needs it
        raise ImportError("reading MNE's FIF files needs MNE-Python: pip install 'nxr-convert[mne]'") from e
    mne.set_log_level("ERROR")
    return mne


def package_files(src: str | Path, prefix: str) -> dict[str, Path]:
    """The package's files for ``prefix``; the EEG is its first part when MNE split it (MNE follows the parts itself)."""
    src = Path(src)
    eeg = [f for f in sorted(src.glob(f"{prefix}_*desc-preproc*_eeg*.fif"))      # not a later part: MNE's `-1.fif`, BIDS' split-02…
           if not re.search(r"-\d+$", f.stem) and ("_split-" not in f.name or "_split-01" in f.name)]
    if len(eeg) != 1:
        raise FileNotFoundError(f"{src}: {len(eeg)} preprocessed EEG files for {prefix} (want 1: {prefix}_task-<t>_desc-preproc_eeg.fif)")
    out = {"eeg": eeg[0]}
    for key, name in (("trans", "eeg-trans.fif"), ("inv", "eeg-inv.fif"), ("kernel", "kernel.json"), ("bandpower", "stage_bandpower.npz"),
                      ("provenance", "mne_eeg.json")):
        f = src / f"{prefix}_{name}"
        if f.is_file():
            out[key] = f
    if "kernel" not in out:
        raise FileNotFoundError(f"{src}: no {prefix}_kernel.json — the kernels cannot be placed")
    return out


def recording_name(eeg: Path) -> str:
    """``<p>_task-sleep_split-01_desc-preproc_eeg.fif`` → ``<p>_task-sleep_desc-preproc_eeg``: the parts are one recording."""
    return sanitize_node_name(eeg.stem.replace("_split-01", ""))


def ico_cortex(fsaverage: str | Path, vertices: dict[str, list[int]]) -> dict[str, Any]:
    """fsaverage's white surface at the kernel's vertices: positions (m), faces (MNE's ico-5 triangles, 0-based, on the kept
    vertices), sphere, hemispheres and parcellations, lh then rh."""
    from .freesurfer import atlas_members, read_surf
    from .surface import FS_ATLASES
    mne = _mne()
    fs = Path(fsaverage)
    src = mne.read_source_spaces(fs / "bem" / "fsaverage-ico-5-src.fif")
    pos, sph, faces, full, keep = [], [], [], [], []
    off = 0
    for h, s in zip(("lh", "rh"), src):
        v = np.asarray(vertices[h], dtype=np.int64)
        if not np.isin(v, s["vertno"]).all():
            raise ValueError(f"{h}: the kernel names vertices outside fsaverage's ico-5 source space")
        white, _ = read_surf(fs / "surf" / f"{h}.white")
        sphere, _ = read_surf(fs / "surf" / f"{h}.sphere.reg")
        if len(white) != s["np"]:
            raise ValueError(f"{h}.white has {len(white)} vertices, the source space {s['np']} — not the same fsaverage")
        lut = np.full(s["np"], -1, dtype=np.int64)
        lut[v] = np.arange(v.size)
        tri = lut[np.asarray(s["use_tris"], dtype=np.int64)]
        faces.append(tri[(tri >= 0).all(axis=1)] + off)           # a triangle with a dropped vertex (MNE's mindist) goes with it
        pos.append(white[v] / 1000.0)
        sph.append(sphere[v] / 1000.0)
        full.append(len(white))
        keep.append(v)
        off += v.size
    members = [(fam, codes[np.concatenate([keep[0], keep[1] + full[0]])], levels, colors)
               for fam, codes, levels, colors in atlas_members(fs, full[0], full[1], FS_ATLASES)]
    return {"vertices": np.concatenate(pos), "faces": np.concatenate(faces), "sphere": np.concatenate(sph),
            "hemis": [("left", 0, len(keep[0])), ("right", len(keep[0]), len(keep[1]))], "atlas_members": members}


def _events(raw) -> list[dict]:
    """The annotations in Brainstorm's ``Events`` shape (``events.parse_events``): one entry a description, its occurrences
    extended ``[onset, onset + duration]`` in seconds from the first sample, the channels a mark names per occurrence."""
    a = raw.annotations
    onset = np.asarray(a.onset, dtype=np.float64) - (raw.first_time if a.orig_time is not None else 0.0)
    out = []
    for label in dict.fromkeys(a.description):
        k = np.flatnonzero(a.description == label)
        chans = [list(a.ch_names[i]) for i in k]
        out.append({"label": str(label), "times": np.vstack([onset[k], onset[k] + a.duration[k]]), "epochs": np.ones(k.size),
                    "channels": chans if any(chans) else None})
    return out


def _channels(sub: Subject, raw, trans, eeg_file: Path, *, session: str) -> dict:
    """The EEG channels at their electrodes, carried head → MRI (the cortex's frame); shared with an earlier session when
    the montage, the bad channels and the transform are the same (D121)."""
    from .sensors import export_sensors
    mne = _mne()
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    loc = np.array([raw.info["chs"][i]["loc"][:3] for i in picks], dtype=np.float64)
    loc[~np.isfinite(loc).all(axis=1) | (np.abs(loc).sum(axis=1) == 0)] = np.nan
    if trans["from"] != mne.io.constants.FIFF.FIFFV_COORD_HEAD:
        trans = mne.transforms.invert_transform(trans)
    pos = mne.transforms.apply_trans(trans["trans"], loc)
    flags = np.array([-1 if n in raw.info["bads"] else 1 for n in names])
    fp = source_fingerprint([], json.dumps([names, np.round(pos, 7).tolist(), flags.tolist()], allow_nan=True).encode())
    name, shared = claim_node_name(sub, "timeseries", CHANNELS, session, fp)
    if shared:
        return {"name": name, "id": sub.find("manifold", f"timeseries/{name}")["id"], "names": names, "picks": picks, "flags": flags}
    chans = [{"Name": n, "Type": "EEG", "Loc": p.reshape(3, 1), "Group": ""} for n, p in zip(names, pos)]
    si = export_sensors(sub, {"Channel": chans}, flags, name=name, session=session, source_sha1=fp, cs_key="freesurfer-tkr",
                        producer={"electrodes": "MNE's montage (head frame) carried to fsaverage's MRI frame by the package's transform",
                                  "head_to_mri": np.asarray(trans["trans"]).tolist(), "source": eeg_file.name})
    return {**si, "picks": picks, "flags": flags}


def import_mne_eeg(sub: Subject, src: str | Path, *, fsaverage: str | Path, prefix: str | None = None, progress=None) -> dict:
    """``nxr-convert mne-eeg``: one subject's package into ``sub`` (the cortex the first time, a session per recording)."""
    from .convert import session_rows
    from .inverse import export_inverse
    from .surface import add_surface
    from .timeseries import _recording
    mne = _mne()
    emit = progress or (lambda e: None)
    prefix = prefix or sub.row["name"]
    files = package_files(src, prefix)
    kj = json.loads(files["kernel"].read_text())
    prov = json.loads(files["provenance"].read_text()) if "provenance" in files else {}
    rec = recording_name(files["eeg"])
    session = session_label(rec.removeprefix(prefix))
    if session_rows(sub, session):
        raise ValueError(f"session {session} already exists in {sub.name} — delete it first to re-import")

    # ---- THE CORTEX the kernels' rows index ----
    have = sub.find("manifold", f"sources/{CORTEX}")
    n_v = len(kj["vertices"]["lh"]) + len(kj["vertices"]["rh"])
    if have is None:
        cx = ico_cortex(fsaverage, kj["vertices"])
        info = add_surface(sub, name=CORTEX, folder="sources", vertices=cx["vertices"], faces=cx["faces"], sphere=cx["sphere"],
                           source=str(Path(fsaverage).name), comment="fsaverage white, MNE's ico-5 source space", hemis=cx["hemis"],
                           atlas_members=cx["atlas_members"], primary=True, cs_key="freesurfer-tkr",
                           producer={"fsaverage": {"surface": "white", "source_space": "bem/fsaverage-ico-5-src.fif", "units": "m",
                                                   "nV_lh": len(kj["vertices"]["lh"]), "nV_rh": len(kj["vertices"]["rh"])}})
        have = sub.db.read("manifold", info["id"])
        emit({"stage": "surface", "name": CORTEX, "n_vertices": info["n_vertices"], "atlases": info["atlases"]})
    if int(have["n_vertices"]) != n_v:
        raise ValueError(f"{CORTEX} has {have['n_vertices']} vertices, the kernels {n_v} — another source space")

    # ---- THE CHANNELS and THE RECORDING ----
    raw = mne.io.read_raw_fif(files["eeg"], preload=False)
    if "trans" not in files:
        raise FileNotFoundError(f"{src}: no {prefix}_eeg-trans.fif — the electrodes cannot be placed on the cortex")
    si = _channels(sub, raw, mne.read_trans(files["trans"]), files["eeg"], session=session)
    emit({"stage": "sensors", "name": si["name"], "n_channels": len(si["names"])})
    picks, n_t, sfreq = si["picks"], int(raw.n_times), float(raw.info["sfreq"])

    def write(arr) -> None:
        for s0 in range(0, n_t, BLOCK):
            arr[:, s0:s0 + BLOCK] = raw.get_data(picks=picks, start=s0, stop=min(n_t, s0 + BLOCK)).astype(np.float32)
            emit({"stage": "stream", "name": rec, "written": min(n_t, s0 + BLOCK), "of": n_t})
    ri = _recording(sub, rec, n_chan=len(picks), n_samples=n_t, sfreq=sfreq, t0=0.0, session=session, channels_id=si["id"], write=write,
                    flags=si["flags"], events=_events(raw), channel_names=si["names"], comment=prov.get("preprocess", {}).get("reference"),
                    kind="sensor", source=files["eeg"].name, unit="V",
                    producer={"format": "FIF", "software": "mne-python", **({"preprocess": prov["preprocess"]} if "preprocess" in prov else {})})
    emit({"stage": "timeseries", **{k: ri[k] for k in ("name", "n_samples", "sfreq", "events", "bad_segments")}})

    # ---- THE KERNELS, one operator a method, on the cortex ----
    index = {n: i for i, n in enumerate(si["names"])}
    missing = [c for c in kj["channels"] if c not in index]
    if missing:
        raise ValueError(f"the kernels name channels the EEG does not hold: {missing[:5]}")
    good = np.array([index[c] for c in kj["channels"]]) + 1                        # 1-based, as Brainstorm's GoodChannel
    cov = None
    if "inv" in files:
        nc = mne.minimum_norm.read_inverse_operator(files["inv"])["noise_cov"]
        at = {n: i for i, n in enumerate(nc["names"])}
        if all(c in at for c in kj["channels"]):
            c = np.diag(nc["data"]) if nc["diag"] else np.asarray(nc["data"])
            k = [at[n] for n in kj["channels"]]
            cov = c[np.ix_(k, k)]
    lam = float(kj.get("lambda2", 1 / 9))
    inv_opts = kj.get("inverse", {})
    kernels = []
    for m in kj.get("methods", list(METHODS)):
        f = Path(src) / f"{prefix}_kernel-{m}_ori-normal.npy"
        if not f.is_file():
            emit({"stage": "skip-kernel", "file": f.name, "reason": "missing"})
            continue
        K = np.load(f)
        measure, units = METHODS.get(m, (m.lower(), ""))
        name, shared = claim_node_name(sub, "", sanitize_node_name(f.stem.removeprefix(prefix)), session,
                                       source_fingerprint([f], f"{si['id']}|{have['id']}".encode()))
        if shared:
            kernels.append(name)
            continue
        kern = {"ImagingKernel": K, "GoodChannel": good, "nComponents": 1, "Comment": f"{m}: MNE-Python, cortex-normal (EEG)",
                "Options": {"InverseMethod": "minnorm", "InverseMeasure": measure, "SourceOrient": ["normal"], "SnrMethod": "fixed",
                            "SnrFixed": float(np.sqrt(1 / lam)), "Loose": inv_opts.get("loose"), "WeightExp": inv_opts.get("depth"),
                            "UseDepth": inv_opts.get("depth") is not None, "NoiseMethod": "adhoc" if "ad hoc" in str(kj.get("noise_covariance", "")) else None,
                            **({"NoiseCovMat": {"NoiseCov": cov}} if cov is not None else {})}}
        ii = export_inverse(sub, kern, session=session, name=name, channels_id=si["id"], surface_id=have["id"], channel_names=si["names"],
                            n_vertices=n_v, source_sha1=source_fingerprint([f], f"{si['id']}|{have['id']}".encode()),
                            producer_extra={"method": m, "method_comment": m, "units": kj.get("units", {}).get(m) or units,
                                            "software": "mne-python", "pick_ori": "normal", "lambda2": lam, "apply": kj.get("apply"),
                                            "template": kj.get("template"), "noise_covariance": kj.get("noise_covariance"),
                                            "source_file": f.name, **({"mne": prov["mne"]} if "mne" in prov else {})})
        kernels.append(ii["name"])
        emit({"stage": "inverse", "name": ii["name"], "method": m})

    # ---- THE BAND POWER per scored stage: Fields on cortex × bands (and channels × bands) ----
    maps = _bandpower(sub, files.get("bandpower"), cortex_id=have["id"], si=si, session=session, kernel=kj.get("methods", ["MNE"])[0])
    for mp in maps:
        emit({"stage": "field", "name": mp})
    return {"subject": sub.name, "session": session, "surface": CORTEX, "channels": si["name"], "recording": rec, "kernels": kernels,
            "maps": maps, "events": ri["events"], "bad_segments": ri["bad_segments"]}


def _bandpower(sub: Subject, f: Path | None, *, cortex_id: str, si: dict, session: str, kernel: str) -> list[str]:
    """``<p>_stage_bandpower.npz`` → a source-power Field per stage (absolute and relative) and a sensor-power one; a stage
    with too little clean signal (all NaN) is not written."""
    from .timefreq import _bands_partition
    if f is None:
        return []
    z = np.load(f)
    bands = [{"name": str(b), "low_hz": float(e[0]), "high_hz": float(e[1]), "function": "mean"} for b, e in zip(z["bands"], z["band_edges"])]
    bands_id = _bands_partition(sub, bands)
    on_cortex = sub.product([cortex_id, bands_id], session=None)
    on_channels = sub.product([si["id"], bands_id], session=None)
    row = {n: i for i, n in enumerate(si["names"])}
    cols = np.array([row[str(c)] for c in z["channels"]])
    out = []
    for i, stage in enumerate(str(s) for s in z["stages"]):
        if np.isnan(z["source_power"][i]).all():
            continue
        for key, suffix, unit in (("source_power", "", "(A.m)^2"), ("source_relative", "_relative", None)):
            name = sanitize_node_name(f"bandpower_{stage}{suffix}")
            sub.field(name=name, path=f"frequency/{name}", kind="source power", manifold_id=on_cortex, data=z[key][i].T.astype(np.float32),
                      session=session, unit=unit, source=f.name, description=f"source power by band in {stage}, the clean epochs "
                      f"({'relative to the bands summed' if suffix else 'diag(K C Kᵀ), K the ' + kernel + ' kernel'})",
                      producer_json=jdump({"stage": stage, "relative": bool(suffix), "kernel": kernel, "measure": "power"}))
            out.append(name)
        sp = np.full((len(si["names"]), len(bands)), np.nan, dtype=np.float32)
        sp[cols] = z["sensor_power"][i].T
        name = sanitize_node_name(f"sensorpower_{stage}")
        sub.field(name=name, path=f"frequency/{name}", kind="sensor power", manifold_id=on_channels, data=sp, session=session, unit="V^2",
                  source=f.name, description=f"sensor power by band in {stage}, the clean epochs (a bad channel NaN)",
                  producer_json=jdump({"stage": stage, "relative": False, "measure": "power"}))
        out.append(name)
    return out
