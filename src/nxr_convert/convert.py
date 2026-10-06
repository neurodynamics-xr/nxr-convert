"""ONE BRAINSTORM CONDITION into a subject — a SESSION of it, as entities (D141/D142).

A subject holds every session; a session is an ATTRIBUTE (``session``) on the rows it produced, never a folder. The
first condition brings the anatomy (the surface the kernels index, the primary); a later one adds a session to it.
Session-scoped nodes (channels, head models, kernels) are named by Brainstorm's stems and SHARED when a later session's
source is byte-identical, qualified otherwise (D121, ``naming.claim_node_name``).

    convert_condition(sub, …)     into a Subject being composed (``nxr-convert subject``, ``convert``)
    add_inverse(sub, …)           ONE more kernel into an existing subject (``nxr-convert inverse``)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crud import Subject
from .inverse import export_forward, export_inverse
from .matio import load_mat, load_mat_vars
from .naming import (channel_node_name, claim_node_name, headmodel_node_name, kernel_node_name, recording_node_name,
                     source_fingerprint, surface_node_name)
from .sensors import _channel_list, export_sensors
from .surface import export_surface, folder_for_surface
from .timeseries import export_raw_timeseries, export_timeseries

Progress = Callable[[dict[str, Any]], None]


def _emit(progress: Progress | None, **kw: Any) -> None:
    if progress:
        progress(kw)
    else:
        print(json.dumps(kw, default=str), file=sys.stderr)


def resolve_raw_binary(raw_link: str | Path, sfile: dict, progress: Progress | None = None) -> Path:
    """The binary a raw link reads (D122). Brainstorm records it as an ABSOLUTE path from the machine that wrote it, and
    relinks to the file beside the link when that path is gone (a protocol exported, copied or moved); so does this."""
    recorded = Path(str(sfile.get("filename", "")))
    if recorded.is_file():
        return recorded
    beside = Path(raw_link).parent / recorded.name
    if recorded.name and beside.is_file():
        _emit(progress, stage="relinked", link=Path(raw_link).name, **{"from": str(recorded), "to": str(beside)})
        return beside
    raise FileNotFoundError(f"{Path(raw_link).name}: its binary is neither at the recorded path {recorded} nor beside the link ({beside})")


def session_rows(sub: Subject, session: str) -> int:
    """How many node rows carry ``session`` — how a session is FOUND (it is an attribute, not a folder)."""
    return sum(sub.db.value(f"SELECT count(*) FROM {t} WHERE subject_id = ? AND session = ?", sub.id, session)
               for t in ("manifold", "field", "operator", "selection"))


def _spatial_head_models(cond_dir: Path) -> list[Path]:
    out = []
    for c in sorted(cond_dir.glob("headmodel_*.mat")):
        gl = load_mat_vars(c, ["GridLoc"]).get("GridLoc")
        if gl is not None and np.size(gl):
            out.append(c)
    return out


def convert_condition(sub: Subject, *, bst_root: str | Path, condition: str, surface_file: str | Path,
                      extra_surfaces: list[Path] | None = None, data_files: list[str | Path], channel_file: str | Path,
                      kernel_files: list[str | Path] | None = None, raw_data_file: str | Path | None = None,
                      head_model_map: dict[str, str] | None = None, write_gain: bool = True, require_source_grid: bool = True,
                      progress: Progress | None = None, all_surfaces: bool = False) -> dict:
    """Mirror one Brainstorm condition into the subject ``sub`` as the session ``condition``."""
    bst_root = Path(bst_root)
    if not data_files and raw_data_file is None:
        raise ValueError("nothing to convert: no data files and no raw link")
    if session_rows(sub, condition):
        raise ValueError(f"session {condition} already exists in {sub.name} — delete it first to re-import")
    cond_dir = Path(data_files[0]).parent if data_files else Path(channel_file).parent
    anat_dir = Path(surface_file).parent
    session = condition

    # ---- the anatomy: THE ONE SURFACE the leadfield is defined on (the canonical import, 2026-08-14) ----
    primary_surf = surface_node_name(surface_file)
    present = lambda f: sub.find("manifold", f"{folder_for_surface(surface_node_name(f))}/{surface_node_name(f)}") is not None
    if all_surfaces:
        surf_files = [f for f in sorted(anat_dir.glob("tess_*.mat")) if not f.name.startswith(("tess_fibers", "._"))]
    else:
        surf_files = [Path(surface_file), *(extra_surfaces or [])]
    names = []
    for sf in surf_files:
        if present(sf):
            continue
        if not Path(sf).exists():
            raise ValueError(f"subject has no anatomy ({Path(sf).name} not found under anat/) — a sensor-only subject cannot "
                             "import yet: the store is manifold-centric.")
        info = export_surface(sub, sf, bst_root=str(bst_root), primary=(surface_node_name(sf) == primary_surf))
        names.append(info["name"])
        _emit(progress, stage="surface", name=info["name"], folder=info["folder"], n_vertices=info["n_vertices"],
              hemispheres=info["hemispheres"])
    surface = sub.find("manifold", f"{folder_for_surface(primary_surf)}/{primary_surf}")
    if surface is None:
        raise ValueError(f"primary surface {primary_surf} not among exported {names}")
    surface_id, n_vertices = surface["id"], int(surface["n_vertices"])

    # SOURCE POWER BY BAND (D55/O31): the condition's timefreq_srcpow_*.mat on the cortex the kernel indexes
    from .timefreq import export_timefreq
    for tf in sorted({*Path(cond_dir).glob("timefreq_*.mat"), *(Path(cond_dir).parent / f"@raw{condition}").glob("timefreq_*.mat")}):
        if tf.name.startswith("._"):
            continue
        info = export_timefreq(sub, tf, session=session)
        if info:
            _emit(progress, stage="field", **info)

    # ---- the channels ----
    chan = load_mat(channel_file)
    first = load_mat(data_files[0]) if data_files else None
    if first is not None:
        flags = np.asarray(first["ChannelFlag"])
    else:
        raw_link = load_mat(raw_data_file)
        cf = raw_link.get("ChannelFlag")
        if cf is None and isinstance(raw_link.get("F"), dict):
            cf = raw_link["F"].get("channelflag")
        flags = np.asarray(cf) if cf is not None else np.ones(len(np.atleast_1d(chan["Channel"])))
    # SESSION-SCOPED NAMES (D121): shared when the file and the flags are identical, qualified otherwise
    chan_fp = source_fingerprint([channel_file], np.asarray(flags, dtype=np.int8).tobytes())
    chan_name, chan_shared = claim_node_name(sub, "timeseries", channel_node_name(channel_file), session, chan_fp)
    if chan_shared:
        row = sub.find("manifold", f"timeseries/{chan_name}")
        ch_names = [str(c["Name"]) for c in _channel_list(chan)]
        si = {"name": chan_name, "id": row["id"], "n_channels": int(row["n_vertices"]), "names": ch_names}
        _emit(progress, stage="sensors-shared", name=chan_name, n_channels=si["n_channels"])
    else:
        si = export_sensors(sub, chan, flags, name=chan_name, session=session, source_sha1=chan_fp)
        _emit(progress, stage="sensors", name=si["name"], n_channels=si["n_channels"])

    # ---- the recording(s) ----
    for i, df in enumerate(data_files):
        data = first if (i == 0 and first is not None) else load_mat(df)
        ti = export_timeseries(sub, recording_node_name(df), data, session=session, channels_id=si["id"], channel_names=si["names"])
        _emit(progress, stage="timeseries", **{k: ti[k] for k in ("name", "n_samples", "sfreq", "events")})
    if raw_data_file is not None:
        raw_mat = load_mat(raw_data_file)
        sfile = raw_mat["F"]
        if not isinstance(sfile, dict):
            raise ValueError(f"{raw_data_file} is not a raw link (F is a matrix)")
        ri = export_raw_timeseries(sub, recording_node_name(raw_data_file), sfile, raw_mat.get("ChannelFlag"), session=session,
                                   channels_id=si["id"], comment=raw_mat.get("Comment"),
                                   bst_path=resolve_raw_binary(raw_data_file, sfile, progress), channel_names=si["names"],
                                   progress=progress)
        _emit(progress, stage="timeseries-raw", **{k: v for k, v in ri.items() if k not in ("id", "time_id")})

    # ---- kernels (+ head models, deduped per file) ----
    if kernel_files is None:
        kernel_files = sorted(cond_dir.glob("results_*KERNEL*.mat"))
    fwd_done: dict[str, dict] = {}
    inv_names: list[str] = []
    for kf in kernel_files:
        if Path(kf).name.startswith("._"):
            continue
        kern = load_mat(kf)
        if "ImagingKernel" not in kern or not np.size(kern.get("ImagingKernel")):
            _emit(progress, stage="skip-kernel", file=str(kf), reason="no ImagingKernel")
            continue
        hm_link = str(kern.get("HeadModelFile") or "")
        hm_path = (head_model_map or {}).get(hm_link) or (str(bst_root / "data" / hm_link) if hm_link else "")
        if hm_path and not Path(hm_path).is_file():
            # Brainstorm never rewrites a kernel's HeadModelFile when the head model is renamed: stale links are NORMAL.
            # One spatial head model beside it is the answer; several stay a skip.
            spatial = _spatial_head_models(cond_dir)
            if len(spatial) == 1:
                _emit(progress, stage="stale-headmodel-link", file=str(kf), link=hm_link, resolved=spatial[0].name)
                hm_path = str(spatial[0])
            else:
                _emit(progress, stage="skip-kernel", file=str(kf), reason=f"head model missing: {hm_link}",
                      candidates=[c.name for c in spatial])
                continue
        gl = go = None
        fwd_id: str | None = None
        if hm_path:
            hm = load_mat(hm_path)
            gl_arr = np.asarray(hm.get("GridLoc")) if hm.get("GridLoc") is not None else np.empty(0)
            if require_source_grid and not gl_arr.size:
                _emit(progress, stage="skip-kernel", file=str(kf), reason="modal head model (empty GridLoc)")
                continue
            if hm_path not in fwd_done:
                hm_fp = source_fingerprint([hm_path], f"{si['id']}|{surface_id}|{write_gain}".encode())
                hm_name, hm_shared = claim_node_name(sub, "", headmodel_node_name(hm_path), session, hm_fp)
                if hm_shared:
                    fi = {"name": hm_name, "id": sub.find("operator", hm_name)["id"]}
                    _emit(progress, stage="forward-shared", name=hm_name)
                else:
                    fi = export_forward(sub, hm, session=session, name=hm_name, channels_id=si["id"], surface_id=surface_id,
                                        n_channels=si["n_channels"], n_vertices=n_vertices, head_model_file=hm_path,
                                        bst_root=str(bst_root), write_gain=write_gain, source_sha1=hm_fp)
                    _emit(progress, stage="forward", name=fi["name"])
                fwd_done[hm_path] = {"name": fi["name"], "id": fi["id"], "GridLoc": hm.get("GridLoc"), "GridOrient": hm.get("GridOrient")}
            f = fwd_done[hm_path]
            fwd_id, gl, go = f["id"], f["GridLoc"], f["GridOrient"]
        k_fp = source_fingerprint([kf], f"{si['id']}|{surface_id}|{fwd_id}".encode())
        k_name, k_shared = claim_node_name(sub, "", kernel_node_name(kf), session, k_fp)
        if k_shared:
            ii = {"name": k_name, "method": json.loads(sub.find("operator", k_name)["producer_json"] or "{}").get("method", "unknown")}
        else:
            ii = export_inverse(sub, kern, session=session, name=k_name, channels_id=si["id"], surface_id=surface_id,
                                forward_id=fwd_id, kernel_file=str(kf), bst_root=str(bst_root), channel_names=si["names"],
                                n_vertices=n_vertices, grid_loc=np.asarray(gl) if gl is not None else None,
                                grid_orient=np.asarray(go) if go is not None else None, source_sha1=k_fp)
        inv_names.append(ii["name"])
        _emit(progress, stage="inverse", name=ii["name"], method=ii["method"])
    if not inv_names:
        # NO KERNEL IS NOT AN ERROR (2026-08-13): a legitimate anatomy + timeseries session
        _emit(progress, stage="no-kernel", message="no inverse kernel in this condition — the session is anatomy + timeseries")
    return {"subject": sub.name, "surfaces": names, "inverses": inv_names, "session": session}


# ---------------------------------------------------------------------------
# ONE MORE KERNEL INTO AN EXISTING SUBJECT
# ---------------------------------------------------------------------------

def _kernel_head_model(kernel_file: Path, kern: dict, bst_root: Path, progress: Progress | None) -> Path | None:
    """The head model a kernel names: the recorded link, else — a stale link is normal — the one spatial head model beside it."""
    link = str(kern.get("HeadModelFile") or "")
    if not link:
        return None
    p = bst_root / "data" / link
    if p.is_file():
        return p
    spatial = _spatial_head_models(kernel_file.parent)
    if len(spatial) == 1:
        _emit(progress, stage="stale-headmodel-link", file=str(kernel_file), link=link, resolved=spatial[0].name)
        return spatial[0]
    raise FileNotFoundError(f"{kernel_file.name}: head model {link} is missing and {len(spatial)} candidates are beside it")


def add_inverse(sub: Subject, *, kernel_file: str | Path, session: str, bst_root: str | Path, channels_id: str | None = None,
                surface_id: str | None = None, forward_id: str | None = None, name: str | None = None,
                channel_file: str | Path | None = None, progress: Progress | None = None) -> dict:
    """ONE inverse kernel (+ its channel Indices, covariance, whitener and source geometry) into an existing subject.

    References are rows of the subject: by id when given, else found — the session's channels, the cortex the kernel's
    own ``SurfaceFile`` names, the session's head model of the kernel's ``HeadModelFile``. Its name is Brainstorm's stem,
    session-qualified if taken (D121); a source already in this session is REFUSED — this never overwrites."""
    kernel_file, bst_root = Path(kernel_file), Path(bst_root)
    kern = load_mat(kernel_file)
    if "ImagingKernel" not in kern or not np.size(kern.get("ImagingKernel")):
        raise ValueError(f"{kernel_file.name}: not an inverse KERNEL results file (no ImagingKernel)")
    if channels_id is None:
        ch = sub.db.one("SELECT id FROM manifold WHERE subject_id = ? AND session = ? AND type = 'points' AND path LIKE 'timeseries/%' "
                        "ORDER BY path LIMIT 1", sub.id, session)
        if ch is None:
            raise ValueError(f"{sub.name}: no channels in session {session}")
        channels_id = ch["id"]
    sf = str(kern.get("SurfaceFile") or "")
    if surface_id is None:
        sname = surface_node_name(Path(sf).name) if sf else None
        row = sub.find("manifold", f"{folder_for_surface(sname)}/{sname}") if sname else None
        if row is None:
            raise ValueError(f"{kernel_file.name}: its surface {sf or '(none)'} is not a manifold of {sub.name}")
        surface_id = row["id"]
    surface = sub.db.read("manifold", surface_id)
    hm = _kernel_head_model(kernel_file, kern, bst_root, progress)
    gl = go = None
    if hm is not None:
        g = load_mat_vars(hm, ["GridLoc", "GridOrient"])
        gl = np.asarray(g["GridLoc"]) if np.size(g.get("GridLoc")) else None
        go = np.asarray(g["GridOrient"]) if np.size(g.get("GridOrient")) else None
        if forward_id is None:
            row = sub.db.one("SELECT id FROM operator WHERE subject_id = ? AND kind = 'forward' AND session = ? AND name LIKE ? "
                             "ORDER BY name LIMIT 1", sub.id, session, headmodel_node_name(hm) + "%")
            forward_id = row["id"] if row else None
    fp = source_fingerprint([kernel_file], f"{channels_id}|{surface_id}|{forward_id}".encode())
    if name is None:
        stem = kernel_node_name(kernel_file)
        had = sub.find("operator", stem)
        if had and (had["session"] == session or json.loads(had["producer_json"] or "{}").get("source_sha1") == fp):
            raise ValueError(f"{sub.name}: {stem} is already in session {session} — refusing to add it twice")
        name, _ = claim_node_name(sub, "", stem, session, fp)
    parts = [name, f"{name}_channels", f"{name}_whitener", f"{name}_noise_cov", f"{name}_source_rr", f"{name}_source_nn"]
    taken = [p for p in parts if any(sub.find(t, p) for t in ("operator", "selection", "field", "manifold"))]
    if taken:
        raise ValueError(f"{sub.name}: refusing to overwrite {', '.join(taken)}")
    if channel_file is None:
        chans = sorted(p for p in kernel_file.parent.glob("channel*.mat") if not p.name.startswith("._"))
        if len(chans) != 1:
            raise FileNotFoundError(f"{kernel_file.parent}: {len(chans)} channel*.mat — pass --channel-file")
        channel_file = chans[0]
    channel_names = [str(c["Name"]) for c in _channel_list(load_mat(channel_file))]
    info = export_inverse(sub, kern, session=session, name=name, channels_id=channels_id, surface_id=surface_id,
                          forward_id=forward_id, kernel_file=str(kernel_file), bst_root=str(bst_root), channel_names=channel_names,
                          n_vertices=int(surface["n_vertices"]), grid_loc=gl, grid_orient=go, source_sha1=fp)
    _emit(progress, stage="inverse", **info)
    return info
