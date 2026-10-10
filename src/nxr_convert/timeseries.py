"""A RECORDING — a FIELD on channels × time, and everything that describes it, as entities (D141).

    timeseries/<rec>_time       Manifold  line (lattice, rank 1) in closed form: n samples, spacing 1/sfreq, origin t0 (s)
    channels × <rec>_time       Manifold  the PRODUCT the recording is on (D37) — no node, its subject's root carries it
    timeseries/<rec>            Field     float32 [C, T] on the product, sharded on the grid (``grid.py``)
    <rec>_chunks                Selection the CHUNK GRID (D46/D57): a closed-form spans tiling of the time Line in the
                                          stored inner chunk — no node
    timeseries/<rec>_envelope   array     min/max per 32-sample window [C, W, 2]; <rec>_envelope (no node) the closed-form
                                          tiling and its two ARRAY-BACKED measurements, min and max (D96)
    timeseries/<rec>_flags      Selection the recording's own channel flags, a partition of the channels (good/bad)
    timeseries/<rec>_events     Selection its Brainstorm events, spans on the time Line (D143); _bad_segments the bad ones

A raw link (``Data.F`` is a struct pointing at the original binary) goes to ``export_raw_timeseries``; an imported excerpt
to ``export_timeseries``. Both share one writer of everything after the bytes, so they cannot drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crud import Subject, jdump, lattice_counts, populate_array
from .db import ensure_groups
from .entities import flag_levels, labelling
from .events import export_events, parse_events
from .grid import BLOCK, WINDOW, EnvelopeTee, envelope_path, recording_meta


def _time_line(sub: Subject, path: str, *, session: str, n_samples: int, sfreq: float, t0: float) -> str:
    """The Time Line: three numbers, exactly as a Volume's voxel positions are — a closed form, no array."""
    return sub.manifold(name=path.rsplit("/", 1)[-1], path=path, type="line", n_vertices=int(n_samples), session=session,
                        topology={"kind": "lattice", "rank": 1}, **lattice_counts([n_samples]),
                        geometry={"form": "closed_form", "cs_key": "clock",
                                  "dimensions": [{"name": "time", "n_vertices": int(n_samples), "spacing": 1.0 / float(sfreq),
                                                  "origin": float(t0), "unit": "s"}]})


def _write_flags(sub: Subject, rec_path: str, flags: Any, *, session: str, channels_id: str, of_id: str) -> None:
    f = np.asarray(flags).ravel()
    if not f.size:
        return
    codes, levels, colors = flag_levels(f, f"{rec_path}: ")
    labelling(sub, f"{rec_path}_flags", codes, levels, manifold_id=channels_id,
              dictionary=f"{rec_path.rsplit('/', 1)[-1]}_flags", scope="subject", colors=colors,
              session=session, description="the recording's own channel flags", of_field_id=of_id)


def _recording(sub: Subject, name: str, *, n_chan: int, n_samples: int, sfreq: float, t0: float, session: str,
               channels_id: str, write: Callable[[Any], None], flags: Any, events: Any, channel_names: list[str] | None,
               comment: str | None, kind: str, source: str, producer: dict[str, Any], progress=None) -> dict:
    rec = f"timeseries/{name}"
    time_id = _time_line(sub, f"{rec}_time", session=session, n_samples=n_samples, sfreq=sfreq, t0=t0)
    product = sub.product([channels_id, time_id], session=session)
    meta = recording_meta(n_chan, n_samples)

    def populate(at: Path) -> None:
        ensure_groups(sub.store, envelope_path(rec), self_too=False)
        write(EnvelopeTee(populate_array(at), sub.at(envelope_path(rec))))
        if progress:
            progress({"stage": "envelope", "name": name})
    rid = sub.field(name=name, path=rec, kind="recording", manifold_id=product, meta=meta, populate=populate, session=session,
                    description=f"{session} — the {kind} recording", comment=str(comment) if comment else None, source=source,
                    producer_json=jdump({"kind": kind, **producer}))
    dt = 1.0 / float(sfreq)
    # THE CHUNK GRID (D46): closed form — four numbers state a regular grid, `at(k)` derives every member
    chunk = meta.chunks[1]
    n = -(-n_samples // chunk)
    sub.selection(name=f"{name}_chunks", path=None, type="spans", manifold_id=time_id, cell="1", of_field_id=rid, session=session,
                  description=f"the {n} stored chunks of {chunk} samples", n_elements=int(n_samples), n_members=n,
                  params_json=jdump({"tiling": {"origin": 0, "width": chunk * dt, "hop": chunk * dt, "count": n}, "end": n_samples * dt}))
    # THE ENVELOPE (D96): a closed-form tiling of the time Line with two array-backed measurements
    w = -(-n_samples // WINDOW)
    env = envelope_path(rec)
    sub.selection(name=f"{name}_envelope", path=None, type="spans", manifold_id=time_id, cell="1", of_field_id=rid, session=session,
                  description=f"the {w} windows of {WINDOW} samples the recording's min and max are measured over",
                  n_elements=int(n_samples), n_members=w,
                  params_json=jdump({"tiling": {"origin": t0, "width": WINDOW * dt, "hop": WINDOW * dt, "count": w}, "end": t0 + n_samples * dt}),
                  measurements=[{"measure": "min", "of_field_id": rid, "array_path": env, "component": 0},
                                {"measure": "max", "of_field_id": rid, "array_path": env, "component": 1}])
    if flags is not None and np.size(flags):
        _write_flags(sub, rec, flags, session=session, channels_id=channels_id, of_id=rid)
    ev = export_events(sub, rec, parse_events(events, channel_names), line_id=time_id, of_id=rid, session=session)
    return {"name": name, "id": rid, "time_id": time_id, "n_channels": int(n_chan), "n_samples": int(n_samples), "sfreq": sfreq,
            "events": ev["n_events"], "event_types": ev["n_types"], "bad_segments": ev["bad_segments"]}


def export_raw_timeseries(sub: Subject, name: str, sfile: dict, channel_flag: np.ndarray | None, *, session: str,
                          channels_id: str, comment: str | None = None, bst_path: str | Path | None = None,
                          block: int = BLOCK, channel_names: list[str] | None = None, progress=None) -> dict:
    """A RAW LINK's native-rate recording, block-streamed from the BST-BIN binary. Byte-faithful: the ``.bst`` already
    carries the processed (SSP-applied, notch-filtered) signal (measured: skipping the projector reproduces the MATLAB
    export to 6e-13 through three cascaded decimations — the bytes are the truth)."""
    from .raw import read_bst

    # THE PREMISE, enforced: byte-faithful ONLY for Brainstorm's own preprocessed binary
    fmt = str(sfile.get("format", "")) or "unknown"
    if fmt != "BST-BIN":
        raise ValueError(f"raw link format '{fmt}' is not supported — only a Brainstorm-preprocessed continuous (BST-BIN) "
                         "imports byte-faithfully. Run the recording through a Brainstorm process first, which writes one.")
    prop = sfile.get("prop", {})
    cur, dest = prop.get("currCtfComp"), prop.get("destCtfComp")
    cur_a = np.asarray(cur) if cur is not None else None             # EEG: an EMPTY array, not a missing field
    dest_a = np.asarray(dest) if dest is not None else None
    if (cur_a is not None and dest_a is not None and cur_a.size == 1 and dest_a.size == 1 and int(cur_a.item()) != int(dest_a.item())):
        raise ValueError(f"raw link's CTF compensation is not at destination order (curr={cur}, dest={dest}) — apply "
                         "compensation in Brainstorm first.")
    header = sfile["header"]
    n_chan, n_t = int(header["nchannels"]), int(header["nsamples"])
    sfreq = float(np.asarray(sfile["prop"]["sfreq"]).item())
    t0 = float(np.atleast_1d(np.asarray(sfile["prop"]["times"]))[0])

    def write(arr) -> None:
        for s0 in range(0, n_t, block):
            n = min(block, n_t - s0)
            arr[:, s0:s0 + n] = read_bst(sfile, s0=s0, n=n, path=bst_path)
            if progress:
                progress({"stage": "raw-stream", "name": name, "written": s0 + n, "of": n_t})
    return _recording(sub, name, n_chan=n_chan, n_samples=n_t, sfreq=sfreq, t0=t0, session=session, channels_id=channels_id,
                      write=write, flags=channel_flag, events=sfile.get("events"), channel_names=channel_names, comment=comment,
                      kind="sensor", source="raw-link", progress=progress,
                      producer={"format": fmt, "device": str(sfile.get("device", "")), **({"bst_file": str(bst_path)} if bst_path else {})})


def export_timeseries(sub: Subject, name: str, data_mat: dict, *, kind: str = "sensor", session: str, channels_id: str,
                      channel_names: list[str] | None = None) -> dict:
    """An IMPORTED excerpt (``data_*.mat``, ``F`` a matrix) — written a shard at a time."""
    f = data_mat.get("F")
    if isinstance(f, dict):
        raise NotImplementedError(f"timeseries {name!r} is a RAW LINK (F is an sFile struct) — pass it to export_raw_timeseries")
    f = np.asarray(f)
    t = np.asarray(data_mat["Time"], dtype=np.float64).ravel()
    if t.size < 2:
        raise ValueError(f"{name}: not a Brainstorm data file (need .Time with >=2 samples)")
    sfreq = 1.0 / float(np.mean(np.diff(t)))
    n_chan, n_t = f.shape
    f32 = f.astype(np.float32, copy=False)

    def write(arr) -> None:
        for s0 in range(0, n_t, BLOCK):
            arr[:, s0:s0 + BLOCK] = f32[:, s0:s0 + BLOCK]
    return _recording(sub, name, n_chan=n_chan, n_samples=n_t, sfreq=sfreq, t0=float(t[0]), session=session, channels_id=channels_id,
                      write=write, flags=data_mat.get("ChannelFlag"), events=data_mat.get("Events"), channel_names=channel_names,
                      comment=data_mat.get("Comment"), kind=kind, source="imported",
                      producer={"data_type": str(data_mat.get("DataType", ""))})
