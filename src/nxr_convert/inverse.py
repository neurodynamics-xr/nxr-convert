"""The inverse kernel and the head model — OPERATORS at the subject root, and what belongs to them (D40/D106/D107).

    <kernel>                Operator  float32 [nsrc, M]  channels → cortex (vector3 when free), from its channel Indices;
                                      its METHOD recorded (``params_json.inverse``), PAIRED with the leadfield it was
                                      computed from (``inverts_id``, regularized) when the ends swap
    <kernel>_channels       Selection indices: the M channels it maps, in column order
    <kernel>_noise_cov      Operator  the noise covariance C [M, M], channels → channels on those channels
    <kernel>_whitener       Operator  W = C^(-1/2) [M, M], likewise, derived from the covariance
    <kernel>_source_rr/_nn  Field     of the kernel (``of_operator``): kernel-row-ordered geometry on the cortex
    <headmodel>             Operator  float32 [M, 3V]    cortex (vector3) → channels (declared without bytes when no gain)
    <headmodel>_GridLoc/_GridOrient   Fields of the head model, on the cortex

Names are Brainstorm's file stems (``naming.py``). Brainstorm's ``Comment`` goes in ``comment``; the solver's options and
the Brainstorm links in ``producer_json``.

The inverse's source GEOMETRY lives with the kernel, not the head model: a free-
orientation kernel interleaves three components per vertex, so ``source_rr``/
``source_nn`` are kernel-row-ordered — a property of the KERNEL. SSP/compensation
are ABSORBED in ``ImagingKernel`` upstream (K·P = K to 6e-16), so no separate
projector step exists or is written.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, default_chunks, jdump


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_METHOD = {
    ("minnorm", "amplitude"): ("MNE", "MN"),
    ("minnorm", "dspm2018"): ("dSPM", "dSPM-unscaled"),
    ("minnorm", "sloreta"): ("sLORETA", "sLORETA"),
}
_UNITS = {"amplitude": "A.m", "dspm2018": "dSPM", "sloreta": "sLORETA"}


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.size(value) == 0
    return False


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def _set_if(attrs: dict, key: str, value: Any) -> None:
    if _empty(value):
        return
    attrs[key] = _scalar(value)


def _history(hist: Any) -> list[str] | None:
    if hist is None or (hasattr(hist, "__len__") and len(hist) == 0):
        return None
    rows = np.atleast_2d(np.asarray(hist, dtype=object))
    return [" | ".join(str(c) for c in row) for row in rows]


def _rel(p: Any, root: str) -> Any:
    s = str(p) if p is not None else ""
    if root and s.startswith(root):
        return s[len(root):].lstrip("/")
    return s or None


def inverse_method_of(p: dict[str, Any]) -> dict[str, Any] | None:
    """THE INVERSE METHOD, RECORDED (D107) — ``inverseMethodOf`` of ``inverse-method.ts``: what kind of inverse a kernel is,
    read once from its producer into ONE shape (``params_json.inverse``)."""
    if p.get("inverse_method") is None and p.get("inverse_measure") is None:
        return None
    s = lambda v: v if isinstance(v, str) else (v[0] if isinstance(v, list) and v and isinstance(v[0], str) else None)
    num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) else None
    m = (s(p.get("inverse_measure")) or "").lower()
    measure = ("MNE" if m in ("amplitude", "current") else "dSPM" if m.startswith("dspm") else "sLORETA" if m == "sloreta"
               else "eLORETA" if m == "eloreta" else s(p.get("method")) or (m or "unknown"))
    orientation = s(p.get("source_orient")) or s(p.get("source_ori")) or "unknown"
    snr = num(p.get("snr_fixed")) if s(p.get("snr_method")) == "fixed" else None
    depth = None
    if p.get("use_depth") is not False and num(p.get("weight_exp")) is not None:
        lim = num(p.get("weight_limit"))
        depth = {"exponent": num(p.get("weight_exp")), "limit": lim if lim is not None else None}
    return {"measure": measure, "estimate": "current" if measure == "MNE" else "statistic",
            "solver": s(p.get("inverse_method")) or "unknown", "orientation": orientation,
            "loose": num(p.get("loose")) if orientation == "loose" else None, "depth": depth,
            "noise": {"method": s(p.get("noise_method")) or "unknown", "reg": num(p.get("noise_reg"))},
            "snr": snr, "lambda2": 1 / (snr * snr) if snr else None,
            "from": "brainstorm" if isinstance(p.get("bst_results_file"), str) else "producer"}


def export_inverse(sub: Subject, kernel: dict, *, session: str, name: str, channels_id: str, surface_id: str,
                   forward_id: str | None = None, kernel_file: str = "", bst_root: str = "",
                   channel_names: list[str] | None = None, n_vertices: int | None = None,
                   grid_loc: np.ndarray | None = None, grid_orient: np.ndarray | None = None,
                   source_sha1: str | None = None) -> dict:
    k = np.asarray(kernel["ImagingKernel"])
    if k.size == 0:
        raise ValueError("not an inverse KERNEL results file (need .ImagingKernel)")
    nsrc, m = k.shape
    # WHICH CHANNELS — `GoodChannel` is 1-BASED (MATLAB); the Indices are 0-based, in the kernel's column order
    good = np.asarray(kernel.get("GoodChannel", []), dtype=np.int64).ravel()
    if good.size != m:
        raise ValueError(f"{name}: GoodChannel names {good.size} channels for a kernel with {m} columns")
    picked = (good - 1).astype(np.int32)
    ch_names = [channel_names[i] for i in picked] if channel_names is not None else []
    n_comp = int(kernel.get("nComponents", 1) or 1)
    is_free = n_comp == 3
    if n_vertices is not None and nsrc != n_vertices * max(1, n_comp):
        raise ValueError(f"ImagingKernel rows ({nsrc}) != nComponents*nV ({n_vertices * max(1, n_comp)})")

    opts = kernel.get("Options") or {}
    im = str(opts.get("InverseMethod", "")).lower()
    ims = str(opts.get("InverseMeasure", "")).lower()
    method, comment_tag = _METHOD.get((im, ims), ("unknown", ""))
    du = kernel.get("DisplayUnits")
    units = _UNITS.get(ims, "" if _empty(du) else str(du))
    producer: dict[str, Any] = {
        "method": method, "method_comment": comment_tag, "units": units,
        "ch_names": ch_names, "n_channels": int(m), "n_sources": int(nsrc),
        "n_components": n_comp, "source_ori": "free" if is_free else "fixed",
    }
    _set_if(producer, "inverse_method", opts.get("InverseMethod"))
    _set_if(producer, "inverse_measure", opts.get("InverseMeasure"))
    _set_if(producer, "function", kernel.get("Function") or opts.get("FunctionName"))
    so = opts.get("SourceOrient")
    _set_if(producer, "source_orient", [so] if isinstance(so, str) else so)
    for key, src in (("loose", "Loose"), ("weight_exp", "WeightExp"), ("weight_limit", "WeightLimit"),
                     ("noise_method", "NoiseMethod"), ("noise_reg", "NoiseReg"),
                     ("snr_method", "SnrMethod"), ("snr_rms", "SnrRms"), ("snr_fixed", "SnrFixed")):
        _set_if(producer, key, opts.get(src))
    _set_if(producer, "use_depth", None if _empty(opts.get("UseDepth")) else bool(_scalar(opts.get("UseDepth"))))
    for key, src in (("n_avg", "nAvg"), ("leff", "Leff"), ("display_units", "DisplayUnits"),
                     ("colormap_type", "ColormapType"), ("head_model_type", "HeadModelType"),
                     ("bst_data_file", "DataFile"), ("bst_head_model_file", "HeadModelFile"),
                     ("bst_surface_file", "SurfaceFile")):
        _set_if(producer, key, kernel.get(src))
    if kernel_file:
        producer["bst_results_file"] = _rel(kernel_file, str(bst_root))
    _set_if(producer, "bst_history", _history(kernel.get("History")))
    if source_sha1:
        producer["source_sha1"] = source_sha1
    producer = _jsonable(producer)
    stamp = _utc_now()

    sel_id = sub.selection(name=f"{name}_channels", path=f"{name}_channels", type="indices", manifold_id=channels_id,
                           data=picked, picked=True, session=session,
                           description=f"{name} — the {m} channels it maps, in column order")
    # THE PAIR (D106): W of G — a kernel derived from a leadfield whose ends are its swapped is its regularized inverse
    pair = {}
    if forward_id:
        g = sub.db.read("operator", forward_id)
        if (g and g["kind"] == "forward" and g["to_manifold_id"] == channels_id and g["from_manifold_id"] == surface_id
                and (g["from_value_type"] or ("vector3" if is_free else "scalar")) in (None, "vector3" if is_free else "scalar")
                and g["to_value_type"] in (None, "scalar")):
            pair = {"inverts_id": forward_id, "inverse_kind": "regularized"}
    inv = inverse_method_of(producer)
    kf32 = k.astype(np.float32)
    kid = sub.operator(name=name, path=name, kind="inverse kernel", from_manifold_id=channels_id, to_manifold_id=surface_id,
                       from_selection_id=sel_id, from_value_type="scalar", to_value_type="vector3" if is_free else "scalar",
                       layout="dense", n_rows=int(nsrc), n_cols=int(m), data=kf32, chunks=default_chunks(kf32.shape, 4),
                       compress=False, session=session, derived_from_id=forward_id,
                       description=f"{name} — sensor field to source field on the cortex ({method})",
                       comment=str(_scalar(kernel["Comment"])) if not _empty(kernel.get("Comment")) else None,
                       params_json=jdump({"inverse": inv}) if inv else None, producer_json=jdump(producer),
                       created_utc=stamp, **pair)

    # THE NOISE COVARIANCE AND THE WHITENER — operators on the kernel's channels (channels → channels)
    ncm = opts.get("NoiseCovMat")
    ncov = ncm.get("NoiseCov") if isinstance(ncm, dict) else None
    cov_id = None
    on_channels = {"from_manifold_id": channels_id, "to_manifold_id": channels_id, "from_selection_id": sel_id,
                   "to_selection_id": sel_id, "from_value_type": "scalar", "to_value_type": "scalar", "layout": "dense",
                   "session": session, "compress": False}
    if ncov is not None and np.size(ncov):
        c = np.asarray(ncov, dtype=np.float64)
        cov_id = sub.operator(name=f"{name}_noise_cov", path=f"{name}_noise_cov", kind="noise covariance", n_rows=c.shape[0],
                              n_cols=c.shape[1], data=c, producer_json=jdump({"kernel": name}),
                              description=f"the noise covariance {name} was computed with, on its {m} channels", **on_channels)
    whit = kernel.get("Whitener")
    if whit is not None and np.size(whit):
        w = np.asarray(whit, dtype=np.float64)
        sub.operator(name=f"{name}_whitener", path=f"{name}_whitener", kind="whitener", n_rows=w.shape[0], n_cols=w.shape[1],
                     data=w, derived_from_id=cov_id, producer_json=jdump({"kernel": name}),
                     description=f"the whitener of {name}'s noise covariance, on its {m} channels", **on_channels)

    # THE SOURCE GEOMETRY — kernel-row-ordered, fields OF the kernel on the cortex
    if grid_loc is not None and np.size(grid_loc):
        gl = np.asarray(grid_loc, dtype=np.float64)
        if is_free:
            rr, nn = np.repeat(gl, 3, axis=0), np.tile(np.eye(3), (gl.shape[0], 1))
        else:
            rr = gl
            nn = np.asarray(grid_orient, dtype=np.float64) if grid_orient is not None else None
        if rr.shape[0] != nsrc:
            raise ValueError(f"source geometry ({rr.shape[0]}) != kernel rows ({nsrc})")
        vt = "matrix3" if is_free else "vector3"        # rows 3V × 3 (one row a component) · V × 3
        sub.field(name=f"{name}_source_rr", path=f"{name}_source_rr", kind="source_rr", manifold_id=surface_id, value_type=vt,
                  data=rr.astype(np.float32), compress=False, of_operator_id=kid, session=session, unit="m")
        if nn is not None and np.size(nn):
            sub.field(name=f"{name}_source_nn", path=f"{name}_source_nn", kind="source_nn", manifold_id=surface_id,
                      value_type=vt, data=nn.astype(np.float32), compress=False, of_operator_id=kid, session=session)
    return {"name": name, "id": kid, "channels_id": sel_id, "method": method, "units": units, "nsrc": int(nsrc), "M": int(m),
            "n_components": n_comp}


def export_forward(sub: Subject, head_model: dict, *, session: str, name: str, channels_id: str, surface_id: str,
                   n_channels: int, n_vertices: int, head_model_file: str = "", bst_root: str = "", write_gain: bool = True,
                   source_sha1: str | None = None) -> dict:
    gl = np.asarray(head_model.get("GridLoc")) if head_model.get("GridLoc") is not None else np.empty((0, 3))
    producer: dict[str, Any] = {"units": "m", "n_sources": int(gl.shape[0]), "has_source_grid": bool(gl.size)}
    for key, src in (("head_model_type", "HeadModelType"), ("meg_method", "MEGMethod"),
                     ("eeg_method", "EEGMethod"), ("bst_surface_file", "SurfaceFile")):
        _set_if(producer, key, head_model.get(src))
    hmf = head_model.get("FileName") or (head_model_file or None)
    if hmf:
        producer["bst_head_model_file"] = _rel(hmf, str(Path(bst_root) / "data") if bst_root else "")
    if source_sha1:
        producer["source_sha1"] = source_sha1
    gain = head_model.get("Gain")
    g = np.asarray(gain).astype(np.float32) if (write_gain and gain is not None and np.size(gain)) else None
    # DECLARED, BYTES ABSENT — a head model without its gain is still the head model the kernel names
    hid = sub.operator(name=name, path=name, kind="forward", from_manifold_id=surface_id, to_manifold_id=channels_id,
                       from_value_type="vector3", to_value_type="scalar", layout="dense",
                       n_rows=int(g.shape[0]) if g is not None else int(n_channels), n_cols=int(g.shape[1]) if g is not None else 3 * int(n_vertices),
                       data=g, compress=True if g is not None else None, session=session,
                       description=f"{name} — the leadfield, source field to sensor field",
                       comment=str(_scalar(head_model["Comment"])) if not _empty(head_model.get("Comment")) else None,
                       producer_json=jdump(_jsonable(producer)), created_utc=_utc_now())
    if gl.size:
        sub.field(name=f"{name}_GridLoc", path=f"{name}_GridLoc", kind="GridLoc", manifold_id=surface_id, value_type="vector3",
                  data=gl.astype(np.float32), of_operator_id=hid, session=session, unit="m")
    go = head_model.get("GridOrient")
    if go is not None and np.size(go):
        sub.field(name=f"{name}_GridOrient", path=f"{name}_GridOrient", kind="GridOrient", manifold_id=surface_id,
                  value_type="vector3", data=np.asarray(go, dtype=np.float64).astype(np.float32), of_operator_id=hid, session=session)
    return {"name": name, "id": hid, "n_sources": int(gl.shape[0])}


def _jsonable(obj: Any) -> Any:
    """Plain JSON values; an integral float as an int (MATLAB's jsonencode writes 300, not 300.0)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and obj.is_integer() and abs(obj) < 2**53:
        return int(obj)
    return obj
