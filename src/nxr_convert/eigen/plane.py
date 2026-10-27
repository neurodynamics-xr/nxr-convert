"""A SUBJECT'S ROTATION PLANE, stored — ``nxr-convert rotationplane``.

The top rotation plane of the band's analytic cross-spectrum, as two SENSOR filters: the same computation as ``research/results/fsaverage-two-subjects/native.py``
(``top_plane``), read through the subject's fused weights.

    Wc = [a ; μ b]   [2K × C]   the strength form of the subject's ``<kernel>_hodgeDirac_weights``
    F  = Wc · factor            factor: C = Xa Xaᴴ / T = V diag(e) Vᴴ, F = Wc V √e   (Xa the band's analytic signal, 2 s trimmed off each end)
    x, y = the top plane of Im(F Fᴴ)  (2K each);   fx, fy = Wcᵀ x, Wcᵀ y   (C each)

A recording read through them is z = fx·X + i fy·X. The sense is the rotary spectrum's: power at +f over −f in the band; when −f wins, fy is negated, so
+f is forward (x → y).

Stored on the subject:  ``<kernel>_rotationPlane``  Operator  kind 'rotation plane'  [2 × C]  rows fx, fy over the kernel's channels.
"""
from __future__ import annotations

import json

import numpy as np
import zarr
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from ..crud import array_meta, jdump, remove_node
from ..qc import _factors


def _top_plane(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The top rotation plane of a complex matrix's columns (native.py `top_plane`): two orthonormal real vectors."""
    R, J = F.real, F.imag
    Q, Rq = np.linalg.qr(np.hstack([R, J])); n = R.shape[1]
    M = np.zeros((2 * n, 2 * n)); M[n:, :n] = np.eye(n); M[:n, n:] = -np.eye(n)
    S = Rq @ M @ Rq.T; S /= np.linalg.norm(S)
    w, Z = np.linalg.eig(S); i = int(np.argmax(w.imag))
    v = Q @ Z[:, i]; x = v.real / np.linalg.norm(v.real); y = v.imag - (v.imag @ x) * x
    return x, y / np.linalg.norm(y)


def rotation_plane(W: np.ndarray, mu: np.ndarray, X: np.ndarray, fs: float, band=(8.0, 12.0)) -> tuple[np.ndarray, np.ndarray, int]:
    """The plane's sensor filters (fx, fy: [C] each) and the sense (+1 when the rotary peak was at +f; else fy is negated).

    W [2K × C] the fused weights, mu [K], X [C × T] the recording's channels."""
    K = W.shape[0] // 2
    Wc = np.vstack([W[:K], np.asarray(mu)[:, None] * W[K:]]).astype(np.float64)
    X = np.asarray(X, np.float64)
    Xa = hilbert(sosfiltfilt(butter(4, band, "bandpass", fs=fs, output="sos"), X, axis=1), axis=1)[:, int(2 * fs): -int(2 * fs)]
    e, V = np.linalg.eigh(Xa @ Xa.conj().T / Xa.shape[1])
    x, y = _top_plane(Wc @ (V * np.sqrt(np.clip(e, 0, None))))
    fx, fy = Wc.T @ x, Wc.T @ y
    Xb = sosfiltfilt(butter(4, (1.0, 30.0), "bandpass", fs=fs, output="sos"), X, axis=1)[:, int(2 * fs): -int(2 * fs)]
    f, P = welch(fx @ Xb + 1j * (fy @ Xb), fs=fs, nperseg=min(1024, Xb.shape[1]), return_onesided=False)
    a = (np.abs(f) >= band[0]) & (np.abs(f) <= band[1])
    sense = 1 if P[a & (f > 0)].sum() >= P[a & (f < 0)].sum() else -1
    return fx, sense * fy, sense


def store_rotation_plane(sub, *, kernel: str, recording: str, band=(8.0, 12.0), replace: bool = False) -> dict:
    """Compute and store ``<kernel>_rotationPlane`` on the subject (recording: its name; its kernel's channels are the rows' columns)."""
    db = sub.db
    k = db.one("SELECT * FROM operator WHERE subject_id = ? AND name = ?", sub.id, kernel)
    wr = db.one("SELECT * FROM operator WHERE subject_id = ? AND name = ?", sub.id, f"{kernel}_hodgeDirac_weights")
    rec = db.one("SELECT * FROM field WHERE subject_id = ? AND name = ?", sub.id, recording)
    if k is None or wr is None or rec is None:
        raise KeyError(f"{sub.row['name']}: kernel {kernel!r}, its _hodgeDirac_weights and recording {recording!r} must all exist")
    rd = lambda p: np.asarray(zarr.open_array(str(sub.at(p)), mode="r")[...])
    cols = rd(db.one("SELECT array_path FROM selection WHERE id = ?", k["from_selection_id"])["array_path"]).ravel().astype(int)
    _, t_id = _factors(db, rec["manifold_id"])
    fs = 1.0 / db.one("SELECT d.spacing FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.id = ?", t_id)["spacing"]
    fx, fy, sense = rotation_plane(rd(wr["path"]).astype(np.float64), np.asarray(json.loads(wr["params_json"])["mu"]), np.asarray(zarr.open_array(str(sub.at(rec["path"])), mode="r").oindex[cols, :]), fs, tuple(band))
    name = f"{kernel}_rotationPlane"
    old = db.one("SELECT * FROM operator WHERE subject_id = ? AND name = ?", sub.id, name)
    if old and not replace:
        raise FileExistsError(f"{sub.row['name']}: {name} exists (--replace)")
    if old:
        remove_node(db, sub.root, "operator", old["id"])
    data = np.stack([fx, fy]).astype(np.float32)
    oid = sub.operator(name=name, path=name, kind="rotation plane", from_manifold_id=k["from_manifold_id"], from_selection_id=k["from_selection_id"],
                       to_manifold_id=k["from_manifold_id"], layout="dense", n_rows=2, n_cols=len(cols), data=data, meta=array_meta(np.float32, data.shape, compress=False),
                       session=k["session"],
                       params_json=jdump({"kernel": k["id"], "band": list(band), "sense": sense, "recording": rec["id"], "names": ["pattern x", "pattern y"]}),
                       description=f"{kernel}'s top rotation plane in {tuple(band)} Hz: rows 0 and 1 the sensor filters of the two patterns (x → y forward)",
                       producer_json=jdump({"producer": "nxr-convert rotationplane (nxr_convert.eigen.plane)", "recording_name": recording}))
    return {"operator": name, "id": oid, "sense": sense, "channels": int(len(cols))}
