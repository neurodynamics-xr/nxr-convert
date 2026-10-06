"""
The HCP-1065 average tractography in the group space — the fibres of the dataset's average cortex.

Source: DSI Studio's HCP-1065 population-average tractography (``hcp1065.trk``, 105 221
streamlines on the ICBM152 2 mm grid). Its cortex is ICBM152's, as Brainstorm ships it in the
default anatomy (``@default_subject``): the fibre ends are assigned on the dense WHITE surface
(``tess_cortex_white_high``; tractography stops at the grey/white boundary) and carried to
fsaverage through that surface's registered sphere (``Reg.Sphere``, FreeSurfer's sphere.reg), the
same way every subject reaches the group tree. So a fibre end gets the fsaverage vertex and the
group-tree code of the place it lands, and the connectome of any level is the leaf × leaf count
with both codes shifted: ``C_ℓ[i >> k, j >> k] += 1`` (k = D − ℓ) — rolled up, never stored per level.

Coordinates. The .trk is read with the transform validated for the Brainstorm import
(``build_hcp1065_default_fibers.m``): MNI_mm = (−x + 79.5, −y + 81.5, z − 72) of the raw voxmm
points, i.e. nibabel's TrackVis reading shifted by (−1, −1, +1) mm (half a voxel; measured: the
ends sit closer to the cortex this way). MNI → the default anatomy's SCS is the affine fitted
from the SAME fibres as Brainstorm stored them (``tess_fibers_tess_hcp1065.mat``, 40 points,
ends exact): residual 0.0 mm, so the fit is Brainstorm's own ``cs_convert``.

``build_hcp`` answers the streamlines and, per fibre end [F, 2]: ``vertex`` (fsaverage, hemisphere-local, −1 off the
cortex), ``hemi`` (0 left, 1 right, −1), ``code`` (the group-tree code at the template's finest level, h·2^L + t) and
``dist_mm`` (end → ICBM152 white). The default subject stores them in our dialect (``default_subject``): the curve
manifold ``sources/hcp1065``, the csr ``hcp1065_ends`` and ``sources/hcp1065_connectome_L<k>``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .trees import HEMIS, Template, _unit

MNI_SHIFT_MM = np.array([-1.0, -1.0, 1.0])     # nibabel TrackVis RAS → the validated Brainstorm MNI


def read_trk(path: str | Path):
    """(streamlines points [P,3] MNI mm, offsets [F+1]) of a DSI Studio .trk."""
    import nibabel as nib
    t = nib.streamlines.load(str(path))
    s = t.tractogram.streamlines
    pts = np.asarray(s.get_data(), dtype=np.float64) + MNI_SHIFT_MM
    offsets = np.r_[0, np.cumsum(s._lengths)].astype(np.int64)
    return pts, offsets


def mni_to_scs_affine(trk_pts: np.ndarray, offsets: np.ndarray, bst_fibers_mat: str | Path) -> tuple[np.ndarray, float]:
    """The 4×3 affine MNI mm → default-anatomy SCS m, fitted on the fibre ends Brainstorm stored."""
    import scipy.io as sio
    P = sio.loadmat(str(bst_fibers_mat), squeeze_me=True, variable_names=["Points"])["Points"]
    if len(P) != len(offsets) - 1:
        raise ValueError(f"{bst_fibers_mat} has {len(P)} fibres, the .trk {len(offsets) - 1}")
    first, last = trk_pts[offsets[:-1]], trk_pts[offsets[1:] - 1]
    mni = np.stack([first, last], 1).reshape(-1, 3)
    scs = np.stack([P[:, 0, :], P[:, -1, :]], 1).reshape(-1, 3).astype(np.float64)
    X = np.c_[mni, np.ones(len(mni))]
    A, *_ = np.linalg.lstsq(X, scs, rcond=None)
    return A, float(np.abs(X @ A - scs).max() * 1000)


def _bst_surface(path: str | Path):
    """Vertices, registered sphere and per-vertex hemisphere (0/1, −1 none) of a Brainstorm cortex."""
    import scipy.io as sio
    c = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, variable_names=["Vertices", "Reg", "Atlas"])
    V = np.asarray(c["Vertices"], dtype=np.float64)
    sph = np.asarray(c["Reg"].Sphere.Vertices, dtype=np.float64)
    hemi = np.full(len(V), -1, dtype=np.int8)
    st = next(a for a in np.atleast_1d(c["Atlas"]) if a.Name == "Structures")
    for sc in np.atleast_1d(st.Scouts):
        h = 0 if sc.Label.endswith(" L") else 1 if sc.Label.endswith(" R") else None
        if h is not None:
            hemi[np.atleast_1d(sc.Vertices).astype(np.int64) - 1] = h
    return V, sph, hemi


def assign_ends(template: Template, trk_pts: np.ndarray, offsets: np.ndarray, A: np.ndarray,
                white_mat: str | Path, gate_mm: float = 5.0) -> dict[str, np.ndarray]:
    """Each fibre end → nearest ICBM152 white vertex (≤ gate) → its sphere point → nearest fsaverage vertex."""
    from scipy.spatial import cKDTree
    V, sph, hemi_of = _bst_surface(white_mat)
    ends = np.stack([trk_pts[offsets[:-1]], trk_pts[offsets[1:] - 1]], 1).reshape(-1, 3)
    scs = np.c_[ends, np.ones(len(ends))] @ A
    d, j = cKDTree(V).query(scs)
    d_mm = d * 1000
    h = hemi_of[j].astype(np.int8)
    ok = (d_mm <= gate_mm) & (h >= 0)
    vertex = np.full(len(ends), -1, dtype=np.int32)
    code = np.full(len(ends), -1, dtype=np.int32)
    L = template.finest
    for hh, name in enumerate(HEMIS):
        m = ok & (h == hh)
        _, k = cKDTree(template.sphere[name]).query(_unit(sph[j[m]]))
        vertex[m] = k
        code[m] = hh * 2 ** L + template.codes[name][k]
    h = np.where(ok, h, -1).astype(np.int8)
    shape = (len(offsets) - 1, 2)
    return {"vertex": vertex.reshape(shape), "hemi": h.reshape(shape), "code": code.reshape(shape),
            "dist_mm": d_mm.astype(np.float32).reshape(shape)}


def connectome(codes: np.ndarray, depth: int, level: int, weights: np.ndarray | None = None):
    """Symmetric sparse [2·2^level]² counts of the fibres whose BOTH ends are on the cortex.

    ``codes`` [F, 2] at ``depth``; the node of an end at ``level`` is code >> (depth − level).
    """
    from scipy.sparse import coo_matrix
    c = np.asarray(codes)
    both = (c >= 0).all(1)
    a, b = c[both, 0] >> (depth - level), c[both, 1] >> (depth - level)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)[both]
    n = 2 * 2 ** level
    C = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return C


def build_hcp(template: Template, trk: str | Path, bst_fibers_mat: str | Path, white_mat: str | Path,
              gate_mm: float = 5.0) -> dict:
    pts, offsets = read_trk(trk)
    A, resid = mni_to_scs_affine(pts, offsets, bst_fibers_mat)
    if resid > 0.01:
        raise ValueError(f"MNI → SCS fit residual {resid:.3f} mm: the .trk and the Brainstorm fibres disagree")
    ends = assign_ends(template, pts, offsets, A, white_mat, gate_mm)
    sha = hashlib.sha256(Path(trk).read_bytes()).hexdigest()[:16]
    cortical = (ends["code"] >= 0).all(1)
    meta = {"source": str(trk), "source_sha": sha, "fibres": int(len(offsets) - 1), "points": int(len(pts)),
            "cortical_fibres": int(cortical.sum()), "gate_mm": gate_mm,
            "end_to_white_median_mm": float(np.median(ends["dist_mm"])),
            "mni_shift_mm": MNI_SHIFT_MM.tolist(), "mni_to_scs": A.tolist(), "mni_to_scs_residual_mm": resid,
            "white_surface": str(white_mat), "brainstorm_fibres": str(bst_fibers_mat)}
    return {"points": pts.astype(np.float32), "offsets": offsets, "ends": ends, "meta": meta}
