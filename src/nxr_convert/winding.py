"""Face-winding canonicalization — the port of ``canonicalize_winding.m``.

Brainstorm winds faces CW-for-outward; nxr-compute (geometry-central) and
three.js both read CCW as front. Canonicalizing at export makes the stored
faces the single source of truth: after this, ``cross(B-A, C-A)`` is the
outward normal and no reader needs a flip or stored vertex normals.

Detection is purely geometric, per connected component: each hemisphere is a
watertight closed surface, so a NEGATIVE signed volume means that component's
winding is inward, and the flip is GLOBAL within the component (a partial flip
would corrupt the oriented manifold geometry-central requires). Idempotent —
an already-CCW-outward component is untouched. Faces are 0-BASED here (the
caller converts from Brainstorm's 1-based before or after; this module does
not care, only consistency with ``vertices`` indexing matters).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def canonicalize_winding(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return ``(faces_canonical, info)`` with CCW-outward winding per component.

    ``vertices``: float array [nV, 3]. ``faces``: int array [nF, 3], 0-based.
    ``info``: {"n_components": int, "flipped": [component indices]}.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces).copy()
    n_v = v.shape[0]

    # Vertex adjacency from the three directed edges of every face.
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    adj = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_v, n_v))
    n_comp, comp = connected_components(adj, directed=False)
    fcomp = comp[f[:, 0]]

    flipped: list[int] = []
    for c in range(n_comp):
        idx = np.nonzero(fcomp == c)[0]
        if idx.size == 0:
            continue
        fi = f[idx]
        v0, v1, v2 = v[fi[:, 0]], v[fi[:, 1]], v[fi[:, 2]]
        vol = float(np.sum(v0 * np.cross(v1, v2)) / 6.0)
        if abs(vol) < np.finfo(np.float32).eps:
            # Open / non-watertight component: no orientation fact to act on.
            continue
        if vol < 0:
            f[idx[:, None], [1, 2]] = fi[:, [2, 1]]
            flipped.append(c)

    return f, {"n_components": int(n_comp), "flipped": flipped}
