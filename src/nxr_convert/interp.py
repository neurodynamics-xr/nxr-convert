"""The INTERPOLATION operator between two surfaces of one subject, through their FreeSurfer registration spheres.

``<DST>_from_<SRC>``  Operator  csr [V_DST × V_SRC]: each DST vertex in barycentric weights of the SRC triangle that holds it
                                ON THE SPHERE (3 weights a row, ≥ 0, summing to 1; a vertex both surfaces share is one 1).
Both hemispheres lie on the one sphere, so each DST vertex is located among the SAME hemisphere's SRC triangles (the
hemisphere components ``<name>_1``, ``<name>_2``, matched by label)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import zarr

from .crud import Subject, jdump, remove_node

EDGE_TOL = 1e-7                                          # barycentric slack for a point on an edge
GATHER_TOL = 1e-9                                        # on the unit sphere: a DST vertex that IS a SRC vertex


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _locate(P: np.ndarray, V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For unit points P: (the triangle of F that holds each, its barycentric weights [n × 3]). The point is projected from
    the sphere's centre onto the triangle's plane (so a great-circle edge stays an edge); the k nearest centroids are tried,
    k growing from 16 until every point has a triangle."""
    from scipy.spatial import cKDTree
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nrm = np.cross(B - A, C - A)
    d0 = (nrm * A).sum(1)
    tree = cKDTree(_unit(A + B + C))
    tri, w = np.full(len(P), -1), np.zeros((len(P), 3))
    todo, k = np.arange(len(P)), 16
    while todo.size:
        k = min(k, len(F))
        _, cand = tree.query(P[todo], k=k)
        cand = cand.reshape(len(todo), -1)
        p = P[todo][:, None, :]
        t = d0[cand] / np.maximum((nrm[cand] * p).sum(2), 1e-300)           # the ray p·t meets the plane
        q = p * t[..., None] - A[cand]
        e1, e2 = (B - A)[cand], (C - A)[cand]
        a, b, c = (e1 * e1).sum(2), (e1 * e2).sum(2), (e2 * e2).sum(2)
        r1, r2 = (q * e1).sum(2), (q * e2).sum(2)
        den = a * c - b * b
        u, v = (c * r1 - b * r2) / den, (a * r2 - b * r1) / den
        bary = np.stack([1 - u - v, u, v], -1)
        best = bary.min(-1)
        j = best.argmax(1)
        ok = (best[np.arange(len(todo)), j] >= -EDGE_TOL) & (t[np.arange(len(todo)), j] > 0)
        sel = todo[ok]
        tri[sel], w[sel] = cand[ok, j[ok]], bary[ok, j[ok]][:]
        todo = todo[~ok]
        if todo.size and k == len(F):
            raise ValueError(f"{todo.size} vertices lie in no triangle of the source sphere")
        k *= 4
    w = np.clip(w, 0, None)
    return tri, w / w.sum(1, keepdims=True)


def _surface(s: Subject, name: str) -> dict:
    row = s.db.one("SELECT m.*, g.positions_path FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                   "WHERE m.subject_id = ? AND m.type = 'surface' AND m.name = ?", s.id, name)
    if row is None:
        have = [r["name"] for r in s.db.all("SELECT name FROM manifold WHERE subject_id = ? AND type = 'surface' ORDER BY name", s.id)]
        raise KeyError(f"{s.row['name']}: no surface named {name!r}; it has {', '.join(have) or 'none'}")
    sph = s.db.one("SELECT m.*, g.positions_path FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                   "WHERE m.subject_id = ? AND m.name = ?", s.id, f"{name}_sphere")
    if sph is None:
        raise ValueError(f"{name}: no registration sphere ({name}_sphere); interpolation goes through the spheres")
    hemis = s.db.all("SELECT label, from_index, n_vertices FROM manifold WHERE subject_id = ? AND component_of_id = ? "
                     "ORDER BY component_ordinal", s.id, row["id"])
    if len(hemis) != 2 or {h["label"] for h in hemis} != {"left", "right"}:
        raise ValueError(f"{name}: no left/right hemisphere components ({name}_1, {name}_2); both are needed to keep the hemispheres apart")
    rd = lambda path: np.asarray(zarr.open_array(str(s.store / path), mode="r")[:])
    return {"row": row, "sphere": _unit(rd(sph["positions_path"]).astype(np.float64)), "faces": rd(f"{row['path']}/faces").astype(np.int64),
            "hemis": {h["label"]: (int(h["from_index"]), int(h["n_vertices"])) for h in hemis}}


def add_interpolation(s: Subject, src: str, dst: str, *, replace: bool = False) -> dict:
    """Write ``<dst>_from_<src>`` and return its checks."""
    from scipy.spatial import cKDTree
    name = f"{dst}_from_{src}"
    have = s.find("operator", name)
    if have and not replace:
        raise FileExistsError(f"{s.row['name']}: {name} exists; pass --replace")
    S, D = _surface(s, src), _surface(s, dst)
    rows, cols, vals, exact = [], [], [], 0
    for side in ("left", "right"):
        (s0, ns), (d0, nd) = S["hemis"][side], D["hemis"][side]
        V, P = S["sphere"][s0:s0 + ns], D["sphere"][d0:d0 + nd]
        F = S["faces"][(S["faces"] >= s0).all(1) & (S["faces"] < s0 + ns).all(1)] - s0
        dist, near = cKDTree(V).query(P)
        gather = dist <= GATHER_TOL
        tri, w = _locate(P[~gather], V, F)
        i = np.where(~gather)[0]
        rows += [np.repeat(d0 + i, 3), d0 + np.where(gather)[0]]
        cols += [(s0 + F[tri]).ravel(), s0 + near[gather]]
        vals += [w.ravel(), np.ones(int(gather.sum()))]
        exact += int(gather.sum())
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(int(D["row"]["n_vertices"]), int(S["row"]["n_vertices"])))
    M.sum_duplicates()
    M.sort_indices()
    checks = {"max_row_sum_error": float(np.abs(np.asarray(M.sum(1)).ravel() - 1).max()), "min_weight": float(M.data.min()),
              "exact_gather_share": exact / M.shape[0]}
    x = M @ S["sphere"][:, 0]                            # a smooth field, SRC → DST, against DST's own
    checks["roundtrip_rms_over_sd"] = float(np.sqrt(((x - D["sphere"][:, 0]) ** 2).mean()) / D["sphere"][:, 0].std())
    if have:
        remove_node(s.db, s.root, "operator", have["id"])
    s.operator(name=name, path=name, kind="interpolation", from_manifold_id=S["row"]["id"], to_manifold_id=D["row"]["id"],
               layout="csr", n_rows=M.shape[0], n_cols=M.shape[1], nnz=int(M.nnz),
               sparse={"indptr": M.indptr.astype(np.int64), "indices": M.indices.astype(np.int32), "data": M.data.astype(np.float64)},
               description=f"each {dst} vertex in barycentric weights of the {src} triangle holding it on the registration sphere",
               params_json=jdump({"method": "barycentric on the registration sphere (centre projection, k-nearest centroids)",
                                  "hemispheres": "each destination vertex is located among the same hemisphere's source triangles",
                                  "checks": checks}))
    return {"operator": name, "n_rows": M.shape[0], "n_cols": M.shape[1], "nnz": int(M.nnz), **checks}
