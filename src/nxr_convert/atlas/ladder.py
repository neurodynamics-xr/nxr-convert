"""
The dyadic surface ladder — cortical-flow's ``surfaceLadder`` (backend/src/db/surface-ladder.ts), in numpy.

Each level halves every tile of the level above by EQUAL MASS across its geodesic diameter (the
geodesic stand-in for Fiedler bisection), so the ladder is nested by construction and the tiles
of a level carry equal area. Level ℓ has 2^ℓ tiles; a tile t at level ℓ has children 2t and 2t+1,
so THE LEVELS ARE ARITHMETIC: the tile of a vertex at level ℓ ≤ L is its level-L tile >> (L − ℓ).
The ladder stops before a level that would leave a tile empty (the mesh's own Nyquist).

The port follows the TypeScript step for step, so a stored code is the tile the app shows:

  * mass is the barycentric lumped vertex area (a third of each incident triangle);
  * the diameter of a tile is found with two Dijkstra sweeps inside the tile (from its first
    vertex to the farthest one ``a``, from ``a`` to the farthest one ``b``; "farthest" is the
    first vertex, in vertex order, at the largest finite distance);
  * the tile's vertices are ordered by d_a − d_b, a STABLE sort (ties keep vertex order), and
    cut where the running mass first reaches half the tile's mass (both halves keep a vertex).

One difference in HOW, never in what: the sweeps of all the tiles of a level run as ONE
multi-source Dijkstra over the graph with the edges between tiles removed (tiles are then
disconnected, so each vertex's distance is from its own tile's source) — the per-tile sweeps
of the TypeScript, batched. ``tests/test_atlas_ladder.py`` pins the result against the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra


def vertex_areas(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The barycentric lumped vertex areas: a third of each incident triangle; they total the surface area."""
    p = np.asarray(positions, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    u = p[f[:, 1]] - p[f[:, 0]]
    v = p[f[:, 2]] - p[f[:, 0]]
    area = 0.5 * js_hypot(np.cross(u, v))
    mass = np.zeros(len(p))
    # face by face, a then b then c — the order the TypeScript accumulates in (np.add.at is sequential)
    np.add.at(mass, f.ravel(), np.repeat(area / 3, 3))
    return mass


def js_hypot(d: np.ndarray) -> np.ndarray:
    """Row-wise ``Math.hypot`` as V8 computes it (scale by the largest |x|, Kahan-sum the squares,
    √ then rescale) — bit-for-bit, so equal-length paths tie exactly as they do in the app."""
    x = np.abs(np.asarray(d, dtype=np.float64))
    mx = x.max(axis=1)
    safe = np.where(mx > 0, mx, 1.0)
    s = np.zeros(len(x))
    c = np.zeros(len(x))
    for k in range(x.shape[1]):
        n = x[:, k] / safe
        summand = n * n - c
        pre = s + summand
        c = (pre - s) - summand
        s = pre
    return np.where(mx > 0, np.sqrt(s) * mx, 0.0)


def edge_graph(positions: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """The symmetric edge graph of a triangle mesh, weighted by Euclidean edge length."""
    p = np.asarray(positions, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    e = np.unique(np.sort(np.stack([a, b], 1), axis=1), axis=0)
    w = js_hypot(p[e[:, 0]] - p[e[:, 1]])
    n = len(p)
    g = coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(n, n))
    return g.tocsr()


def component_mesh(positions: np.ndarray, faces: np.ndarray, start: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """One component (a hemisphere) out of a surface: vertices [start, start+n), the faces wholly inside, re-indexed."""
    f = np.asarray(faces, dtype=np.int64) - start
    keep = np.all((f >= 0) & (f < n), axis=1)
    return np.asarray(positions)[start:start + n], f[keep]


def _first_argmax(labels: np.ndarray, values: np.ndarray, n_tiles: int) -> np.ndarray:
    """Per tile, the first vertex (in vertex order) at the tile's largest FINITE value; −1 for a tile with none."""
    v = np.where(np.isfinite(values), values, -np.inf)
    best = np.full(n_tiles, -np.inf)
    np.maximum.at(best, labels, v)
    hit = np.flatnonzero((v == best[labels]) & np.isfinite(v))
    out = np.full(n_tiles, -1, dtype=np.int64)
    # vertices are scanned in increasing order, so the first hit per tile is the smallest index
    tiles, first = np.unique(labels[hit], return_index=True)
    out[tiles] = hit[first]
    return out


def _sweep(graph: csr_matrix, labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Dijkstra from one source per tile, confined to the tile: the edges between tiles are dropped."""
    g = graph.tocoo()
    keep = labels[g.row] == labels[g.col]
    inner = csr_matrix((g.data[keep], (g.row[keep], g.col[keep])), shape=graph.shape)
    src = sources[sources >= 0]
    return dijkstra(inner, directed=False, indices=src, min_only=True)


@dataclass
class Ladder:
    """The levels of one surface: ``labels[ℓ]`` is the tile of each vertex at level ℓ (2^ℓ tiles)."""
    area: float
    mass: np.ndarray
    labels: list[np.ndarray] = field(default_factory=list)

    @property
    def finest(self) -> int:
        return len(self.labels) - 1

    def tile_areas(self, level: int) -> np.ndarray:
        return np.bincount(self.labels[level], weights=self.mass, minlength=2 ** level)


def surface_ladder(positions: np.ndarray, faces: np.ndarray, max_tiles: int | None = None) -> Ladder:
    """Equal-mass geodesic bisection, one level per power of two of tiles (cortical-flow's ``surfaceLadder``)."""
    p = np.asarray(positions, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = len(p)
    if n < 1 or len(f) < 1:
        raise ValueError(f"surface_ladder: a mesh of {n} vertices and {len(f)} faces")
    mass = vertex_areas(p, f)
    graph = edge_graph(p, f)
    cap = 2 ** int(np.floor(np.log2(n)))
    max_tiles = min(max_tiles or cap, cap)
    vidx = np.arange(n)
    labels = [np.zeros(n, dtype=np.int32)]
    tiles = 1
    while tiles * 2 <= max_tiles:
        prev = labels[-1]
        counts = np.bincount(prev, minlength=tiles)
        first = np.full(tiles, -1, dtype=np.int64)
        present, idx = np.unique(prev, return_index=True)
        first[present] = idx
        # sweep 1 from each tile's first vertex → a; sweep 2 from a → da; b = the far end of da
        d0 = _sweep(graph, prev, first)
        a = _first_argmax(prev, d0, tiles)
        a = np.where(a >= 0, a, first)
        da = _sweep(graph, prev, a)
        b = _first_argmax(prev, da, tiles)
        b = np.where(b >= 0, b, first)
        # the TypeScript starts this scan at the tile's first vertex; when that vertex is out of
        # a's reach (da = ∞) nothing beats it, so it stays the far end
        stuck = (first >= 0) & ~np.isfinite(da[np.maximum(first, 0)])
        b = np.where(stuck, first, b)
        db = _sweep(graph, prev, b)
        with np.errstate(invalid="ignore"):
            key = da - db                       # NaN where both ends are out of reach (∞ − ∞)
        nxt = np.empty(n, dtype=np.int32)
        # a tile of one vertex (or none) is not halved: its vertex goes to the even child
        small = counts[prev] < 2
        nxt[small] = 2 * prev[small]
        big = ~small
        if big.any():
            order = _order_within_tiles(prev, key, big)
            # the tile's mass, summed in vertex order as the TypeScript's reduce does
            totals = np.zeros(tiles)
            for t, members in _groups(prev, big):
                totals[t] = np.cumsum(mass[members])[-1]
            nxt[order] = _halves(prev[order], mass[order], totals)
        new_counts = np.bincount(nxt, minlength=2 * tiles)
        if np.any(new_counts == 0):
            break
        labels.append(nxt)
        tiles *= 2
    return Ladder(area=float(np.cumsum(mass)[-1]), mass=mass, labels=labels)   # summed in order, as the app does


def _order_within_tiles(labels: np.ndarray, key: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The vertices of the masked tiles, grouped by tile, each tile STABLY sorted by key (ties: vertex order).

    A key is non-finite only where the tile is disconnected (a distance is ∞); the TypeScript
    comparator can then return NaN, which ``Array.prototype.sort`` reads as 0. Those tiles are
    sorted as V8 sorts them (``v8_sort``); every other tile takes the vectorised stable sort,
    which is identical.
    """
    v = np.flatnonzero(mask)
    order = v[np.lexsort((v, key[v], labels[v]))]
    bad = np.unique(labels[v][~np.isfinite(key[v])])   # ∞ − ∞ is NaN too
    if bad.size == 0:
        return order
    out = []
    for t in np.unique(labels[order]):
        members = order[labels[order] == t]
        if t in bad:
            members = np.asarray(v8_sort(list(np.sort(members)), lambda x, y: key[x] - key[y]), dtype=np.int64)
        out.append(members)
    return np.concatenate(out)


def v8_sort(items: list, compare) -> list:
    """``Array.prototype.sort`` as V8 runs it, for a comparator that may return NaN (read as 0).

    Below 8 elements V8 runs a plain binary insertion; from 8 to 63 its TimSort is one run
    (CountAndMakeRun) extended by binary insertion (array-sort.tq). Both are emulated step for
    step — with an inconsistent comparator the result depends on the exact steps — and pinned
    against node by ``tests/test_atlas_ladder.py``. Longer arrays fall back to Python's sort, the
    algorithm V8 ported — identical for a consistent comparator.
    """
    def cmp(x, y) -> float:
        c = float(compare(x, y))
        return 0.0 if np.isnan(c) else c

    a = list(items)
    n = len(a)
    if n < 2:
        return a
    if n >= 64:
        return sorted(a, key=cmp_to_key(lambda x, y: -1 if cmp(x, y) < 0 else (1 if cmp(x, y) > 0 else 0)))
    run = 1                         # V8 (node ≥ 22): below 8 elements, a plain binary insertion
    if n >= 8:                      # else TimSort's first run (CountAndMakeRun), then insertion
        run = 2
        desc = cmp(a[1], a[0]) < 0
        prev = a[1]
        for i in range(2, n):
            o = cmp(a[i], prev)
            if (desc and o >= 0) or (not desc and o < 0):
                break
            prev = a[i]
            run += 1
        if desc:
            a[:run] = a[:run][::-1]
    for start in range(run, n):
        pivot = a[start]
        left, right = 0, start
        while left < right:
            mid = left + ((right - left) >> 1)
            if cmp(pivot, a[mid]) < 0:
                right = mid
            else:
                left = mid + 1
        a[left + 1:start + 1] = a[left:start]
        a[left] = pivot
    return a


def _groups(labels: np.ndarray, mask: np.ndarray):
    """(tile, its masked vertices in increasing order) for every tile with a masked vertex."""
    v = np.flatnonzero(mask)
    v = v[np.argsort(labels[v], kind="stable")]
    lab = labels[v]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(v)]):
        yield int(lab[s]), v[s:e]


def _halves(tile_of: np.ndarray, mass: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Given the vertices of each tile in bisection order, the child each goes to: cut at half the tile's mass."""
    starts = np.flatnonzero(np.r_[True, tile_of[1:] != tile_of[:-1]])
    ends = np.r_[starts[1:], len(tile_of)]
    child = np.empty(len(tile_of), dtype=np.int32)
    for s, e in zip(starts, ends):
        total = totals[int(tile_of[s])]
        acc = np.cumsum(mass[s:e])
        hit = np.flatnonzero(acc >= total / 2)
        cut = int(hit[0]) + 1 if hit.size else (e - s) - 1
        cut = min(max(cut, 1), (e - s) - 1)
        t = int(tile_of[s])
        child[s:s + cut] = 2 * t
        child[s + cut:e] = 2 * t + 1
    return child
