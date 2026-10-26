"""GLUE BY ANATOMY — the Editor's ``glueMesh`` (editor/src/editor/glue.ts), ported line for line: a surface of two closed pieces (a cortex's
hemispheres) made ONE closed surface by cutting each MEDIAL WALL (the atlas) and growing a bridge between the rims. The order of everything follows
the TypeScript's, so the faces come out identical — what the parity test holds."""
from __future__ import annotations

import math

import numpy as np


def _labels(n: int, F: np.ndarray, on: np.ndarray | None = None) -> np.ndarray:
    """each vertex's component through the faces' edges, among ``on``: the union-find ROOT (as the TypeScript's), −1 off ``on``"""
    up = np.arange(n)

    def find(i: int) -> int:
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i
    for f in range(len(F)):
        for k in range(3):
            a, b = int(F[f, k]), int(F[f, (k + 1) % 3])
            if on is not None and not (on[a] and on[b]):
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                up[ra] = rb
    out = np.full(n, -1, np.int64)
    for i in range(n):
        if on is None or on[i]:
            out[i] = find(i)
    return out


def _largest(n: int, F: np.ndarray, on: np.ndarray) -> np.ndarray:
    """the members of ``on`` in its largest component (a tie: the one seen first, by vertex)"""
    lab = _labels(n, F, on)
    count: dict[int, int] = {}
    for i in range(n):
        if lab[i] >= 0:
            count[int(lab[i])] = count.get(int(lab[i]), 0) + 1
    best, most = -1, 0
    for k, c in count.items():
        if c > most:
            best, most = k, c
    return (lab >= 0) & (lab == best)


def _rims(F: np.ndarray, n: int):
    """the faces' half-edges used once, chained into loops in the faces' winding; ``("pinch", v)`` for a vertex on two rims"""
    have = {(int(F[f, k]), int(F[f, (k + 1) % 3])) for f in range(len(F)) for k in range(3)}
    nxt: dict[int, int] = {}
    for f in range(len(F)):
        for k in range(3):
            a, b = int(F[f, k]), int(F[f, (k + 1) % 3])
            if (b, a) in have:
                continue
            if a in nxt:
                return ("pinch", a)
            nxt[a] = b
    loops = []
    while nxt:
        a0 = next(iter(nxt))
        loop, a = [a0], nxt.pop(a0)
        while a != a0:
            loop.append(a)
            a = nxt.pop(a)
        loops.append(loop)
    return ("loops", loops)


def _cut_disk(F: np.ndarray, piece: np.ndarray, start: np.ndarray, which: int):
    n = len(piece)
    cut = _largest(n, F, start)
    if not cut.any():
        raise ValueError(f"glue: hemisphere {which} has no medial wall in the atlas — the hole is cut by anatomy only")
    for _ in range(200):
        keep = _largest(n, F, piece & ~cut)
        cut |= piece & ~keep                                         # fill its islands: the patch is a disk
        inside = piece[F[:, 0]] & ~cut[F[:, 0]] & ~cut[F[:, 1]] & ~cut[F[:, 2]]
        faces = F[inside]
        kind, r = _rims(faces, n)
        if kind == "pinch":
            cut[r] = True
            continue
        if len(r) == 1:
            return faces, r[0]
        for loop in sorted(r, key=len)[:-1]:
            cut[loop] = True
    raise ValueError("glue: the medial wall did not become a disk")


def _dist(a, b) -> float:
    x, y, z = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(x * x + y * y + z * z)


def _arc(pts):
    cum = [0.0]
    for i in range(1, len(pts) + 1):
        cum.append(cum[i - 1] + _dist(pts[i - 1], pts[i % len(pts)]))
    return cum, cum[len(pts)]


def _sample(pts, cum, per, s):
    L, k = s * per, 0
    while k < len(pts) - 1 and cum[k + 1] <= L:
        k += 1
    a, b = pts[k], pts[(k + 1) % len(pts)]
    seg = cum[k + 1] - cum[k]
    t = (L - cum[k]) / seg if seg > 0 else 0.0
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _stitch(A: list[int], B0: list[int], at) -> list[int]:
    """the band of least total edge length between two loops running the same way (the Editor's ``stitch``)"""
    j0 = 0
    for j in range(1, len(B0)):
        if _dist(at(B0[j]), at(A[0])) < _dist(at(B0[j0]), at(A[0])):
            j0 = j
    B = B0[j0:] + B0[:j0]
    m, nb = len(A), len(B)
    W = nb + 1
    a = lambda i: A[i % m]
    b = lambda j: B[j % nb]
    cost = [math.inf] * ((m + 1) * W)
    frm = [0] * ((m + 1) * W)
    cost[0] = 0.0
    for i in range(m + 1):
        for j in range(nb + 1):
            c = cost[i * W + j]
            if i < m:
                x, k = c + _dist(at(a(i + 1)), at(b(j))), (i + 1) * W + j
                if x < cost[k] or (x == cost[k] and frm[k] != 1):
                    cost[k], frm[k] = x, 1
            if j < nb:
                x, k = c + _dist(at(a(i)), at(b(j + 1))), i * W + j + 1
                if x < cost[k]:
                    cost[k], frm[k] = x, 2
    T: list[int] = []
    i, j = m, nb
    while i > 0 or j > 0:
        if frm[i * W + j] == 1:
            T += [a(i - 1), a(i), b(j)]
            i -= 1
        else:
            T += [a(i), b(j), b(j - 1)]
            j -= 1
    return T


def _mean_edge(P: np.ndarray, F: np.ndarray) -> float:
    s = 0.0
    for f in range(len(F)):
        for k in range(3):
            s += _dist(P[F[f, k]], P[F[f, (k + 1) % 3]])
    return s / F.size


def _euler(F: np.ndarray, n: int) -> int:
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    return n - len(e) + len(F)


def glue_mesh(P: np.ndarray, F: np.ndarray, wall: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(positions, faces, keep, bridge) of the surface glued BY ANATOMY (the Editor's ``glueMesh``): each hemisphere's medial wall (``wall``, the atlas'
    unassigned vertices) cut out as one disk, the two rims oriented by geometry, resampled to one count, aligned, and joined by rings at the mesh's mean
    edge length, each band stitched optimally. ``keep[i]`` the input vertex of glued vertex i, −1 on the bridge (its vertices come last). Checked as the
    Editor checks: every half-edge once, every edge on two faces, one piece, χ = χ₁ + χ₂ − 2."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    wall = np.asarray(wall).astype(bool).ravel()
    n = len(P)
    lab = _labels(n, F)
    roots = list(dict.fromkeys(lab.tolist()))
    if len(roots) != 2:
        raise ValueError(f"glue: the surface has {len(roots)} pieces, not two")
    if len(wall) != n:
        raise ValueError(f"glue: the medial-wall mask has {len(wall)} entries for {n} vertices")
    pieces = [lab == r for r in roots]
    chi = []
    for p in pieces:
        fs = F[p[F[:, 0]]]
        vs, inv = np.unique(fs, return_inverse=True)
        chi.append(_euler(inv.reshape(-1, 3), len(vs)))
    disks = [_cut_disk(F, p, p & wall, i) for i, p in enumerate(pieces)]
    pt = lambda v: (float(P[v, 0]), float(P[v, 1]), float(P[v, 2]))

    def centre(L):
        c = [0.0, 0.0, 0.0]
        for v in L:
            p = pt(v)
            for k in range(3):
                c[k] += p[k] / len(L)
        return c
    ca, cb = centre(disks[0][1]), centre(disks[1][1])
    axis = (cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2])

    def orient(L):
        c, s = centre(L), 0.0
        for i in range(len(L)):
            p, q = pt(L[i]), pt(L[(i + 1) % len(L)])
            p = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
            q = (q[0] - c[0], q[1] - c[1], q[2] - c[2])
            s += (p[1] * q[2] - p[2] * q[1]) * axis[0] + (p[2] * q[0] - p[0] * q[2]) * axis[1] + (p[0] * q[1] - p[1] * q[0]) * axis[2]
        o = L[::-1] if s < 0 else list(L)
        k = o.index(min(o))                                          # from its lowest-numbered vertex: the start is not the winding's either
        return o[k:] + o[:k]
    A, B = orient(disks[0][1]), orient(disks[1][1])
    h = _mean_edge(P, F)
    pa, pb = [pt(v) for v in A], [pt(v) for v in B]
    (ca_, pera), (cb_, perb) = _arc(pa), _arc(pb)
    N = max(3, int(math.floor((pera + perb) / 2 / h + 0.5)))         # Math.round
    SA = [_sample(pa, ca_, pera, i / N) for i in range(N)]
    SB = [_sample(pb, cb_, perb, i / N) for i in range(N)]
    shift, best = 0, math.inf
    for j in range(N):
        c = 0.0
        for i in range(N):
            d = _dist(SA[i], SB[(i + j) % N])
            c += d * d
        if c < best:
            best, shift = c, j
    gaps = sorted(_dist(SA[i], SB[(i + shift) % N]) for i in range(N))
    R = max(1, int(math.floor(gaps[N // 2] / h + 0.5)) - 1)
    used = np.unique(np.concatenate([disks[0][0].ravel(), disks[1][0].ravel()]))
    new = np.full(n, -1, np.int64)
    new[used] = np.arange(len(used))
    m = len(used)
    total = m + R * N
    X = np.zeros((total, 3))
    X[:m] = P[used]
    for r in range(1, R + 1):
        t = r / (R + 1)
        for i in range(N):
            a, b = SA[i], SB[(i + shift) % N]
            X[m + (r - 1) * N + i] = [a[k] + t * (b[k] - a[k]) for k in range(3)]
    at = lambda v: (X[v, 0], X[v, 1], X[v, 2])
    ring = lambda r: [m + (r - 1) * N + i for i in range(N)]
    bridge = _stitch([int(new[v]) for v in A], ring(1), at)
    for r in range(1, R):
        bridge += _stitch(ring(r), ring(r + 1), at)
    bridge += _stitch(ring(R), [int(new[v]) for v in B], at)
    la = disks[0][1]
    x0, x1 = int(new[la[0]]), int(new[la[1]])
    if any(bridge[f + k] == x0 and bridge[f + (k + 1) % 3] == x1 for f in range(0, len(bridge), 3) for k in range(3)):
        bridge = [v for f in range(0, len(bridge), 3) for v in (bridge[f], bridge[f + 2], bridge[f + 1])]
    disk_faces = new[np.concatenate([disks[0][0], disks[1][0]])]
    faces = np.concatenate([disk_faces, np.asarray(bridge, np.int64).reshape(-1, 3)])
    keep = np.concatenate([used, np.full(R * N, -1)])
    half = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    hk = half[:, 0] * total + half[:, 1]
    if len(np.unique(hk)) != len(hk):
        raise ValueError("glue: a half-edge is used twice")
    if not np.isin(half[:, 1] * total + half[:, 0], hk).all():
        raise ValueError("glue: an edge has one face")
    if len(set(_labels(total, faces).tolist())) != 1:
        raise ValueError("glue: the result is not one piece")
    want, got = chi[0] + chi[1] - 2, _euler(faces, total)
    if got != want:
        raise ValueError(f"glue: χ = {got}, not {want}")
    B_ = np.asarray(bridge, np.int64).reshape(-1, 3)
    area = float(0.5 * np.linalg.norm(np.cross(X[B_[:, 1]] - X[B_[:, 0]], X[B_[:, 2]] - X[B_[:, 0]]), axis=1).sum())
    longest = float(max(np.linalg.norm(X[B_[:, k]] - X[B_[:, (k + 1) % 3]], axis=1).max() for k in range(3)))
    return X, faces.astype(np.int32), keep.astype(np.int32), {"vertices": R * N, "rings": R, "area": area, "longestEdge": longest}


def mesh_key(faces: np.ndarray) -> str:
    """the Editor's ``meshKey``: FNV-1a over the faces' int32 bytes (little-endian), ``<nF>:<hash>`` — what a stored product records of its mesh"""
    h = 0x811C9DC5
    for byte in np.ascontiguousarray(np.asarray(faces, "<i4")).tobytes():
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"{np.asarray(faces).size // 3}:{h:08x}"
