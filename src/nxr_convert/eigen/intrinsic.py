"""THE INTRINSIC DELAUNAY MESH — the Editor's ``intrinsicDelaunay.ts`` (engine/toolbox/differential), ported line for line (itself ported from
geometry-central, MIT, Sharp et al.): halfedges h = 3f + k from F[h] to F[3f + (k+1) % 3], each edge's length; MOLLIFY (Sharp & Crane 2020),
the mixed-Voronoi LUMPED MASS of the input triangulation (taken before the first flip), then FLIP TO DELAUNAY (the length-based queue).
Signposts are not ported (only the Editor's tangent frames read them). Arrays in, arrays out; the flip queue is sequential, as the Editor's."""
from __future__ import annotations

import math
from collections import deque

import numpy as np


def next_he(h: int) -> int:
    return h - h % 3 + (h % 3 + 1) % 3


class IntrinsicMesh:
    def __init__(self, positions: np.ndarray, faces: np.ndarray):
        P = np.asarray(positions, np.float64).reshape(-1, 3)
        self.F = np.asarray(faces, np.int64).ravel().copy()
        self.nV, self.nF = len(P), len(self.F) // 3
        n3 = 3 * self.nF
        self.twin = np.full(n3, -1, np.int64)
        self.edge_of = np.full(n3, -1, np.int64)
        open_: dict[tuple[int, int], int] = {}
        lengths, edge_he = [], []
        F = self.F
        for h in range(n3):
            a, b = int(F[h]), int(F[next_he(h)])
            t = open_.pop((b, a), None)
            if t is not None:
                self.twin[h], self.twin[t] = t, h
                self.edge_of[h] = self.edge_of[t]
            else:
                if (a, b) in open_:
                    raise ValueError(f"intrinsicMesh: the edge {a}→{b} appears twice in one orientation — not oriented or not manifold")
                open_[(a, b)] = h
                self.edge_of[h] = len(lengths)
                edge_he.append(h)
                lengths.append(math.hypot(P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]))
        self.boundary = np.zeros(self.nV, np.uint8)
        for h in open_.values():
            self.boundary[F[h]] = 1
            self.boundary[F[next_he(h)]] = 1
        self.length = np.asarray(lengths, np.float64)
        self.edge_he = np.asarray(edge_he, np.int64)
        self.mass: np.ndarray | None = None

    @property
    def nE(self) -> int:
        return len(self.length)

    def L(self, h: int) -> float:
        return float(self.length[self.edge_of[h]])


def triangle_area(a: float, b: float, c: float) -> float:
    s = (a + b + c) / 2
    return math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))


def _half_cot(m: IntrinsicMesh, h: int) -> float:
    h1 = next_he(h)
    h2 = next_he(h1)
    lij, ljk, lki = m.L(h), m.L(h1), m.L(h2)
    A = triangle_area(lij, ljk, lki)
    return (-lij * lij + ljk * ljk + lki * lki) / (4 * A) / 2


def edge_weight(m: IntrinsicMesh, e: int) -> float:
    """½(cot α + cot β); a boundary edge's its one triangle's"""
    h = int(m.edge_he[e])
    t = int(m.twin[h])
    return _half_cot(m, h) + (_half_cot(m, t) if t >= 0 else 0.0)


def mollify(m: IntrinsicMesh, relative_factor: float = 1e-6) -> float:
    delta = float(m.length.mean()) * relative_factor
    eps = 0.0
    for h in range(3 * m.nF):
        lA, lB, lC = m.L(h), m.L(next_he(h)), m.L(next_he(next_he(h)))
        eps = max(eps, lC - lA - lB + delta)
    if eps > 0:
        m.length += eps
    return eps


def _flipped_length(lab: float, lbc: float, lca: float, lad: float, ldb: float) -> float:
    ang = lambda x, y, opp: math.acos(max(-1.0, min(1.0, (x * x + y * y - opp * opp) / (2 * x * y))))
    theta = ang(lab, lca, lbc) + ang(lab, lad, ldb)
    return math.sqrt(max(0.0, lca * lca + lad * lad - 2 * lca * lad * math.cos(theta)))


def flip(m: IntrinsicMesh, e: int) -> bool:
    h = int(m.edge_he[e])
    t = int(m.twin[h])
    if t < 0:
        return False
    h1, t1 = next_he(h), next_he(t)
    h2, t2 = next_he(h1), next_he(t1)
    if h1 == t or t1 == h:
        return False
    F = m.F
    a, b, c, d = int(F[h]), int(F[h1]), int(F[h2]), int(F[t2])
    ln = _flipped_length(float(m.length[e]), m.L(h1), m.L(h2), m.L(t1), m.L(t2))
    if not math.isfinite(ln) or ln <= 0:
        return False
    f1, f2 = h - h % 3, t - t % 3
    outer = [(int(m.edge_of[t2]), int(m.twin[t2])), (int(m.edge_of[h1]), int(m.twin[h1])),
             (int(m.edge_of[h2]), int(m.twin[h2])), (int(m.edge_of[t1]), int(m.twin[t1]))]
    if any(tw in (h, t) for _, tw in outer):
        return False
    F[f1], F[f1 + 1], F[f1 + 2] = d, b, c
    F[f2], F[f2 + 1], F[f2 + 2] = c, a, d
    slots = [f1, f1 + 1, f2, f2 + 1]
    moved = {t2: f1, h1: f1 + 1, h2: f2, t1: f2 + 1}
    for (edge, tw), s in zip(outer, slots):
        tw2 = moved.get(tw, tw)
        m.edge_of[s], m.twin[s], m.edge_he[edge] = edge, tw2, s
        if tw not in moved and tw >= 0:
            m.twin[tw] = s
    m.edge_of[f1 + 2] = m.edge_of[f2 + 2] = e
    m.twin[f1 + 2], m.twin[f2 + 2] = f2 + 2, f1 + 2
    m.edge_he[e] = f1 + 2
    m.length[e] = ln
    return True


def lumped_mass(m: IntrinsicMesh) -> np.ndarray:
    """the MIXED VORONOI area of each vertex (Meyer et al. 2003) of the triangulation as it is (the input's, when called before the flips)"""
    if m.mass is not None:
        return m.mass.copy()
    M = np.zeros(m.nV)
    for f in range(m.nF):
        h = 3 * f
        v = [int(m.F[h]), int(m.F[h + 1]), int(m.F[h + 2])]
        l = [m.L(h), m.L(h + 1), m.L(h + 2)]
        A = triangle_area(*l)
        cos = [(l[k] ** 2 + l[(k + 2) % 3] ** 2 - l[(k + 1) % 3] ** 2) / (2 * l[k] * l[(k + 2) % 3]) for k in range(3)]
        obtuse = next((k for k in range(3) if cos[k] < 0), -1)
        if obtuse >= 0:
            for k in range(3):
                M[v[k]] += A / 2 if k == obtuse else A / 4
            continue
        cot = [c / math.sqrt(max(1e-300, 1 - c * c)) for c in cos]
        for k in range(3):
            e2 = (k + 2) % 3
            M[v[k]] += (l[k] ** 2 * cot[(k + 2) % 3] + l[e2] ** 2 * cot[(e2 + 2) % 3]) / 8
    return M


def flip_to_delaunay(m: IntrinsicMesh, eps: float = 1e-6) -> int:
    nE = m.nE
    queue, in_queue = deque(range(nE)), np.ones(nE, bool)
    if m.mass is None:
        m.mass = lumped_mass(m)
    flips = 0
    while queue:
        e = queue.popleft()
        in_queue[e] = False
        if not (edge_weight(m, e) < -eps):
            continue
        if not flip(m, e):
            continue
        flips += 1
        h = int(m.edge_he[e])
        t = int(m.twin[h])
        for x in (next_he(h), next_he(next_he(h)), next_he(t), next_he(next_he(t))):
            n = int(m.edge_of[x])
            if not in_queue[n]:
                in_queue[n] = True
                queue.append(n)
    return flips


def prepared(positions: np.ndarray, faces: np.ndarray) -> IntrinsicMesh:
    """the Editor's preparation of a surface for its operators: the mesh, mollified, its mass taken, flipped to Delaunay"""
    m = IntrinsicMesh(positions, faces)
    mollify(m)
    flip_to_delaunay(m)
    return m
