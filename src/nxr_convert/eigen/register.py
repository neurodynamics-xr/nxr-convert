"""REGISTRATION BY THE TRANSFER MATRIX — a subject's Hodge–Dirac mode coefficients carried onto a template's (fsaverage): T [K_template × K_subject].

A field on the subject, given by its coefficients c (synthesis: f = Pᵀ c), is carried as T c on the template:

    T = A_template · R · Pᵀ_subject        (block-diagonal by form kind: exact → exact, coexact → coexact)

- ``Pᵀ_subject``  synthesis: each subject mode's vertex potential on the subject's glued surface;
- ``R``           the REGISTRATION, sparse [V_template_glued × V_subject_glued]: each template vertex's position on the registered sphere, barycentric
                  in the subject sphere triangle containing it, a hemisphere at a time, on the subject's CORTEX only (its kept vertices — the bridge is not
                  cortex and sits on no sphere); nearest triangle where none contains it (the subject's cut medial wall). A template BRIDGE vertex is the
                  blend of its 3 nearest cut-out wall vertices (``bridge_rows``: the Editor's carry rule, glue.ts ``sourcesOf``), those pulled from the
                  subject the same way;
- ``A_template``  analysis on the template package (exact modes: 2 Pᵀ M; coexact, whose vertex potentials are not orthonormal: G⁻¹ P M, G = P M Pᵀ),
                  as the Editor's ``hodgeProjection.ts`` reads a field.

Both packages are ``nxr-convert modes`` solves (``hodge_dirac.solve_input``): glued by anatomy, ``keep`` = the input vertex of a glued vertex (kept
first, ascending; −1 on the bridge), the 0-form mass, the glued positions. A field projection, not mode matching: T is exact for whatever the two
bands share, and ``check_transfer`` measures it on smooth fields (spherical harmonics of the sphere positions, l = 1…3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

K_WALL = 3                                                    # a bridge vertex: its K_WALL nearest cut-out wall vertices (the Editor's)


@dataclass
class Solve:
    """one ``nxr-convert modes`` solve, its first k modes"""
    P: np.ndarray            # [k × V_glued] float32 vertex potentials, mode-major
    mu: np.ndarray           # [k] eigenvalues (1/m)
    kind: np.ndarray         # [k] 0 exact · 2 coexact
    keep: np.ndarray         # [V_glued] the input vertex, −1 on the bridge (kept first)
    mass: np.ndarray         # [V_glued] the 0-form (vertex) mass
    positions: np.ndarray    # [V_glued × 3] the glued positions
    mesh: str | None         # the glued mesh's fingerprint
    sha: str                 # sha256 of P's bytes: the package's identity (the Editor checks it)

    @property
    def kept(self) -> int:
        return int((self.keep >= 0).sum())


def load_solve(folder: str | Path, prefix: str, k: int) -> Solve:
    """the first ``k`` modes of the solve written to ``folder`` with file prefix ``prefix``"""
    from ..hodge import package
    pk = package(folder, prefix)
    if k > pk["mu"].size:
        raise ValueError(f"{folder}: {k} modes asked, the solve has {pk['mu'].size}")
    keep = pk["keep"].astype(np.int64)
    m = int((keep >= 0).sum())
    if not ((keep[:m] >= 0).all() and (keep[m:] < 0).all()):
        raise ValueError(f"{folder}: keep is not the kept vertices first, then the bridge's")
    P = np.ascontiguousarray(pk["potentials"][:k], dtype="<f4")
    mass = np.load(Path(folder) / f"{prefix}_hodge_mass.npy", mmap_mode="r")[: len(keep)].astype(np.float64)
    return Solve(P, pk["mu"][:k], pk["kind"][:k].astype(int), keep, mass, np.asarray(pk["glued_V"], np.float64), pk["meta"].get("mesh"),
                 hashlib.sha256(P.tobytes()).hexdigest())


def hemispheres(positions: np.ndarray, faces: np.ndarray, left: str) -> np.ndarray:
    """'L' / 'R' per vertex: the two largest connected components; ``left`` names the axis on which the left one lies — ``+y`` (Brainstorm SCS),
    ``-x`` (MNI, fsaverage)"""
    sign, axis = (1 if left[0] == "+" else -1), "xyz".index(left[1])
    n = len(positions)
    comp = connected_components(sp.coo_matrix((np.ones(faces.size), (faces.ravel(), np.roll(faces, 1, 1).ravel())), shape=(n, n)), directed=False)[1]
    big = np.argsort(np.bincount(comp))[::-1][:2]
    side = [sign * positions[comp == c, axis].mean() for c in big]
    h = np.full(n, "?")
    h[comp == big[int(np.argmax(side))]] = "L"
    h[comp == big[int(np.argmin(side))]] = "R"
    return h


def _barycentric(X: np.ndarray, F: np.ndarray, Y: np.ndarray, faces_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """each point of Y (unit sphere) in the X triangle (unit sphere, faces ``faces_ok``) containing it — else the one it is nearest inside:
    (the triangle's vertices [n × 3], the weights [n × 3], ≥ 0, summing to 1)"""
    F = F[faces_ok[F].all(1)]
    C = X[F].mean(1)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    _, near = cKDTree(C).query(Y, k=12)
    best_w, best_f, best_m = np.zeros((len(Y), 3)), np.zeros(len(Y), int), np.full(len(Y), -np.inf)
    for j in range(near.shape[1]):
        T = F[near[:, j]]
        a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
        n = np.cross(b - a, c - a)
        P = Y * ((n * a).sum(1) / (n * Y).sum(1))[:, None]                       # Y projected onto the triangle's plane, from the centre
        w = np.stack([(np.cross(c - b, P - b) * n).sum(1), (np.cross(a - c, P - c) * n).sum(1), (np.cross(b - a, P - a) * n).sum(1)], 1) / (n * n).sum(1)[:, None]
        m = w.min(1)
        better = m > best_m
        best_m[better], best_w[better], best_f[better] = m[better], w[better], near[better, j]
    best_w = np.clip(best_w, 0, None)
    return F[best_f], best_w / best_w.sum(1, keepdims=True)


def sphere_registration(src_sphere: np.ndarray, src_faces: np.ndarray, src_hemi: np.ndarray, src_cortex: np.ndarray, src_columns: np.ndarray,
                        dst_points: np.ndarray, dst_hemi: np.ndarray, sampling: str = "barycentric") -> sp.csr_matrix:
    """R [len(dst_points) × n_columns]: each destination point (unit sphere) from the source's CORTEX vertices (``src_cortex``, a mask on the
    source's input vertices), a hemisphere at a time; column = ``src_columns[input vertex]`` (its glued index). ``sampling``: ``barycentric``
    (the blend in the containing triangle, Workbench's BARYCENTRIC) or ``nearest`` (the triangle's nearest vertex, weight 1 — FreeSurfer's
    mri_surf2surf nearest-neighbour mapping)"""
    if sampling not in ("barycentric", "nearest"):
        raise ValueError(f"sampling: barycentric or nearest, not {sampling!r}")
    unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    Xs, Yd = unit(src_sphere), unit(dst_points)
    I, J, W = [], [], []
    for h in "LR":
        pts = np.where(dst_hemi == h)[0]
        if not len(pts):
            continue
        T, w = _barycentric(Xs, src_faces, Yd[pts], src_cortex & (src_hemi == h))
        if sampling == "nearest":                                                  # the triangle's vertex nearest the point: its largest weight
            T, w = T[np.arange(len(T)), w.argmax(1)][:, None], np.ones((len(T), 1))
        I.append(np.repeat(pts, T.shape[1])); J.append(src_columns[T.ravel()]); W.append(w.ravel())
    ncol = int(src_columns.max()) + 1
    return sp.csr_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(len(dst_points), ncol))


def bridge_rows(solve: Solve, input_positions: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
    """the walls the bridge replaces (the input vertices the glue cut) and B [n_bridge × n_cut]: each bridge vertex the inverse-distance blend of its
    ``K_WALL`` nearest cut vertices (3-D, the input positions) — glue.ts ``sourcesOf``"""
    cut = np.setdiff1d(np.arange(len(input_positions)), solve.keep[: solve.kept])
    nb = len(solve.keep) - solve.kept
    if not nb:
        return cut, sp.csr_matrix((0, len(cut)))
    d, nn = cKDTree(input_positions[cut]).query(solve.positions[solve.kept:], k=K_WALL)
    w = 1 / np.maximum(d, 1e-9)
    w /= w.sum(1, keepdims=True)
    return cut, sp.csr_matrix((w.ravel(), (np.repeat(np.arange(nb), K_WALL), nn.ravel())), shape=(nb, len(cut)))


def fold(solve: Solve, X, block: int = 20000) -> np.ndarray:
    """A X, the analysis of fields X on the solve's glued vertices, without forming A: ``X(s)`` gives the rows of vertex slice s (exact: 2 P M x;
    coexact: G⁻¹ P M x with G = P M Pᵀ on the coexact modes)"""
    co = solve.kind == 2
    G = E = 0
    for i in range(0, solve.P.shape[1], block):
        s = slice(i, i + block)
        Pc = np.asarray(solve.P[:, s], np.float64) * solve.mass[s]
        E = E + Pc @ X(s)
        G = G + Pc[co] @ np.asarray(solve.P[co, s], np.float64).T
    E = np.array(E, dtype=np.float64)
    E[~co] *= 2
    if co.any():
        E[co] = la.cho_solve(la.cho_factor(G), E[co])
    return E


@dataclass
class Registration:
    """what a transfer is built from beyond the two solves: the two surfaces' input positions, faces, registered spheres and left axes"""
    subject_positions: np.ndarray
    subject_faces: np.ndarray
    subject_sphere: np.ndarray
    template_positions: np.ndarray
    template_faces: np.ndarray
    template_sphere: np.ndarray
    subject_left: str = "+y"                                  # Brainstorm SCS
    template_left: str = "-x"                                 # MNI (fsaverage)
    sampling: str = "barycentric"                             # or "nearest" (FreeSurfer's mri_surf2surf default)


def registration_matrix(subject: Solve, template: Solve, reg: Registration) -> sp.csr_matrix:
    """R [V_template_glued × V_subject_glued]: the template's kept vertices pulled from the subject's cortex through the spheres; its bridge vertices
    from its cut walls, pulled the same way"""
    if np.abs(reg.template_positions[template.keep[: template.kept]] - template.positions[: template.kept]).max() > 1e-6:
        raise ValueError("the template solve is not on the template surface given (positions differ)")
    if np.abs(reg.subject_positions[subject.keep[: subject.kept]] - subject.positions[: subject.kept]).max() > 1e-6:
        raise ValueError("the subject solve is not on the subject surface given (positions differ)")
    hs = hemispheres(reg.subject_positions, reg.subject_faces, reg.subject_left)
    ht = hemispheres(reg.template_positions, reg.template_faces, reg.template_left)
    cortex = np.zeros(len(reg.subject_positions), bool)
    cortex[subject.keep[: subject.kept]] = True
    columns = np.full(len(reg.subject_positions), -1, np.int64)
    columns[subject.keep[: subject.kept]] = np.arange(subject.kept)
    pull = lambda pts: sphere_registration(reg.subject_sphere, reg.subject_faces, hs, cortex, columns, reg.template_sphere[pts], ht[pts], reg.sampling)
    kt = template.keep[: template.kept]
    cut, B = bridge_rows(template, reg.template_positions)
    Rk, Rc = pull(kt), pull(cut)
    R = sp.vstack([Rk, B @ Rc]).tocsr()
    R.resize((len(template.keep), len(subject.keep)))
    return R


def transfer_matrix(subject: Solve, template: Solve, reg: Registration) -> tuple[np.ndarray, sp.csr_matrix]:
    """T [K_template × K_subject] (float64) and the registration R it was built through"""
    R = registration_matrix(subject, template, reg)
    PsT = np.asarray(subject.P, np.float64).T
    T = fold(template, lambda s: R[s] @ PsT) * (template.kind[:, None] == subject.kind[None, :])
    return T, R


def _harmonics(X: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """real spherical harmonics l = 1…3 of unit positions X: [V × 15] and their (l, m)"""
    from scipy.special import sph_harm_y
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    th, ph = np.arccos(np.clip(X[:, 2], -1, 1)), np.arctan2(X[:, 1], X[:, 0])
    out, names = [], []
    for l in (1, 2, 3):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), th, ph)
            out.append(y.real if m == 0 else np.sqrt(2) * (y.imag if m < 0 else y.real)); names.append((l, m))
    return np.stack(out, 1), names


def check_transfer(T: np.ndarray, R: sp.csr_matrix, subject: Solve, template: Solve, reg: Registration) -> list[dict]:
    """smooth fields (spherical harmonics of the sphere positions) through T against the truth: per (l, m) ``err_reg`` (the registration: R's pull
    analysed on the template, against the field analysed there), ``err_trunc`` (T against R's pull: the subject's band), ``err_T`` and ``corr_T``
    (the synthesised result against the field, area-weighted) — gate err_reg ≤ 0.05, target err_T < 0.15 and corr_T > 0.95"""
    Ys, names = _harmonics(reg.subject_sphere[subject.keep[: subject.kept]])
    Ys = np.vstack([Ys, np.zeros((len(subject.keep) - subject.kept, Ys.shape[1]))])      # the subject's bridge: not cortex, no data
    cut, B = bridge_rows(template, reg.template_positions)
    Yt = np.vstack([_harmonics(reg.template_sphere[template.keep[: template.kept]])[0], B @ _harmonics(reg.template_sphere[cut])[0]])
    cs = fold(subject, lambda s: Ys[s])
    ct, creg = fold(template, lambda s: Yt[s]), fold(template, lambda s: R[s] @ Ys)
    Tc = T @ cs
    rec = np.asarray(template.P, np.float64).T @ Tc
    nrm = lambda x, y: np.linalg.norm(x - y, axis=0) / np.linalg.norm(y, axis=0)
    ereg, etr, eT = nrm(creg, ct), nrm(Tc, creg), nrm(Tc, ct)
    m = template.mass
    out = []
    for i, (l, mm) in enumerate(names):
        x, y = rec[:, i] - m @ rec[:, i] / m.sum(), Yt[:, i] - m @ Yt[:, i] / m.sum()
        r = float((m * x * y).sum() / np.sqrt((m * x * x).sum() * (m * y * y).sum()))
        out.append({"l": l, "m": mm, "err_reg": float(ereg[i]), "err_trunc": float(etr[i]), "err_T": float(eT[i]), "corr_T": r,
                    "gate": bool(ereg[i] <= 0.05), "target": bool(eT[i] < 0.15 and r > 0.95)})
    return out


def write_transfer(out: Path, T: np.ndarray, subject: Solve, template: Solve, *, subject_name: str, template_name: str, template_surface: str) -> Path:
    """T as the Editor reads it (adapters/eigenmodes, ``operator: 'hodge-transfer'``): ``hodge-to-<template>.f32`` [K_to × K_from] float32 + ``.json``"""
    import json
    out.mkdir(parents=True, exist_ok=True)
    nm = template_name.lstrip("@")
    T.astype("<f4").tofile(out / f"hodge-to-{nm}.f32")
    (out / f"hodge-to-{nm}.json").write_text(json.dumps({
        "operator": "hodge-transfer", "from": {"package": subject.sha, "K": int(subject.P.shape[0]), "mu": subject.mu.tolist(), "mesh": subject.mesh},
        "flow": "fields",   # a reader carries the flow weights b as flow fields: μ_to⁻¹ T μ_from b (fused.registered_weights); a and the potentials' T otherwise
        "to": {"package": template.sha, "K": int(template.P.shape[0]), "mu": template.mu.tolist(), "kind": template.kind.tolist(), "mesh": template.mesh},
        "registration": "the registered spheres, sampled on the subject's cortex; the template bridge from its cut walls (nxr_convert.eigen.register)",
        "subject": subject_name, "template": nm, "template_surface": template_surface}))
    return out / f"hodge-to-{nm}.json"
