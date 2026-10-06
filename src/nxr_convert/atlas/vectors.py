"""
Vector reductions — a tangent field reduced over a tile, in the canonical frames (``frames``).

A tangent vector v at vertex i is written in its frame: c1 = v·e1 ("north"), c2 = v·e2 ("west"),
cn = v·n (normal). Two kinds of reduction:

CLOSED-FORM (mergeable, stored per leaf, rolled up like scalars)
  The frame e1 is parallel for a connection that is flat away from the poles, so frame components add
  across vertices, tiles, frames and subjects:
    static field  w, Σa·c1, Σa·c2, Σa·cn, Σa·c1², Σa·c2², Σa·c1·c2, Σa·|v_t|        (mean vector, spread)
    MEG current   the Hermitian orientation tensor of the band's analytic current J (complex 3-vector):
                  Σa·|J1|², Σa·|J2|², Σa·|Jn|², Σa·Re/Im(J1·J2*), Σa·Re/Im(J1·Jn*), Σa·Re/Im(J2·Jn*)
                  per band × leaf × frame (its principal axis is the tile's dominant orientation).

LEVI-CIVITA (explicit per level — stored for each level asked, never rolled up)
  Each node of level ℓ has a CENTRE (the member vertex nearest its area-weighted centroid). The vector
  heat method transports the centre's e1 to every vertex; the frame angle it arrives with at vertex i is
  ρ_ℓ(i). A vector at i with frame angle φ reaches the centre with angle φ − ρ_ℓ(i), so the node's
  Levi-Civita resultant is Σa·(c1 + i·c2)·e^{−iρ_ℓ(i)}, and its tensor rotates the same way. The
  difference from the closed-form reduction is the curvature the tile encloses.
  The same solves give the level's CONNECTION: Ω_ℓ[a, b] = the frame angle at centre b of e1(centre a)
  transported — a vector with angle φ at a arrives at b with φ + Ω_ℓ[a, b] (phase 3's transport).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import face_gradient, face_normals_areas, run_compute
from .scalars import analytic_bands, leaf_operator
from .trees import Tree


@dataclass
class Frames:
    """Per-vertex frames of the whole surface (both hemispheres), NaN where singular."""
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray
    singular: np.ndarray


def vertex_gradient(p: np.ndarray, f: np.ndarray, x: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The tangent gradient of a vertex scalar: area-weighted face gradients, projected on the vertex plane."""
    g = face_gradient(p, f, x)
    _, a = face_normals_areas(p, f)
    acc = np.zeros_like(p)
    wsum = np.zeros(len(p))
    for k in range(3):
        np.add.at(acc, f[:, k], g * a[:, None])
        np.add.at(wsum, f[:, k], a)
    acc /= np.maximum(wsum, 1e-300)[:, None]
    return acc - (acc * normals).sum(1, keepdims=True) * normals


def components(fr: Frames, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (v * fr.e1).sum(1), (v * fr.e2).sum(1), (v * fr.normal).sum(1)


# ── closed form ─────────────────────────────────────────────────────────────────────────────

VECTOR_MERGE = {k: "sum" for k in ("w", "c1", "c2", "cn", "c11", "c22", "c12", "mag")}


def vector_stats(tree: Tree, area: np.ndarray, fr: Frames, v: np.ndarray) -> dict[str, np.ndarray]:
    """Mergeable frame-component sums of a static tangent field over each leaf (singular vertices excluded)."""
    c1, c2, cn = components(fr, v)
    ok = ~fr.singular & np.isfinite(c1) & np.isfinite(c2)
    a = np.where(ok, area, 0.0)
    c1, c2, cn = (np.where(ok, x, 0.0) for x in (c1, c2, cn))
    M = leaf_operator(tree, np.ones(len(a)))
    return {"w": M @ a, "c1": M @ (a * c1), "c2": M @ (a * c2), "cn": M @ (a * cn),
            "c11": M @ (a * c1 * c1), "c22": M @ (a * c2 * c2), "c12": M @ (a * c1 * c2),
            "mag": M @ (a * np.hypot(c1, c2))}


TENSOR_KEYS = ("t11", "t22", "tnn", "t12r", "t12i", "t1nr", "t1ni", "t2nr", "t2ni", "total")
TENSOR_MERGE = {k: "sum" for k in TENSOR_KEYS}


def _tensor_terms(J1r, J1i, J2r, J2i, Jnr, Jni):
    """The Hermitian products of the frame components, from their real and imaginary parts."""
    return {"t11": J1r * J1r + J1i * J1i, "t22": J2r * J2r + J2i * J2i, "tnn": Jnr * Jnr + Jni * Jni,
            "t12r": J1r * J2r + J1i * J2i, "t12i": J1i * J2r - J1r * J2i,
            "t1nr": J1r * Jnr + J1i * Jni, "t1ni": J1i * Jnr - J1r * Jni,
            "t2nr": J2r * Jnr + J2i * Jni, "t2ni": J2i * Jnr - J2r * Jni}


def _lc_rotate(sums: dict, c: np.ndarray, s: np.ndarray) -> dict:
    """The per-vertex tensor rotated by −ρ into the centre's frame, from frame sums (the rotation is
    constant in time, so it commutes with summing samples): J1' = c·J1 + s·J2, J2' = −s·J1 + c·J2."""
    t11, t22, t12r = sums["t11"], sums["t22"], sums["t12r"]
    c2, s2, cs = (c * c)[:, None], (s * s)[:, None], (c * s)[:, None]
    return {"t11": c2 * t11 + s2 * t22 + 2 * cs * t12r,
            "t22": s2 * t11 + c2 * t22 - 2 * cs * t12r,
            "t12r": cs * (t22 - t11) + (c2 - s2) * t12r,
            "t12i": sums["t12i"],
            "tnn": sums["tnn"],
            "t1nr": c[:, None] * sums["t1nr"] + s[:, None] * sums["t2nr"],
            "t1ni": c[:, None] * sums["t1ni"] + s[:, None] * sums["t2ni"],
            "t2nr": -s[:, None] * sums["t1nr"] + c[:, None] * sums["t2nr"],
            "t2ni": -s[:, None] * sums["t1ni"] + c[:, None] * sums["t2ni"]}


def band_tensor(trees: "Tree | dict[str, Tree]", area: np.ndarray, fr: Frames, kernel3: np.ndarray, data: np.ndarray,
                sfreq: float, bands: list[tuple[float, float]], frame_s: float = 0.25, chunk_frames: int = 16, good: np.ndarray | None = None,
                lc: "LeviCivita | dict[str, LeviCivita] | None" = None, lc_levels: tuple[int, ...] = (),
                progress=None) -> dict:
    """The orientation tensor of each band's cortical current per band × leaf × frame (+ Levi-Civita per level).

    ``kernel3`` is [3·V, channels] with rows (x, y, z) per vertex (Brainstorm's free orientation). It is
    projected once onto the frames (K1 = e1·K, K2 = e2·K, Kn = n·K), so each chunk is six real GEMMs; the
    tensor terms are summed per vertex over each frame first, and everything after that (leaf sums, the
    Levi-Civita rotation, node sums) works on [V × frames] — exact, because both are linear. ``trees``
    (and ``lc``) may be one or {name: …}; every tree is filled from the same projection.
    """
    from .frames import tangent_basis
    single = isinstance(trees, Tree)
    tset = {"_": trees} if single else dict(trees)
    lcs = {"_": lc} if single else (lc or {})
    V = len(area)
    spf = int(round(frame_s * sfreq))
    n = data.shape[-1]
    n_frames = int(np.ceil(n / spf))
    g_all = np.ones(n, bool) if good is None else np.asarray(good, bool)
    K = np.asarray(kernel3, dtype=np.float32)
    if K.shape[0] != 3 * V:
        raise ValueError(f"kernel has {K.shape[0]} rows, expected 3·{V}")
    K = K.reshape(V, 3, -1)
    # singular vertices keep an arbitrary tangent basis: out of the tensor sums, still in the total
    b1, b2 = tangent_basis(np.nan_to_num(fr.normal) + np.where(np.isnan(fr.normal), [0, 0, 1.0], 0))
    e1 = np.where(fr.singular[:, None], b1, np.nan_to_num(fr.e1)).astype(np.float32)
    e2 = np.where(fr.singular[:, None], b2, np.nan_to_num(fr.e2)).astype(np.float32)
    nn = np.nan_to_num(fr.normal).astype(np.float32)
    K1, K2, Kn = (np.einsum("vk,vkc->vc", e, K) for e in (e1, e2, nn))
    ok = ~fr.singular
    a_ok = np.where(ok, area, 0.0)
    ops = {k: (leaf_operator(t, a_ok), leaf_operator(t, area)) for k, t in tset.items()}
    out = {k: {key: np.zeros((len(bands), t.n_leaves, n_frames)) for key in TENSOR_KEYS} for k, t in tset.items()}
    lc_ops = {k: {} for k in tset}
    lc_out = {k: {} for k in tset}
    for k, t in tset.items():
        L = lcs.get(k)
        for lvl in (lc_levels if L is not None else ()):
            rho = L.rho[lvl]
            good = np.isfinite(rho) & ok
            lc_ops[k][lvl] = (_node_operator(t.at(lvl), np.where(good, area, 0.0), 2 * 2 ** lvl),
                              np.cos(np.nan_to_num(rho)), np.sin(np.nan_to_num(rho)))
            lc_out[k][lvl] = {key: np.zeros((len(bands), 2 * 2 ** lvl, n_frames)) for key in TENSOR_KEYS[:-1]}
    step = spf * chunk_frames
    for b, A in analytic_bands(data, sfreq, bands):
        Ar, Ai = np.ascontiguousarray(A.real), np.ascontiguousarray(A.imag)
        del A                                                      # memory: the record is held once, as Ar / Ai
        for s in range(0, n, step):
            e = min(s + step, n)
            k0, k1 = s // spf, int(np.ceil(e / spf))
            pad = k1 * spf - e

            def fsum(x):
                if pad:
                    x = np.pad(x, ((0, 0), (0, pad)))
                return x.reshape(len(x), k1 - k0, spf).sum(-1, dtype=np.float64)
            gm = g_all[s:e].astype(np.float32)
            ar, ai = Ar[:, s:e] * gm, Ai[:, s:e] * gm                  # bad samples → 0
            parts = [Kx @ y for Kx in (K1, K2, Kn) for y in (ar, ai)]       # J1r, J1i, J2r, J2i, Jnr, Jni
            sums = {key: fsum(v) for key, v in _tensor_terms(*parts).items()}    # [V × frames]
            total = sums["t11"] + sums["t22"] + sums["tnn"]
            for k in tset:
                M, Mall = ops[k]
                for key, x in sums.items():
                    out[k][key][b, :, k0:k1] = M @ x
                out[k]["total"][b, :, k0:k1] = Mall @ total
                for lvl, (Ml, c, s_) in lc_ops[k].items():
                    for key, x in _lc_rotate(sums, c, s_).items():
                        lc_out[k][lvl][key][b, :, k0:k1] = Ml @ x
        if progress:
            progress(b)
    # good samples per frame: a bad sample carries no current and no count, so every sum stays mergeable
    samples = np.add.reduceat(g_all.astype(np.int64), np.arange(0, n, spf))
    res = {k: {"tensor": out[k], "lc": lc_out[k], "samples": samples, "spf": spf} for k in tset}
    return res["_"] if single else res


def _node_operator(node: np.ndarray, w: np.ndarray, n_nodes: int):
    from scipy.sparse import csr_matrix
    v = np.flatnonzero(node >= 0)
    return csr_matrix((w[v], (node[v], v)), shape=(n_nodes, len(node)))


# ── Levi-Civita: centres, transport, connection ─────────────────────────────────────────────

@dataclass
class LeviCivita:
    levels: tuple[int, ...]
    centres: dict[int, np.ndarray]      # level → [nodes] centre vertex (−1: empty node)
    rho: dict[int, np.ndarray]          # level → [V] frame angle at i of e1(centre of i's node), transported
    omega: dict[int, np.ndarray]        # level → [nodes × nodes] Ω[a, b] (NaN across hemispheres)


def node_centres(tree: Tree, positions: np.ndarray, area: np.ndarray, level: int, singular: np.ndarray) -> np.ndarray:
    """Per node of ``level``, the (non-singular) member vertex nearest the node's area-weighted centroid."""
    node = tree.at(level)
    n_nodes = 2 * 2 ** level
    out = np.full(n_nodes, -1, dtype=np.int64)
    for k in range(n_nodes):
        m = np.flatnonzero((node == k) & ~singular)
        if m.size:
            c = (positions[m] * area[m, None]).sum(0) / area[m].sum()
            out[k] = m[np.argmin(np.linalg.norm(positions[m] - c, axis=1))]
    return out


def levi_civita(tree: Tree, positions: np.ndarray, faces: np.ndarray, area: np.ndarray, fr: Frames,
                runs: list[tuple[int, int]], levels: tuple[int, ...]) -> LeviCivita:
    """Vector heat transport from every node centre of each level, per hemisphere mesh."""
    from .ladder import component_mesh
    centres, rho, omega = {}, {}, {}
    V = len(positions)
    for lvl in levels:
        cen = node_centres(tree, positions, area, lvl, fr.singular)
        centres[lvl] = cen
        r = np.full(V, np.nan)
        om = np.full((len(cen), len(cen)), np.nan)
        node = tree.at(lvl)
        for start, n in runs:
            p, f = component_mesh(positions, faces, start, n)
            mine = np.flatnonzero((cen >= start) & (cen < start + n))
            if mine.size == 0:
                continue
            src = (cen[mine] - start).astype("<i4")
            vec = np.nan_to_num(fr.e1[cen[mine]]).astype("<f8")
            T = run_compute("transport", p, f, {}, {"sources.i32": src, "vectors.f64": vec},
                            outputs=("transport.f64",))["transport.f64"].reshape(len(mine), n, 3)
            e1, e2 = fr.e1[start:start + n], fr.e2[start:start + n]
            ang = np.arctan2((T * e2[None]).sum(2), (T * e1[None]).sum(2))       # [sources, n]
            # calibrate: transport is the identity at its own source (the heat solution is smoothed
            # there, and its tangent plane is not exactly ours — measured median 0.1–0.5°, tail ~20°)
            ang = np.angle(np.exp(1j * (ang - ang[np.arange(len(mine)), src][:, None])))
            local = node[start:start + n]
            idx = {int(k): i for i, k in enumerate(mine)}
            for vtx in range(n):
                k = int(local[vtx])
                if k in idx:
                    r[start + vtx] = ang[idx[k], vtx]
            for i, ka in enumerate(mine):
                om[ka, mine] = ang[i, cen[mine] - start]
        rho[lvl] = r
        omega[lvl] = om
    return LeviCivita(tuple(levels), centres, rho, omega)


def lc_resultant(tree: Tree, area: np.ndarray, fr: Frames, lc: LeviCivita, v: np.ndarray, level: int) -> np.ndarray:
    """Per node of ``level``, Σa·(c1 + i·c2)·e^{−iρ} — the field transported to the centre (complex, centre frame)."""
    c1, c2, _ = components(fr, v)
    rho = lc.rho[level]
    ok = np.isfinite(rho) & np.isfinite(c1) & np.isfinite(c2)
    z = np.where(ok, area * (c1 + 1j * c2) * np.exp(-1j * np.nan_to_num(rho)), 0)
    node = tree.at(level)
    out = np.zeros(2 * 2 ** level, dtype=np.complex128)
    np.add.at(out, node[node >= 0], z[node >= 0])
    return out
