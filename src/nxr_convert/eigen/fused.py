"""THE FUSED INVERSE — an imaging kernel read into the Hodge–Dirac modes: the fused weights [2K × C] that take a sensor column to the modes' synthesis
weights (rows k < K: the normal-current weight a_k; rows K + k: the flow weight b_k). The Editor's ``hodgeSynthesisWeights`` (editor/engine/hodgeProjection.ts),
ported: the same operators, so a kernel fused here is the kernel the Editor fuses (to float32: it runs on the device).

    FORMS        each channel's current J [3V] on the glued surface → its NORMAL current n·J, and its tangent part's DIVERGENCE and CURL: Brainstorm's ambient
                 face operator (∇φᵢ = N × opposite edge / 2A; div = Σ ∇φᵢ·Jᵢ, curl = (Σ ∇φᵢ × Jᵢ)·N per face, area-weighted to the vertices — faceGradient.ts)
    READINGS     the forms against the modes under the vertex mass: e = P_co M (n·J), o_co = P_co M curl, o_ex = P_ex M div
    WEIGHTS      a (coexact only — an MEG current has no charge) = G⁻¹ e, G = P_co M P_coᵀ (the coexact vertex potentials are not orthonormal);
                 b = A⁻¹ [o_ex; o_co], A the FLOW SYSTEM: column k the divergence (exact rows) and curl (coexact rows) of mode k's flow — −∇P_k exact,
                 −n × ∇P_k coexact (the vertex gradient: area-weighted face gradients) — read the same way. So reading a synthesised flow returns its
                 weights exactly, whatever the discretisation.

The kernel is carried onto the glued surface as the Editor carries it (glue.ts): a kept vertex its own rows, a bridge vertex the inverse-distance blend of
its 3 nearest cut-out wall vertices (``register.bridge_rows``).

REGISTERED: the weights are LINEAR in the kernel and the transfer T is linear in the coefficients, so a subject's kernel registered onto its template is
``registered = [T a; μ_t⁻¹ T μ_s b]`` [2K_template × C] — sensor columns straight to the template's synthesis weights (``registered_weights``): the flow
modes carried as flow fields, so the template's smaller cortex does not steepen them.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .register import Solve, bridge_rows

AREA_EPS = 1e-12


class FaceOperators:
    """the face gradient of hat functions on a mesh (faceGradient.ts ``assembleFaceGradient``) as sparse matrices"""

    def __init__(self, positions: np.ndarray, faces: np.ndarray):
        P = np.asarray(positions, np.float32).astype(np.float64)   # the Editor's positions are float32
        F = np.asarray(faces, np.int64).reshape(-1, 3)
        nV, nF = len(P), len(F)
        a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n = np.cross(b - a, c - a)
        twoA = np.linalg.norm(n, axis=1)
        live = twoA > AREA_EPS
        N = np.zeros_like(n)
        N[live] = n[live] / twoA[live, None]
        area = np.where(live, twoA / 2, 0.0)
        safe = np.where(live, twoA, 1.0)
        g = [np.cross(N, c - b) / safe[:, None], np.cross(N, a - c) / safe[:, None], np.cross(N, b - a) / safe[:, None]]   # ∇φ₀ ∇φ₁ ∇φ₂
        for x in g:
            x[~live] = 0
        rows, cols = np.repeat(np.arange(nF), 3), F.ravel()
        gs = np.stack(g, 1)                                         # [nF × 3 slots × 3 xyz]
        self.grad = [sp.csr_matrix((gs[:, :, d].ravel(), (rows, cols)), shape=(nF, nV)) for d in range(3)]      # face gradient of a scalar
        cg = np.cross(N[:, None, :], gs)                            # (N × ∇φ) · J = N · (∇φ × J): the curl's rows
        self.curl = [sp.csr_matrix((cg[:, :, d].ravel(), (rows, cols)), shape=(nF, nV)) for d in range(3)]
        inc = sp.csr_matrix((np.repeat(area, 3), (cols, rows)), shape=(nV, nF))
        ws = np.asarray(inc.sum(1)).ravel()
        self.to_vertex = sp.diags(np.where(ws > 0, 1 / np.where(ws > 0, ws, 1), 0)) @ inc                     # area-weighted face → vertex mean
        vn = inc @ N
        ln = np.linalg.norm(vn, axis=1)
        self.normals = np.where(ln[:, None] > 0, vn / np.where(ln > 0, ln, 1)[:, None], 0)                     # area-weighted vertex normals
        self.nV, self.nF = nV, nF

    def tangent(self, J: np.ndarray) -> np.ndarray:
        """J [V × 3 × m] without its normal part"""
        return J - (J * self.normals[:, :, None]).sum(1, keepdims=True) * self.normals[:, :, None]

    def div_curl(self, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """the tangent current's divergence and curl at the vertices, [V × m] each (flowFieldCpu with ``tangential``)"""
        Jt = self.tangent(J)
        d = sum(self.grad[k] @ Jt[:, k, :] for k in range(3))
        c = sum(self.curl[k] @ Jt[:, k, :] for k in range(3))
        return self.to_vertex @ d, self.to_vertex @ c

    def vertex_gradient(self, x: np.ndarray) -> np.ndarray:
        """∇x at the vertices [V × 3 × m]: the area-weighted mean of the face gradients (gradPairByGather)"""
        return np.stack([self.to_vertex @ (self.grad[k] @ x) for k in range(3)], 1)


def carry_kernel(W: np.ndarray, solve: Solve, input_positions: np.ndarray) -> np.ndarray:
    """a kernel [3·V_input × C] on the glued surface [3·V_glued × C]: kept vertices their rows, a bridge vertex its cut walls' blend (glue.ts)"""
    C = W.shape[1]
    Wv = np.asarray(W, np.float64).reshape(-1, 3, C)
    kept = Wv[solve.keep[: solve.kept]]
    cut, B = bridge_rows(solve, np.asarray(input_positions, np.float32).astype(np.float64))
    bridge = (B @ Wv[cut].reshape(len(cut), -1)).reshape(-1, 3, C)
    return np.concatenate([kept, bridge]).reshape(-1, C)


def fused_weights(solve: Solve, faces: np.ndarray, W: np.ndarray, block: int = 256, progress=None) -> np.ndarray:
    """the fused weights [2K × C] (float64) of the kernel W [3·V_glued × C] on the solve's glued surface (``faces``: its glued faces)"""
    ops = FaceOperators(solve.positions, faces)
    V, K = ops.nV, solve.P.shape[0]
    C = W.shape[1]
    P = np.asarray(solve.P, np.float64)
    M = solve.mass
    mu = np.asarray(solve.mu, np.float64)
    co, ex = np.where(solve.kind == 2)[0], np.where(solve.kind == 0)[0]
    J = W.reshape(V, 3, C)
    normal = (J * ops.normals[:, :, None]).sum(1)                   # [V × C]
    div, curl = ops.div_curl(J)
    e = P[co] @ (M[:, None] * normal)                               # the readings
    o_co = P[co] @ (M[:, None] * curl)
    o_ex = P[ex] @ (M[:, None] * div)
    G = P[co] @ (M[:, None] * P[co].T)
    a = la.cho_solve(la.cho_factor(0.5 * (G + G.T)), e) if len(co) else np.zeros((0, C))
    order = np.concatenate([ex, co])
    A = np.zeros((len(order), len(order)))
    for k0 in range(0, len(order), block):                          # THE FLOW SYSTEM, a block of modes at a time
        ks = order[k0: k0 + block]
        x = P[ks].T                                                 # [V × w]
        g = ops.vertex_gradient(x)                                  # [V × 3 × w]
        flow = np.where((solve.kind[ks] == 0)[None, None, :], -g, -np.cross(ops.normals[:, :, None], g, axis=1))
        d, c = ops.div_curl(flow)
        A[: len(ex), k0: k0 + len(ks)] = P[ex] @ (M[:, None] * d)
        A[len(ex):, k0: k0 + len(ks)] = P[co] @ (M[:, None] * c)
        if progress:
            progress({"stage": "flow system", "done": int(k0 + len(ks)), "of": int(len(order))})
    b = la.lu_solve(la.lu_factor(A), np.vstack([o_ex, o_co]))
    out = np.zeros((2 * K, C))
    out[co] = a
    out[K + order] = b
    del mu
    return out


def registered_weights(weights: np.ndarray, T: np.ndarray, mu_from: np.ndarray | None = None, mu_to: np.ndarray | None = None) -> np.ndarray:
    """a subject's fused weights [2K_s × C] carried onto the template by the transfer T [K_t × K_s] — linearity.

    The normal-current weights a are a scalar's coefficients: T a. The flow weights b are POTENTIALS, and a potential carried as it is keeps the drop
    between corresponding points, so on a smaller template its flow is steeper by the length ratio — a bias a subject (it tracks brain size). With the
    two spectra (``mu_from`` the subject's μ, ``mu_to`` the template's, the Dirac eigenvalues), each flow mode is carried as a FLOW FIELD of unit
    strength instead (|∇ψ_k| = μ_k): b_t = μ_t⁻¹ T μ_s b_s. Without them, the potentials (T b)."""
    Ks = T.shape[1]
    if weights.shape[0] != 2 * Ks:
        raise ValueError(f"weights for {weights.shape[0] // 2} modes, the transfer is from {Ks}")
    if (mu_from is None) != (mu_to is None):
        raise ValueError("the flow-field transfer needs both spectra (mu_from and mu_to)")
    if mu_from is None:
        return np.vstack([T @ weights[:Ks], T @ weights[Ks:]])
    mf, mt = np.asarray(mu_from, np.float64)[:Ks], np.asarray(mu_to, np.float64)[: T.shape[0]]
    return np.vstack([T @ weights[:Ks], (T @ (mf[:, None] * weights[Ks:])) / mt[:, None]])


def synthesis(solve: Solve, faces: np.ndarray, weights: np.ndarray, ops: FaceOperators | None = None) -> np.ndarray:
    """the CURRENT [3·V_glued × C] the weights [2K × C] synthesise (the Editor's HodgeBrushFilter sums + ``currentFromSums``): s₂ = Σ_coexact a P (the
    normal current), A = Σ_exact b P and B = Σ_coexact b P (the flow's potentials); J = −∇A − n × ∇B, its normal part set to s₂"""
    ops = ops or FaceOperators(solve.positions, faces)
    K = solve.P.shape[0]
    P = np.asarray(solve.P, np.float64)
    co, ex = solve.kind == 2, solve.kind == 0
    s2 = P[co].T @ weights[:K][co]
    A, B = P[ex].T @ weights[K:][ex], P[co].T @ weights[K:][co]
    n = ops.normals[:, :, None]
    J = -ops.vertex_gradient(A) - np.cross(n, ops.vertex_gradient(B), axis=1)
    J += (s2[:, None, :] - (J * n).sum(1, keepdims=True)) * n
    return J.reshape(-1, weights.shape[1])


def registered_kernel(subject: Solve, subject_faces: np.ndarray, W: np.ndarray, T: np.ndarray, template: Solve, template_faces: np.ndarray,
                      progress=None) -> tuple[np.ndarray, np.ndarray]:
    """a subject's kernel W [3·V_subject_glued × C] REGISTERED onto the template: (the kernel on the template's glued surface [3·V_template_glued × C] —
    sensors straight to the template's current —, the registered weights [2K_t × C]). Linear all the way: fuse, transfer, synthesise."""
    w = fused_weights(subject, subject_faces, W, progress=progress)
    wt = registered_weights(w, T)
    return synthesis(template, template_faces, wt), wt
