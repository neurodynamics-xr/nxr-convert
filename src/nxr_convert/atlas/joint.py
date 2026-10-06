"""
The joint spatiotemporal atlas — tiles over cortex × time, and transport along trajectories.

A JOINT CELL is (space node s at level ℓ, time tile k at level m): 2^(D−ℓ) leaves × 2^m frames. Every
mergeable array ([…, leaves, frames]) reduces to any joint cell by ``rollup.space`` then
``rollup.time`` (they commute), so scalar reductions (band power) and closed-form vector reductions
(frame-component orientation tensors) are available on every cell with nothing stored per level.

TRANSPORT needs more: a vector carried from cell to cell along a TRAJECTORY must be parallel-
transported between the cells' centres, and that depends on the path. It uses the level's discrete
connection on the tile graph (``vectors.levi_civita``): Ω_ℓ[a, b] is the frame angle at centre b of
centre a's e1 after vector-heat transport, antisymmetrised (Ω[a,b] − Ω[b,a])/2 so that going there
and back is the identity. A tangent vector with frame angle φ at a arrives at b with φ + Ω[a, b].

Trajectories and what is measured along them (explicit, per level — path-dependent, never rolled up):
  peak trajectory   per band and time tile, the node of the level with the largest power density
                    (a path through joint cells); hops across hemispheres break the path.
  transported angle Φ_k: the frame angle picked up by a vector carried from step 0 to step k.
  orientation drift δ_k: how much the dominant current orientation (principal axis of the tile's
                    Levi-Civita orientation tensor, axial) turns from step k to k+1 BEYOND what
                    transport accounts for: δ_k = wrap_π(ψ_{k+1} − ψ_k − Ω[a_k, a_{k+1}]).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from . import rollup
from .trees import Tree


def tile_adjacency(tree: Tree, faces: np.ndarray, level: int) -> np.ndarray:
    """[nodes × nodes] bool: two nodes of ``level`` touch when a mesh edge joins them."""
    node = tree.at(level)
    n = 2 * 2 ** level
    e = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    a, b = node[e[:, 0]], node[e[:, 1]]
    ok = (a >= 0) & (b >= 0) & (a != b)
    A = coo_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(n, n)).toarray() > 0
    return A | A.T


def antisymmetric(omega: np.ndarray) -> np.ndarray:
    """(Ω − Ωᵀ)/2 on the circle: there-and-back is the identity."""
    z = np.exp(1j * omega)
    return np.angle(z * np.conj(z.T)) / 2


def cell_density(power: np.ndarray, w: np.ndarray, samples: np.ndarray, depth: int, level: int, time_level: int) -> np.ndarray:
    """Power per unit area per sample on every joint cell: [bands, nodes, time tiles]."""
    P = rollup.time(rollup.space(power, depth, level, axis=1), time_level, axis=2)
    A = rollup.space(w, depth, level)
    N = rollup.time(samples.astype(np.float64), time_level)
    return P / (A[None, :, None] * N[None, None, :])


def peak_trajectory(density: np.ndarray) -> np.ndarray:
    """[bands, time tiles]: the node with the largest density in each time tile."""
    return np.nanargmax(np.where(np.isfinite(density), density, -np.inf), axis=1)


def hemisphere_of(nodes: np.ndarray, level: int) -> np.ndarray:
    return nodes >> level


def transported_angle(path: np.ndarray, omega: np.ndarray, level: int) -> np.ndarray:
    """Φ_k along one path: the frame angle accumulated by transport from its last restart (NaN at a hemisphere hop)."""
    phi = np.zeros(len(path))
    for k in range(1, len(path)):
        a, b = path[k - 1], path[k]
        if hemisphere_of(a, level) != hemisphere_of(b, level) or not np.isfinite(omega[a, b]):
            phi[k] = np.nan
        else:
            prev = 0.0 if not np.isfinite(phi[k - 1]) else phi[k - 1]
            phi[k] = prev + omega[a, b]
    return phi


def principal_orientation(t11: np.ndarray, t22: np.ndarray, t12: np.ndarray) -> np.ndarray:
    """The axial angle (−π/2, π/2] of the tangent tensor's principal axis in the centre's frame."""
    return 0.5 * np.arctan2(2 * t12, t11 - t22)


def orientation_drift(path: np.ndarray, psi: np.ndarray, omega: np.ndarray, level: int) -> np.ndarray:
    """δ_k = wrap_π(ψ_{k+1} − ψ_k − Ω[a_k, a_{k+1}]), axial (NaN at hemisphere hops); length len(path) − 1."""
    out = np.full(len(path) - 1, np.nan)
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        if hemisphere_of(a, level) != hemisphere_of(b, level) or not np.isfinite(omega[a, b]):
            continue
        d = psi[k + 1] - psi[k] - omega[a, b]
        out[k] = np.angle(np.exp(2j * d)) / 2                  # axial: wrap to (−π/2, π/2]
    return out
