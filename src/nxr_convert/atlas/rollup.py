"""
Roll-up by arithmetic — the atlas never stores a level it can compute.

Space: leaves are the codes ``h·2^D + t`` (2·2^D of them, both hemispheres); the node of a leaf at
level ℓ ≤ D is ``code >> (D − ℓ)``, so the leaves of a node are one contiguous block of 2^(D−ℓ)
and a roll-up is a reshape plus a reduction. Time: frame k at level 0 has parent k >> 1; a level-m
tile is 2^m consecutive frames (the last one may be partial; a missing frame is neutral).

Only MERGEABLE statistics roll up — sums (and counts), maxima, minima. Everything derived
(means, variances, densities) is computed from rolled-up sums, never rolled up itself.
"""
from __future__ import annotations

import numpy as np

MERGE = {"sum": np.add, "max": np.maximum, "min": np.minimum}
NEUTRAL = {"sum": 0.0, "max": -np.inf, "min": np.inf}


def space(x: np.ndarray, depth: int, level: int, op: str = "sum", axis: int = -1) -> np.ndarray:
    """Leaves (2·2^depth along ``axis``) → the 2·2^level nodes of ``level`` (−1: the whole cortex)."""
    x = np.moveaxis(np.asarray(x), axis, -1)
    n = x.shape[-1]
    if n != 2 * 2 ** depth:
        raise ValueError(f"{n} leaves along axis, expected 2·2^{depth}")
    groups = 1 if level < 0 else 2 * 2 ** level
    if level > depth:
        raise ValueError(f"level {level} is finer than the stored depth {depth}")
    y = MERGE[op].reduce(x.reshape(*x.shape[:-1], groups, n // groups), axis=-1)
    return np.moveaxis(y, -1, axis)


def time(x: np.ndarray, level: int, op: str = "sum", axis: int = -1) -> np.ndarray:
    """Frames (level 0) → tiles of 2^level frames; a partial last tile keeps what it has."""
    x = np.moveaxis(np.asarray(x), axis, -1)
    k = 2 ** level
    pad = (-x.shape[-1]) % k
    if pad:
        x = np.concatenate([x, np.full(x.shape[:-1] + (pad,), NEUTRAL[op], dtype=x.dtype)], axis=-1)
    y = MERGE[op].reduce(x.reshape(*x.shape[:-1], x.shape[-1] // k, k), axis=-1)
    return np.moveaxis(y, -1, axis)


def time_levels(n_frames: int) -> int:
    """The coarsest time level: one tile covers the whole record."""
    return int(np.ceil(np.log2(max(n_frames, 1))))
