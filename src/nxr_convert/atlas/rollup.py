"""
Roll-up by arithmetic — the atlas never stores a level it can compute.

Space: leaves are the codes ``h·2^D + t`` (2·2^D of them, both hemispheres); the node of a leaf at
level ℓ ≤ D is ``code >> (D − ℓ)``, so the leaves of a node are one contiguous block of 2^(D−ℓ)
and a roll-up is a reshape plus a reduction. Time: frames are cycles of the tower (level ``frame_level``,
code ``frame_code0 + k``); the parent of code c is c >> 1, so a level-m tile is 2^m consecutive frames
aligned on the tower (the first and last may be partial; a missing frame is neutral).

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


def time(x: np.ndarray, level: int, op: str = "sum", axis: int = -1, code0: int = 0) -> np.ndarray:
    """Frames (level 0) → tiles of 2^level frames; partial first and last tiles keep what they have.
    ``code0``: the tower code of frame 0 (the atlas's ``frame_code0``) — tiles are then the tower's own
    (code >> level), not counted from the first frame; tile i of the result is code (code0 >> level) + i."""
    x = np.moveaxis(np.asarray(x), axis, -1)
    k = 2 ** level
    front = code0 % k
    pad = (-(front + x.shape[-1])) % k
    if front or pad:
        x = np.concatenate([np.full(x.shape[:-1] + (front,), NEUTRAL[op], dtype=x.dtype), x,
                            np.full(x.shape[:-1] + (pad,), NEUTRAL[op], dtype=x.dtype)], axis=-1)
    y = MERGE[op].reduce(x.reshape(*x.shape[:-1], x.shape[-1] // k, k), axis=-1)
    return np.moveaxis(y, -1, axis)


def time_levels(n_frames: int) -> int:
    """The coarsest time level: one tile covers the whole record."""
    return int(np.ceil(np.log2(max(n_frames, 1))))
