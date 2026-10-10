"""THE TIMESERIES GRID (2026-09-25) — one dyadic grid for the raw recording and its envelope.

Raw chunks of 2^15 samples, envelope windows of 2^5, both sharded on a shared 2^17-sample time block. Mirrors
``backend/src/db/envelope.ts``:

    raw        [channels × samples]        inner chunk [1 × 2^15]   shard [channels × 2^17]
    envelope   [channels × windows × 2]    inner chunk [1 × 2^12 × 2]   shard [channels × 2^12 × 2]
                                           a window 2^5 samples, so an envelope chunk IS a 2^17 block

    window 2^5  ⊂  raw chunk 2^15  ⊂  block 2^17 — every boundary shared.

ONE CHANNEL per inner chunk: the timeseries view reads one channel at any zoom as a few small reads. The SHARD keeps the
file count small and lets a multi-channel read take a whole block. UNCOMPRESSED — the renderer reads byte ranges. The
recording's row states its INNER chunk (``chunk_shape_json``, the unit the app loads, D57).

The envelope is ONE array (no ladder, no format of its own): the database states it as a closed-form tiling of the time
Line with array-backed ``min`` and ``max`` measurements (D96); every coarser window is a roll-up at read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .crud import ArrayMeta, array_meta, layout_array, populate_array

RAW_CHUNK = 1 << 15
BLOCK = 1 << 17
WINDOW = 1 << 5
ENV_CHUNK = BLOCK // WINDOW          # 4096 windows — one block


def envelope_path(recording_path: str) -> str:
    """``timeseries/<rec>`` → ``timeseries/<rec>_envelope``."""
    return f"{recording_path}_envelope"


def raw_grid(n_chan: int, n_samples: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(inner chunk, shard) for a recording — clipped for one shorter than a chunk."""
    c = min(RAW_CHUNK, max(1, n_samples))
    s = BLOCK if n_samples >= RAW_CHUNK else c
    return (1, c), (max(1, n_chan), s)


def env_grid(n_chan: int, n_windows: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    c = min(ENV_CHUNK, max(1, n_windows))
    return (1, c, 2), (max(1, n_chan), c, 2)


def recording_meta(n_chan: int, n_samples: int) -> ArrayMeta:
    """The recording's layout, as its row states it: float32, sharded on the grid, uncompressed."""
    chunks, shards = raw_grid(n_chan, n_samples)
    return array_meta(np.float32, (n_chan, n_samples), chunks=chunks, shards=shards, compress=False)


def reduce_windows(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """``[C, n]`` samples → ``[C, ceil(n/window), 2]`` (min, max); the last window may be partial."""
    starts = np.arange(0, x.shape[1], window)
    return np.stack([np.minimum.reduceat(x, starts, axis=1), np.maximum.reduceat(x, starts, axis=1)], axis=-1)


class EnvelopeTee:
    """The recording's zarr array, its ENVELOPE (at ``dest``) reduced from each block AS IT IS WRITTEN — one pass over
    the raw, never a read back. A write must be a whole-channel block starting on a window boundary, whole windows
    long except at the end (every writer's is: a BLOCK, a multiple of WINDOW)."""

    def __init__(self, arr, dest: Path):
        n_chan, n_samples = arr.shape
        n_win = -(-n_samples // WINDOW)
        chunks, shards = env_grid(n_chan, n_win)
        layout_array(dest, array_meta(np.float32, (n_chan, n_win, 2), chunks=chunks, shards=shards, compress=False))
        self.arr, self.env, self.shape = arr, populate_array(dest), arr.shape

    def __setitem__(self, key: tuple[slice, slice], x) -> None:
        rows, cols = key
        x = np.asarray(x, dtype=np.float32)
        s0 = cols.start or 0
        if rows != slice(None) or s0 % WINDOW or (x.shape[1] % WINDOW and s0 + x.shape[1] != self.shape[1]):
            raise ValueError(f"envelope: a write at {key} is not a whole-channel block on window boundaries")
        self.arr[key] = x
        w0 = s0 // WINDOW
        self.env[:, w0:w0 + -(-x.shape[1] // WINDOW), :] = reduce_windows(x)
