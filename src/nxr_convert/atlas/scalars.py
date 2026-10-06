"""
Scalar reductions — a measurement reduced over a tile to one number, stored as MERGEABLE sums.

Every statistic below is a sum, a maximum or a minimum over the vertices (and samples) of a leaf,
weighted by the vertex's lumped area a_v, so the value of any coarser node — in space, in time, or
across subjects — is the same reduction over its leaves (``rollup``). Means and spreads are derived
when read: mean = s1 / w, variance = s2 / w − mean².

  maps      a scalar per vertex (PET SUVR …)          w = Σ a,  s1 = Σ a·x,  s2 = Σ a·x²,
                                                      n = #finite, min, max          [leaf]
  band power  MEG source power per octave band        power = Σ a·|J|²  (Σ over the frame's
            (cortical current J = K·A, A the band's   samples), env = Σ a·|J|, envmax = max |J|,
            analytic sensor signal)                   w = Σ a (per leaf), samples (per frame)
                                                      [band, leaf, frame]

The octave filter bank is a POWER PARTITION: its squared gains sum to one across the covered
range, so the band powers of a frame add up to its broadband power (Parseval) — the bands are
themselves a mergeable axis.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.sparse import csr_matrix

from .trees import Tree


def leaf_operator(tree: Tree, weights: np.ndarray) -> csr_matrix:
    """Sparse [leaves × vertices]: the weight of each vertex in its leaf (vertices off every leaf drop out)."""
    on = tree.codes >= 0
    v = np.flatnonzero(on)
    return csr_matrix((np.asarray(weights, dtype=np.float64)[v], (tree.codes[v], v)),
                      shape=(tree.n_leaves, len(tree.codes)))


def _leaf_extreme(tree: Tree, x: np.ndarray, op) -> np.ndarray:
    """Per leaf, the max (or min) of ``x`` over its vertices along axis 0 (x: [V, ...]); ±inf when empty."""
    on = np.flatnonzero(tree.codes >= 0)
    order = on[np.argsort(tree.codes[on], kind="stable")]
    codes = tree.codes[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    fill = -np.inf if op is np.maximum else np.inf
    out = np.full((tree.n_leaves,) + x.shape[1:], fill, dtype=np.float64)
    out[codes[starts]] = op.reduceat(x[order], starts, axis=0)
    return out


def map_stats(tree: Tree, area: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """The mergeable statistics of one scalar map over each leaf."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    a = np.where(ok, area, 0.0)
    xz = np.where(ok, x, 0.0)
    M = leaf_operator(tree, np.ones_like(a))
    return {
        "w": M @ a,
        "s1": M @ (a * xz),
        "s2": M @ (a * xz * xz),
        "n": M @ ok.astype(np.float64),
        "min": _leaf_extreme(tree, np.where(ok, x, np.inf), np.minimum),
        "max": _leaf_extreme(tree, np.where(ok, x, -np.inf), np.maximum),
    }


MAP_MERGE = {"w": "sum", "s1": "sum", "s2": "sum", "n": "sum", "min": "min", "max": "max"}


# ── the octave filter bank ──────────────────────────────────────────────────────────────────

def octave_bands(f_lo: float = 1.0, n_bands: int = 6) -> list[tuple[float, float]]:
    """Octaves [f_lo·2^b, f_lo·2^(b+1)), b = 0 … n_bands−1."""
    return [(f_lo * 2 ** b, f_lo * 2 ** (b + 1)) for b in range(n_bands)]


def band_gains(freqs: np.ndarray, bands: list[tuple[float, float]]) -> np.ndarray:
    """[bands, freqs] amplitude gains whose SQUARES sum to 1 inside [first low, last high].

    Adjacent octaves cross over with a cos/sin taper in log2-frequency, half an octave wide,
    centred on the shared edge; the outer edges taper to zero the same way, so power partitions.
    """
    f = np.asarray(freqs, dtype=np.float64)
    lf = np.log2(np.maximum(f, 1e-12))
    edges = np.log2([bands[0][0]] + [hi for _, hi in bands])
    half = 0.25                                   # half the crossover width, in octaves
    g = np.zeros((len(bands), len(f)))

    def rise(e):                                  # 0 → 1 across [e − half, e + half]
        t = np.clip((lf - (e - half)) / (2 * half), 0, 1)
        return np.sin(0.5 * np.pi * t)

    for b in range(len(bands)):
        up = rise(edges[b])
        down = np.sqrt(np.clip(1 - rise(edges[b + 1]) ** 2, 0, 1))
        g[b] = up * down
    g[:, f <= 0] = 0
    return g


def analytic_bands(x: np.ndarray, sfreq: float, bands: list[tuple[float, float]]):
    """Yield (band index, analytic signal [channels, samples] complex64) — one FFT, zero-phase."""
    n = x.shape[-1]
    X = sfft.fft(np.asarray(x, dtype=np.float32), axis=-1, workers=-1)
    f = sfft.fftfreq(n, 1.0 / sfreq)
    gains = band_gains(np.abs(f), bands)
    pos = (f > 0).astype(np.float32) * 2.0      # analytic: positive frequencies doubled, negatives removed
    for b in range(len(bands)):
        h = (gains[b] * pos).astype(np.float32)
        yield b, sfft.ifft(X * h, axis=-1, workers=-1).astype(np.complex64, copy=False)


# ── MEG band power on the cortex ────────────────────────────────────────────────────────────

def band_power(trees: "Tree | dict[str, Tree]", area: np.ndarray, kernel: np.ndarray, data: np.ndarray, sfreq: float,
               bands: list[tuple[float, float]], frame_s: float = 0.25, chunk_frames: int = 16, good: np.ndarray | None = None,
               progress=None) -> dict:
    """Per band × leaf × frame: Σ a·|J|², Σ a·|J|, max |J|, with J = K·A the cortical current of each band.

    ``kernel`` is [vertices, channels], ``data`` [channels, samples] in the kernel's column order.
    The band's analytic signal is taken over the WHOLE record at the sensors (exact: the inverse is
    linear), then projected in chunks of frames. ``trees`` may be one tree or {name: tree}: every tree
    is filled from the same projection (the costly part), and the result is keyed the same way.
    """
    single = isinstance(trees, Tree)
    tset = {"_": trees} if single else dict(trees)
    spf = int(round(frame_s * sfreq))
    n = data.shape[-1]
    n_frames = int(np.ceil(n / spf))
    K = np.asarray(kernel, dtype=np.float32)
    ops = {k: leaf_operator(t, area) for k, t in tset.items()}
    res = {k: {"power": np.zeros((len(bands), t.n_leaves, n_frames)),
               "env": np.zeros((len(bands), t.n_leaves, n_frames)),
               "envmax": np.full((len(bands), t.n_leaves, n_frames), -np.inf)} for k, t in tset.items()}
    g_all = np.ones(n, bool) if good is None else np.asarray(good, bool)
    # good samples per frame: a bad sample carries no current and no count, so every sum stays mergeable
    samples = np.add.reduceat(g_all.astype(np.int64), np.arange(0, n, spf))
    step = spf * chunk_frames
    for b, A in analytic_bands(data, sfreq, bands):
        Ar, Ai = np.ascontiguousarray(A.real), np.ascontiguousarray(A.imag)
        del A                                                      # memory: the record is held once, as Ar / Ai
        for s in range(0, n, step):
            e = min(s + step, n)
            gm = g_all[s:e].astype(np.float32)
            Jr, Ji = K @ (Ar[:, s:e] * gm), K @ (Ai[:, s:e] * gm)  # [V, T]: two real GEMMs; bad samples → 0
            p = Jr * Jr + Ji * Ji
            m = np.sqrt(p)
            k0, k1 = s // spf, int(np.ceil(e / spf))
            # frame sums: pad the chunk to whole frames (the last frame may be partial)
            pad = k1 * spf - e
            if pad:
                p = np.pad(p, ((0, 0), (0, pad)))
                m = np.pad(m, ((0, 0), (0, pad)))
            ps = p.reshape(len(p), k1 - k0, spf).sum(-1, dtype=np.float64)
            mf = m.reshape(len(m), k1 - k0, spf)
            ms, mx = mf.sum(-1, dtype=np.float64), mf.max(-1)
            for k, t in tset.items():
                res[k]["power"][b, :, k0:k1] = ops[k] @ ps
                res[k]["env"][b, :, k0:k1] = ops[k] @ ms
                res[k]["envmax"][b, :, k0:k1] = _leaf_extreme(t, mx, np.maximum)
        if progress:
            progress(b)
    for k in tset:
        res[k].update({"w": ops[k] @ np.ones(len(area)), "samples": samples, "frame_s": frame_s, "spf": spf})
    return res["_"] if single else res


BAND_MERGE = {"power": "sum", "env": "sum", "envmax": "max"}
