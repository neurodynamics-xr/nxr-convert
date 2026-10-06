"""
The filter bank of the tower of cycles — one complex analytic FIR per level, so chunked filtering is EXACT.

A level L of the tower holds the band [1/P(L), 2/P(L)) Hz, P(L) = 86 400·2^L s (``band_hz``; nsp's dynamics-atlas
``grid`` names the levels, and only these two lines of it are needed here — the dynamics atlas is not ported, D135). Its filter is the band's
gain of the octave power partition (``scalars.band_gains``: cos/sin crossovers half an octave wide,
squared gains summing to one across the bank), realised as a finite impulse response:

  h_L[k], k = −H … H     the analytic (positive-frequency) gain, inverse-transformed, cut to ±H samples
                         and tapered by a Hann window. H = ``cycles`` periods of the band's lowest
                         passed frequency (its lower edge less the crossover), so the response is local
                         in time at that band's own scale.

Because h_L is finite, filtering a block padded by H on each side gives exactly the samples a whole-record
filter would (overlap-save): the chunked engine and a whole-record pass agree to float rounding. The cut
and the taper make the partition approximate (``partition_error`` measures it); the exactness is not.

A sample within H of a record edge or of a bad sample is not clean for that band (``clean``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.signal import fftconvolve

from .scalars import band_gains

DAY_S = 86400.0                             # time level 0: one day (nsp's tower of cycles)


def period_s(level: int) -> float:
    """The period of a level-L cycle (s): 86 400·2^L."""
    return DAY_S * 2.0 ** level


def band_hz(level: int) -> tuple[float, float]:
    """The octave band of level L: [1/P(L), 2/P(L)) Hz."""
    lo = 1.0 / period_s(level)
    return lo, 2.0 * lo


CYCLES = 4.0
CROSSOVER_OCT = 0.25                        # scalars.band_gains: half the crossover width, in octaves
DEFAULT_LEVELS = tuple(range(-22, -15))     # −22 … −16: 0.76 – 97 Hz


def levels_for(sfreq: float, f_min: float = 0.5, f_max: float | None = None) -> list[int]:
    """The tower levels whose whole band (crossover included) lies in [f_min, min(f_max, Nyquist)]."""
    top = min(f_max or math.inf, sfreq / 2.0)
    out = []
    for L in range(-40, 20):
        lo, hi = band_hz(L)
        if lo * 2 ** -CROSSOVER_OCT >= f_min and hi * 2 ** CROSSOVER_OCT <= top:
            out.append(L)
    return out


@dataclass(frozen=True)
class Filter:
    level: int
    band_hz: tuple[float, float]
    half: int                     # H: the filter spans samples −H … H
    h: np.ndarray                 # complex128 [2H + 1]


def analytic_fir(level: int, sfreq: float, levels=DEFAULT_LEVELS, cycles: float = CYCLES) -> Filter:
    """The complex analytic FIR of ``level`` in the power partition over ``levels``."""
    levels = sorted(levels, reverse=True)            # ascending frequency (band_gains' order)
    bands = [band_hz(L) for L in levels]
    lo = band_hz(level)[0] * 2 ** -CROSSOVER_OCT
    H = int(math.ceil(cycles / lo * sfreq))
    n = 1 << int(math.ceil(math.log2(8 * (2 * H + 1))))
    f = sfft.fftfreq(n, 1.0 / sfreq)
    g = band_gains(np.abs(f), bands)[levels.index(level)] * 2.0 * (f > 0)
    full = sfft.ifft(g)
    k = np.arange(-H, H + 1)
    h = full[k % n] * np.hanning(2 * H + 3)[1:-1]
    return Filter(level, band_hz(level), H, h)


def response(filt: Filter, freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """The realised complex gain of the FIR at ``freqs`` (Hz)."""
    k = np.arange(-filt.half, filt.half + 1)
    return np.exp(-2j * np.pi * np.outer(np.asarray(freqs) / sfreq, k)) @ filt.h


def partition_error(filters: list[Filter], sfreq: float, n: int = 512) -> float:
    """max |Σ |H_L(f)|²/4 − 1| over the bank's inner range (the analytic gain is 2 on positive frequencies)."""
    lo = min(F.band_hz[0] for F in filters) * 2 ** CROSSOVER_OCT
    hi = max(F.band_hz[1] for F in filters) * 2 ** -CROSSOVER_OCT
    f = np.geomspace(lo, hi, n)
    tot = sum(np.abs(response(F, f, sfreq)) ** 2 for F in filters) / 4.0
    return float(np.max(np.abs(tot - 1.0)))


def filter_block(x: np.ndarray, filt: Filter) -> np.ndarray:
    """Filter a block padded by H samples on each side ([channels, T + 2H], zeros outside the record):
    returns the T analytic samples of its interior (exact overlap-save)."""
    return fftconvolve(np.asarray(x), filt.h[None, :], mode="valid", axes=-1)


def clean(good: np.ndarray, half: int) -> np.ndarray:
    """Samples whose ±H neighbourhood lies inside the record and holds no bad sample."""
    bad = np.r_[np.ones(half, bool), ~np.asarray(good, bool), np.ones(half, bool)].astype(np.int64)
    c = np.r_[0, np.cumsum(bad)]
    w = 2 * half + 1
    return (c[w:] - c[:-w]) == 0
