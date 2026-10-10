"""
The time convention of the atlas (D135, nsp ``feat/atlas`` decision TOWER): frames are CYCLES of the day-anchored tower of
nested cycles. Level L is one cycle of 86 400·2^L s (the day is level 0); the atlas's base frame is level ``FRAME_LEVEL``
(−18: ≈ 0.3296 s). A frame is generally not a whole number of samples (395.5 at 1200 Hz), so each sample is assigned to its
tile, floor((start_s·fs + i) / (tile_s·fs)), and each frame carries its own sample count. Frame k is tower code
``frame_code0 + k``; a level-m tile is code >> m (``rollup.time(..., code0=frame_code0)``).

Ported from nsp's ``atlas/grid.py`` (``period_s``) and ``atlas/placement.py`` (``Placement``), only the time half.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DAY_S = 86400.0
FRAME_LEVEL = -18


def period_s(level: int) -> float:
    """The period of a level-L cycle (s): 86 400·2^L."""
    return DAY_S * 2.0 ** level


def level_for_seconds(seconds: float) -> int:
    """The finest level whose cycle is at least ``seconds`` long."""
    return int(math.ceil(math.log2(seconds / DAY_S)))


@dataclass(frozen=True)
class Placement:
    """A recording on the tower: anchor "recording" (tiles counted from its first sample), "day" (``start_s``, seconds from
    local midnight) or "birth" (also ``day``, days since birth). Tile lengths are the same in every case."""
    recording: str
    sfreq: float
    n_samples: int
    start_s: float | None = None
    day: int | None = None

    @property
    def anchor(self) -> str:
        if self.start_s is None:
            return "recording"
        return "day" if self.day is None else "birth"

    @property
    def coarsest(self) -> int | None:
        if self.anchor == "recording":
            return level_for_seconds(self.n_samples / self.sfreq)
        return 0 if self.anchor == "day" else None

    def sample_codes(self, level: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        """The tile (at ``level``) of each sample in [start, stop), computed in samples so a boundary on a sample is exact."""
        top = self.coarsest
        if top is not None and level > top:
            raise ValueError(f"{self.recording}: level {level} is above what is known (anchor {self.anchor}, top level {top})")
        stop = self.n_samples if stop is None else stop
        offset = 0.0 if self.start_s is None else self.start_s * self.sfreq
        pos = offset + np.arange(start, stop, dtype=np.float64)
        if level > 0:
            days = int(self.day) + np.floor(pos / (DAY_S * self.sfreq)).astype(np.int64)
            return days >> level
        codes = np.floor(pos / (period_s(level) * self.sfreq)).astype(np.int64)
        return codes + (int(self.day) * 2 ** -level if self.anchor == "birth" else 0)


def recording_placement(store_path: Path, name: str, sfreq: float, n_samples: int) -> Placement:
    """The recording's stored placement (nsp's ``dynamics_atlas/placements/<recording>``, aligned to the day when its
    acquisition time was known), else its own timeline."""
    z = Path(store_path) / "dynamics_atlas" / "placements" / name / "zarr.json"
    if z.exists():
        m = json.loads(z.read_text()).get("attributes", {})
        if m.get("n_samples") == n_samples and m.get("sfreq") == sfreq:
            return Placement(name, sfreq, n_samples, m.get("start_s"), m.get("day"))
    return Placement(name, sfreq, n_samples)
