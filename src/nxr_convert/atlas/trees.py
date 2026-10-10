"""
The two bookkeeping trees of the atlas — one code per cortical vertex, rolled up by arithmetic.

Both trees are cortical-flow's dyadic surface ladder (``ladder.surface_ladder``) run per
hemisphere; a vertex's code at depth D is ``h·2^D + t`` (h = 0 left, 1 right; t its tile at level
D), and its node at any coarser level ℓ ≤ D is ``code >> (D − ℓ)``. Codes are contiguous per node,
so a roll-up is a reshape and a sum, and nothing per level is ever stored.

  SUBJECT tree  the ladder on the subject's own pial surface — equal-area on THAT cortex, for
                within-subject analysis. Identical to the tiles cortical-flow shows for it.
  GROUP tree    the ladder built ONCE on fsaverage (equal-area on the average cortex); each subject
                vertex takes the code of its nearest fsaverage vertex on the registered sphere
                (FreeSurfer's sphere.reg, which the nxr store carries as ``<surface>_sphere``), so
                node s is the same place in every subject — for group analysis.

The group tree's template is the dataset's default subject (``default_subject``): its tiles and sphere are
read back from that store (``default_subject.load_default_template``), never from a private file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ladder import Ladder, component_mesh, surface_ladder

HEMIS = ("lh", "rh")


@dataclass
class Tree:
    """One code per vertex at depth ``depth`` (−1 off the cortex) and what built it."""
    kind: str                       # "subject" | "group"
    depth: int                      # D: codes are h·2^D + t
    codes: np.ndarray               # int32 [n_vertices]
    finest: int                     # the deepest level the ladder reached on every hemisphere
    meta: dict = field(default_factory=dict)

    @property
    def n_leaves(self) -> int:
        return 2 * 2 ** self.depth

    def at(self, level: int) -> np.ndarray:
        """The node of every vertex at ``level`` ≤ depth (−1 stays −1)."""
        if not 0 <= level <= self.depth:
            raise ValueError(f"level {level} outside 0..{self.depth}")
        c = self.codes
        return np.where(c >= 0, c >> (self.depth - level), -1)

    def coarsen(self, depth: int) -> "Tree":
        """The same tree cut at a shallower depth."""
        return Tree(self.kind, depth, self.at(depth).astype(np.int32), self.finest, dict(self.meta))


def hemisphere_runs(structures: np.ndarray, names: list[str] | None = None) -> list[tuple[int, int]]:
    """(start, length) of the left then right hemisphere: the contiguous runs of the Structures labels.

    Labels are read by name when the LUT is given ("Cortex L" / "Cortex R"), else 1 = left, 2 = right.
    A label that is not one contiguous run is refused: a hemisphere must be one block of vertices.
    """
    s = np.asarray(structures)
    want = [1, 2]
    if names:
        low = [n.lower() for n in names]
        want = [next(i for i, n in enumerate(low) if n.endswith(" l") or "left" in n),
                next(i for i, n in enumerate(low) if n.endswith(" r") or "right" in n)]
    runs = []
    for lab in want:
        idx = np.flatnonzero(s == lab)
        if idx.size == 0:
            raise ValueError(f"no vertex labelled {lab} in Structures")
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError(f"hemisphere label {lab} is not contiguous in vertex order")
        runs.append((int(idx[0]), int(idx.size)))
    return runs


def _codes_from_ladders(n: int, runs: list[tuple[int, int]], ladders: list[Ladder], depth: int | None) -> tuple[np.ndarray, int, int]:
    finest = min(l.finest for l in ladders)
    d = finest if depth is None else depth
    if not 0 <= d <= finest:
        raise ValueError(f"depth {d} outside 0..{finest} (the finest level every hemisphere reaches)")
    codes = np.full(n, -1, dtype=np.int32)
    for h, ((start, length), lad) in enumerate(zip(runs, ladders)):
        codes[start:start + length] = h * 2 ** d + lad.labels[d]
    return codes, d, finest


def subject_tree(positions: np.ndarray, faces: np.ndarray, structures: np.ndarray,
                 names: list[str] | None = None, depth: int | None = None) -> Tree:
    """The subject tree: the ladder on the subject's own surface, per hemisphere."""
    runs = hemisphere_runs(structures, names)
    ladders = [surface_ladder(*component_mesh(positions, faces, s, n)) for s, n in runs]
    codes, d, finest = _codes_from_ladders(len(positions), runs, ladders, depth)
    areas = [float(l.area) for l in ladders]
    return Tree("subject", d, codes, finest, {"hemisphere_runs": runs, "hemisphere_area_m2": areas})


# ── the group tree: built on fsaverage, assigned through the registered sphere ─────────────────

def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@dataclass
class Template:
    """The group tree on fsaverage: per hemisphere, the unit sphere positions and the finest codes."""
    name: str
    finest: int
    sphere: dict[str, np.ndarray]    # hemi → [n, 3] unit vectors (fsaverage sphere.reg)
    codes: dict[str, np.ndarray]     # hemi → [n] tile at level ``finest`` (no hemisphere offset)
    meta: dict = field(default_factory=dict)
    resolutions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # name → hemi → parent vertex

    def tree_for(self, sphere: np.ndarray, structures: np.ndarray, names: list[str] | None = None,
                 depth: int | None = None) -> Tree:
        """Assign a subject's vertices to the group tree: nearest fsaverage vertex on the sphere."""
        from scipy.spatial import cKDTree
        runs = hemisphere_runs(structures, names)
        d = self.finest if depth is None else depth
        if not 0 <= d <= self.finest:
            raise ValueError(f"depth {d} outside 0..{self.finest}")
        codes = np.full(len(sphere), -1, dtype=np.int32)
        dist = np.full(len(sphere), np.nan)
        for h, (hemi, (start, n)) in enumerate(zip(HEMIS, runs)):
            q = _unit(sphere[start:start + n])
            dd, j = cKDTree(self.sphere[hemi]).query(q)
            codes[start:start + n] = h * 2 ** d + (self.codes[hemi][j] >> (self.finest - d))
            dist[start:start + n] = dd
        return Tree("group", d, codes, self.finest,
                    {"template": self.name, "template_sha": self.meta.get("sha"), "hemisphere_runs": runs,
                     "sphere_match_median": float(np.nanmedian(dist)), "sphere_match_max": float(np.nanmax(dist))})
