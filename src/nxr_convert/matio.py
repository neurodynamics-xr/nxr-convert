"""Brainstorm ``.mat`` reading — one loader for both MAT v5/v7 and v7.3 (HDF5), through
pymatreader. Returns plain dicts of numpy arrays / strings, squeezed the way Brainstorm
code expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pymatreader import read_mat


def load_mat(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    return read_mat(str(p))


def load_mat_vars(path: str | Path, names: list[str]) -> dict[str, Any]:
    """Read only the named variables — the cheap PROBE (a head-model file
    carries an ~84 MB Gain; probing its GridLoc must not decode that). Both
    formats come back in `load_mat`'s shape: structs are dicts."""
    return read_mat(str(path), variable_names=names)


def struct_rows(d: Any, key: str) -> list[dict]:
    """A Brainstorm struct array as a list of per-element dicts, whatever pymatreader
    made of it: a dict of per-field lists (one entry per element, `key` naming a field
    every element has), a single struct (its `key` a string), or already a list."""
    if d is None:
        return []
    if not isinstance(d, dict):
        return [dict(e) for e in d]
    k = d.get(key)
    if k is None:
        return []
    if isinstance(k, str):
        return [d]
    n = len(k)
    return [{f: (v[i] if isinstance(v, (list, np.ndarray)) and len(v) == n else v) for f, v in d.items()}
            for i in range(n)]
