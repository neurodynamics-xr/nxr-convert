"""Brainstorm ``.mat`` reading — one loader for both MAT v5/v7 (scipy via
pymatreader) and v7.3 (HDF5 via h5py, same package). Returns plain dicts of
numpy arrays / strings, squeezed the way Brainstorm code expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy.io
from pymatreader import read_mat


def load_mat(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    return read_mat(str(p))


def load_mat_vars(path: str | Path, names: list[str]) -> dict[str, Any]:
    """Read only the named variables — the cheap PROBE (a head-model file
    carries an ~84 MB Gain; probing its GridLoc must not decode that)."""
    try:
        return scipy.io.loadmat(str(path), variable_names=names,
                                squeeze_me=True, struct_as_record=False)
    except NotImplementedError:      # MAT v7.3 — fall back to a full HDF5 read
        full = read_mat(str(path))
        return {k: full[k] for k in names if k in full}
