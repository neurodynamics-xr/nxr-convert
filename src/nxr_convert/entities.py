"""The Brainstorm readers' shapes as ENTITIES — labellings, label sets, time spans — composed on a ``crud.Subject``.

A LABELLING is a partition Selection (D48): one code per element, its codes read with a DICTIONARY (D34) — an atlas's is
app-wide (every subject's Desikan-Killiany codes alike), a subject's own names (its channels, a recording's flags, its
events) are the subject's. A LABEL SET is a ``set`` Selection whose members are labellings (O9). A list of time intervals
(events, bad segments, D143) is a ``spans`` Selection on a recording's time Line, a member per interval, coded by type.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .crud import Subject, jdump


def remap(codes: np.ndarray, code_of: Sequence[int]) -> np.ndarray:
    """A labelling's level indices → the dictionary's codes (identity for a new dictionary)."""
    lut = np.asarray(code_of, dtype=np.int64)
    out = lut[np.asarray(codes, dtype=np.int64)] if lut.size else np.asarray(codes, dtype=np.int64)
    return out.astype(np.int32)


def labelling(sub: Subject, path: str, codes, levels: Sequence[str], *, manifold_id: str, dictionary: str,
              scope: str = "subject", colors=None, session: str | None = None, description: str | None = None,
              member_of: tuple[str, int] | None = None, of_field_id: str | None = None, dtype=np.int32,
              element_rows: bool = True, entries: Sequence[dict] | None = None, dictionary_id: str | None = None,
              name: str | None = None) -> str:
    """One PARTITION Selection at ``path``: its codes array (laid out from its row, then populated), its element rows,
    its dictionary (made or extended by name in its scope; or ``dictionary_id``, whose codes ``codes`` already are)."""
    if dictionary_id is not None:
        did, data = dictionary_id, np.asarray(codes).astype(dtype)
    else:
        did, code_of = sub.dictionary(dictionary, list(levels), scope=scope, colors=colors, entries=entries)
        data = remap(codes, code_of).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.unsignedinteger) and data.size and data.max() > np.iinfo(dtype).max:
        raise ValueError(f"{path}: a code exceeds {np.dtype(dtype)}")
    name = name or path.rsplit("/", 1)[-1]
    kw = {}
    if member_of is not None:
        kw = {"member_of_id": member_of[0], "member_ordinal": member_of[1]}
    return sub.selection(name=name, path=path, type="set", manifold_id=manifold_id, data=data, dictionary_id=did,
                         elements=None if element_rows else False, n_elements=int(data.size),
                         session=session, description=description, of_field_id=of_field_id, **kw)


def label_set(sub: Subject, set_path: str, members: Sequence[tuple[str, np.ndarray, Sequence[str], np.ndarray | None]], *,
              manifold_id: str, description: str, session: str | None = None,
              dictionary: Callable[[str], tuple[str, str]] = lambda fam: (fam, "app"),
              element_rows: bool = True) -> tuple[str, list[str]]:
    """A SET of labellings: the set at ``set_path`` (a group node), each member its own partition ``<set_path>/<name>``.
    ``dictionary(member) → (name, scope)``. Returns (set id, member ids)."""
    set_id = sub.selection(name=set_path.rsplit("/", 1)[-1], path=set_path, type="set", manifold_id=manifold_id,
                           n_members=len(members), session=session, description=description)
    ids = []
    for k, (fam, codes, levels, colors) in enumerate(members):
        dname, scope = dictionary(fam)
        ids.append(labelling(sub, f"{set_path}/{fam}", codes, levels, manifold_id=manifold_id, dictionary=dname, scope=scope,
                             colors=colors, session=session, description=f"{fam} — one level per element",
                             member_of=(set_id, k), element_rows=element_rows))
    return set_id, ids


def rgb255(c) -> list[int]:
    """Brainstorm's [0, 1] colour as 0–255 (a dictionary entry's ``color_json``)."""
    c = np.ravel(np.asarray(c, dtype=float))[:3]
    if c.size < 3:
        c = np.array([0.5, 0.5, 0.5])
    scale = 255.0 if float(np.max(c)) <= 1.0 else 1.0
    return [int(round(float(x) * scale)) for x in c]


def time_spans(sub: Subject, path: str, spans: Sequence[tuple[float, float, int]], types: Sequence[str], *, line_id: str,
               of_field_id: str, session: str | None, colors=None, description: str, params: dict | None = None) -> str:
    """A ``spans`` Selection on a time Line (D87/D143): a member per interval, ``[start, stop)`` in seconds (an
    instantaneous one start = stop), ``code`` = its type in the subject's dictionary named after the selection."""
    did, code_of = sub.dictionary(path.rsplit("/", 1)[-1], list(types), scope="subject",
                                  colors=[rgb255(c) for c in colors] if colors is not None else None)
    rows = [(float(a), float(b), int(code_of[t])) for a, b, t in spans]
    return sub.selection(name=path.rsplit("/", 1)[-1], path=path, type="spans", manifold_id=line_id, cell="1", spans=rows,
                         dictionary_id=did, unit="s", of_field_id=of_field_id, session=session, description=description,
                         params_json=jdump(params) if params else None)
