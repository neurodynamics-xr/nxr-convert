"""A recording's Brainstorm EVENTS — read into a table, written as SELECTIONS (D143).

    timeseries/<rec>_events         Selection  spans on the recording's time Line: a member per occurrence, [start, stop)
                                               in seconds (an instantaneous event start = stop), code = its event type in
                                               the subject's ``<rec>_events`` dictionary (names, colours)
    timeseries/<rec>_bad_segments   Selection  the same for the EXTENDED events whose label contains 'bad' — what the atlas
                                               and the views exclude

THE FRAMING: an event table is to a timeseries what an atlas is to a manifold — categorical labels over a domain with a
dictionary. Events overlap, are sparse and coexist, so they are not a partition: they are SPANS, a member per occurrence.
A per-occurrence channel scope (Brainstorm's ``channels``) rides in the selection's ``params_json`` (``channels``: the
member ordinal → its channel indices).

Brainstorm's own shape is per-TYPE: one struct entry per label, holding a
``1×N`` (simple) or ``2×N`` (extended) times matrix. This flattens that to
occurrence rows sorted by onset, which is what makes a range query a binary
search; the type grouping survives losslessly in ``type``/``type_names``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .matio import struct_rows


@dataclass(frozen=True)
class EventTable:
    type: np.ndarray                       # int32 [N] -> index into type_names
    onset: np.ndarray                      # float64 [N] seconds
    duration: np.ndarray                   # float64 [N] seconds; 0 == simple
    channel_offsets: np.ndarray            # int32 [N+1] CSR row pointers
    channel_index: np.ndarray              # int32 [M] indices into channel_names
    type_names: list[str] = field(default_factory=list)
    type_colors: list[list[float]] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return int(self.onset.size)


def rejoin_cellstr(value: Any) -> list[str]:
    """A Brainstorm cell-of-strings as a list of strings.

    pymatreader collapses a SINGLE-element cell into a char array, so a
    one-channel reference `'P11'` reads back as `['P','1','1']` while a genuine
    two-element cell `['PO7','PO3']` reads correctly. The tell is that every
    element is one character: a real montage never scopes an event to three
    channels literally named 'P', '1' and '1'.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    items = [str(v) for v in np.atleast_1d(np.asarray(value, dtype=object)).ravel().tolist()]
    items = [s for s in items if s != ""]
    if len(items) > 1 and all(len(s) == 1 for s in items):
        return ["".join(items)]
    return items


def _color(entry: dict) -> list[float]:
    c = np.ravel(np.asarray(entry.get("color", []), dtype=float))
    if c.size < 3:
        return [0.5, 0.5, 0.5]
    return [float(c[0]), float(c[1]), float(c[2])]


def _times_matrix(entry: dict) -> np.ndarray:
    """A type's times as a ``(1 or 2) x nOccurrences`` matrix — Brainstorm's
    own shape, recovered from whatever pymatreader's ``squeeze_me`` did to it.

    Squeeze erases EVERY singleton dimension, including the ROW axis that
    distinguishes simple from extended. A single-occurrence EXTENDED event
    (2 rows, 1 column: onset, offset) round-trips to a bare 1-D array of
    length 2 — bitwise identical to what a 1-row, 2-column SIMPLE event (two
    onsets) round-trips to. ``np.atleast_2d`` cannot tell these apart; it
    always prepends a row axis, so it silently reads the real file's ``L``
    event (AnphySleep/EPCTL01: one 150 s span, 28560.0 -> 28710.0) as two
    zero-duration marks instead of one 150 s span — 29149 occurrences
    counted where truth is 29148.

    ``epochs`` is the disambiguator: Brainstorm always sizes it
    ``1 x nOccurrences``, so its element count is the occurrence count
    regardless of what squeeze did to ``times``. Reshape by it whenever the
    counts are consistent AND the implied row count is 1 or 2 — Brainstorm's
    own contract for ``times`` (simple or extended, never more). ``n_occ ==
    1`` makes ``raw.size % n_occ == 0`` true unconditionally, which is NOT a
    check: it would happily reshape a genuine 2x15 extended matrix down to
    ``(30, 1)`` — one occurrence reported, 14 silently lost, with a duration
    fabricated from the first and last onset — and it would just as happily
    merge two independent simple onsets into one fake extended span, the
    exact inverse of the bug this function exists to fix. When the implied
    row count is inconsistent with ``epochs`` (not 0-size-divisible) OR would
    imply more than 2 rows, this raises rather than truncate silently — a
    times matrix genuinely is never 3+ rows for a real Brainstorm event, so a
    3-row implication means ``epochs`` and ``times`` disagree in a way this
    parser cannot resolve, and the old ``atleast_2d`` fallback would silently
    drop rows 2+ with no error at all. Otherwise fall back to the old guess
    (e.g. no ``epochs`` field at all, or a non-Brainstorm construction of
    ``times``). Do not revert this to a bare ``atleast_2d`` — that is
    precisely the bug this exists to fix.
    """
    raw = np.asarray(entry.get("times", []), dtype=float)
    epochs = entry.get("epochs")
    if epochs is not None:
        n_occ = int(np.size(epochs))
        if n_occ > 0 and raw.size % n_occ == 0:
            n_rows = raw.size // n_occ
            if n_rows > 2:
                label = str(entry.get("label", "<unknown>"))
                raise ValueError(
                    f"event {label!r}: times has {raw.size} elements over "
                    f"{n_occ} occurrence(s) (epochs), implying {n_rows} rows — "
                    f"Brainstorm's times matrix is always 1 (simple) or 2 "
                    f"(extended) rows, so this shape cannot be reshaped safely; "
                    f"refusing rather than silently dropping rows 2+")
            if n_rows in (1, 2):
                return raw.reshape(n_rows, n_occ)
    return np.atleast_2d(raw)


def parse_events(events: Any, channel_names: list[str] | None = None) -> EventTable:
    """One Brainstorm ``Events`` / ``F.events`` struct as an EventTable."""
    index = {str(n): i for i, n in enumerate(channel_names or [])}
    types: list[str] = []
    colors: list[list[float]] = []
    rows: list[tuple[float, float, int, list[int]]] = []

    for entry in struct_rows(events, "label"):
        label = str(entry.get("label", ""))
        times = _times_matrix(entry)
        if times.size == 0:
            continue
        ti = len(types)
        types.append(label)
        colors.append(_color(entry))
        onsets = times[0]
        # A 2-row times matrix is an EXTENDED event: row 0 onset, row 1 offset.
        offsets = times[1] if times.shape[0] >= 2 else onsets
        raw_ch = entry.get("channels")
        # Per-occurrence channel scoping can arrive as a Python list (scipy) or
        # a numpy object array (h5py/MAT v7.3) — either counts, as long as its
        # length matches the occurrence count. A shorter/absent/scalar value
        # broadcasts to every occurrence (i.e. "no per-occurrence data").
        if raw_ch is None or (isinstance(raw_ch, np.ndarray) and raw_ch.size == 0):
            per_occ = [None] * onsets.size
        elif isinstance(raw_ch, (list, np.ndarray)) and len(raw_ch) == onsets.size:
            per_occ = list(raw_ch)
        else:
            per_occ = [raw_ch] * onsets.size
        for k in range(onsets.size):
            names = rejoin_cellstr(per_occ[k]) if k < len(per_occ) else []
            idx = []
            for nm in names:
                if nm not in index:
                    raise ValueError(
                        f"event {label!r} references channel {nm!r}, which is not in the "
                        f"recording's channel list — refusing rather than dropping the "
                        f"reference (known channels: {len(index)})")
                idx.append(index[nm])
            duration = float(offsets[k]) - float(onsets[k])
            if duration < 0:
                raise ValueError(
                    f"event {label!r} occurrence {k} has offset {float(offsets[k])!r} "
                    f"before onset {float(onsets[k])!r} (duration {duration!r} < 0) — "
                    f"refusing rather than producing a negative duration")
            rows.append((float(onsets[k]), duration, ti, idx))

    # Sorted by ONSET, so a range query is a binary search. Sorting by type
    # instead would buy an O(1) per-type slice; you cannot have both, and at
    # ~30k rows a per-type scan costs nothing.
    rows.sort(key=lambda r: r[0])

    n = len(rows)
    onset = np.array([r[0] for r in rows], dtype=np.float64)
    duration = np.array([r[1] for r in rows], dtype=np.float64)
    tcol = np.array([r[2] for r in rows], dtype=np.int32)
    offs = np.zeros(n + 1, dtype=np.int32)
    flat: list[int] = []
    for i, r in enumerate(rows):
        flat.extend(r[3])
        offs[i + 1] = len(flat)
    return EventTable(type=tcol, onset=onset, duration=duration,
                      channel_offsets=offs,
                      channel_index=np.asarray(flat, dtype=np.int32),
                      type_names=types, type_colors=colors)



from .crud import Subject  # noqa: E402
from .entities import time_spans  # noqa: E402


def export_events(sub: Subject, rec_path: str, table: EventTable, *, line_id: str, of_id: str,
                  session: str | None = None) -> dict:
    """``<rec>_events`` and ``<rec>_bad_segments`` as spans Selections on the time Line ``line_id``. A table with no rows
    writes NOTHING: absence means 'no events'."""
    n = table.n_events
    if n == 0:
        return {"n_events": 0, "n_types": 0, "bad_segments": 0}
    spans = [(float(table.onset[k]), float(table.onset[k] + table.duration[k]), int(table.type[k])) for k in range(n)]
    scoped = {str(k): table.channel_index[table.channel_offsets[k]:table.channel_offsets[k + 1]].tolist()
              for k in range(n) if table.channel_offsets[k + 1] > table.channel_offsets[k]}
    time_spans(sub, f"{rec_path}_events", spans, table.type_names, line_id=line_id, of_field_id=of_id, session=session,
               colors=table.type_colors, description=f"the recording's {n} Brainstorm events — one span per occurrence, "
                                                     "[onset, onset + duration) in seconds; code = the event type",
               params={"channels": scoped} if scoped else None)
    bad_types = [i for i, nm in enumerate(table.type_names) if "bad" in nm.lower()]
    bad = [(a, b, bad_types.index(t)) for a, b, t in spans if t in bad_types and b > a]
    if bad:
        time_spans(sub, f"{rec_path}_bad_segments", bad, [table.type_names[i] for i in bad_types], line_id=line_id,
                   of_field_id=of_id, session=session, colors=[table.type_colors[i] for i in bad_types],
                   description="the recording's bad segments (Brainstorm events labelled bad*), one span per segment; "
                               "code = the event type")
    return {"n_events": n, "n_types": len(table.type_names), "bad_segments": len(bad)}
