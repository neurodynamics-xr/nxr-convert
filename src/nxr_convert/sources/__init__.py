"""Reading a Brainstorm database as a SOURCE VIEW.

DISPATCH, not a parser: the reader picks by what is on disk and every reader returns
the same `SourceView`. (Brainstorm's coming SQLite database is a third reader, added
when it lands.)

    protocol.mat  today's index — one 78 KB read        (protocol_mat.py)
    walk          the filesystem traversal, as fallback (walk.py)

The walk is KEPT rather than replaced. It is the honest answer for a protocol with no
index (half-copied, hand-assembled), and it is the cross-check: the index describes
what Brainstorm believes, the disk holds what is actually there, and where they
disagree is a real condition worth reporting rather than papering over.
"""
from __future__ import annotations

from pathlib import Path

from .model import SourceView, SourceSubject, SourceCondition, SourceRecording, SCHEMA

__all__ = ["read_source", "SourceView", "SourceSubject", "SourceCondition", "SourceRecording", "SCHEMA"]


def read_source(root: str | Path, prefer: str | None = None) -> SourceView:
    """The view of a Brainstorm protocol at `root`.

    `prefer` forces one reader ('protocol.mat' | 'walk') — used by the
    cross-check, which reads BOTH and diffs them, and by tests that must pin which
    path they are exercising rather than depend on what happens to be on disk.
    """
    root = Path(root)
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"not a Brainstorm protocol (no data/): {root}")

    if prefer == "walk":
        from .walk import read_walk
        return read_walk(root)
    if prefer == "protocol.mat":
        from .protocol_mat import read_protocol_mat
        return read_protocol_mat(root)

    if (root / "data" / "protocol.mat").is_file():
        from .protocol_mat import read_protocol_mat
        return read_protocol_mat(root)
    from .walk import read_walk
    return read_walk(root)

