"""Reading a Brainstorm database as a SOURCE VIEW.

DISPATCH, not a parser: Brainstorm is moving its database to SQLite, so the reader
picks by what is on disk and every reader returns the same `SourceView`.

    sqlite        the new database, when it lands       (protocol_db.py)
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

    `prefer` forces one reader ('protocol.mat' | 'sqlite' | 'walk') — used by the
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
    if prefer == "sqlite":
        from .protocol_db import read_protocol_db
        return read_protocol_db(root)

    # Newest first — a protocol mid-migration may carry both, and the database is
    # then the one Brainstorm is actually maintaining.
    db = _sqlite_path(root)
    if db is not None:
        from .protocol_db import read_protocol_db
        return read_protocol_db(root)
    if (root / "data" / "protocol.mat").is_file():
        from .protocol_mat import read_protocol_mat
        return read_protocol_mat(root)
    from .walk import read_walk
    return read_walk(root)


def _sqlite_path(root: Path) -> Path | None:
    """Brainstorm's SQLite database, if this protocol has one.

    The filename is not settled upstream yet, so this matches the candidates rather
    than hard-coding one — and returns None rather than guessing when several match,
    because picking arbitrarily between two databases is worse than falling back to
    an index we can read correctly.
    """
    hits = sorted(p for p in (root / "data").glob("*.db") if p.is_file())
    hits += sorted(p for p in (root / "data").glob("*.sqlite") if p.is_file())
    return hits[0] if len(hits) == 1 else None
