"""Brainstorm's SQLite database — NOT YET IMPLEMENTED, on purpose.

Brainstorm is migrating its protocol index to SQLite. The dispatch in `__init__`
already prefers it, so the day a protocol carries one this is the only file that
needs writing, and every consumer above it is unchanged — which is the whole reason
the reader is a dispatch and the model is separate.

It raises rather than silently degrading to `protocol.mat`. A source view whose
`kind` says "sqlite" while the numbers came from somewhere else is exactly the class
of confident-wrong-answer this codebase keeps finding, and the fallback is one
argument away (`read_source(root, prefer='protocol.mat')`) for anyone who wants it.
"""
from __future__ import annotations

from pathlib import Path

from .model import SourceView


def read_protocol_db(root: str | Path) -> SourceView:  # pragma: no cover - not built
    raise NotImplementedError(
        "Brainstorm's SQLite protocol database is not readable yet. When it lands, "
        "implement it here and return the same SourceView the other readers do; the "
        "dispatch in sources/__init__.py already prefers it. Until then use "
        "read_source(root, prefer='protocol.mat') or prefer='walk'."
    )
