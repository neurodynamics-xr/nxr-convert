"""Filesystem-compatibility shim for zarr-python's LocalStore.

zarr >= 3.3 makes exclusive node creation atomic via a HARDLINK
(``_safe_move``: ``os.link`` then ``unlink``) — which raises
``OSError(EOPNOTSUPP/EPERM)`` on filesystems without hardlinks (exFAT is the
live case: an external drive chosen as the datastore, 2026-08-13). Upstream
has no fallback as of 3.3.0 (nor on main), so every write into such a
datastore crashes on the FIRST node.

The fallback below keeps the semantics that matter here: ``FileExistsError``
when the destination exists (what ``os.link`` guaranteed), then a plain
``os.replace``. The existence check is not atomic against a concurrent
creator — acceptable by construction: nxr-convert is the datastore's SINGLE
writer (the app's convert service refuses two subprocesses, and the CLI is
one process).

Imported for its side effect by ``store`` — every writer passes through it.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import zarr.storage._local as _local

_HARDLINK_ERRNOS = {errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM, errno.EXDEV}

_orig_safe_move = getattr(_local, "_safe_move", None)   # private: a zarr without it is left alone


def _safe_move_with_fallback(src: Path, dst: Path) -> None:
    try:
        _orig_safe_move(src, dst)
    except OSError as e:
        if e.errno not in _HARDLINK_ERRNOS:
            raise
        if os.path.exists(dst):
            os.unlink(src)
            raise FileExistsError(str(dst)) from e
        os.replace(src, dst)


if _orig_safe_move is not None:
    _local._safe_move = _safe_move_with_fallback
