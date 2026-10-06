"""Raw-link reading — Phase ②.

A Brainstorm raw LINK's ``F`` is an sFile struct whose samples live in a
separate binary. This protocol's links are **BST-BIN** (``.bst`` — Brainstorm's
own format, written when the notch-filtered continuous was saved): a
``hdrsize``-byte header, then EPOCHS of ``[nchannels × epochsize]`` float32,
each channel's samples contiguous within its epoch (the ``in_fread_bst.m``
layout, ported here byte-for-byte). The header fields ride in the LINK's own
``F.header``, so the binary's header is never parsed.

CTF compensation is a no-op on these links (``currCtfComp == destCtfComp``);
the SSP projector question is settled by the oracle gate (the recording was
saved projected, so applying P is idempotent — whether the float32 cast
absorbs the 1e-16 is measured, not assumed).

Original-format raw links (CTF ``.ds``, FIF) go through MNE when they appear;
this module reads what this protocol actually contains.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_bst(sfile: dict, *, s0: int = 0, n: int | None = None,
             channels: tuple[int, int] | None = None,
             path: str | Path | None = None) -> np.ndarray:
    """Read ``[nchannels, n]`` float64 from a BST-BIN link, samples
    ``[s0, s0+n)`` relative to the file start. ``channels=(c0, c1)`` reads a
    contiguous 0-based channel range — channels are contiguous within an
    epoch, so a range costs one seek per epoch, and it is what keeps the
    ladder within this machine's no-swap memory ceiling."""
    header = sfile["header"]
    n_chan_total = int(header["nchannels"])
    c0, c1 = channels if channels is not None else (0, n_chan_total)
    n_chan = c1 - c0
    epoch_size = int(header["epochsize"])
    hdr = int(header["hdrsize"])
    n_total = int(header["nsamples"])
    if n is None:
        n = n_total - s0
    if s0 < 0 or s0 + n > n_total:
        raise ValueError(f"read [{s0}, {s0+n}) outside [0, {n_total})")
    p = Path(path or sfile["filename"])

    out = np.empty((n_chan, n), dtype=np.float64)
    got = 0
    with open(p, "rb") as f:
        e0 = s0 // epoch_size
        e1 = (s0 + n - 1) // epoch_size
        for e in range(e0, e1 + 1):
            lo = max(e * epoch_size, s0)
            hi = min((e + 1) * epoch_size, s0 + n)
            count = hi - lo
            f.seek(hdr + (e * n_chan_total + c0) * epoch_size * 4)
            block = np.fromfile(f, dtype="<f4", count=epoch_size * n_chan)
            block = block.reshape(n_chan, epoch_size)   # channel-major within the epoch
            out[:, got:got + count] = block[:, lo - e * epoch_size: hi - e * epoch_size]
            got += count
    assert got == n
    return out
