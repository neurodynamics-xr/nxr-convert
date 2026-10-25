"""FreeSurfer's binary files, read without a dependency: triangle surfaces (``surf/lh.pial``) and annotations
(``label/lh.aparc.annot``). All integers and floats are big-endian."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

TRIANGLE_MAGIC = b"\xff\xff\xfe"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_surf(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [n × 3] float64 in the file's units — mm, faces [m × 3] int64) of a triangle surface file."""
    b = Path(path).read_bytes()
    if b[:3] != TRIANGLE_MAGIC:
        raise ValueError(f"{path}: not a FreeSurfer triangle surface (magic {b[:3].hex()})")
    at = 3
    for _ in range(2):                                   # two newline-terminated comment lines
        at = b.index(b"\n", at) + 1
    nv, nf = np.frombuffer(b, ">i4", 2, at)
    at += 8
    v = np.frombuffer(b, ">f4", 3 * nv, at).reshape(nv, 3).astype(np.float64)
    f = np.frombuffer(b, ">i4", 3 * nf, at + 12 * nv).reshape(nf, 3).astype(np.int64)
    return v, f


def read_annot(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(entry index per vertex — -1 where no entry owns the vertex, entry names, colours [k × 3]) of an annotation."""
    b = Path(path).read_bytes()
    at = 0

    def ints(n):
        nonlocal at
        out = np.frombuffer(b, ">i4", n, at)
        at += 4 * n
        return out

    n = int(ints(1)[0])
    pairs = ints(2 * n).reshape(n, 2)
    if int(ints(1)[0]) != 1:
        raise ValueError(f"{path}: no colour table")
    ver = int(ints(1)[0])
    if ver > 0:                                          # the old layout: the count first, no ids
        k, ids = ver, None
        skip = int(ints(1)[0])                           # its original file name
        at += skip
    else:
        if ver != -2:
            raise ValueError(f"{path}: colour table version {-ver} is not read")
        ints(1)                                          # the number of entries the table spans
        skip = int(ints(1)[0])                           # its original file name
        at += skip
        k, ids = int(ints(1)[0]), True
    names, rgb, annot = [], [], []
    for i in range(k):
        if ids:
            ints(1)
        ln = int(ints(1)[0])
        names.append(b[at:at + ln].split(b"\0")[0].decode())
        at += ln
        r, g, bl, _ = ints(4)
        rgb.append((r, g, bl))
        annot.append(int(r) + (int(g) << 8) + (int(bl) << 16))
    lut = {a: i for i, a in enumerate(annot)}
    return np.array([lut.get(int(a), -1) for a in pairs[np.argsort(pairs[:, 0]), 1]]), names, np.array(rgb, dtype=np.uint8)


def atlas_members(fs: Path, n_lh: int, n_rh: int, families) -> list:
    """The annotations present for BOTH hemispheres, as label-set members (family, codes, level names, colours): code 0
    ``unassigned``, then each entry that owns a vertex, named ``<entry> L`` / ``<entry> R`` as Brainstorm names scouts."""
    out = []
    for stem, family in families:
        files = [fs / "label" / f"{h}.{stem}.annot" for h in ("lh", "rh")]
        if not all(f.is_file() for f in files):
            continue
        codes, levels, colors = np.zeros(n_lh + n_rh, np.int32), ["unassigned"], [[0, 0, 0]]
        for f, off, n, side in ((files[0], 0, n_lh, "L"), (files[1], n_lh, n_rh, "R")):
            idx, names, rgb = read_annot(f)
            if len(idx) != n:
                raise ValueError(f"{f.name}: {len(idx)} vertices, the surface's hemisphere has {n}")
            for e in np.unique(idx[idx >= 0]):
                codes[off + np.where(idx == e)[0]] = len(levels)
                levels.append(f"{names[e]} {side}")
                colors.append(rgb[e].tolist())
        out.append((family, codes, levels, np.asarray(colors, np.uint8)))
    return out
