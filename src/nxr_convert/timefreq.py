"""Brainstorm ``timefreq_*.mat`` results — SOURCE POWER by band, per sleep stage — as Fields
(register D55/O30/O31).

A subject's ``timefreq_srcpow_<stage>__*.mat`` (in its ``@raw<condition>`` folder) and the
group's ``timefreq_average_*.mat`` (``data/Group_analysis/@intra``) have one shape:
``TF [nSources × nBands]`` on the cortex the kernel indexes (``SurfaceFile``, the template's
when the subject inherits the default anatomy), ``Freqs`` a list of bands
``[name, 'lo, hi', function]``, and — for an average — a ``History`` naming the averaged
files and ``nAvg``. The rows: the Field on the product of its SURFACE (the template's across subjects) and the BANDS (a
points manifold + its partition, coded by the dataset's ``bands`` dictionary), its STAGE and STATISTIC, its CONTRIBUTIONS.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .matio import load_mat
from .naming import surface_node_name


def _bands(freqs: Any) -> list[dict[str, Any]]:
    out = []
    for row in freqs:
        name, rng, fn = (str(row[0]), str(row[1]), str(row[2]) if len(row) > 2 else "mean")
        lo, hi = [float(x) for x in re.split(r"[,\s]+", rng.strip()) if x]
        out.append({"name": name, "low_hz": lo, "high_hz": hi, "function": fn})
    return out


def _stage(comment: str, stem: str) -> str | None:
    m = re.search(r"power:\s*(\w+)", comment) or re.search(r"srcpow_(\w+?)__", stem)
    return m.group(1) if m else None


def _contributions(history: Any) -> list[str]:
    """The averaged files, as Brainstorm lists them: ``<subject>/<condition>/<file>``."""
    out: list[str] = []
    listing = False
    for row in history or []:
        text = str(row[2]) if len(row) > 2 else ""
        if "List of averaged files" in text:
            listing = True
            continue
        if listing and text.strip().startswith("- "):
            out.append(text.strip()[2:].strip())
        elif listing and text.strip() and not text.strip().startswith("- "):
            listing = False
    return out


def _bands_partition(sub: Subject, bands: list[dict[str, Any]]) -> str:
    """The subject's BANDS: a points manifold (one vertex a band, no node) and its partition, coded by the dataset's
    ``bands`` dictionary (D55)."""
    have = sub.by_name("manifold", "bands")
    if have is not None:
        return have["id"]
    did, _ = sub.dictionary("bands", scope="dataset", entries=[
        {"code": k, "name": b["name"], "ordinal": k, "measurement": (b["low_hz"] + b["high_hz"]) / 2, "measurement_unit": "Hz",
         "function": b["function"], "attributes_json": {"low_hz": b["low_hz"], "high_hz": b["high_hz"]}} for k, b in enumerate(bands)])
    mid = sub.manifold(name="bands", path=None, type="points", n_vertices=len(bands), topology={"kind": "none", "rank": 0},
                       comment="the frequency bands, in order", producer_json=jdump({"bands": bands}))
    sub.selection(name="bands", path=None, type="set", manifold_id=mid, dictionary_id=did, elements=np.arange(len(bands)),
                  description="each band its own code")
    return mid


def export_timefreq(sub: Subject, mat_path: str | Path, *, session: str | None, group: bool = False) -> dict[str, Any] | None:
    """One results-level timefreq file → ``frequency/<stem>`` [nSources × nBands] float32, a Field on its surface × the
    bands (the surface this subject's, else the dataset's template's — D55); a group average's contributions are the
    members' fields. None when the file is not source-level."""
    p = Path(mat_path)
    m = load_mat(p)
    if str(m.get("DataType", "")) != "results":
        return None
    tf = np.asarray(m["TF"])
    if tf.ndim == 3 and tf.shape[1] == 1:
        tf = tf[:, 0, :]
    if tf.ndim != 2:
        return None
    comment = str(m.get("Comment", ""))
    stem = p.stem
    surface = surface_node_name(Path(str(m.get("SurfaceFile", ""))).name) if m.get("SurfaceFile") else None
    bands = _bands(m.get("Freqs", []))
    statistic = "median" if "MEDIAN" in comment.upper() else ("mean" if group else None)
    surf = sub.db.one("SELECT * FROM manifold WHERE subject_id = ? AND type = 'surface' AND name = ? AND component_of_id IS NULL",
                      sub.id, surface) if surface else None
    if surf is None and surface:
        t = sub.ds.template()
        surf = sub.db.one("SELECT * FROM manifold WHERE subject_id = ? AND type = 'surface' AND name = ? AND component_of_id IS NULL",
                          t["id"], surface) if t else None
    if surf is None:
        raise ValueError(f"{p.name}: its surface '{surface}' is neither in this subject nor the dataset's template — "
                         "convert the template first (D55)")
    bands_id = _bands_partition(sub, bands)
    contributions = []
    n_avg = None
    if group:
        names = _contributions(m.get("History"))
        n_avg = int(np.ravel(np.asarray(m.get("nAvg", 0)))[0]) if m.get("nAvg") is not None else len(names)
        for c in names:
            subject_name, _, file = (c.split("/") + ["", ""])[:3]
            s = sub.ds.subject(subject_name)
            if s is None:
                continue                   # a contributor not (yet) in the dataset is not a row — n_avg says how many
            src = sub.db.one("SELECT id FROM field WHERE subject_id = ? AND source = ?", s["id"], file) if file else None
            contributions.append({"subject_id": s["id"], "source_field_id": src["id"] if src else None})
    fid = sub.field(name=stem, path=f"frequency/{stem}", kind="source power", manifold_id=sub.product([surf["id"], bands_id], session=None),
                    data=tf.astype(np.float32), description=comment, source=p.name, session=session, function=statistic,
                    contributions=contributions,
                    producer_json=jdump({"stage": _stage(comment, stem), "relative": stem.endswith("_relative"),
                                         "method": str(m.get("Method", "")), "measure": str(m.get("Measure", "")), "n_avg": n_avg}))
    return {"name": stem, "id": fid, "n_sources": int(tf.shape[0]), "n_bands": len(bands), "sleep_stage": _stage(comment, stem), "group": group}
