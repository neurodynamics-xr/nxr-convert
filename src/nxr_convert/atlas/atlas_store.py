"""THE ATLAS AS ROWS (D129–D134): there is no atlas format. An atlas is MEASUREMENTS ON TILE PARTITIONS (D68), array-backed
(D96) on plain arrays under the subject's ``atlas/`` folder — the folder holds only arrays, and what they are is the
measurement rows that name them (``array_path``). The default atlas's DEFINITIONS (depth, frame, bands, Levi-Civita
levels, trajectory, the gauge) are the ``atlas`` block of the default subject's tile SET (``params_json``).

    <subject>/atlas/trees/<tree>/{codes, w}                  a tree's leaf codes (the partition's array) and leaf areas
    <subject>/atlas/trees/<tree>/lc/L<k>/{centre,omega,adjacency}   explicit per level (path-dependent: not rows)
    <subject>/atlas/spatial/<tree>/maps/<field>/{w,s1,s2,n,min,max}, grad/{…}, grad_lc/L<k>/{re,im}
    <subject>/atlas/time/<recording>__<kernel>/samples, <tree>/{power,env,envmax | tensor/…, tensor_lc/…, trajectory/…}
    <default subject>/atlas/spatial/group/maps/<kind>/…, per_subject/mean       the group sums (``reduce``)
    <default subject>/atlas/time/<task>__<method>__<orientation>/group/…, per_subject/density

Each array name maps to a MEASURE (``ATLAS_MEASURES``, the app's ``measurements.ts``), whose reduction says how it merges
across members, leaves and frames (``MERGE``).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

DEFAULT_SUBJECT = "@default_subject"
DEFAULT_BANDS = [(1.0 * 2 ** b, 2.0 * 2 ** b) for b in range(6)]

#: an atlas array (relative to its statistics group) → the app's measure (``atlas.ts`` / ``measurements.ts``)
ATLAS_MEASURES = {
    "w": "weight", "s1": "sum", "s2": "sum_square", "n": "count", "min": "min", "max": "max",
    "power": "power_sum", "env": "envelope_sum", "envmax": "envelope_max", "samples": "count",
    "grad/w": "gradient_weight", "grad/c1": "gradient_1", "grad/c2": "gradient_2", "grad/cn": "gradient_n",
    "grad/c11": "gradient_11", "grad/c22": "gradient_22", "grad/c12": "gradient_12", "grad/mag": "gradient_magnitude",
    "tensor/t11": "tensor_11", "tensor/t22": "tensor_22", "tensor/tnn": "tensor_nn", "tensor/t12r": "tensor_12_re",
    "tensor/t12i": "tensor_12_im", "tensor/t1nr": "tensor_1n_re", "tensor/t1ni": "tensor_1n_im", "tensor/t2nr": "tensor_2n_re",
    "tensor/t2ni": "tensor_2n_im", "tensor/total": "tensor_total",
}
#: how a measure merges (its reduction in the app's ``measure`` dictionary): sums add, extremes take the extreme
MERGE = {m: ("max" if m in ("max", "envelope_max") else "min" if m == "min" else "sum") for m in ATLAS_MEASURES.values()}


def merge_of(rel: str) -> str | None:
    """How the array at ``rel`` (relative to its statistics group) merges — None for an explicit array."""
    m = ATLAS_MEASURES.get(rel)
    return MERGE[m] if m else None


def default_subject(ds) -> dict | None:
    """The dataset's DEFAULT SUBJECT (D127): the template named ``@default_subject``, else its template."""
    return (ds.db.one("SELECT * FROM subject WHERE dataset_id = ? AND name = ? AND status = 'complete'", ds.id, DEFAULT_SUBJECT)
            or ds.template())


def definition(ds) -> dict[str, Any]:
    """The dataset's default atlas DEFINITIONS — the ``atlas`` block of the default subject's tile set (``{}`` when none)."""
    d = default_subject(ds)
    if d is None:
        return {}
    row = ds.db.one("SELECT params_json FROM selection WHERE subject_id = ? AND type = 'set' AND member_of_id IS NULL AND "
                    "json_extract(params_json, '$.atlas.default') = 1", d["id"])
    return (json.loads(row["params_json"]) or {}).get("atlas", {}) if row and row["params_json"] else {}


def put(sub, path: str, data, **_ignored) -> None:
    """A plain atlas array at the subject's store path (laid out canonically, no attributes: the rows say what it is)."""
    sub.write_array(path, np.ascontiguousarray(np.asarray(data)))
