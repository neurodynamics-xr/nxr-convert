"""The dataset's default atlas: the GROUP SUMS of its subjects' group-tree atlases, written into the DEFAULT SUBJECT as one
composition (D130/D131) — arrays under ``<default subject>/atlas/`` and the rows that say what they are.

Every mergeable array adds across subjects cell for cell, because every subject's group tree is the default subject's
tiles. Spatial maps are matched across subjects by their field's KIND (a field's name carries a per-subject stamp, its kind
does not); spatiotemporal atlases by (task, kernel method, orientation), with time collapsed — subjects share no clock, so
the group sum is over each subject's frames (and runs). Per-subject stacks (means / densities per leaf) are kept for
statistics, and the members are the group fields' CONTRIBUTIONS (``field_contribution``).

    <default>/atlas/spatial/group/maps/<kind>/{w,s1,s2,n,min,max}, grad/{…}      Σ over members
    <default>/atlas/spatial/group/maps/<kind>/per_subject/mean  [subjects, leaves]
    <default>/atlas/time/<task>__<method>__<orientation>/group/{power|tensor/…}  [bands, leaves]  Σ over members, runs, frames
    <default>/atlas/time/<task>__<method>__<orientation>/group/samples           Σ samples
    <default>/atlas/time/<task>__<method>__<orientation>/group/per_subject/density [subjects, bands, leaves]

The rows: the default subject's tile partition at depth D carries the ``atlas`` params (tree group · level · gauge) and,
per group field (pathless — its reductions are its only bytes), the array-backed measurements, band arrays ``within`` the
default subject's ``atlas bands``.

ACROSS DATASETS (``sources``): members may also come from OTHER datasets (read only, no lock) — the case of a pipeline that
keeps one dataset per subject. Each source must carry a default subject whose group tree at depth D is the target's, code
for code, and the same atlas definitions; else it is refused. A member of another dataset is named ``<dataset>/<subject>``;
its contribution is recorded in the group field's ``producer_json`` (``sources``), since ``field_contribution`` can only
name subjects of this database.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..crud import Subject, jdump, open_dataset, remove_node
from ..db import remove_folder
from .atlas_store import ATLAS_MEASURES, MERGE, default_subject, definition, put
from .build import BY, _bands_partition, _measure
from .nxr_store import SubjectStore

MERGE_OPS = {"sum": np.add, "max": np.maximum, "min": np.minimum}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def _merge(acc: dict, arrays: dict) -> None:
    for name, data in arrays.items():
        op = MERGE.get(ATLAS_MEASURES.get(name, ""))
        if op is None:
            continue
        acc[name] = data if name not in acc else MERGE_OPS[op](acc[name], data)


def _group_partition(db, sid: str, surface: str) -> dict | None:
    """A member's group-tree partition at its depth (the one carrying ``atlas.tree = group``)."""
    return db.one("SELECT * FROM selection WHERE subject_id = ? AND json_extract(params_json, '$.atlas.tree') = 'group' "
                  "AND json_extract(params_json, '$.joint') IS NULL AND name LIKE '% group tiles L%'", sid)


#: what makes two datasets' default atlases the same (ids such as the frames field's differ between databases)
DEFINING = ("depth", "frame_s", "frame_level", "bands_hz", "lc_levels", "trajectory")


def _group_tree(ds, dsub: dict, D: int) -> tuple[dict, np.ndarray]:
    """The default subject's tile partition at depth D, and its codes (one tile per vertex)."""
    part = ds.db.one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'set'", dsub["id"], f"cortex_pial tiles L{D}")
    if part is None:
        raise ValueError(f"{ds.row['name']}/{dsub['name']}: the group atlas needs 'cortex_pial tiles L{D}' — the default subject's tiles at its depth")
    return part, SubjectStore(ds, dsub).array(part["path"])


def _open_default(dataset, lock: bool):
    ds = open_dataset(dataset, lock=lock)
    dsub = default_subject(ds)
    if dsub is None:
        ds.close()
        raise FileNotFoundError(f"{dataset}: the dataset has no default subject (nxr-convert atlas default-subject)")
    return ds, dsub


def reduce_dataset(dataset: str | Path, sources: tuple = (), log=print) -> dict:
    """Sum the members' group-tree atlases into the default subject (its rows, its ``atlas/`` arrays). A re-run replaces.
    ``sources``: other datasets whose members are summed too (read only; their group tree must be this one's)."""
    ds, dsub = _open_default(dataset, lock=True)
    opened = []
    try:
        d = definition(ds)
        D = int(d.get("depth", 8))
        db = ds.db
        _, codes = _group_tree(ds, dsub, D)
        members, skipped = [], []
        for src in sources:
            sds, sdsub = _open_default(src, lock=False)
            opened.append(sds)
            if any(definition(sds).get(k) != d.get(k) for k in DEFINING):
                raise ValueError(f"{src}: its atlas definitions differ from {ds.row['name']}'s — not the same group atlas")
            if not np.array_equal(_group_tree(sds, sdsub, D)[1], codes):
                raise ValueError(f"{src}: its default subject's group tree differs from {ds.row['name']}'s — tiles would not add")
        for mds in [ds, *opened]:
            mdsub = default_subject(mds)
            for s in mds.db.all("SELECT * FROM subject WHERE dataset_id = ? AND id <> ? AND status = 'complete' ORDER BY name", mds.id, mdsub["id"]):
                name = s["name"] if mds is ds else f"{mds.row['name']}/{s['name']}"
                st = SubjectStore(mds, s)
                part = _group_partition(mds.db, s["id"], st.surface)
                if part is None or not (st.path / "atlas" / "trees" / "group").exists():
                    skipped.append({"subject": name, "reason": "no group-tree atlas (nxr-convert atlas build)"})
                    continue
                members.append(({**s, "name": name}, st, part, mds))
        spatial, spatial_stack = defaultdict(dict), defaultdict(list)
        temporal, temporal_stack, samples_total = defaultdict(dict), defaultdict(list), defaultdict(float)
        sources = defaultdict(list)                                 # group key → the members' fields it sums
        units = {}
        for s, st, part, mds in members:
            mdb = mds.db
            # a member of another dataset: named in the producer (field_contribution names only this database's subjects)
            ref = ({"subject_id": s["id"]} if mds is ds else
                   {"dataset": mds.row["name"], "subject": s["name"].split("/", 1)[1], "subject_id": s["id"]})
            ms = mdb.all("SELECT m.*, f.kind AS fkind, f.unit AS funit FROM selection_measurement m JOIN field f ON f.id = m.of_field_id "
                        "WHERE m.selection_id = ? AND m.array_path IS NOT NULL", part["id"])
            # SPATIAL — by the field's kind
            by_field = defaultdict(list)
            for m in ms:
                by_field[(m["of_field_id"], m["fkind"])].append(m)
            for (fid, fkind), rows in by_field.items():
                kind = _safe(fkind)
                prefix = rows[0]["array_path"].split("/maps/", 1)[0] + "/maps/" + rows[0]["array_path"].split("/maps/", 1)[1].split("/", 1)[0]
                arrs = {r["array_path"][len(prefix) + 1:]: st.array(r["array_path"]) for r in rows}
                _merge(spatial[kind], arrs)
                spatial_stack[kind].append((s["name"], arrs["s1"] / np.where(arrs["w"] > 0, arrs["w"], np.nan)))
                sources[("spatial", kind)].append({**ref, "source_field_id": fid})
                units[kind] = rows[0]["funit"]
            # TIME — every source estimate's joint partition on the group tree
            w = st.array("atlas/trees/group/w")
            per_sub = {}
            for j in mdb.all("SELECT x.*, f.producer_json AS fp, f.id AS fid, r.name AS rec FROM selection x JOIN field f ON f.id = x.of_field_id "
                            "JOIN field r ON r.id = f.derived_from_id WHERE x.subject_id = ? AND json_extract(x.params_json, '$.joint') IS NOT NULL "
                            "AND json_extract(x.params_json, '$.atlas.tree') = 'group'", s["id"]):
                p = json.loads(j["fp"] or "{}")
                task = re.search(r"task-([A-Za-z0-9]+)", j["rec"])
                key = f"{task.group(1) if task else 'rec'}__{p.get('method')}__{p.get('orientation')}"
                paths = {r["array_path"] for r in mdb.all("SELECT DISTINCT array_path FROM selection_measurement WHERE selection_id = ?", j["id"])}
                base = sorted(paths)[0].rsplit("/group/", 1)[0] + "/group"
                arrs = {}
                for path in paths:
                    rel = path[len(base) + 1:]
                    a = st.array(path)
                    arrs[rel] = a.max(-1) if MERGE.get(ATLAS_MEASURES.get(rel, "")) == "max" else a.sum(-1)      # frames collapsed
                total = arrs.get("power", arrs.get("tensor/total"))
                _merge(temporal[key], arrs)
                smp = float(np.asarray(st.array(base.rsplit("/group", 1)[0] + "/samples")).sum())
                samples_total[key] += smp
                prev = per_sub.get(key)
                per_sub[key] = (total if prev is None else prev[0] + total, smp if prev is None else prev[1] + smp)
                sources[("time", key)].append({**ref, "source_field_id": j["fid"]})
            for key, (tot, smp) in per_sub.items():
                temporal_stack[key].append((s["name"], tot / (np.where(w > 0, w, np.nan)[None, :] * smp)))

        sub = Subject.open(ds, dsub["id"], created_by=BY)
        # a re-run REPLACES: the group fields, their measurements and the arrays
        for r in db.all("SELECT id FROM field WHERE subject_id = ? AND path IS NULL AND producer_json LIKE '%\"atlas\"%'", sub.id):
            remove_node(db, ds.root, "field", r["id"])
        remove_folder(sub.at("atlas"))
        part, _ = _group_tree(ds, dsub, D)
        frames = db.one("SELECT id FROM field WHERE subject_id = ? AND name = 'cortex_pial_canonical_frames'", sub.id)
        bands = [tuple(b) for b in d.get("bands_hz", [])]
        with sub.composition():
            p = {**(json.loads(part["params_json"]) if part["params_json"] else {})}
            p["atlas"] = {**p.get("atlas", {}), "tree": "group", "level": D, "gauge": frames["id"] if frames else None,
                          "members": [m[0]["name"] for m in members]}
            with db.tx():
                db.execute("UPDATE selection SET params_json = ? WHERE id = ?", jdump(p), part["id"])
            bands_id = _bands_partition(sub, bands) if (bands and temporal) else None

            def group_field(name: str, kind: str, unit, producer: dict, contributions: list) -> str:
                foreign = [c for c in contributions if "dataset" in c]
                if foreign:
                    producer = {**producer, "sources": foreign}
                contributions = [c for c in contributions if "dataset" not in c]
                return sub.field(name=name, path=None, kind=kind, manifold_id=part["manifold_id"], data_type="float32", unit=unit,
                                 function="sum", description="the group sum over the dataset's members — its reductions only (D130)",
                                 producer_json=jdump(producer), contributions=contributions)
            for kind, arrs in spatial.items():
                g = f"atlas/spatial/group/maps/{kind}"
                for name, data in arrs.items():
                    put(sub, f"{g}/{name}", data)
                subs, stack = zip(*spatial_stack[kind])
                put(sub, f"{g}/per_subject/mean", np.stack(stack))
                fid = group_field(f"group {kind}", "group map", units.get(kind), {"atlas": "spatial", "key": kind, "subjects": list(subs)},
                                  sources[("spatial", kind)])
                _measure(sub, part["id"], [{"measure": ATLAS_MEASURES[n], "of_field_id": fid, "array_path": f"{g}/{n}",
                                            "unit": units.get(kind) if not n.startswith("grad/") else f"{units.get(kind) or 'value'}/m"}
                                           for n in arrs if n in ATLAS_MEASURES])
            for key, arrs in temporal.items():
                g = f"atlas/time/{key}/group"
                for name, data in arrs.items():
                    put(sub, f"{g}/{name}", data)
                put(sub, f"{g}/samples", np.array([samples_total[key]]))
                subs, stack = zip(*temporal_stack[key])
                put(sub, f"{g}/per_subject/density", np.stack(stack))
                condition, method, orientation = key.split("__")
                fid = group_field(f"group {key}", "source estimate", None,
                                  {"atlas": "time", "condition": condition, "method": method, "orientation": orientation,
                                   "collapsed": "frames and runs summed: subjects share no clock", "subjects": list(subs)},
                                  sources[("time", key)])
                _measure(sub, part["id"], [{"measure": ATLAS_MEASURES[n], "of_field_id": fid, "array_path": f"{g}/{n}",
                                            "within_selection_id": bands_id, "within_code": b}
                                           for n in arrs if n in ATLAS_MEASURES for b in range(len(bands))])
        log(f"dataset atlas: {len(members)} members; spatial {sorted(spatial)}; time {sorted(temporal)}; skipped {skipped}")
        return {"members": len(members), "skipped": skipped, "spatial": sorted(spatial), "time": sorted(temporal)}
    finally:
        for o in opened:
            o.close()
        ds.close()
