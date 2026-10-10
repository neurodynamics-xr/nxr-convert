"""One subject's atlases, written INTO the subject as ONE composition (D141), following the dataset's default atlas.

The subject side first (entities of the subject):
  sources/<surface>_canonical_frames [V,3,3]   the default's canonical field pushed forward (e1, e2, n)
  <surface>_frame_singularities                 vertices without a frame (indices)
  <surface>_to_default                          csr: each subject vertex ← its default vertex (registered sphere), from the
                                                default subject's cortex (a cross-subject reference, D128)
  sources/<surface>_group_tiles/L<k>            the subject's vertices on the GROUP tree, read with the default subject's
                                                ``cortex_pial tiles L<k>`` (D128); member L<D> is the group-tree partition

then the ATLAS — plain arrays under ``atlas/`` and the rows that say what they are (``atlas.ts`` of the app, D129–D134):
  the two trees' partitions at depth D (``<surface> tiles L<D>`` · ``<surface> group tiles L<D>``) with their ``atlas``
  params (tree · level · finest · gauge · areas); each map's mergeable sums as array-backed measurements on them; per
  recording a closed-form FRAME tiling of its time Line; per (recording, kernel) the SOURCE ESTIMATE — a Field with no
  bytes, ``by_operator`` the kernel, ``derived_from`` the recording, on cortex × time — and one JOINT partition a tree
  (``code = (tile << B) | frame``) whose rows are the band arrays, within the dataset's ``atlas bands``.

Ported from nsp's ``nsp atlas build`` (D135); the dataset's default atlas is the default subject's (D131).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..crud import Subject, jdump, open_dataset, remove_node
from ..db import drain, remove_folder, verify
from ..entities import labelling
from . import joint, rollup, scalars, vectors
from .tower import FRAME_LEVEL, period_s, recording_placement
from .atlas_store import ATLAS_MEASURES, default_subject, definition, put
from .bad_segments import sample_mask
from .default_subject import load_default_template
from .frames import hemisphere_frames, poles_on_sphere
from .ladder import component_mesh, vertex_areas
from .nxr_store import SubjectStore
from .trees import Template, Tree, hemisphere_runs, subject_tree

BY = "nxr-convert atlas"


def make_tree(store: SubjectStore, kind: str, template: Template | None = None, depth: int | None = None) -> Tree:
    structures, names = store.labels("Structures")
    if kind == "subject":
        return subject_tree(store.positions, store.faces, structures, names, depth)
    if kind == "group":
        if template is None:
            raise ValueError("the group tree needs the dataset's default subject")
        return template.tree_for(store.sphere, structures, names, depth)
    raise ValueError(f"tree must be subject|group, not {kind!r}")


def make_frames(store: SubjectStore, domain: str = "sphere") -> tuple[vectors.Frames, list, dict]:
    structures, names = store.labels("Structures")
    runs = hemisphere_runs(structures, names)
    P, F, S = store.positions, store.faces, store.sphere
    V = len(P)
    e1, e2, nn = (np.full((V, 3), np.nan) for _ in range(3))
    sg = np.zeros(V, dtype=bool)
    meta = {}
    for h, (s0, n) in zip(("lh", "rh"), runs):
        p, f = component_mesh(P, F, s0, n)
        sp = S[s0:s0 + n]
        su = sp / np.linalg.norm(sp, axis=1, keepdims=True)
        N, So = poles_on_sphere(su, np.array([0, 0, 1.0]), np.array([0, 0, -1.0]))
        H = hemisphere_frames(p, f, sp, N, So, domain=domain)
        e1[s0:s0 + n], e2[s0:s0 + n], nn[s0:s0 + n], sg[s0:s0 + n] = H.e1, H.e2, H.normal, H.singular
        meta[h] = {"north": s0 + N, "south": s0 + So, **H.meta}
    return vectors.Frames(e1, e2, nn, sg), runs, meta


def _kernel_selected(kernels: str, method: str, n_orient: int) -> bool:
    if kernels == "all":
        return True
    if kernels == "constrained":
        return n_orient == 1
    if kernels == "free":
        return n_orient == 3
    return method in kernels.split(",")


def _measures(prefix: str, arrays: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    """(measure, array path) for the arrays of a statistics group that ARE measures (explicit ones are not rows)."""
    return [(ATLAS_MEASURES[rel], f"{prefix}/{rel}") for rel in arrays if rel in ATLAS_MEASURES]


def remove_atlas(sub: Subject, surface: str) -> int:
    """A REBUILD replaces: every row a previous build made (subject side, partitions it made, its measurements, the
    estimates, the frame tilings, the bands) and the ``atlas/`` folder."""
    db, sid = sub.db, sub.id
    n = 0
    for t, where in (("field", "path = ? OR (kind = 'source estimate' AND path IS NULL)"),
                     ("operator", "path = ?"),
                     ("selection", "path = ? OR json_extract(params_json, '$.atlas.made') = 1 OR json_extract(params_json, '$.joint') IS NOT NULL "
                                   "OR (name = 'atlas bands' AND path IS NULL)"),
                     ("manifold", "name = 'atlas bands' AND path IS NULL AND 1 = ?")):
        arg = {"field": f"sources/{surface}_canonical_frames", "operator": f"{surface}_to_default", "manifold": 1}.get(t)
        for path in ([f"{surface}_frame_singularities", f"sources/{surface}_group_tiles"] if t == "selection" else [arg]):
            for r in db.all(f"SELECT id FROM {t} WHERE subject_id = ? AND ({where})", sid, path):
                remove_node(db, sub.root, t, r["id"])
                n += 1
    with db.tx():
        db.execute("DELETE FROM selection_measurement WHERE computed_by = ? AND selection_id IN (SELECT id FROM selection WHERE subject_id = ?)", BY, sid)
        for r in db.all("SELECT id, params_json FROM selection WHERE subject_id = ? AND json_extract(params_json, '$.atlas') IS NOT NULL", sid):
            p = {k: v for k, v in __import__("json").loads(r["params_json"]).items() if k != "atlas"}
            db.execute("UPDATE selection SET params_json = ? WHERE id = ?", jdump(p) if p else None, r["id"])
    remove_folder(sub.at("atlas"))
    return n


def _subject_side(sub: Subject, store: SubjectStore, fr: vectors.Frames, fmeta: dict, tpl: Template | None, default: dict | None,
                  frames_domain: str) -> dict:
    """The rows a subject gains: frames, singularities, the map to the default, the group tiles."""
    surf = f"sources/{store.surface}"
    surf_id = store.surface_row["id"]
    db = sub.db
    d_frames = db.one("SELECT id FROM field WHERE subject_id = ? AND name = 'cortex_pial_canonical_frames'", default["id"]) if default else None
    fid = sub.field(name=f"{store.surface}_canonical_frames", path=f"{surf}_canonical_frames", kind="canonical frames",
                    manifold_id=surf_id, value_type="matrix3", data=np.stack([fr.e1, fr.e2, fr.normal], 1).astype(np.float32),
                    description="per vertex rows e1 (north), e2 = n × e1 (west), n (outward); NaN at singular vertices",
                    producer_json=jdump({"method": "the default subject's canonical field (trivial connection at the registered sphere's poles)",
                                         "domain": frames_domain, "default": d_frames["id"] if d_frames else None, "hemispheres": fmeta}))
    sub.selection(name=f"{store.surface}_frame_singularities", path=f"{store.surface}_frame_singularities", type="indices",
                  manifold_id=surf_id, data=np.flatnonzero(fr.singular).astype(np.int32), picked=True,
                  description="the vertices with no canonical frame (poles, ill-conditioned push-forwards)")
    out = {"frames": fid}
    if tpl is not None:
        from scipy.spatial import cKDTree
        structures, names = store.labels("Structures")
        runs = hemisphere_runs(structures, names)
        n_lh_default = len(tpl.sphere["lh"])
        V_default = n_lh_default + len(tpl.sphere["rh"])
        dv = np.empty(store.n_vertices, np.int64)
        for h, (hemi, (s0, n)) in enumerate(zip(("lh", "rh"), runs)):
            u = store.sphere[s0:s0 + n]
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            _, j = cKDTree(tpl.sphere[hemi]).query(u)
            dv[s0:s0 + n] = j + (0 if h == 0 else n_lh_default)
        d_cortex = db.one("SELECT id FROM manifold WHERE subject_id = ? AND name = 'cortex_pial' AND component_of_id IS NULL", default["id"])
        op = f"{store.surface}_to_default"
        out["to_default"] = sub.operator(
            name=op, path=op, kind="registration", from_manifold_id=d_cortex["id"], to_manifold_id=surf_id, layout="csr",
            n_rows=int(store.n_vertices), n_cols=int(V_default), nnz=int(store.n_vertices),
            sparse={"indptr": np.arange(store.n_vertices + 1, dtype=np.int32), "indices": dv.astype(np.int32),
                    "data": np.ones(store.n_vertices, np.float32)},
            description="each subject vertex ← its nearest default-subject vertex on the registered sphere (per hemisphere)")
        gt = f"{surf}_group_tiles"
        set_id = sub.selection(name=f"{store.surface}_group_tiles", path=gt, type="set", manifold_id=surf_id, n_members=tpl.finest + 1,
                               description="this subject's vertices on the dataset's GROUP tree (the default subject's tiles, through the "
                                           "registered sphere); member L<k> codes h·2^k + t")
        for l in range(tpl.finest + 1):
            dd = db.one("SELECT id FROM dictionary WHERE scope = 'subject' AND subject_id = ? AND name = ?", default["id"], f"cortex_pial tiles L{l}")
            if dd is None:
                dd = {"id": sub.dictionary(f"{store.surface} group tiles L{l}", [f"tile {k}" for k in range(2 * 2 ** l)])[0]}
            codes = tpl.tree_for(store.sphere, structures, names, l).codes
            labelling(sub, f"{gt}/L{l}", codes, [], manifold_id=surf_id, dictionary="", dictionary_id=dd["id"], member_of=(set_id, l),
                      name=f"{store.surface} group tiles L{l}", element_rows=False, description=f"L{l} — the group tile of every vertex")
            db.execute("UPDATE selection SET n_members = ? WHERE subject_id = ? AND path = ?", 2 * 2 ** l, sub.id, f"{gt}/L{l}")
        out["group_tiles"] = set_id
    return out


def _tree_partition(sub: Subject, store: SubjectStore, tree: Tree, kind: str, frames_id: str, codes_path: str, areas_path: str) -> dict:
    """The tile partition at depth D on a tree — the existing one by name (the app's ladder level, or the group tiles'
    member), else made from the atlas's own codes array; its ``atlas`` params recorded."""
    db = sub.db
    surf = store.surface_row
    name = f"{store.surface} tiles L{tree.depth}" if kind == "subject" else f"{store.surface} group tiles L{tree.depth}"
    params = {"tree": kind, "level": tree.depth, "finest": tree.finest, "gauge": frames_id, "areas": areas_path}
    row = db.one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND name = ? AND type = 'set'", sub.id, surf["id"], name)
    if row is None:
        comps = db.all("SELECT name FROM manifold WHERE component_of_id = ? ORDER BY component_ordinal", surf["id"])
        per = 2 ** tree.depth
        did, _ = sub.dictionary(name, entries=[{"code": c, "name": f"{comps[c // per]['name'] if c // per < len(comps) else f'component {c // per}'} tile {c % per}",
                                                "ordinal": c, "attributes_json": {"level": tree.depth, "tile": c % per}} for c in range(tree.n_leaves)],
                                description=f"the tiles of level {tree.depth} of the surface ladder, every hemisphere")
        sid = sub.selection(name=name, path=None, type="set", manifold_id=surf["id"], dictionary_id=did, n_elements=store.n_vertices,
                            n_members=tree.n_leaves, array_path=codes_path,
                            description=f"level {tree.depth} of the surface ladder over every hemisphere — {tree.n_leaves} tiles",
                            params_json=jdump({"atlas": {**params, "made": True}}))
        return db.read("selection", sid)
    import json
    p = {**(json.loads(row["params_json"]) if row["params_json"] else {}), "atlas": params}
    with db.tx():
        db.execute("UPDATE selection SET params_json = ? WHERE id = ?", jdump(p), row["id"])
    return db.read("selection", row["id"])


def _measure(sub: Subject, selection_id: str, rows: list[dict]) -> None:
    with sub.db.tx():
        for m in rows:
            sub.db.insert("selection_measurement", {"selection_id": selection_id, "computed_by": BY, **m})


def _bands_partition(sub: Subject, bands: list[tuple[float, float]]) -> str:
    """The ATLAS BANDS — a points manifold of the bands and its partition, coded by the dataset's ``atlas bands``."""
    have = sub.db.one("SELECT id FROM selection WHERE subject_id = ? AND name = 'atlas bands' AND type = 'set'", sub.id)
    if have:
        return have["id"]
    did, _ = sub.dictionary("atlas bands", scope="dataset", description="the atlas's band bank: an octave power partition, code = the band's index",
                            entries=[{"code": k, "name": f"{b[0]:g}–{b[1]:g} Hz", "ordinal": k, "measurement": math.sqrt(b[0] * b[1]),
                                      "measurement_unit": "Hz", "attributes_json": {"low_hz": b[0], "high_hz": b[1]}} for k, b in enumerate(bands)])
    mid = sub.manifold(name="atlas bands", path=None, type="points", n_vertices=len(bands), topology={"kind": "none", "rank": 0},
                       comment="the atlas's frequency bands, in order", producer_json=jdump({"bands_hz": [list(b) for b in bands]}))
    return sub.selection(name="atlas bands", path=None, type="set", manifold_id=mid, dictionary_id=did, elements=np.arange(len(bands)),
                         description="each band its own code")


def build_subject(dataset_dir: str | Path, subject: str, *, trees: tuple[str, ...] = ("subject", "group"), depth: int | None = None,
                  maps: bool = True, meg: bool = True, kernels: str = "all", bands: list[tuple[float, float]] | None = None,
                  frame_level: int | None = None, frames_domain: str = "sphere", lc_levels: tuple[int, ...] | None = None,
                  trajectory: tuple[int, int] | None = None, recordings: int | None = None, chunk_frames: int = 4,
                  replace: bool = False, log=print) -> dict:
    """The subject's atlases, as one composition into its dataset (``<datastore>/<dataset>``). ``chunk_frames``: frames
    projected at a time (memory only — a frame cut by a chunk edge is completed by the next chunk). ``replace``: a previous build's rows
    and arrays are removed first; without it a subject that has an atlas is refused."""
    t0 = time.time()
    ds = open_dataset(dataset_dir)
    try:
        store = SubjectStore(ds, subject)
        default = default_subject(ds)
        d = definition(ds)
        depth = depth if depth is not None else int(d.get("depth", 8))
        # definitions written before the tower convention say frame_level 0 (0.25 s frames): they get the tower's
        fl = d.get("frame_level")
        frame_level = frame_level if frame_level is not None else (int(fl) if fl is not None and fl < 0 else FRAME_LEVEL)
        frame_s = period_s(frame_level)
        bands = bands or [tuple(b) for b in d.get("bands_hz", scalars.octave_bands())]
        lc_levels = tuple(lc_levels if lc_levels is not None else d.get("lc_levels", (4, 6)))
        if trajectory is None:
            tj = d.get("trajectory") or {"space_level": 4, "time_level": 2}
            trajectory = (int(tj["space_level"]), int(tj["time_level"]))
        lc_levels = tuple(sorted(set(l for l in lc_levels if l <= depth) | {trajectory[0]}))
        tpl = load_default_template(ds, default) if (default is not None and "group" in trees) else None
        if "group" in trees and tpl is None:
            raise FileNotFoundError(f"{ds.row['name']}: no default subject (nxr-convert atlas default-subject)")
        sub = Subject.open(ds, store.subject, created_by=BY)
        if sub.db.one("SELECT 1 FROM selection WHERE subject_id = ? AND json_extract(params_json, '$.atlas') IS NOT NULL", sub.id) \
                or sub.at("atlas").exists():
            if not replace:
                raise FileExistsError(f"{store.subject}: already has an atlas — pass replace to rebuild it")
            log(f"{store.subject}: removed {remove_atlas(sub, store.surface)} rows of the previous atlas")
        T = {k: make_tree(store, k, tpl, depth) for k in trees}
        P, F = store.positions, store.faces
        area = vertex_areas(P, F)
        log(f"{store.subject}: trees {', '.join(f'{k} (depth {t.depth}, finest {t.finest})' for k, t in T.items())}")
        with sub.composition():
            out = _compose(sub, store, T, P, F, area, tpl, default, frames_domain=frames_domain, maps=maps, meg=meg, kernels=kernels,
                           bands=bands, frame_s=frame_s, frame_level=frame_level, lc_levels=lc_levels, trajectory=trajectory, recordings=recordings,
                           chunk_frames=chunk_frames, log=log)
        # a replaced estimate leaves the group fields that summed it (another subject's root) behind its row: re-sync them
        stale = verify(ds.db, ds.root)
        if stale:
            drain(ds.db, ds.root)
        return {"subject": store.subject, "atlas": str(sub.at("atlas")), "trees": list(T), **out, "seconds": round(time.time() - t0, 1)}
    finally:
        ds.close()


def _compose(sub, store, T, P, F, area, tpl, default, *, frames_domain, maps, meg, kernels, bands, frame_s, frame_level, lc_levels, trajectory,
             recordings, chunk_frames, log) -> dict:
    t1 = time.time()
    fr, runs, fmeta = make_frames(store, frames_domain)
    side = _subject_side(sub, store, fr, fmeta, tpl, default, frames_domain)
    lc = {k: vectors.levi_civita(t, P, F, area, fr, runs, lc_levels) for k, t in T.items()}
    log(f"  frames ({frames_domain}, {int(fr.singular.sum())} singular), subject-side rows, Levi-Civita {list(lc_levels)}: {time.time() - t1:.1f}s")

    parts = {}
    for k, t in T.items():
        base = f"atlas/trees/{k}"
        put(sub, f"{base}/codes", t.codes.astype(np.int32))
        put(sub, f"{base}/w", scalars.leaf_operator(t, area) @ np.ones(len(area)))
        for l in lc_levels:
            put(sub, f"{base}/lc/L{l}/centre", lc[k].centres[l].astype(np.int32))
            put(sub, f"{base}/lc/L{l}/omega", joint.antisymmetric(lc[k].omega[l]))
            put(sub, f"{base}/lc/L{l}/adjacency", joint.tile_adjacency(t, F, l))
        parts[k] = _tree_partition(sub, store, t, k, side["frames"], f"{base}/codes", f"{base}/w")

    catalog = []
    if maps:
        for name, f in store.maps().items():
            x = store.array(f["path"]).astype(np.float64)
            grad = vectors.vertex_gradient(P, F, np.nan_to_num(x), np.nan_to_num(fr.normal))
            for k, t in T.items():
                g = f"atlas/spatial/{k}/maps/{name}"
                stats = scalars.map_stats(t, area, x)
                vstats = vectors.vector_stats(t, area, fr, grad)
                for key, v in stats.items():
                    put(sub, f"{g}/{key}", v)
                for key, v in vstats.items():
                    put(sub, f"{g}/grad/{key}", v)
                for l in lc_levels:
                    z = vectors.lc_resultant(t, area, fr, lc[k], grad, l)
                    put(sub, f"{g}/grad_lc/L{l}/re", z.real)
                    put(sub, f"{g}/grad_lc/L{l}/im", z.imag)
                unit = f["unit"]
                _measure(sub, parts[k]["id"], [{"measure": m, "of_field_id": f["id"], "array_path": p, "unit": unit}
                                               for m, p in _measures(g, stats)]
                         + [{"measure": m, "of_field_id": f["id"], "array_path": p, "unit": f"{unit or 'value'}/m"}
                            for m, p in _measures(g, {f"grad/{key}": v for key, v in vstats.items()})])
            catalog.append({"kind": "spatial", "field": name, "field_kind": f["kind"]})
            log(f"  spatial: {name} (+ gradient)")

    if meg:
        ks = store.kernels()
        sessions = {k.session for k in ks}
        recs = [r for r in store.recordings() if r.session in sessions]
        recs = recs[:recordings] if recordings else recs
        bands_id = _bands_partition(sub, bands) if recs else None
        for rec in recs:
            frames_id = None
            raw = None                      # read ONCE a recording, when a kernel first needs it (a raw can be GBs)
            for k in (k for k in ks if k.session == rec.session):
                n_orient = k.n_vertices // store.n_vertices if k.n_vertices % store.n_vertices == 0 else 0
                if n_orient not in (1, 3) or not _kernel_selected(kernels, k.method, n_orient):
                    continue
                t1 = time.time()
                K = store.array(k.path)
                if raw is None:
                    raw = store.recording_data(rec)
                    good, badinfo = sample_mask(store, rec)
                    pl = recording_placement(store.path, rec.name, rec.sfreq, raw.shape[-1])
                    codes = pl.sample_codes(frame_level)              # the tower tile of every sample (D135, TOWER)
                    code0 = int(codes[0])
                ch = store.kernel_channels(k)
                data = raw if np.array_equal(ch, np.arange(raw.shape[0])) else raw[ch]
                g = f"atlas/time/{rec.name}__{k.method}_{k.stamp}"
                orientation = "constrained" if n_orient == 1 else "free"
                if n_orient == 1:
                    res = scalars.band_power(T, area, K, data, rec.sfreq, bands, chunk_frames=chunk_frames, good=good, codes=codes)
                else:
                    res = vectors.band_tensor(T, area, fr, K, data, rec.sfreq, bands, chunk_frames=chunk_frames, codes=codes,
                                              lc=lc, lc_levels=lc_levels, good=good)
                del data, K
                samples = next(iter(res.values()))["samples"]
                n_frames = int(samples.size)
                put(sub, f"{g}/samples", samples)
                # the FRAME tiling, closed form on the recording's own time Line (one a recording, shared by its kernels): frame k
                # is tower tile frame_code0 + k, so frame 0 starts where that tile does (at or before the first sample)
                if frames_id is None:
                    origin = rec.origin + code0 * frame_s - (pl.start_s or 0.0)
                    frames_id = sub.selection(
                        name=f"{rec.name} frames {frame_s:g} s", path=None, type="spans", manifold_id=rec.time_id, cell="1",
                        of_field_id=rec.id, session=rec.session, n_elements=rec.n_samples, n_members=n_frames,
                        description=f"the atlas's time tiles: {n_frames} cycles of tower level {frame_level} ({frame_s:g} s), "
                                    f"frame k is code {code0} + k, a level-m tile code >> m",
                        params_json=jdump({"tiling": {"origin": origin, "width": frame_s, "hop": frame_s, "count": n_frames},
                                           "end": origin + n_frames * frame_s,
                                           "atlas": {"frame_s": frame_s, "frame_level": frame_level, "frame_code0": code0,
                                                     "anchor": pl.anchor, "start_s": pl.start_s, "made": True}}))
                # cortex × time, and the SOURCE ESTIMATE on it: no bytes — its reductions are the atlas
                cortex = sub.db.read("operator", k.id)["to_manifold_id"] or store.surface_row["id"]
                product = sub.product([cortex, rec.time_id], session=rec.session)
                est = sub.field(name=f"{rec.name} × {k.name}", path=None, kind="source estimate", label=f"{k.method} ({orientation})",
                                session=rec.session, manifold_id=product, value_type="vector3" if n_orient == 3 else "scalar",
                                data_type="float32", by_operator_id=k.id, derived_from_id=rec.id,
                                description="the kernel applied to the recording — stored as its atlas reductions only, never W·x itself",
                                producer_json=jdump({"method": k.method, "orientation": orientation,
                                                     "filterbank": "octave power partition (cos/sin crossover, ½ octave)",
                                                     "bad_segments": {**badinfo, "rule": "samples inside a bad span carry no weight; "
                                                                                         "samples = good samples per frame"}}))
                _measure(sub, frames_id, [{"measure": "count", "of_field_id": est, "array_path": f"{g}/samples"}])
                for tk, r in res.items():
                    tg = f"{g}/{tk}"
                    arrays = {}
                    if n_orient == 1:
                        for key in ("power", "env", "envmax"):
                            put(sub, f"{tg}/{key}", r[key])
                            arrays[key] = r[key]
                        power, psi_src = r["power"], None
                    else:
                        for key, v in r["tensor"].items():
                            put(sub, f"{tg}/tensor/{key}", v)
                            arrays[f"tensor/{key}"] = v
                        for l, terms in r["lc"].items():
                            for key, v in terms.items():
                                put(sub, f"{tg}/tensor_lc/L{l}/{key}", v)
                        power, psi_src = r["tensor"]["total"], r["lc"]
                    sl, tl = trajectory
                    w = scalars.leaf_operator(T[tk], area) @ np.ones(len(area))
                    path_nodes = joint.peak_trajectory(joint.cell_density(power, w, samples, T[tk].depth, sl, tl, code0=code0))
                    om = joint.antisymmetric(lc[tk].omega[sl])
                    trg = f"{tg}/trajectory/L{sl}_T{tl}"
                    put(sub, f"{trg}/node", path_nodes.astype(np.int32))
                    put(sub, f"{trg}/phi", np.stack([joint.transported_angle(pn, om, sl) for pn in path_nodes]))
                    if psi_src is not None:
                        idx = path_nodes[:, None, :]
                        tt = {key: np.take_along_axis(rollup.time(psi_src[sl][key], tl, axis=2, code0=code0), idx, axis=1)[:, 0, :] for key in ("t11", "t22", "t12r")}
                        psi = joint.principal_orientation(tt["t11"], tt["t22"], tt["t12r"])
                        put(sub, f"{trg}/psi", psi)
                        put(sub, f"{trg}/drift", np.stack([joint.orientation_drift(path_nodes[b], psi[b], om, sl) for b in range(len(bands))]))
                    # one JOINT partition a tree: code = (tile << B) | frame; its rows the band arrays [band × tile × frame]
                    part = parts[tk]
                    B = max(1, math.ceil(math.log2(max(2, n_frames))))
                    n_t = T[tk].n_leaves
                    sub.selection(name=f"{rec.name} × {k.name} {part['name']} × {frame_s:g} s", path=None, type="set", session=rec.session,
                                  description=f"the joint cells of {part['name']} × the {n_frames} frames: code = (tile << {B}) | frame",
                                  manifold_id=product, cell="0,1", of_field_id=est, n_members=n_t * n_frames,
                                  params_json=jdump({"joint": {"space": part["id"], "time": frames_id, "bits": B, "tiles": n_t, "frames": n_frames,
                                                               "layout": ["band", "tile", "frame"]}, "atlas": {"tree": tk, "gauge": side["frames"]}}),
                                  measurements=[{"measure": m, "of_field_id": est, "array_path": p, "within_selection_id": bands_id, "within_code": b,
                                                 "computed_by": BY} for m, p in _measures(tg, arrays) for b in range(len(bands))])
                catalog.append({"kind": "spatiotemporal", "recording": rec.name, "kernel": k.name, "orientation": orientation, "frames": n_frames})
                log(f"  time: {rec.name} × {k.method}_{k.stamp} ({orientation}): {time.time() - t1:.0f}s")
    return {"atlases": len(catalog), "catalog": catalog}
