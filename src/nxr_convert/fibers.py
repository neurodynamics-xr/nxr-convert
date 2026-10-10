"""TRACTOGRAPHY — a Brainstorm fibres surface (``tess_fibers_*.mat``) as a bundle of
curves, the map of their ends onto the cortex, and the connectomes Brainstorm
assigned (register D125, D126).

The 2026-07-20 ruling: fibres are a MANIFOLD (a bundle of 1-manifolds), on which
fields may later live — not a field. The 2026-09-23 ladder sketch: store the rawest
thing, the fibre ENDS on the cortex; a connectome is a binning of those ends through
a labelling. Both are written; the connectome Brainstorm computed is kept too, because
it is what the protocol says was assigned.

    sources/<fibers>             Manifold  curve (path topology) — nF components of nP points
      points                     float32 [nF·nP, 3]  SCS metres, fibre-major (its geometry's positions)
      colors                     uint8   [nF·nP, 3]  (when Brainstorm has them)
    <fibers>_ends                Operator  csr (indptr · indices · data), fibre vertices -> cortex vertices: each
                                 fibre's first and last point to its nearest cortex vertex
                                 (one 1 per END, 2·nF nonzeros) — push any fibre field
                                 through it and it lands on the cortex
    sources/<surface>_<atlas>_regions
                                 Manifold  points — one element per CODE of that atlas's
                                 labelling on the cortex (0 = unassigned), partition = it
    sources/<fibers>_connectome_<atlas>
                                 Field     float32 [(N+1) × (N+1)] on the product regions × regions —
                                 streamline counts; row/column 0 counts fibres with an
                                 unassigned end (Brainstorm's own matrix is the [1:,1:] block)

A fibres file has ``Points`` and no ``Vertices``/``Faces``, which is what routes it
here rather than into ``export_surface`` (where it failed with KeyError 'Vertices').
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .matio import load_mat, struct_rows
from .naming import sanitize_node_name, surface_node_name


def is_fibers_file(path: str | Path) -> bool:
    return Path(path).name.startswith("tess_fibers")


def export_fibers(sub: Subject, fibers_file: str | Path, *, subject: str, cortex_path: str, cortex_file: str | Path,
                  progress: Any = None) -> dict[str, Any]:
    """The fibres, their ends on the cortex at ``cortex_path`` (``sources/<surface>``, already a manifold of the subject,
    converted from ``cortex_file``) and one connectome per atlas Brainstorm assigned."""
    p = Path(fibers_file)
    m = load_mat(p)
    pts = np.asarray(m.get("Points"))
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError(f"{p.name}: Points is {pts.shape}, expected [nFibers, nPoints, 3]")
    n_f, n_p, _ = pts.shape
    name = sanitize_node_name(surface_node_name(p.name))
    node = f"sources/{name}"
    cortex = sub.find("manifold", cortex_path)
    if cortex is None:
        raise ValueError(f"{p.name}: the cortex {cortex_path} is not in the subject — export it first")

    # ---- the curves: a bundle of 1-manifolds (D125) ----
    flat = pts.reshape(n_f * n_p, 3).astype(np.float32)
    cols = m.get("Colors")

    def arrays(at: Path) -> None:
        sub.write_array(f"{node}/points", flat, compress=True)
        if cols is not None and np.size(cols) == pts.size:
            sub.write_array(f"{node}/colors", np.asarray(cols).reshape(n_f * n_p, 3).astype(np.uint8), compress=True)
    fid = sub.manifold(name=name, path=node, type="curve", n_vertices=int(n_f * n_p), n_edges=int(n_f * (n_p - 1)),
                       topology={"kind": "path", "rank": 1, "n_components": int(n_f)},
                       geometry={"form": "stored", "cs_key": "scs", "positions_path": f"{node}/points"},
                       comment=str(m.get("Comment", "")) or None, source=p.name,
                       producer_json=jdump({"points_per_component": int(n_p),
                                            "description": f"{n_f} fibres of {n_p} points each (fibre-major: fibre f is rows f·{n_p}…)"}),
                       populate=arrays)
    if progress:
        progress({"stage": "fibers", "name": name, "n_fibers": int(n_f), "n_points": int(n_p)})

    # ---- the ends on the cortex: nearest vertex to each fibre's first and last point ----
    from scipy.spatial import cKDTree
    import zarr
    verts = np.asarray(zarr.open_array(str(sub.at(f"{cortex_path}/vertices")), mode="r")[...], dtype=np.float64)
    ends = np.concatenate([pts[:, 0, :], pts[:, -1, :]]).astype(np.float64)
    _, nearest = cKDTree(verts).query(ends)
    end_rows = np.concatenate([np.arange(n_f) * n_p, np.arange(n_f) * n_p + (n_p - 1)])   # fibre vertices
    order = np.lexsort((end_rows, nearest))                       # CSR over cortex vertices
    indptr = np.r_[0, np.cumsum(np.bincount(nearest, minlength=len(verts)))]
    op = f"{name}_ends"
    sub.operator(name=op, path=op, kind="fibre ends", from_manifold_id=fid, to_manifold_id=cortex["id"], layout="csr",
                 n_rows=int(len(verts)), n_cols=int(n_f * n_p), nnz=int(2 * n_f),
                 sparse={"indptr": indptr.astype(np.int32 if indptr[-1] < 2**31 else np.int64),
                         "indices": end_rows[order].astype(np.int32 if n_f * n_p < 2**31 else np.int64),
                         "data": np.ones(2 * n_f, dtype=np.float32)},
                 description="each fibre's two ends -> its nearest cortex vertex (1 per end)",
                 source=p.name, producer_json=jdump({"method": "nearest vertex (scipy cKDTree), SCS metres"}))
    if progress:
        progress({"stage": "fibre-ends", "name": op, "nnz": int(2 * n_f)})

    # ---- the connectomes Brainstorm assigned (Scouts(k).Assignment, per atlas) ----
    connectomes = []
    levels_of = {sanitize_node_name(str(e.get("Name", ""))): ["unassigned", *(str(x["Label"]) for x in struct_rows(e.get("Scouts"), "Label"))]
                 for e in struct_rows(load_mat(cortex_file).get("Atlas"), "Name")}
    atlases = sub.find("selection", f"{cortex_path}_atlases")
    families = {r["name"]: r for r in sub.db.all("SELECT * FROM selection WHERE member_of_id = ?", atlases["id"])} if atlases else {}
    for sc in struct_rows(m.get("Scouts"), "ConnectFile"):
        asg = np.asarray(sc.get("Assignment"))
        label = str(sc.get("ConnectFile") or sc.get("Label") or "")
        if asg.size == 0 or asg.ndim != 2 or asg.shape[1] != 2 or asg.shape[0] != n_f:
            continue
        atlas = label[len(subject) + 1:] if label.startswith(subject + "_") else label
        fam = sanitize_node_name(atlas)
        if fam not in families:
            if progress:
                progress({"stage": "skipped", "file": p.name, "reason": f"connectome '{label}': no atlas '{fam}' on {cortex_path}"})
            continue
        member = families[fam]
        # Brainstorm's assignment is a region INDEX (the scout's row, 1-based; 0 = none); the partition's CODES are its
        # dictionary's — the index → the scout's name → its code
        levels = levels_of.get(fam)
        if levels is None:
            continue
        code = {r["name"]: r["code"] for r in sub.db.all("SELECT name, code FROM dictionary_entry WHERE dictionary_id = ?", member["dictionary_id"])}
        code_of = np.asarray([code[nm] for nm in levels], dtype=np.int64)
        a = asg.astype(np.int64)
        if a.max() >= len(levels) or a.min() < 0:
            raise ValueError(f"{p.name}: {label} assigns region {int(a.max())} of a {len(levels) - 1}-region atlas")
        a = code_of[a]
        n_codes = int(code_of.max()) + 1
        C = np.zeros((n_codes, n_codes), dtype=np.float64)   # the tally import_fibers_subject.m makes, code 0 kept
        np.add.at(C, (a[:, 0], a[:, 1]), 1)
        off = a[:, 0] != a[:, 1]
        np.add.at(C, (a[off, 1], a[off, 0]), 1)
        reg = f"{cortex_path}_{fam}_regions"
        have = sub.find("manifold", reg)
        rid = have["id"] if have else sub.manifold(
            name=reg.rsplit("/", 1)[-1], path=reg, type="points", n_vertices=n_codes, topology={"kind": "none", "rank": 0},
            partition_id=member["id"], comment=f"the {fam} regions of {cortex_path.split('/')[-1]} — element k is code k (0 = unassigned)")
        sub.field(name=f"{name}_connectome_{fam}", path=f"sources/{name}_connectome_{fam}", kind="connectome",
                  manifold_id=sub.product([rid, rid], session=None), data=C.astype(np.float32), unit="streamlines",
                  description=f"streamline counts between {fam} regions; row/col 0 = an unassigned end", source=p.name,
                  producer_json=jdump({"measure": "streamline_count", "assignment": label, "fibres": int(n_f),
                                       "assigned_both_ends": int(np.all(a > 0, axis=1).sum())}))
        connectomes.append(fam)
        if progress:
            progress({"stage": "connectome", "atlas": fam, "regions": len(levels) - 1, "assigned": int(np.all(a > 0, axis=1).sum())})
    return {"name": name, "id": fid, "n_fibers": int(n_f), "n_points": int(n_p), "connectomes": connectomes}
