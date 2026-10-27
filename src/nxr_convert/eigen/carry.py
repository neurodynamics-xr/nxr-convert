"""A SUBJECT'S MEASUREMENT CARRIED TO ITS TEMPLATE — ``nxr-convert carry``.

Measured on the subject's own cortex (physical units, the subject's lengths), then carried; never the current itself (a flow re-derived on the
template's smaller cortex would be steeper — a size bias):

    a MAP (a scalar a vertex: normal current, divergence, curl, speed, power …)   sampled through the registered spheres, as the transfer matrix
                                                                                   samples (``register.sphere_registration``, barycentric or nearest)
    POINTS (a vertex selection: vortex cores, sources, peaks)                     each to the template vertex nearest it on the registered sphere

Stored on the TEMPLATE, as Brainstorm stores sources projected to the default anatomy: ``<subject> · <name>`` at ``carried/<subject>/<name>``,
``derived_from`` the native map. Nothing is normalised here: a per-subject normalisation is the analysis' choice, made on the carried values.
"""
from __future__ import annotations

import numpy as np
import zarr

from ..crud import jdump
from .register import hemispheres, sphere_registration
from .registered import _surface


def _registration(sub, tpl, subject_surface, template_surface, subject_left, template_left):
    scx, Ps, Fs, Ss = _surface(sub, subject_surface)
    tcx, Pt, Ft, St = _surface(tpl, template_surface)
    return scx, tcx, (Ps, Fs, Ss, hemispheres(Ps, Fs, subject_left)), (Pt, Ft, St, hemispheres(Pt, Ft, template_left))


def carry(sub, tpl, *, field: str | None = None, selection: str | None = None, subject_surface: str | None = None, template_surface: str | None = None,
          subject_left: str = "+y", template_left: str = "-x", sampling: str = "barycentric", replace: bool = False) -> dict:
    """the subject ``sub``'s map ``field`` or point selection ``selection`` (on its surface) carried onto ``tpl``'s surface and stored on ``tpl``"""
    from ..crud import remove_node
    if (field is None) == (selection is None):
        raise ValueError("carry one thing: a map (field) or points (selection)")
    scx, tcx, (Ps, Fs, Ss, hs), (Pt, Ft, St, ht) = _registration(sub, tpl, subject_surface, template_surface, subject_left, template_left)
    Vs, Vt = len(Ps), len(Pt)
    table, name = ("field", field) if field is not None else ("selection", selection)
    row = sub.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND name = ? AND manifold_id = ?", sub.id, name, scx["id"])
    if row is None:
        raise KeyError(f"{sub.row['name']}: no {table} {name!r} on {scx['name']}")
    read = lambda p: np.asarray(zarr.open_array(str(sub.at(p)), mode="r")[...])
    out_name, out_path = f"{sub.row['name']} · {name}", f"carried/{sub.row['name']}/{name}"
    old = tpl.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND path = ?", tpl.id, out_path)
    if old and not replace:
        raise FileExistsError(f"{tpl.row['name']}: {out_path} exists (--replace)")
    if old:
        remove_node(tpl.db, tpl.root, table, old["id"])
    how = {"subject": sub.row["name"], "subject_surface": scx["name"], "template_surface": tcx["name"], "registration": "the registered spheres",
           "sampling": sampling if field is not None else "nearest", "of": row["id"]}
    if field is not None:
        if row["value_type"] != "scalar":
            raise ValueError(f"{name}: a {row['value_type']} field — only a scalar map is carried (measure it as a scalar first: normal current, divergence, curl …)")
        x = read(row["path"]).astype(np.float64)
        if x.size != Vs:
            raise ValueError(f"{name}: {x.size} values, not one a vertex of {scx['name']} ({Vs}) — a static map is carried, not a time series")
        R = sphere_registration(Ss, Fs, hs, np.ones(Vs, bool), np.arange(Vs), St, ht, sampling)        # [Vt × Vs]
        y = R @ x.ravel()
        y[R.getnnz(1) == 0] = np.nan                                                                   # a template vertex no subject vertex reaches
        oid = tpl.field(name=out_name, path=out_path, kind=row["kind"], manifold_id=tcx["id"], data=y.astype(np.float32), unit=row["unit"],
                        derived_from_id=row["id"], description=f"{sub.row['name']}'s {name}, measured on its own cortex, carried to {tpl.row['name']}'s "
                        f"{tcx['name']} through the registered spheres ({sampling})", producer_json=jdump({"producer": "nxr-convert carry", **how}))
        return {"carried": out_name, "id": oid, "kind": "map", "vertices": Vt, "unreached": int(np.isnan(y).sum())}
    idx = read(row["array_path"] or row["path"]).astype(np.int64).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= Vs):
        raise ValueError(f"{name}: indices outside {scx['name']}'s {Vs} vertices")
    # each point to its template vertex: the template's sphere sampled AT the point, nearest (FreeSurfer's vertex correspondence)
    R = sphere_registration(St, Ft, ht, np.ones(Vt, bool), np.arange(Vt), Ss[idx], hs[idx], "nearest")   # [n points × Vt], one 1 a row
    if (R.getnnz(1) != 1).any():
        raise ValueError(f"{name}: {(R.getnnz(1) != 1).sum()} points reach no template vertex")
    to = R.indices.astype(np.int32)
    oid = tpl.selection(name=out_name, path=out_path, type=row["type"], manifold_id=tcx["id"], data=to,
                        description=f"{sub.row['name']}'s {name} (points on its own cortex), each at the {tpl.row['name']} vertex nearest it on the registered sphere",
                        params_json=jdump({"carried": {"producer": "nxr-convert carry", **how}}))      # a selection has no producer column
    return {"carried": out_name, "id": oid, "kind": "points", "points": int(idx.size), "distinct": int(np.unique(to).size)}
