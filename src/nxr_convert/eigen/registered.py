"""A SUBJECT'S KERNEL REGISTERED ONTO ITS TEMPLATE, stored — ``nxr-convert registerkernel``.

    fuse        the subject's kernel read into its Hodge–Dirac modes (``fused.fused_weights``)
    transfer    the weights carried to the template's modes by T (``register.transfer_matrix``) — linear, so [T a; T b]
    synthesise  the template's current from them (``fused.synthesis``): sensors straight to current on the template

Stored on the SUBJECT (it is the subject's kernel, as Brainstorm's registration to the default anatomy is):

    <kernel>_to_<template>                      Operator  inverse kernel  [3·V_template × C]: the subject's channels → the template's surface; the
                                                          original's method record kept, ``registered`` naming the template, the transfer and both solves
    <kernel>_to_<template>_hodgeDirac_weights   Operator  hodge-dirac weights  [2K_t × C]: the registered weights, which the Editor's mode view reads
                                                          instead of fusing the registered kernel again (they ARE its fusion: the fusion is a projection)
    <kernel>_hodgeDirac_weights                 Operator  hodge-dirac weights  [2K_s × C]: the subject's own fused weights, replaced

The kernel is used as Brainstorm gives it (a dipole moment a vertex, A·m), as the Editor uses it.

The kernel's rows are on the template's INPUT vertices: a kept vertex the synthesis there; a cut (medial wall) vertex the blend of the 3 nearest bridge
vertices, so the Editor's glue, which carries a wall's rows onto the bridge, gives back the synthesised bridge.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

from ..crud import array_meta, jdump
from .fused import carry_kernel, fused_weights, registered_weights, synthesis
from .register import Registration, load_solve, transfer_matrix, write_transfer


def _surface(sub, name: str | None):
    cx = sub.db.one("SELECT m.*, g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND "
                    + ("m.is_primary = 1" if name is None else "m.type = 'surface' AND m.name = ?"), sub.id, *([] if name is None else [name]))
    if cx is None:
        raise KeyError(f"{sub.row['name']}: no surface {name or '(primary)'}")
    rd = lambda p: np.asarray(zarr.open_array(str(sub.at(p)), mode="r")[...])
    F = rd(sub.db.one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", cx["topology_id"])["path"]).astype(np.int64)
    sph = sub.db.one("SELECT g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.name = ?",
                     sub.id, f"{cx['name']}_sphere")
    if sph is None:
        raise KeyError(f"{sub.row['name']}: {cx['name']} has no registered sphere ({cx['name']}_sphere)")
    return cx, rd(cx["positions_path"]).astype(np.float32).astype(np.float64), F, rd(sph["positions_path"]).astype(np.float64)


def store_solve(sub, k: int):
    """the subject's STORED primary basis (``hodge`` / ``modes`` wrote it) as a solve of its first ``k`` modes, and its glued faces — for a subject whose solve
    folder is gone. The store holds the modes, μ, the kinds, the kept vertices and the glued mesh; the one thing it lacks, the 0-form mass, is rebuilt from
    the glued mesh as the solve took it (``prepared`` → ``lumped_mass``: no eigensolve). P is the stored bytes, so the sha is the stored basis'."""
    import hashlib
    from .glue import mesh_key
    from .intrinsic import lumped_mass, prepared
    from .register import Solve
    cx = sub.db.one("SELECT * FROM manifold WHERE subject_id = ? AND is_primary = 1", sub.id)
    rd = lambda p: np.asarray(zarr.open_array(str(sub.at(p)), mode="r")[...])
    g = f"sources/{cx['name']}_glued"
    if not sub.at(g).exists():
        raise ValueError(f"{sub.row['name']}: {cx['name']}'s stored basis is not on a glued surface ({g} absent)")
    mu, kind = rd("eigen/hodgeDirac"), rd("eigen/hodgeDirac_kind")
    if k > mu.size:
        raise ValueError(f"{sub.row['name']}: {k} modes asked, the stored basis has {mu.size}")
    P = np.ascontiguousarray(np.asarray(zarr.open_array(str(sub.at("hodgeDirac_eigenvectors")), mode="r")[:, :k], "<f4").T)
    V, F = rd(f"{g}/vertices").astype(np.float64), rd(f"{g}/faces").astype(np.int64)
    kept = rd("hodgeDirac_support").astype(np.int64)
    keep = np.r_[kept, np.full(len(V) - kept.size, -1, np.int64)]
    return Solve(P, mu[:k].astype(np.float64), kind[:k].astype(int), keep, lumped_mass(prepared(V, F)), V, mesh_key(F),
                 hashlib.sha256(P.tobytes()).hexdigest()), F


def register_kernel(sub, tpl, *, kernel: str, subject_solve: Path, template_solve: Path, modes: int, out: Path, subject_prefix: str | None = None,
                    template_prefix: str | None = None, subject_surface: str | None = None, template_surface: str | None = None,
                    subject_left: str = "+y", template_left: str = "-x", sampling: str = "barycentric", flow: str = "fields", replace: bool = False, progress=None) -> dict:
    """the subject ``sub``'s kernel ``kernel`` registered onto ``tpl`` and stored on ``sub`` (both open ``crud.Subject``s of one dataset)"""
    from ..crud import remove_node
    k = sub.db.one("SELECT * FROM operator WHERE subject_id = ? AND name = ? AND kind = 'inverse kernel'", sub.id, kernel)
    if k is None:
        raise KeyError(f"{sub.row['name']}: no inverse kernel {kernel!r}")
    scx, Ps, Fs, Ss = _surface(sub, subject_surface)
    tcx, Pt, Ft, St = _surface(tpl, template_surface)
    if k["to_manifold_id"] != scx["id"]:
        raise ValueError(f"{kernel} maps onto {k['to_manifold_id']}, not the subject's surface {scx['name']}")
    if str(subject_solve) == "store":                                 # no solve folder: the stored basis, its mass rebuilt
        sol_s, Fs_g = store_solve(sub, modes)
    else:
        sol_s = load_solve(subject_solve, subject_prefix or sub.row["name"], modes)
        Fs_g = np.load(Path(subject_solve) / f"{subject_prefix or sub.row['name']}_glued_F.npy")
    sol_t = load_solve(template_solve, template_prefix or tpl.row["name"], modes)
    reg = Registration(Ps, Fs, Ss, Pt, Ft, St, subject_left=subject_left, template_left=template_left, sampling=sampling)
    T, R = transfer_matrix(sol_s, sol_t, reg)
    tfile = write_transfer(Path(out), T, sol_s, sol_t, subject_name=sub.row["name"], template_name=tpl.row["name"], template_surface=tcx["name"])
    W = np.asarray(zarr.open_array(str(sub.at(k["path"])), mode="r")[...], np.float64)
    Ft_g = np.load(Path(template_solve) / f"{template_prefix or tpl.row['name']}_glued_F.npy")
    w = fused_weights(sol_s, Fs_g, carry_kernel(W, sol_s, Ps), progress=progress)
    if flow not in ("fields", "potentials"):
        raise ValueError(f"flow {flow!r}: 'fields' (each flow mode carried as a unit flow field) or 'potentials'")
    wt = registered_weights(w, T, *((sol_s.mu, sol_t.mu) if flow == "fields" else ()))
    Kt = synthesis(sol_t, Ft_g, wt)
    C = W.shape[1]
    G = Kt.reshape(-1, 3, C)
    on_input = np.zeros((len(Pt), 3, C))                              # the template's INPUT vertices: kept their synthesis, the cut walls the bridge's
    on_input[sol_t.keep[: sol_t.kept]] = G[: sol_t.kept]
    cut = np.setdiff1d(np.arange(len(Pt)), sol_t.keep[: sol_t.kept])
    if len(cut) and len(sol_t.keep) > sol_t.kept:
        d, nn = cKDTree(sol_t.positions[sol_t.kept:]).query(Pt[cut], k=3)
        blend = 1 / np.maximum(d, 1e-9); blend /= blend.sum(1, keepdims=True)
        on_input[cut] = np.einsum("ij,ijkc->ikc", blend, G[sol_t.kept:][nn])
    name = f"{kernel}_to_{tpl.row['name'].lstrip('@')}"
    wname, sname = f"{name}_hodgeDirac_weights", f"{kernel}_hodgeDirac_weights"
    for n in (wname, name, sname):
        old = sub.db.one("SELECT * FROM operator WHERE subject_id = ? AND name = ?", sub.id, n)
        if old and not replace:
            raise FileExistsError(f"{sub.row['name']}: {n} exists (--replace)")
        if old:
            remove_node(sub.db, sub.root, "operator", old["id"])
    tax = tpl.db.one("SELECT * FROM manifold WHERE subject_id = ? AND path = ?", tpl.id,
                     "eigen/hodgeDirac" if tcx["is_primary"] else f"eigen/hodgeDirac__{tcx['name']}")
    tbasis = tpl.db.one("SELECT id FROM operator WHERE subject_id = ? AND to_manifold_id IN (SELECT id FROM manifold WHERE derived_from_id = ? AND path LIKE '%_glued') AND kind = 'eigenvectors'",
                        tpl.id, tcx["id"])
    if tax is None:
        raise KeyError(f"{tpl.row['name']}: {tcx['name']} stores no Hodge–Dirac basis (nxr-convert modes / hodge)")
    registered = {"template": tpl.row["name"], "template_surface": tcx["name"], "transfer": tfile.name, "modes": modes, "sampling": sampling, "flow": flow,
                  "from_package": sol_s.sha, "to_package": sol_t.sha, "subject_mesh": sol_s.mesh, "template_mesh": sol_t.mesh, "of_kernel": k["id"]}
    params = json.loads(k["params_json"] or "{}")
    kid = sub.operator(name=name, path=name, kind="inverse kernel", from_manifold_id=k["from_manifold_id"], from_selection_id=k["from_selection_id"],
                       to_manifold_id=tcx["id"], from_value_type=k["from_value_type"], to_value_type="vector3", layout="dense",
                       n_rows=3 * len(Pt), n_cols=C, data=on_input.reshape(-1, C).astype(np.float32), meta=array_meta(np.float32, (3 * len(Pt), C), compress=False),
                       session=k["session"], params_json=jdump({**params, "registered": registered}),
                       description=f"{kernel} registered onto {tpl.row['name']}'s {tcx['name']}: fused into the Hodge–Dirac modes, carried by the transfer "
                                   f"matrix, synthesised on the template — the subject's channels straight to current on the template",
                       producer_json=jdump({"producer": "nxr-convert registerkernel (nxr_convert.eigen)", **registered}))
    sub.operator(name=wname, path=wname, kind="hodge-dirac weights", from_manifold_id=k["from_manifold_id"], from_selection_id=k["from_selection_id"],
                 to_manifold_id=tax["id"], n_rows=wt.shape[0], n_cols=C, data=wt.astype(np.float32), meta=array_meta(np.float32, wt.shape, compress=False),
                 description=f"{name} in the template's Hodge–Dirac modes: rows 0..K-1 the normal-current weight a, K..2K-1 the flow weight b",
                 params_json=jdump({"kernel": kid, "basis": tbasis["id"] if tbasis else None, "K": int(sol_t.P.shape[0]), "rows": "a (K) then b (K)",
                                    "mu": sol_t.mu.tolist(), "mesh": sol_t.mesh}),
                 producer_json=jdump({"producer": "nxr-convert registerkernel (nxr_convert.eigen)", **registered}))
    sax = sub.db.one("SELECT * FROM manifold WHERE subject_id = ? AND path = ?", sub.id, "eigen/hodgeDirac" if scx["is_primary"] else f"eigen/hodgeDirac__{scx['name']}")
    sbasis = sub.db.one("SELECT id FROM operator WHERE subject_id = ? AND to_manifold_id IN (SELECT id FROM manifold WHERE derived_from_id = ? AND path LIKE '%_glued') AND kind = 'eigenvectors'",
                        sub.id, scx["id"])
    if sax is not None:                                                # the SUBJECT's own fused weights, which its mode view reads
        assert w.shape == (2 * sol_s.P.shape[0], C), w.shape
        sub.operator(name=sname, path=sname, kind="hodge-dirac weights", from_manifold_id=k["from_manifold_id"], from_selection_id=k["from_selection_id"],
                     to_manifold_id=sax["id"], n_rows=w.shape[0], n_cols=C, data=w.astype(np.float32), meta=array_meta(np.float32, w.shape, compress=False),
                     description=f"{kernel} in {sub.row['name']}'s Hodge–Dirac modes: rows 0..K-1 the normal-current weight a, K..2K-1 the flow weight b",
                     params_json=jdump({"kernel": k["id"], "basis": sbasis["id"] if sbasis else None, "K": int(sol_s.P.shape[0]), "rows": "a (K) then b (K)",
                                        "mu": sol_s.mu.tolist(), "mesh": sol_s.mesh}),
                     producer_json=jdump({"producer": "nxr-convert registerkernel (nxr_convert.eigen)", "package": sol_s.sha, "mesh": sol_s.mesh}))
    return {"kernel": name, "id": kid, "weights": wname, "rows": int(3 * len(Pt)), "channels": int(C), "modes": modes, "transfer": str(tfile),
            "unregistered_rows": int((R.getnnz(1) == 0).sum())}
