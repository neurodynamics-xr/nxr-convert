"""The HODGE–DIRAC eigenmode package — nsp's ``rheome-hodge-dirac`` solve, carried INSIDE a subject (schema 49, no DDL).

An eigenbasis is stored as every eigenbasis is (D39/D40/D43): an EIGEN AXIS manifold whose positions are the eigenvalues,
the EIGENVECTORS operator from that axis to the cortex, and fields on the axis. The Hodge–Dirac solve is on the GLUED primary
cortex (both hemispheres, one closed surface), so there is ONE axis for the union, no per-hemisphere components:

    eigen/hodgeDirac           Manifold  points [K]           μ > 0 in 1/m (``eigenvalue:hodgeDirac``), ascending
    eigen/hodgeDirac_kind      Field     uint8  [K]           0 exact (from a Laplace–Beltrami mode ψ) · 2 coexact (from a Δ₂ mode χ)
    hodgeDirac_eigenvectors    Operator  float32 [V × K]      axis → the PRIMARY cortex: per mode ONE vertex potential —
                                                              A = ψ/√2 (exact) or S = ⟨χ/√2, ω_v⟩ (coexact); the 1-form is
                                                              (∇A + n × ∇S)/μ. Zero on vertices the glued surface cut.
    hodgeDirac_support         Selection indices              the primary-cortex vertices the glued surface keeps

On a closed surface every eigenpair of d + δ with μ ≠ 0 is exact (ψ, dψ/μ, 0)/√2 or coexact (0, δχ/μ, χ)/√2; the −μ partner
flips the 1-form, so only μ > 0 is kept; the two zero modes (constant 0- and 2-form) are dropped. This is
``editor/fixtures/eigenmodes/…/make-hodge-package.py``'s package read off the FULL modes instead of separate LB and Δ₂ solves.
Each surface carries its OWN basis (``surface=NAME``): the primary under the names above, any other surface under the same
names with ``__<surface>`` after ``hodgeDirac`` (``eigen/hodgeDirac__cortex_pial``, ``hodgeDirac__cortex_pial_eigenvectors`` …).
The operator is chunked ONE MODE a chunk [all vertices × 1] inside one SHARD per 64-mode block, so a selected mode reads
alone and a 300k-vertex basis is ~94 files, not ~7000; it is WRITTEN 64 modes at a time (one shard each) from the
memory-mapped modes. Peak memory for V = 300 000, K = 6000: about 0.8 GB (a 64-mode block of the 2-form modes in float64,
~0.3 GB, its vertex potentials ~0.15 GB, the [V × 64] float32 block ~0.08 GB, with copies) — never V × K (7.2 GB float32).
The full modes (0-|1-|2-forms on the glued intrinsic-Delaunay complex) stay in the nsp derivative: the datastore has no row
for that complex. The 1-form part is redundant here (it is dA/μ or δS/μ).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .crud import Subject, array_meta, jdump, populate_array, remove_node, uuid7
from .db import now_utc

AXIS = "eigen/hodgeDirac"
MODE_BLOCK = 64                                          # modes written together, one shard each; the chunk is one mode


def _paths(stem: str) -> dict:
    """The nodes of one surface's package; ``stem`` is ``hodgeDirac`` (the primary) or ``hodgeDirac__<surface>``."""
    return {"axis": f"eigen/{stem}", "kind": f"eigen/{stem}_kind", "op": f"{stem}_eigenvectors", "support": f"{stem}_support"}


def _W(iF: np.ndarray, M2: np.ndarray, nV: int):
    """⟨φ₂, ω_v⟩ = Σ over the faces round v of ⋆₂φ₂, ÷ their count (the reference's W)."""
    nF = len(iF)
    inc = sp.csr_matrix((np.ones(3 * nF), (iF.ravel(), np.repeat(np.arange(nF), 3))), shape=(nV, nF))
    return sp.diags(1 / np.maximum(np.asarray(inc.sum(1)).ravel(), 1)) @ inc @ sp.diags(M2)


def package(d: str | Path, sub: str, lazy: bool = False) -> dict:
    """Read nsp's ``<sub>_hodge_*`` outputs and derive the vertex-potential package (glued vertex indexing).

    Two layouts, found by which files exist: the OLD one (``_hodge_modes.npy``, every mode a full form [K × (nV+nE+nF)]) and
    nsp's SPLIT K=6000 one (``_hodge_exact_modes.npy`` [n0 × nV] and ``_hodge_coexact_modes.npy`` [n2 × nF], with λ each)."""
    d = Path(d)
    if (d / f"{sub}_hodge_exact_modes.npy").exists():
        return _package_split(d, sub, lazy)
    ld = lambda n: np.load(d / f"{sub}_{n}.npy")
    meta = json.loads((d / f"{sub}_hodge.json").read_text())
    nV, nE, nF = (int(meta["hodge"][k]) for k in ("nV", "nE", "nF"))
    mu, X, M = ld("hodge_mu").ravel(), ld("hodge_modes"), ld("hodge_mass").ravel()
    keep, iF = ld("glued_keep").ravel().astype(np.int64), ld("idt_F").astype(np.int64)
    if X.shape != (mu.size, nV + nE + nF) or M.size != nV + nE + nF or keep.size != nV or iF.shape != (nF, 3):
        raise ValueError(f"{sub}: nsp outputs disagree with their own sizes (modes {X.shape}, mass {M.size}, keep {keep.size}, "
                         f"faces {iF.shape}; nV {nV} nE {nE} nF {nF})")
    pos = mu > 1e-6 * np.abs(mu).max()                      # μ > 0; the ± partner flips the 1-form, the zero modes out
    X, mu = X[pos].astype(np.float64), mu[pos]
    w0 = (X[:, :nV] ** 2 * M[:nV]).sum(1)
    w2 = (X[:, nV + nE:] ** 2 * M[nV + nE:]).sum(1)
    mixing = float((np.minimum(w0, w2) / (w0 + w2)).max())
    if mixing > 1e-3:
        raise ValueError(f"{sub}: a mode is neither exact nor coexact (mixing {mixing:.2e}); a degenerate μ needs a split first")
    kind = np.where(w0 >= w2, 0, 2).astype(np.uint8)
    W = _W(iF, M[nV + nE:], nV)
    P = np.where((kind == 0)[:, None], X[:, :nV], (W @ X[:, nV + nE:].T).T)   # [K × nV], ψ/√2 already in the modes
    o = np.argsort(mu, kind="stable")
    P = P[o]
    return {"mu": mu[o], "kind": kind[o], "potentials": P, "block": lambda k0, k1: P[k0:k1].T, "keep": keep, "meta": meta,
            "mixing": mixing, "glued_V": ld("glued_V")}


def _package_split(d: Path, sub: str, lazy: bool = False) -> dict:
    """The split layout. Modes are memmapped (the files are GBs) and read ``block(k0, k1)`` at a time; ``lazy`` leaves
    ``potentials`` ([K × nV] float32, all of them) unbuilt."""
    ld = lambda n, **k: np.load(d / f"{sub}_{n}.npy", **k)
    meta = json.loads((d / f"{sub}_hodge.json").read_text())
    nV, nE, nF = (int(meta["hodge"][k]) for k in ("nV", "nE", "nF"))
    M, keep, iF = ld("hodge_mass").ravel(), ld("glued_keep").ravel().astype(np.int64), ld("idt_F").astype(np.int64)
    ex, co = ld("hodge_exact_modes", mmap_mode="r"), ld("hodge_coexact_modes", mmap_mode="r")
    lam = [ld("hodge_exact_lambda").ravel(), ld("hodge_coexact_lambda").ravel()]
    if (ex.ndim, co.ndim) != (2, 2) or ex.shape != (lam[0].size, nV) or co.shape != (lam[1].size, nF) or M.size != nV + nE + nF \
            or keep.size != nV or iF.shape != (nF, 3):
        raise ValueError(f"{sub}: nsp outputs disagree with their own sizes (exact {ex.shape}, coexact {co.shape}, lambdas "
                         f"{lam[0].size}/{lam[1].size}, mass {M.size}, keep {keep.size}, faces {iF.shape}; nV {nV} nE {nE} nF {nF})")
    # μ = √λ per positive mode; both ladders merged, ascending; the zero modes (constants) out
    lam_all = np.concatenate(lam)
    mu_all = np.sqrt(np.maximum(lam_all, 0))
    use = np.where(mu_all > 1e-6 * mu_all.max())[0]
    o = use[np.argsort(mu_all[use], kind="stable")]
    mu, kind = mu_all[o], np.where(o < lam[0].size, 0, 2).astype(np.uint8)
    # the merged ladders must be nsp's own μ (its positive ± pairs)
    ref = np.sort(ld("hodge_mu").ravel())
    ref = ref[ref > 1e-6 * np.abs(ref).max()]
    if ref.size != mu.size or not np.allclose(ref, mu, rtol=1e-6, atol=0):
        raise ValueError(f"{sub}: the split ladders' √λ do not agree with hodge_mu ({mu.size} vs {ref.size} positive modes)")
    W = _W(iF, M[nV + nE:], nV)
    srcs = {"exact": (ex, M[:nV], 0), "coexact": (co, M[nV + nE:], lam[0].size)}
    scales: dict = {}

    def block(k0: int, k1: int) -> np.ndarray:
        """The vertex potentials of merged modes k0:k1, [nV × (k1 − k0)] float32 (read from the memmaps, k1 − k0 rows a time)."""
        src, out = o[k0:k1], np.zeros((nV, k1 - k0), np.float32)
        for name, (arr, w, base) in srcs.items():
            sel = np.where((src >= base) & (src < base + arr.shape[0]) if name == "coexact" else (src < lam[0].size))[0]
            if not sel.size:
                continue
            X = np.asarray(arr[src[sel] - base], dtype=np.float64)
            nrm = (X ** 2 * w).sum(1)
            if name not in scales:                          # the normalisation is read off the data, not assumed
                scales[name] = 1 / np.sqrt(2) if abs(nrm[-1] - 1) < 1e-3 else 1.0 if abs(nrm[-1] - 0.5) < 1e-3 else None
                if scales[name] is None:
                    raise ValueError(f"{sub}: {name} modes have Σ·⋆ = {nrm[-1]:.4g}, neither 1 (ψ) nor 0.5 (ψ/√2)")
            if np.abs(nrm * scales[name] ** 2 - 0.5).max() > 1e-3:
                raise ValueError(f"{sub}: {name} modes are not all normalised alike (Σ·⋆ {nrm.min():.4g} … {nrm.max():.4g})")
            Y = X.T if name == "exact" else W @ X.T
            out[:, sel] = Y * scales[name]
        return out
    for k in {0, int(np.argmax(kind == 2))}:                # both normalisations are checked now, not mid-write
        block(k, k + 1)
    pk = {"mu": mu, "kind": kind, "block": block, "keep": keep, "meta": meta, "mixing": 0.0, "glued_V": ld("glued_V"),
          "glued_F": ld("glued_F") if (d / f"{sub}_glued_F.npy").exists() else None}
    if not lazy:
        pk["potentials"] = block(0, mu.size).T
    return pk


def add_hodge(s: Subject, d: str | Path, *, sub: str | None = None, replace: bool = False, tol_m: float = 1e-6,
              surface: str | None = None) -> dict:
    """Add (or with ``replace`` re-add) the Hodge–Dirac package from nsp's outputs in ``d`` to a surface of the subject:
    the PRIMARY cortex (``surface=None``, and any name that is the primary), else the named surface under suffixed names."""
    sub = sub or s.row["name"]
    if surface is None:
        cx = s.db.one("SELECT m.*, g.positions_path FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                      "WHERE m.subject_id = ? AND m.is_primary = 1", s.id)
        if cx is None:
            raise KeyError(f"{s.row['name']}: no primary surface")
    else:
        cx = s.db.one("SELECT m.*, g.positions_path FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                      "WHERE m.subject_id = ? AND m.type = 'surface' AND m.name = ?", s.id, surface)
        if cx is None:
            have_names = [r["name"] for r in s.db.all("SELECT name FROM manifold WHERE subject_id = ? AND type = 'surface' ORDER BY name", s.id)]
            raise KeyError(f"{s.row['name']}: no surface named {surface!r}; it has {', '.join(have_names) or 'none'}")
    stem = "hodgeDirac" if cx["is_primary"] else f"hodgeDirac__{cx['name']}"
    P_ = _paths(stem)
    have = [(t, r) for t, p in (("operator", P_["op"]), ("field", P_["kind"]), ("selection", P_["support"]), ("manifold", P_["axis"]))
            if (r := s.find(t, p))]
    if have and not replace:
        raise FileExistsError(f"{s.row['name']}: {cx['name']} already holds a Hodge–Dirac package ({have[0][1]['path']}); pass --replace")
    pk = package(d, sub, lazy=True)
    keep, V = pk["keep"], int(cx["n_vertices"])
    glued = bool((keep < 0).any())                         # glued BY ANATOMY: a bridge of new vertices, stored as a surface of its own
    kept = keep[keep >= 0]
    if kept.max() >= V or np.unique(kept).size != kept.size or (glued and not (keep[:kept.size] >= 0).all()):
        raise ValueError(f"{sub}: glued_keep does not index {cx['name']}'s {V} vertices (kept first, then the bridge)")
    if glued and pk.get("glued_F") is None:
        raise ValueError(f"{sub}: a glued solve without its glued faces (_glued_F.npy)")
    # the same FILES, when both sides say which they are (a FreeSurfer pair: sha256 of lh/rh)
    src = (pk["meta"].get("source") or {})
    mine = json.loads(cx["producer_json"] or "{}")
    for h in ("sha256_lh", "sha256_rh"):
        if src.get(h) and mine.get(h) and src[h] != mine[h]:
            raise ValueError(f"{sub}: the solve read a different {h[7:]} than {cx['name']} holds ({src[h][:12]} vs {mine[h][:12]})")
    # THE SAME SURFACE: the glued vertices are this surface's, at the same positions
    import zarr
    xyz = np.asarray(zarr.open_array(str(s.store / cx["positions_path"]), mode="r")[:], dtype=np.float64)
    err = float(np.abs(xyz[kept] - pk["glued_V"][:kept.size]).max())
    if err > tol_m:
        raise ValueError(f"{sub}: the solved surface is not {cx['name']} (glued vertices off by {err:.3g} m > {tol_m} m)")
    gpath = f"sources/{cx['name']}_glued"
    have += [(t, r) for t, p in (("selection", f"{gpath}_bridge"), ("manifold", gpath)) if (r := s.find(t, p))]
    aside = None
    for t, r in have:                                     # a replaced basis's arrays are MOVED ASIDE, never deleted
        loc = r.get("path") or r.get("array_path")
        if loc and (s.store / loc).exists():
            aside = aside or (Path(s.root) / f"_replaced-{now_utc(ms=False)[:10]}" / s.row["path"])
            (aside / loc).parent.mkdir(parents=True, exist_ok=True)
            (s.store / loc).rename(aside / loc)
        remove_node(s.db, s.root, t, r["id"])
    K = int(pk["mu"].size)
    m = pk["meta"]
    prov = {"producer": m.get("producer", "nsp rheome-hodge-dirac"), **({"mesh": m["mesh"]} if m.get("mesh") else {}),
            "subject": m.get("subject"), "dataset": m.get("dataset"), "job": m.get("job"),
            "rheome_commit": m.get("rheome_commit"), "source": m.get("source"), "glue": {"gap_mm": m.get("glue", {}).get("gap_mm"),
            "nV": len(keep)}, "solve": m.get("solve"), "checks": m.get("checks"),
            "hodge_json_sha256": hashlib.sha256((Path(d) / f"{sub}_hodge.json").read_bytes()).hexdigest(), "mixing_max": pk["mixing"]}
    stamp = now_utc(ms=False)
    axis = uuid7()
    cs = f"eigenvalue:{stem}"
    where = f"the {cx['name']} vertices"
    s.manifold(id=axis, name=stem, path=P_["axis"], type="points", n_vertices=K, topology={"kind": "none", "rank": 0},
                      geometry={"form": "stored", "cs_key": cs, "positions_path": P_["axis"]},
                      derived_from_id=cx["id"], operator="hodgeDirac", producer_json=jdump({"variant": "Hodge-Dirac (d + δ, glued)",
                      "fieldType": "potentials", "kept": "μ > 0", **prov}), created_utc=stamp,
                      populate=lambda at: s.write_array(P_["axis"], pk["mu"].astype(np.float64), table="manifold", id=axis))
    sel = s.selection(name=f"{stem}_support", path=P_["support"], type="indices", manifold_id=cx["id"],
                      data=kept.astype(np.int32), picked=True,
                      description=f"{where} the glued Hodge–Dirac surface keeps (the medial cut is outside)")
    target, rows = cx["id"], V
    if glued:                                              # THE GLUED SURFACE: the kept vertices (in `support` order) then the bridge's
        GV, GF = pk["glued_V"], np.asarray(pk["glued_F"], np.int32)
        csk = s.db.one("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", cx["geometry_id"])
        target = s.manifold(name=f"{cx['name']}_glued", path=gpath, type="surface", n_vertices=len(GV), n_faces=len(GF), derived_from_id=cx["id"],
                            topology={"kind": "simplicial", "rank": 2, "n_components": 1, "winding": "ccw-outward", "cells": [(2, f"{gpath}/faces")]},
                            geometry={"form": "stored", "cs_key": csk["name"] if csk else None, "positions_path": f"{gpath}/vertices"},
                            comment=f"{cx['name']} glued by anatomy: its medial walls cut (the Desikan–Killiany atlas), a bridge grown between the "
                                        f"rims — vertex i < {kept.size} is {stem}_support[i], the rest the bridge",
                            producer_json=jdump({"glued_from": cx["name"], "mesh": m.get("mesh"), "glue": m.get("glue")}),
                            populate=lambda at: (s.write_array(f"{gpath}/vertices", GV.astype(np.float64)), s.write_array(f"{gpath}/faces", GF)))
        s.selection(name=f"{cx['name']}_glued_bridge", path=f"{gpath}_bridge", type="indices", manifold_id=target,
                    data=np.arange(kept.size, len(GV), dtype=np.int32), picked=True,
                    description="the medial BRIDGE (not cortex): to mask it — no sources, a forward zero there")
        rows = len(GV)
    s.field(name=f"{stem}_kind", path=P_["kind"], kind="form degree", manifold_id=axis, value_type="integer",
            data=pk["kind"], description="per mode: 0 exact (a 0-form potential A) · 2 coexact (a 2-form potential S)")

    def write(at: Path) -> None:                            # MODE_BLOCK modes at a time: [V × K] is never in memory
        a = populate_array(at)
        for k0 in range(0, K, MODE_BLOCK):
            blk = pk["block"](k0, min(K, k0 + MODE_BLOCK))
            if glued:                                      # on the glued surface: every row, in its order
                a[:, k0:k0 + blk.shape[1]] = blk.astype(np.float32)
                continue
            E = np.zeros((V, blk.shape[1]), np.float32)
            E[keep] = blk
            a[:, k0:k0 + blk.shape[1]] = E
    op = s.operator(name=f"{stem}_eigenvectors", path=P_["op"], kind="eigenvectors", from_manifold_id=axis,
                    to_manifold_id=target, from_value_type="scalar", to_value_type="scalar", layout="dense", n_rows=rows, n_cols=K,
                    meta=array_meta(np.float32, (rows, K), chunks=(rows, 1), shards=(rows, min(K, MODE_BLOCK)), compress=False),   # uncompressed: the Editor's ranged reader merges a block into a few requests
                    populate=write, created_utc=stamp,
                    description="Hodge–Dirac modes as vertex potentials: A (exact) or S (coexact) per column; 1-form (∇A + n×∇S)/μ",
                    params_json=jdump({"potentials": "A = ψ/√2 (kind 0) · S = ⟨χ/√2, ω_v⟩ (kind 2)", "support": sel,
                                       "solved": "K x = μ M x, d + δ on the glued intrinsic-Delaunay complex"}),
                    producer_json=jdump(prov))
    return {"axis": P_["axis"], "surface": cx["name"], "glued": glued, "rows": rows, "moved_aside": str(aside) if aside else None, "n_modes": K, "n_exact": int((pk["kind"] == 0).sum()),
            "n_coexact": int((pk["kind"] == 2).sum()), "mu": [float(pk["mu"][0]), float(pk["mu"][-1])], "support": int(kept.size),
            "of": V, "position_err_m": err, "replaced": bool(have), "operator_id": op}
