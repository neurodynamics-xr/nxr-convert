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
The full modes (0-|1-|2-forms on the glued intrinsic-Delaunay complex) stay in the nsp derivative: the datastore has no row
for that complex. The 1-form part is redundant here (it is dA/μ or δS/μ).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .crud import Subject, jdump, remove_node, uuid7
from .db import now_utc

AXIS = "eigen/hodgeDirac"
PATHS = (("operator", "hodgeDirac_eigenvectors"), ("field", "eigen/hodgeDirac_kind"), ("selection", "hodgeDirac_support"),
         ("manifold", AXIS))


def package(d: str | Path, sub: str) -> dict:
    """Read nsp's ``<sub>_hodge_*`` outputs and derive the vertex-potential package (glued vertex indexing)."""
    d = Path(d)
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
    # ⟨φ₂, ω_v⟩ = Σ over the faces round v of ⋆₂φ₂, ÷ their count (the reference's W)
    inc = sp.csr_matrix((np.ones(3 * nF), (iF.ravel(), np.repeat(np.arange(nF), 3))), shape=(nV, nF))
    deg = np.asarray(inc.sum(1)).ravel()
    W = sp.diags(1 / np.maximum(deg, 1)) @ inc @ sp.diags(M[nV + nE:])
    P = np.where((kind == 0)[:, None], X[:, :nV], (W @ X[:, nV + nE:].T).T)   # [K × nV], ψ/√2 already in the modes
    o = np.argsort(mu, kind="stable")
    return {"mu": mu[o], "kind": kind[o], "potentials": P[o], "keep": keep, "meta": meta, "mixing": mixing,
            "glued_V": ld("glued_V")}


def add_hodge(s: Subject, d: str | Path, *, sub: str | None = None, replace: bool = False, tol_m: float = 1e-6) -> dict:
    """Add (or with ``replace`` re-add) the Hodge–Dirac package from nsp's outputs in ``d`` to the subject's PRIMARY cortex."""
    sub = sub or s.row["name"]
    have = [(t, r) for t, p in PATHS if (r := s.find(t, p))]
    if have and not replace:
        raise FileExistsError(f"{s.row['name']}: already holds a Hodge–Dirac package ({have[0][1]['path']}); pass --replace")
    cx = s.db.one("SELECT m.*, g.positions_path FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                  "WHERE m.subject_id = ? AND m.is_primary = 1", s.id)
    if cx is None:
        raise KeyError(f"{s.row['name']}: no primary surface")
    pk = package(d, sub)
    keep, V = pk["keep"], int(cx["n_vertices"])
    if keep.min() < 0 or keep.max() >= V or np.unique(keep).size != keep.size:
        raise ValueError(f"{sub}: glued_keep does not index the primary cortex's {V} vertices")
    # THE SAME SURFACE: the glued vertices are the primary cortex's, at the same positions
    import zarr
    xyz = np.asarray(zarr.open_array(str(s.store / cx["positions_path"]), mode="r")[:], dtype=np.float64)
    err = float(np.abs(xyz[keep] - pk["glued_V"]).max())
    if err > tol_m:
        raise ValueError(f"{sub}: the solved surface is not {cx['name']} (glued vertices off by {err:.3g} m > {tol_m} m)")
    for t, r in have:
        remove_node(s.db, s.root, t, r["id"])
    K = int(pk["mu"].size)
    E = np.zeros((V, K), np.float32)
    E[keep] = pk["potentials"].T
    m = pk["meta"]
    prov = {"producer": "nsp rheome-hodge-dirac", "subject": m.get("subject"), "dataset": m.get("dataset"), "job": m.get("job"),
            "rheome_commit": m.get("rheome_commit"), "source": m.get("source"), "glue": {"gap_mm": m.get("glue", {}).get("gap_mm"),
            "nV": len(keep)}, "solve": m.get("solve"), "checks": m.get("checks"),
            "hodge_json_sha256": hashlib.sha256((Path(d) / f"{sub}_hodge.json").read_bytes()).hexdigest(), "mixing_max": pk["mixing"]}
    stamp = now_utc(ms=False)
    axis = uuid7()
    s.manifold(id=axis, name="hodgeDirac", path=AXIS, type="points", n_vertices=K, topology={"kind": "none", "rank": 0},
                      geometry={"form": "stored", "cs_key": "eigenvalue:hodgeDirac", "positions_path": AXIS},
                      derived_from_id=cx["id"], operator="hodgeDirac", producer_json=jdump({"variant": "Hodge-Dirac (d + δ, glued)",
                      "fieldType": "potentials", "kept": "μ > 0", **prov}), created_utc=stamp,
                      populate=lambda at: s.write_array(AXIS, pk["mu"].astype(np.float64), table="manifold", id=axis))
    sel = s.selection(name="hodgeDirac_support", path="hodgeDirac_support", type="indices", manifold_id=cx["id"],
                      data=keep.astype(np.int32), picked=True,
                      description="the primary-cortex vertices the glued Hodge–Dirac surface keeps (the medial cut is outside)")
    s.field(name="hodgeDirac_kind", path="eigen/hodgeDirac_kind", kind="form degree", manifold_id=axis, value_type="integer",
            data=pk["kind"], description="per mode: 0 exact (a 0-form potential A) · 2 coexact (a 2-form potential S)")
    op = s.operator(name="hodgeDirac_eigenvectors", path="hodgeDirac_eigenvectors", kind="eigenvectors", from_manifold_id=axis,
                    to_manifold_id=cx["id"], from_value_type="scalar", to_value_type="scalar", layout="dense", n_rows=V, n_cols=K,
                    data=E, compress=True, created_utc=stamp,
                    description="Hodge–Dirac modes as vertex potentials: A (exact) or S (coexact) per column; 1-form (∇A + n×∇S)/μ",
                    params_json=jdump({"potentials": "A = ψ/√2 (kind 0) · S = ⟨χ/√2, ω_v⟩ (kind 2)", "support": sel,
                                       "solved": "K x = μ M x, d + δ on the glued intrinsic-Delaunay complex"}),
                    producer_json=jdump(prov))
    return {"axis": AXIS, "n_modes": K, "n_exact": int((pk["kind"] == 0).sum()), "n_coexact": int((pk["kind"] == 2).sum()),
            "mu": [float(pk["mu"][0]), float(pk["mu"][-1])], "support": int(keep.size), "of": V, "position_err_m": err,
            "replaced": bool(have), "operator_id": op}
