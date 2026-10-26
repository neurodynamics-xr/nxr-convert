"""THE HODGE–DIRAC PENCIL and its EIGENMODES — the Editor's ``hodgeDirac.ts`` (d + δ on 0-, 1-, 2-forms of the intrinsic Delaunay mesh) ported,
and the solve that prepares a subject's basis from the STORE's surface.

    d₀ E × V  (d₀φ)_e = φ(head) − φ(tail) along ``edge_he[e]``     d₁ F × E  ± round the face     ⋆₀ the lumped mass   ⋆₁ the cotan weight   ⋆₂ 1/area
    K = [[0, d₀ᵀ⋆₁, 0], [⋆₁d₀, 0, d₁ᵀ⋆₂], [0, ⋆₂d₁, 0]],  M = diag(⋆₀, ⋆₁, ⋆₂);  rows 0-forms · 1-forms · 2-forms

THE SOLVE splits, as every nonzero pair on a closed surface is exact or coexact (``make-hodge-package.py``): the Laplace–Beltrami ladder
(d₀ᵀ⋆₁d₀, ⋆₀) gives the exact modes ψ (μ = √λ), the 2-form ladder (⋆₂d₁⋆₁⁻¹d₁ᵀ⋆₂, ⋆₂) the coexact χ; both cut at ONE λ so the band is complete,
the zero modes (constants) out. Written as nsp's SPLIT derivative (``hodge._package_split``'s layout), so ``nxr-convert hodge`` stores it unchanged."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .intrinsic import IntrinsicMesh, edge_weight, lumped_mass, triangle_area


def dec(m: IntrinsicMesh):
    """(d₀, d₁, ⋆₀, ⋆₁, ⋆₂) — the TypeScript's ``dec``"""
    nV, nE, nF = m.nV, m.nE, m.nF
    he = m.edge_he
    tail, head = m.F[he], m.F[he - he % 3 + (he % 3 + 1) % 3]
    d0 = sp.csr_matrix((np.concatenate([np.ones(nE), -np.ones(nE)]), (np.tile(np.arange(nE), 2), np.concatenate([head, tail]))), shape=(nE, nV))
    h = np.arange(3 * nF)
    sign = np.where(m.edge_he[m.edge_of[h]] == h, 1.0, -1.0)
    d1 = sp.csr_matrix((sign, (h // 3, m.edge_of[h])), shape=(nF, nE))
    d0.eliminate_zeros(); d1.eliminate_zeros()                               # a self-edge's d₀ row is empty
    L = m.length[m.edge_of]
    star2 = np.array([1 / triangle_area(L[3 * f], L[3 * f + 1], L[3 * f + 2]) for f in range(nF)])
    star1 = np.array([edge_weight(m, e) for e in range(nE)])
    return d0, d1, lumped_mass(m), star1, star2


def pencil(m: IntrinsicMesh) -> tuple[sp.csr_matrix, np.ndarray]:
    """(K, M) of the header — K symmetric [N × N], M the diagonal mass"""
    d0, d1, s0, s1, s2 = dec(m)
    nV, nE, nF = m.nV, m.nE, m.nF
    B1 = sp.diags(s1) @ d0                                                   # ⋆₁d₀ [E × V]
    B2 = sp.diags(s2) @ d1                                                   # ⋆₂d₁ [F × E]
    Z = None
    K = sp.bmat([[Z, B1.T, Z], [B1, Z, B2.T], [Z, B2, Z]], format="csr", dtype=np.float64)
    K.sum_duplicates(); K.sort_indices()
    return K, np.concatenate([s0, s1, s2])


def _ladder(S: sp.spmatrix, Mv: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """the k smallest eigenpairs of (S, diag Mv), M-orthonormal, by shift-invert just below 0; ascending"""
    lam, X = sla.eigsh(S.tocsc(), k=k, M=sp.diags(Mv).tocsc(), sigma=-1e-3, which="LM")
    o = np.argsort(lam)
    return np.maximum(lam[o], 0), X[:, o]


def solve_hodge(m: IntrinsicMesh, n_modes: int, progress=None) -> dict:
    """about ``n_modes`` positive Hodge–Dirac modes (half exact, half coexact by Weyl), cut at one λ: ψ [n0 × V] and χ [n2 × F] M-normalised"""
    d0, d1, s0, s1, s2 = dec(m)
    want = n_modes // 2 + 1
    t = time.time()
    L0 = (d0.T @ sp.diags(s1) @ d0)
    lam0, psi = _ladder(L0, s0, min(want + 1, m.nV - 2))
    B2 = sp.diags(s2) @ d1
    S2 = B2 @ sp.diags(1 / s1) @ B2.T
    lam2, chi = _ladder(S2, s2, min(want + 1, m.nF - 2))
    k0, k2 = lam0 > 1e-6 * lam0.max(), lam2 > 1e-6 * lam2.max()                 # the constants (μ = 0) out
    top = min(lam0[k0].max(), lam2[k2].max())                                 # ONE λ for both ladders: the band is complete
    k0 &= lam0 <= top
    k2 &= lam2 <= top
    if progress:
        progress({"stage": "solved", "exact": int(k0.sum()), "coexact": int(k2.sum()), "lambda_top": float(top), "seconds": round(time.time() - t, 1)})
    return {"exact_lambda": lam0[k0], "exact_modes": psi[:, k0].T, "coexact_lambda": lam2[k2], "coexact_modes": chi[:, k2].T,
            "mass": np.concatenate([s0, s1, s2]), "seconds": time.time() - t}


def write_derivative(out: Path, sub: str, glued_V: np.ndarray, glued_F: np.ndarray, keep: np.ndarray, m: IntrinsicMesh, sol: dict, meta: dict) -> Path:
    """the solve as nsp's SPLIT layout in ``out`` (``hodge.package`` reads it): modes, λ, mass, the glued mesh and the intrinsic faces"""
    out.mkdir(parents=True, exist_ok=True)
    save = lambda n, a: np.save(out / f"{sub}_{n}.npy", a)
    save("hodge_exact_modes", sol["exact_modes"].astype(np.float64))
    save("hodge_coexact_modes", sol["coexact_modes"].astype(np.float64))
    save("hodge_exact_lambda", sol["exact_lambda"])
    save("hodge_coexact_lambda", sol["coexact_lambda"])
    mu = np.sqrt(np.concatenate([sol["exact_lambda"], sol["coexact_lambda"]]))
    save("hodge_mu", np.sort(np.concatenate([-mu, mu])))
    save("hodge_mass", sol["mass"])
    save("glued_keep", keep.astype(np.int32))
    save("glued_V", glued_V)
    save("glued_F", glued_F.astype(np.int32))
    save("idt_F", m.F.reshape(-1, 3).astype(np.int32))
    (out / f"{sub}_hodge.json").write_text(json.dumps({**meta, "hodge": {"nV": m.nV, "nE": m.nE, "nF": m.nF}}, indent=2))
    return out


def modes_input(sub, surface: str | None = None) -> dict:
    """what a solve needs, read from the STORE: the surface's positions (float32, as the Editor receives them), faces and medial wall (its
    Desikan–Killiany atlas), and who it is — so a solve can run where the store is not (``solve_input``; nsp on fir)"""
    import zarr
    q = ("SELECT m.*, g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND "
         + ("m.is_primary = 1" if surface is None else "m.type = 'surface' AND m.name = ?"))
    cx = sub.db.one(q, sub.id, *([] if surface is None else [surface]))
    if cx is None:
        raise KeyError(f"{sub.row['name']}: no {'primary surface' if surface is None else repr(surface)}")
    faces = sub.db.one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", cx["topology_id"])
    P = np.asarray(zarr.open_array(str(sub.at(cx["positions_path"])), mode="r")[...], np.float32).astype(np.float64).reshape(-1, 3)
    F = np.asarray(zarr.open_array(str(sub.at(faces["path"])), mode="r")[...], np.int64).reshape(-1, 3)
    return {"P": P, "F": F, "wall": medial_wall(sub, cx), "subject": sub.row["name"], "surface": {"id": cx["id"], "name": cx["name"]}}


def save_input(inp: dict, path: Path) -> Path:
    np.savez(path, P=inp["P"], F=inp["F"], wall=inp["wall"], meta=json.dumps({"subject": inp["subject"], "surface": inp["surface"]}))
    return path


def load_input(path: Path) -> dict:
    z = np.load(path)
    return {"P": z["P"], "F": z["F"], "wall": z["wall"], **json.loads(str(z["meta"]))}


def solve_input(inp: dict, *, n_modes: int, work: Path, progress=None) -> dict:
    """glue by anatomy (the Editor's glue), prepare the intrinsic Delaunay mesh, solve the pencil; the split derivative written to ``work``"""
    from .. import __version__
    from .glue import glue_mesh, mesh_key
    from .intrinsic import prepared
    t = time.time()
    V, G, keep, bridge = glue_mesh(inp["P"], inp["F"], inp["wall"])
    m = prepared(V, G)
    key = mesh_key(G)
    if progress:
        progress({"stage": "glued", "surface": inp["surface"]["name"], "nV": m.nV, "nE": m.nE, "nF": m.nF, "mesh": key, "bridge": bridge, "seconds": round(time.time() - t, 1)})
    sol = solve_hodge(m, n_modes, progress)
    name = inp["subject"]
    meta = {"producer": f"nxr-convert {__version__} modes", "subject": name, "surface": inp["surface"],
            "glue": {"by": "anatomy: the Desikan-Killiany medial wall cut, a bridge grown", "bridge": bridge, "nV": m.nV,
                     "positions": "float32, as the Editor receives them"}, "mesh": key,
            "solve": {"route": "split: Laplace–Beltrami (d₀ᵀ⋆₁d₀, ⋆₀) and 2-form (⋆₂d₁⋆₁⁻¹d₁ᵀ⋆₂, ⋆₂) ladders, shift-invert eigsh, one λ cut",
                      "requested": n_modes, "exact": int(len(sol["exact_lambda"])), "coexact": int(len(sol["coexact_lambda"])),
                      "seconds": round(sol["seconds"], 1)}}
    write_derivative(Path(work), name, V, G, keep, m, sol, meta)
    return {"folder": str(work), "prefix": name, "mesh": key, "surface": inp["surface"]["name"], **meta["solve"]}


def prepare_hodge(sub, *, n_modes: int, work: Path, surface: str | None = None, progress=None) -> dict:
    """a subject's Hodge–Dirac basis from its STORED surface (the primary, or ``surface``): ``modes_input`` then ``solve_input``"""
    return solve_input(modes_input(sub, surface), n_modes=n_modes, work=work, progress=progress)


def medial_wall(sub, cx: dict) -> np.ndarray:
    """the surface's MEDIAL WALL from its stored Desikan–Killiany atlas: the vertices coded ``unassigned`` (the Editor's ``medialWall``)"""
    import zarr
    dk = sub.db.one("SELECT s.* FROM selection s JOIN selection p ON p.id = s.member_of_id WHERE p.manifold_id = ? AND p.name = ? AND s.name = 'Desikan-Killiany'",
                    cx["id"], f"{cx['name']}_atlases")
    un = dk and sub.db.one("SELECT code FROM dictionary_entry WHERE dictionary_id = ? AND name IN ('unassigned', 'unknown')", dk["dictionary_id"])
    if not dk or not un:
        raise ValueError(f"{sub.row['name']}: {cx['name']} carries no Desikan–Killiany atlas with its medial wall — the hemispheres are joined by anatomy only")
    codes = np.asarray(zarr.open_array(str(sub.at(dk["array_path"])), mode="r")[...]).ravel()
    return codes == un["code"]
