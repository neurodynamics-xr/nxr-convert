"""One Brainstorm surface → a ``surface`` MANIFOLD and what describes it, as entities (D141).

    sources/<name>              Manifold  surface (simplicial, rank 2): vertices [V,3] f64 · faces [F,3] i32 — scs metres
    sources/<name>_1, _2        Manifold  its HEMISPHERES, components of it (D30): label left/right, from_index, their faces
                                          counted; arrays are the union's (positions sliced from ``from_index``)
    sources/<name>_sphere       Manifold  the FreeSurfer registration — a second geometry of the SAME topology
    sources/<name>_atlases      Selection a set; each parcellation a member partition ``<set>/<atlas>`` coded by the
                                          app-wide dictionary of that atlas (names, scout colours)

A HEAD surface (``head_mask``, scalp, skull) is anatomy derived from the volume: it goes in ``mri/`` (decision 1).
"""
from __future__ import annotations

import re
from .db import now_utc
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .entities import label_set
from .matio import load_mat, struct_rows
from .mesh_health import check_mesh, require_healthy
from .naming import sanitize_node_name, surface_node_name
from .winding import canonicalize_winding

HEAD_SURFACE = ("head", "scalp", "outer", "inner", "skull")


def folder_for_surface(name: str) -> str:
    """``mri/`` for a head surface (decision 1), ``sources/`` for a cortex."""
    low = name.lower()
    return "mri" if any(w in low for w in HEAD_SURFACE) else "sources"


def side_of(name: str) -> str | None:
    """A hemisphere's LABEL from a Structures level name (``componentLabelFrom``): 'Cortex L' → left, 'Cortex R' → right."""
    if re.search(r"\bL\b|left", name, re.I):
        return "left"
    if re.search(r"\bR\b|right", name, re.I):
        return "right"
    return None


def hemispheres(entries: list[dict], n_v: int) -> list[tuple[str, int, int]] | None:
    """The surface's hemispheres from its Structures parcellation — (label, first vertex, count) in vertex order — when
    exactly two left/right scouts tile the vertices in two contiguous runs; None otherwise (no components are stated)."""
    st = next((a for a in entries if str(a.get("Name", "")).lower() == "structures"), None)
    if st is None:
        return None
    runs = []
    for s in struct_rows(st.get("Scouts"), "Label"):
        side = side_of(str(s.get("Label", "")))
        v = np.unique(np.asarray(s["Vertices"], dtype=np.int64).ravel() - 1)
        if side is None or not v.size:
            continue
        if v[-1] - v[0] + 1 != v.size:
            return None
        runs.append((side, int(v[0]), int(v.size)))
    runs.sort(key=lambda r: r[1])
    if len(runs) != 2 or {r[0] for r in runs} != {"left", "right"} or runs[0][1] != 0 or runs[0][2] != runs[1][1] \
            or runs[1][1] + runs[1][2] != n_v:
        return None
    return runs


def export_surface(sub: Subject, surface_file: str | Path, *, folder: str | None = None, bst_root: str = "",
                   atlases: list[str] | None = None, primary: bool = True) -> dict:
    """The surface, its hemispheres, its sphere and its parcellations. ``primary`` is whether it is the surface the
    kernel indexes: the primary is refused when its mesh is unhealthy, a display surface only warns (mesh_health.py),
    and only the primary is marked ``is_primary``."""
    if Path(surface_file).name.startswith("tess_fibers"):
        raise ValueError(f"{Path(surface_file).name}: fibres are curves, not a surface — `nxr-convert subject` converts them")
    surf = load_mat(surface_file)
    name = surface_node_name(surface_file)
    folder = folder or folder_for_surface(name)
    vertices = np.asarray(surf["Vertices"], dtype=np.float64)
    reg = surf.get("Reg")
    sphere = reg["Sphere"].get("Vertices") if isinstance(reg, dict) and isinstance(reg.get("Sphere"), dict) else None
    source = None
    if bst_root:
        try:
            source = str(Path(surface_file).resolve().relative_to((Path(bst_root) / "anat").resolve()))
        except ValueError:
            source = str(surface_file)
    entries = struct_rows(surf.get("Atlas"), "Name")
    hemis = hemispheres(entries, int(vertices.shape[0])) if folder == "sources" else None
    return add_surface(sub, name=name, folder=folder, vertices=vertices, faces=np.asarray(surf["Faces"], dtype=np.int64) - 1,
                       sphere=sphere, source=source, comment=str(surf["Comment"]) if surf.get("Comment") else None,
                       hemis=hemis, entries=entries, wanted=atlases, primary=primary)


def add_surface(sub: Subject, *, name: str, folder: str, vertices: np.ndarray, faces: np.ndarray, sphere=None, source=None,
                comment=None, hemis=None, entries=None, wanted=None, atlas_members=None, primary: bool = True,
                cs_key: str = "scs", producer: dict | None = None) -> dict:
    """The writer both sources share (a Brainstorm .mat, a FreeSurfer directory): arrays in, nodes out. ``faces`` are
    0-based; ``hemis`` (label, first vertex, count) states the components; the parcellations come from Brainstorm's
    ``entries`` or are ready-made ``atlas_members``; ``producer`` rides in the manifold's provenance."""
    path = f"{folder}/{name}"
    for table in ("manifold", "field", "operator", "selection"):
        if sub.find(table, path) is not None:
            raise ValueError(f"{name}: a {table} already occupies {path} — remove it first")
    faces, winfo = canonicalize_winding(vertices, faces)
    health = check_mesh(vertices, faces)                         # BEFORE anything is written
    require_healthy(health, surface=name, primary=primary)
    faces32 = faces.astype(np.int32)
    has_reg = (sphere is not None and np.size(sphere) > 0 and np.ndim(sphere) == 2 and np.shape(sphere)[0] == vertices.shape[0])
    stamp = now_utc(ms=False)

    def arrays(at: Path) -> None:
        sub.write_array(f"{path}/vertices", vertices)
        sub.write_array(f"{path}/faces", faces32)
    sid = sub.manifold(name=name, path=path, type="surface", n_vertices=int(vertices.shape[0]), n_faces=int(faces.shape[0]),
                       topology={"kind": "simplicial", "rank": 2, "n_components": len(hemis) if hemis else 1,
                                 "winding": "ccw-outward", "cells": [(2, f"{path}/faces")]},
                       geometry={"form": "stored", "cs_key": cs_key, "positions_path": f"{path}/vertices"},
                       is_primary=1 if (primary and folder == "sources" and not _has_primary(sub)) else 0,
                       comment=comment, source=source,
                       producer_json=jdump({"mesh_health": health.as_attrs(), **({"flipped": winfo["flipped"]} if winfo.get("flipped") else {}),
                                            **(producer or {})}),
                       created_utc=stamp, populate=arrays)
    topology_id = sub.db.read("manifold", sid)["topology_id"]

    # THE HEMISPHERES — components of the cortex (D30), each its own simplicial manifold over the union's arrays
    for k, (side, s0, n) in enumerate(hemis or []):
        inside = ((faces >= s0) & (faces < s0 + n)).all(axis=1)
        sub.manifold(name=f"{name}_{k + 1}", path=f"{path}_{k + 1}", type="surface", n_vertices=n, n_faces=int(inside.sum()),
                     topology={"kind": "simplicial", "rank": 2, "cells": [(2, f"{path}/faces")]},
                     geometry={"form": "stored", "cs_key": cs_key, "positions_path": f"{path}/vertices"},
                     label=side, component_of_id=sid, component_ordinal=k, from_index=s0)

    if has_reg:
        # THE SAME MESH ON A SPHERE — a second manifold (different positions are a different metric) of the SAME topology
        sp = f"{path}_sphere"

        def sphere_arrays(at: Path) -> None:
            sub.write_array(f"{sp}/vertices", np.asarray(sphere, dtype=np.float64))
            sub.write_array(f"{sp}/faces", faces32)
        sub.manifold(name=f"{name}_sphere", path=sp, type="surface", n_vertices=int(vertices.shape[0]), n_faces=int(faces.shape[0]),
                     topology=topology_id, geometry={"form": "stored", "cs_key": "freesurfer-sphere", "positions_path": f"{sp}/vertices"},
                     source="Surf.Reg.Sphere" if entries is not None else "sphere.reg", populate=sphere_arrays)

    n_atlas = _atlas_set(sub, path, sid, atlas_members) if atlas_members else \
        _export_atlas_set(sub, path, sid, entries or [], int(vertices.shape[0]), wanted)
    return {"name": name, "id": sid, "folder": folder, "n_vertices": int(vertices.shape[0]), "n_faces": int(faces.shape[0]),
            "atlases": n_atlas, "registration": bool(has_reg), "hemispheres": len(hemis or []),
            "flipped": winfo["flipped"], "mesh_health": health.as_attrs()}


# Brainstorm's names for FreeSurfer's annotations (what its own import calls them)
FS_ATLASES = (("aparc", "Desikan-Killiany"), ("aparc.a2009s", "Destrieux"), ("aparc.DKTatlas", "DKT"))


def export_freesurfer(sub: Subject, fs_dir: str | Path, *, name: str = "cortex_pial", scs_from: str | Path | None = None) -> dict:
    """A FreeSurfer subject's pial pair (``surf/lh|rh.pial``, else ``.pial.T1``) as ONE secondary surface, lh then rh, in
    FreeSurfer's surface RAS (``freesurfer-tkr``) in metres, with its registration spheres and annotations.
    ``scs_from``: a Brainstorm tess .mat of the same vertices; the affine tkr → SCS is fitted to it and stored."""
    from .freesurfer import atlas_members, read_surf, sha256
    fs = Path(fs_dir)
    got = {}
    for h in ("lh", "rh"):
        pial = next((f for f in (fs / "surf" / f"{h}.pial", fs / "surf" / f"{h}.pial.T1") if f.is_file()), None)
        if pial is None:
            raise FileNotFoundError(f"{fs}: no surf/{h}.pial (or {h}.pial.T1)")
        reg = fs / "surf" / f"{h}.sphere.reg"
        if not reg.is_file():
            raise FileNotFoundError(f"{reg}: the registration sphere is missing")
        (v, f), (sv, _) = read_surf(pial), read_surf(reg)
        if len(sv) != len(v):
            raise ValueError(f"{h}: the sphere has {len(sv)} vertices, the pial {len(v)}")
        got[h] = (v, f, sv, pial, reg)
    nl, nr = len(got["lh"][0]), len(got["rh"][0])
    vertices = np.concatenate([got["lh"][0], got["rh"][0]]) / 1000.0
    faces = np.concatenate([got["lh"][1], got["rh"][1] + nl])
    sphere = np.concatenate([got["lh"][2], got["rh"][2]]) / 1000.0
    prod = {"freesurfer": {"dir": str(fs), "surface": "pial", "units": "m", "nV_lh": nl, "nV_rh": nr,
                           "sha256_lh": sha256(got["lh"][3]), "sha256_rh": sha256(got["rh"][3]),
                           "sha256_sphere_lh": sha256(got["lh"][4]), "sha256_sphere_rh": sha256(got["rh"][4])}}
    prod.update({k: v for k, v in prod["freesurfer"].items() if k.startswith(("sha256_lh", "sha256_rh", "nV_"))})
    members = atlas_members(fs, nl, nr, FS_ATLASES)
    if scs_from is not None:
        bst = np.asarray(load_mat(scs_from)["Vertices"], dtype=np.float64)
        if bst.shape != vertices.shape:
            raise ValueError(f"{Path(scs_from).name}: {len(bst)} vertices, the FreeSurfer pair has {len(vertices)}")
        T, *_ = np.linalg.lstsq(np.c_[vertices, np.ones(len(vertices))], bst, rcond=None)
        rms = float(np.sqrt(((np.c_[vertices, np.ones(len(vertices))] @ T - bst) ** 2).sum(1).mean()))
        if rms > 1e-5:
            raise ValueError(f"{Path(scs_from).name}: no affine takes the FreeSurfer pial onto it (rms {rms:.3g} m > 1e-5)")
    info = add_surface(sub, name=name, folder="sources", vertices=vertices, faces=faces, sphere=sphere, source=str(fs),
                       comment=f"FreeSurfer {fs.name} pial", hemis=[("left", 0, nl), ("right", nl, nr)], atlas_members=members,
                       primary=False, cs_key="freesurfer-tkr", producer=prod)
    if scs_from is not None:
        sub.operator(name=f"{name}_scs", path=f"{name}_scs", kind="transition", from_manifold_id=info["id"],
                     from_coordinate_system_id=sub.cs("freesurfer-tkr"), to_coordinate_system_id=sub.cs("scs"),
                     layout="dense", n_rows=3, n_cols=4, data=T.T, description="FreeSurfer tkr metres → SCS metres (3x4 affine, fitted)",
                     producer_json=jdump({"fitted_to": Path(scs_from).name, "rms_m": rms, "convention": "scs = A @ tkr + t"}))
        info["scs_rms_m"] = rms
    return info


def _has_primary(sub: Subject) -> bool:
    return sub.db.value("SELECT count(*) FROM manifold WHERE subject_id = ? AND type = 'surface' AND is_primary = 1", sub.id) > 0


def _export_atlas_set(sub: Subject, surface_path: str, surface_id: str, entries: list[dict], n_v: int,
                      wanted: list[str] | None) -> int:
    """THE PARCELLATIONS ARE ONE SET (O9): each atlas a member partition, its codes the app-wide dictionary's."""
    if not entries:
        return 0
    names_all = [str(a.get("Name", "")) for a in entries]
    if wanted:
        missing = [w for w in wanted if w not in names_all]
        if missing:
            raise ValueError(f"unknown atlas name(s): {missing}")
        idx = [names_all.index(w) for w in wanted]
    else:
        idx = [i for i, a in enumerate(entries) if len(struct_rows(a.get("Scouts"), "Label")) >= 2]
    for i, n in enumerate(names_all):                       # Structures is the L/R partition — always included
        if n.lower() == "structures" and i not in idx:
            idx.append(i)
    if not idx:
        return 0
    families = [sanitize_node_name(names_all[i]) for i in idx]
    members = []
    for row, i in enumerate(idx):
        scouts = struct_rows(entries[i].get("Scouts"), "Label")
        codes = np.zeros(n_v, dtype=np.int32)
        region_names = ["unassigned"]
        colors = [[0, 0, 0]]                                   # `unassigned` is a real label (0), drawn transparent
        for k, s in enumerate(scouts, start=1):
            v = np.asarray(s["Vertices"], dtype=np.int64).ravel() - 1
            codes[v] = k
            region_names.append(str(s["Label"]))
            c = np.asarray(s.get("Color") if s.get("Color") is not None else [0.5, 0.5, 0.5], dtype=float).ravel()[:3]
            colors.append([int(round(float(x) * 255)) for x in c])
        members.append((families[row], codes, region_names, np.asarray(colors, dtype=np.uint8)))
    return _atlas_set(sub, surface_path, surface_id, members)


def _atlas_set(sub: Subject, surface_path: str, surface_id: str, members) -> int:
    """The set ``<surface>_atlases``: members are (family, codes, level names, colours)."""
    label_set(sub, f"{surface_path}_atlases", members, manifold_id=surface_id,
              description="the cortical parcellations — " + ", ".join(m[0] for m in members))
    return len(members)
