"""
The dataset's DEFAULT SUBJECT — the average anatomy every subject of a dataset is registered to.

Like Brainstorm's ``@default_subject``: one per dataset, a TEMPLATE subject (``kind: template``, D127) written as ONE
composition through the dataset's database (D141) — every row pending, every array laid out from its row, one drain.
Ported from nsp (D135):

    <dataset>/@default_subject.nxr.zarr/
      sources/cortex_pial | cortex_white | cortex_inflated            fsaverage5 (DEFAULT: 10 242 per hemisphere,
                                                                       the subjects' resolution), metres, fsaverage space
      sources/cortex_pial_sphere                                        the registered sphere (sphere.reg)
      sources/cortex_pial_atlases/{Structures,Desikan-Killiany,Destrieux}   fsaverage's parcellations (+ _lut)
      sources/cortex_pial_tiles/L0 … L<finest>                          the GROUP TREE: codes h·2^ℓ + t per level, each read
                                                                       with the subject's ``cortex_pial tiles L<k>``; the SET
                                                                       carries the dataset's default atlas DEFINITIONS
      sources/cortex_pial_<res>… , cortex_pial_<res>_to_default         OPTION resolutions (fsaverage6, fsaverage):
                                                                       same vertex ids on the sphere; the csr operator
                                                                       rolls their values up to the default (parent cells)
      sources/cortex_pial_canonical_frames [V,3,3]                     e1 north, e2 = n × e1, n  (``frames``)
      cortex_pial_frame_singularities                                    the poles (+ ill-conditioned vertices)
      sources/cortex_pial_tiles_L<k>_regions                            one point per tile of level k (partition)
      cortex_pial_lc_centres_L<k>, sources/cortex_pial_lc_connection_L<k>   Levi-Civita connection between tile centres
      sources/hcp1065, hcp1065_ends, sources/hcp1065_connectome_L<k> | _<atlas>   HCP-1065 average tractography
      atlas/                                                           the group sums after ``reduce`` (D131)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..crud import Subject, jdump
from ..entities import label_set, labelling
from . import hcp as _hcp
from .atlas_store import DEFAULT_BANDS, DEFAULT_SUBJECT
from .frames import hemisphere_frames, separate_coincident
from .joint import antisymmetric
from .ladder import vertex_areas
from .ladder import surface_ladder
from .trees import HEMIS, Template, Tree, _unit
from .vectors import Frames, levi_civita

SURFACES = ("pial", "white", "inflated")
ANNOTS = {"Desikan-Killiany": "aparc", "Destrieux": "aparc.a2009s"}
# FreeSurfer's MNI305 (fsaverage) → MNI152 affine (FreeSurfer wiki, "MNI305 to MNI152")
MNI305_TO_MNI152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429],
                             [0.0146, 1.0009, -0.0024, 1.5496],
                             [-0.0130, -0.0093, 0.9971, 1.1840],
                             [0.0, 0.0, 0.0, 1.0]])


def _read_hemis(fs_dir: Path, surface: str):
    import nibabel.freesurfer as fs
    out = []
    for h in HEMIS:
        p, f = fs.read_geometry(str(fs_dir / "surf" / f"{h}.{surface}"))
        out.append((p.astype(np.float64), f.astype(np.int64)))
    return out


def _combined(parts):
    """lh then rh as one surface; the Structures codes 1 (L) / 2 (R)."""
    pos = np.vstack([p for p, _ in parts])
    faces = np.vstack([f + (0 if i == 0 else len(parts[0][0])) for i, (_, f) in enumerate(parts)])
    structures = np.r_[np.ones(len(parts[0][0]), np.int32), 2 * np.ones(len(parts[1][0]), np.int32)]
    return pos, faces, structures


def _palette(n: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.integers(40, 230, size=(n, 3)).astype(np.uint8)
    c[0] = 0
    return c


class DefaultSubjectWriter:
    """The default subject's entities, on a ``crud.Subject`` composition."""

    def __init__(self, sub: Subject, log=print):
        self.sub = sub
        self.log = log

    def surface(self, name: str, pos_m: np.ndarray, faces: np.ndarray, sphere: np.ndarray | None, n_lh: int,
                producer: dict, primary: bool = False) -> tuple[np.ndarray, str]:
        from ..winding import canonicalize_winding
        sub = self.sub
        faces, winfo = canonicalize_winding(pos_m, faces)
        node = f"sources/{name}"
        f32 = faces.astype(np.int32)
        sid = sub.manifold(name=name, path=node, type="surface", n_vertices=len(pos_m), n_faces=len(faces), is_primary=int(primary),
                           topology={"kind": "simplicial", "rank": 2, "n_components": 2, "winding": "ccw-outward", "cells": [(2, f"{node}/faces")]},
                           geometry={"form": "stored", "cs_key": "fsaverage", "positions_path": f"{node}/vertices"},
                           source=producer.get("source"), producer_json=jdump({**{k: v for k, v in producer.items() if k != "source"},
                                                                                "flipped": winfo.get("flipped")}),
                           populate=lambda at: (sub.write_array(f"{node}/vertices", pos_m), sub.write_array(f"{node}/faces", f32)))
        for k, (side, s0, n) in enumerate((("left", 0, n_lh), ("right", n_lh, len(pos_m) - n_lh))):
            inside = ((faces >= s0) & (faces < s0 + n)).all(axis=1)
            sub.manifold(name=f"{name}_{k + 1}", path=f"{node}_{k + 1}", type="surface", n_vertices=n, n_faces=int(inside.sum()),
                         label=side, component_of_id=sid, component_ordinal=k, from_index=s0,
                         topology={"kind": "simplicial", "rank": 2, "cells": [(2, f"{node}/faces")]},
                         geometry={"form": "stored", "cs_key": "fsaverage", "positions_path": f"{node}/vertices"})
        if sphere is not None:
            sp = f"{node}_sphere"
            sub.manifold(name=f"{name}_sphere", path=sp, type="surface", n_vertices=len(pos_m), n_faces=len(faces),
                         topology=sub.db.read("manifold", sid)["topology_id"], source="sphere.reg",
                         geometry={"form": "stored", "cs_key": "freesurfer-sphere", "positions_path": f"{sp}/vertices"},
                         populate=lambda at: (sub.write_array(f"{sp}/vertices", sphere), sub.write_array(f"{sp}/faces", f32)))
        return faces, sid

    def parcellations(self, fs_dir: Path, surface_node: str, surface_id: str, n_lh: int, n: int):
        import nibabel.freesurfer as fs
        members = [("Structures", np.r_[np.ones(n_lh, np.int32), 2 * np.ones(n - n_lh, np.int32)],
                    ["unassigned", "Cortex L", "Cortex R"], np.array([[0, 0, 0], [200, 120, 120], [120, 120, 200]], np.uint8))]
        for fam, annot in ANNOTS.items():
            codes = np.zeros(n, np.int32)
            names, colors = ["unassigned"], [[0, 0, 0]]
            for h, side, off in (("lh", "L", 0), ("rh", "R", n_lh)):
                f = fs_dir / "label" / f"{h}.{annot}.annot"
                if not f.is_file():
                    break
                lab, ctab, nm = fs.read_annot(str(f))
                for k, name in enumerate(nm):
                    name = name.decode() if isinstance(name, bytes) else str(name)
                    if name in ("unknown", "Medial_wall", "corpuscallosum") or not np.any(lab == k):
                        continue
                    names.append(f"{name} {side}")
                    colors.append([int(x) for x in ctab[k, :3]])
                    codes[off + np.flatnonzero(lab == k)] = len(names) - 1
            else:
                members.append((fam, codes, names, np.asarray(colors, np.uint8)))
        label_set(self.sub, f"{surface_node}_atlases", members, manifold_id=surface_id,
                  description="fsaverage's cortical parcellations — " + ", ".join(m[0] for m in members))
        return [m[0] for m in members]

    def tiles(self, surface_node: str, surface_id: str, codes_by_level: list[np.ndarray], params: dict | None = None) -> str:
        """THE GROUP TREE as a set: member L<k> a partition named ``<surface> tiles L<k>`` (as the app names a level's
        partition, D67), read with the subject's dictionary of that name (D128 — a subject's group tiles read the same)."""
        sub = self.sub
        surf = surface_node.rsplit("/", 1)[-1]
        set_id = sub.selection(name=f"{surf}_tiles", path=f"{surface_node}_tiles", type="set", manifold_id=surface_id,
                               n_members=len(codes_by_level), params_json=jdump(params) if params else None,
                               description="the GROUP TREE: cortical-flow's dyadic surface ladder on the default cortex; member L<k> "
                                           "codes h·2^k + t (h 0 = left); level k's tile is the finest's >> (finest − k)")
        for l, c in enumerate(codes_by_level):
            n = 2 * 2 ** l
            did, _ = sub.dictionary(f"{surf} tiles L{l}", [f"tile {k}" for k in range(n)], colors=_palette(n, l).tolist(), scope="subject")
            labelling(sub, f"{surface_node}_tiles/L{l}", c, [], manifold_id=surface_id, dictionary="", dictionary_id=did,
                      member_of=(set_id, l), description=f"L{l} — one tile per vertex", name=f"{surf} tiles L{l}")
        return set_id

    def indices(self, node: str, manifold_id: str, values: np.ndarray, description: str) -> str:
        return self.sub.selection(name=node.rsplit("/", 1)[-1], path=node, type="indices", manifold_id=manifold_id,
                                  data=np.asarray(values, np.int32), picked=True, description=description)

    def regions(self, surface_node: str, level: int, partition_id: str) -> str:
        node = f"{surface_node}_tiles_L{level}_regions"
        have = self.sub.find("manifold", node)
        if have:
            return have["id"]
        return self.sub.manifold(name=node.rsplit("/", 1)[-1], path=node, type="points", n_vertices=int(2 * 2 ** level),
                                 topology={"kind": "none", "rank": 0}, partition_id=partition_id,
                                 comment=f"the {2 * 2 ** level} tiles of the group tree at level {level} — element k is tile code k")

    def csr(self, node: str, rows: int, cols: int, row_of: np.ndarray, col_of: np.ndarray, **cols_):
        order = np.lexsort((col_of, row_of))
        indptr = np.zeros(rows + 1, np.int64)
        np.add.at(indptr, row_of[order] + 1, 1)
        indptr = np.cumsum(indptr)
        return self.sub.operator(name=node, path=node, layout="csr", n_rows=int(rows), n_cols=int(cols), nnz=int(len(row_of)),
                                 sparse={"indptr": indptr.astype(np.int32 if indptr[-1] < 2**31 else np.int64),
                                         "indices": col_of[order].astype(np.int32), "data": np.ones(len(row_of), np.float32)}, **cols_)


def build_default_subject(dataset_dir: str | Path, templates_dir: str | Path, *, default: str = "fsaverage5",
                          options: tuple[str, ...] = ("fsaverage6", "fsaverage"),
                          hcp_trk: str | Path | None = None, hcp_bst_dir: str | Path | None = None,
                          gate_mm: float = 5.0, lc_levels: tuple[int, ...] = (4, 6),
                          connectome_levels: tuple[int, ...] = (2, 4, 6, 8), fibre_points: int = 64,
                          depth: int = 8, frame_s: float = 0.25, log=print) -> dict:
    """The dataset's ``@default_subject`` — one composition into the dataset (``<datastore>/<dataset>``)."""
    from ..crud import open_dataset
    T = Path(templates_dir)
    fs0 = T / default
    ds = open_dataset(dataset_dir)
    try:
        sub = Subject.create(ds, DEFAULT_SUBJECT, kind="template", folder=f"{DEFAULT_SUBJECT}.nxr.zarr", source_format="freesurfer",
                             source_path=str(fs0), template_id=None, created_by="nxr-convert atlas default-subject")
        with sub.composition():
            out = _compose(DefaultSubjectWriter(sub, log), fs0, T, default=default, options=options, hcp_trk=hcp_trk, hcp_bst_dir=hcp_bst_dir,
                           gate_mm=gate_mm, lc_levels=lc_levels, connectome_levels=connectome_levels, fibre_points=fibre_points,
                           depth=depth, frame_s=frame_s, log=log)
        return {"store": str(sub.store), "id": sub.id, **out}
    finally:
        ds.close()


def _compose(W: DefaultSubjectWriter, fs0: Path, T: Path, *, default, options, hcp_trk, hcp_bst_dir, gate_mm, lc_levels,
             connectome_levels, fibre_points, depth, frame_s, log) -> dict:
    sub = W.sub
    # ── the default resolution: surfaces, parcellations, the group tree, frames, connection ──
    sph_parts = _read_hemis(fs0, "sphere.reg")
    n_lh = len(sph_parts[0][0])
    sphere, _, structures = _combined(sph_parts)
    for surf in SURFACES:
        pos, faces, _ = _combined(_read_hemis(fs0, surf))
        pos, moved = separate_coincident(pos, faces)
        faces_c, sid = W.surface(f"cortex_{surf}", pos / 1000.0, faces, sphere if surf == "pial" else None, n_lh,
                                 {"source": f"{default} {surf}", "coincident_vertices_moved": moved}, primary=(surf == "pial"))
        if surf == "pial":
            pial_pos, pial_faces, pial_id = pos / 1000.0, faces_c, sid
    atlases = W.parcellations(fs0, "sources/cortex_pial", pial_id, n_lh, len(pial_pos))
    log(f"  surfaces {SURFACES} ({default}, {len(pial_pos)} vertices), parcellations {atlases}")

    # the group tree: the ladder per hemisphere on the default pial
    ladders = []
    for h, (s0, n) in enumerate(((0, n_lh), (n_lh, len(pial_pos) - n_lh))):
        f = pial_faces[(pial_faces >= s0).all(1) & (pial_faces < s0 + n).all(1)] - s0
        ladders.append(surface_ladder(pial_pos[s0:s0 + n], f))
    finest = min(l.finest for l in ladders)
    levels = []
    for l in range(finest + 1):
        c = np.zeros(len(pial_pos), np.int32)
        c[:n_lh] = ladders[0].labels[l]
        c[n_lh:] = 2 ** l + ladders[1].labels[l]
        levels.append(c)
    tree = Tree("group", depth, levels[depth], finest)
    log(f"  group tree: finest level {finest}, depth {depth}")

    # canonical frames, solved on the sphere, pushed to the default pial
    V = len(pial_pos)
    e1, e2, nn = (np.full((V, 3), np.nan) for _ in range(3))
    sing = np.zeros(V, bool)
    fmeta = {}
    for h, (s0, n) in zip(HEMIS, ((0, n_lh), (n_lh, V - n_lh))):
        f = pial_faces[(pial_faces >= s0).all(1) & (pial_faces < s0 + n).all(1)] - s0
        su = _unit(sphere[s0:s0 + n])
        N, S = int(np.argmax(su[:, 2])), int(np.argmin(su[:, 2]))
        H = hemisphere_frames(pial_pos[s0:s0 + n], f, sphere[s0:s0 + n], N, S, domain="sphere")
        e1[s0:s0 + n], e2[s0:s0 + n], nn[s0:s0 + n], sing[s0:s0 + n] = H.e1, H.e2, H.normal, H.singular
        fmeta[h] = {"north": s0 + N, "south": s0 + S, **H.meta}
    fr = Frames(e1, e2, nn, sing)
    frames_id = sub.field(name="cortex_pial_canonical_frames", path="sources/cortex_pial_canonical_frames", kind="canonical frames",
                          manifold_id=pial_id, value_type="matrix3", data=np.stack([e1, e2, nn], 1).astype(np.float32),
                          description="per vertex rows e1 (north), e2 = n × e1 (west), n (outward); NaN at singular vertices",
                          producer_json=jdump({"method": "trivial connection, index +1 at the registered sphere's poles (nxr-compute)",
                                               "gauge": "vector-heat log map from the north pole", "domain": "sphere", "hemispheres": fmeta}))
    # the tiles, carrying the dataset's default atlas DEFINITIONS (D131: never a dataset attribute)
    definitions = {"atlas": {"default": True, "tree": "group", "depth": depth, "resolution": default, "frames": frames_id,
                             "frame_s": frame_s, "frame_level": 0,
                             "time_levels": "tile of level m = frame_s · 2^(frame_level + m) s — a longer timescale is a coarser level",
                             "bands_hz": [list(b) for b in DEFAULT_BANDS], "lc_levels": list(lc_levels),
                             "trajectory": {"space_level": 4, "time_level": 2}}}
    tiles_id = W.tiles("sources/cortex_pial", pial_id, levels, params=definitions)
    level_ids = {r["member_ordinal"]: r["id"] for r in sub.db.all("SELECT id, member_ordinal FROM selection WHERE member_of_id = ?", tiles_id)}
    W.indices("cortex_pial_frame_singularities", pial_id, np.flatnonzero(sing),
              "the vertices with no canonical frame: the poles, and ill-conditioned push-forwards")
    runs = [(0, n_lh), (n_lh, V - n_lh)]
    area = vertex_areas(pial_pos, pial_faces)
    lc = levi_civita(tree, pial_pos, pial_faces, area, fr, runs, lc_levels)
    for l in lc_levels:
        rid = W.regions("sources/cortex_pial", l, level_ids[l])
        W.indices(f"cortex_pial_lc_centres_L{l}", pial_id, lc.centres[l], f"the centre vertex of each tile of level {l} (−1: empty tile)")
        sub.field(name=f"cortex_pial_lc_connection_L{l}", path=f"sources/cortex_pial_lc_connection_L{l}", kind="connection",
                  manifold_id=sub.product([rid, rid], session=None), data=antisymmetric(lc.omega[l]).astype(np.float32), unit="rad",
                  description="Ω[a,b]: the frame angle at centre b of centre a's e1 after vector-heat transport, antisymmetrised; "
                              "NaN across hemispheres", producer_json=jdump({"level": l}))
    log(f"  canonical frames ({int(sing.sum())} singular) and Levi-Civita connection at levels {list(lc_levels)}")

    # ── option resolutions: same vertex ids on the sphere; parents roll values up to the default ──
    for res in options:
        fsr = T / res
        sph_r, _, _ = _combined(_read_hemis(fsr, "sphere.reg"))
        n_lh_r = len(_read_hemis(fsr, "sphere.reg")[0][0])
        from scipy.spatial import cKDTree
        parent = np.empty(len(sph_r), np.int64)
        for h, (s0, n), (r0, rn) in ((0, (0, n_lh), (0, n_lh_r)), (1, (n_lh, V - n_lh), (n_lh_r, len(sph_r) - n_lh_r))):
            u0, ur = _unit(sphere[s0:s0 + n]), _unit(sph_r[r0:r0 + rn])
            if np.abs(ur[:n] - u0).max() > 1e-6:
                raise ValueError(f"{res}: its first {n} vertices per hemisphere are not {default}'s — they do not nest")
            _, p = cKDTree(u0).query(ur)
            p[:n] = np.arange(n)
            parent[r0:r0 + rn] = s0 + p
        for surf in SURFACES:
            pos, faces, _ = _combined(_read_hemis(fsr, surf))
            pos, moved = separate_coincident(pos, faces)
            _, sid_r = W.surface(f"cortex_{surf}_{res}", pos / 1000.0, faces, sph_r if surf == "pial" else None, n_lh_r,
                                 {"source": f"{res} {surf}", "coincident_vertices_moved": moved})
            if surf == "pial":
                pial_r_id = sid_r
        W.csr(f"cortex_pial_{res}_to_default", V, len(sph_r), parent, np.arange(len(sph_r)), kind="restriction",
              from_manifold_id=pial_r_id, to_manifold_id=pial_id,
              description=f"each {res} vertex → its parent {default} vertex (itself for the shared ones): summing over a parent's "
                          f"cell rolls {res} values up to {default}")
        W.tiles(f"sources/cortex_pial_{res}", pial_r_id, [c[parent] for c in levels])
        log(f"  option {res}: {len(sph_r)} vertices, parents → {default}")

    # ── HCP-1065 ─────────────────────────────────────────────────────────────────────────────
    hcp_info = None
    if hcp_trk:
        d = Path(hcp_bst_dir)
        tpl = Template(default, finest, {"lh": _unit(sphere[:n_lh]), "rh": _unit(sphere[n_lh:])},
                       {"lh": levels[finest][:n_lh], "rh": levels[finest][n_lh:] - 2 ** finest})
        h = _hcp.build_hcp(tpl, hcp_trk, d / "tess_fibers_tess_hcp1065.mat", d / "tess_cortex_white_high.mat", gate_mm)
        pts, off = h["points"].astype(np.float64), h["offsets"]
        to305 = np.linalg.inv(MNI305_TO_MNI152)      # MNI152 mm → fsaverage (MNI305) m, resampled to fibre_points by arc length
        pts = (np.c_[pts, np.ones(len(pts))] @ to305.T)[:, :3] / 1000.0
        nF, P = len(off) - 1, fibre_points
        out = np.empty((nF, P, 3), np.float32)
        for i in range(nF):
            s = pts[off[i]:off[i + 1]]
            if len(s) < 2:
                s = np.vstack([s, s])
            t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))]
            q = np.linspace(0, t[-1], P)
            out[i] = np.stack([np.interp(q, t, s[:, k]) for k in range(3)], 1)
        flat = out.reshape(-1, 3)
        fid = sub.manifold(name="hcp1065", path="sources/hcp1065", type="curve", n_vertices=int(nF * P), n_edges=int(nF * (P - 1)),
                           topology={"kind": "path", "rank": 1, "n_components": int(nF)},
                           geometry={"form": "stored", "cs_key": "fsaverage", "positions_path": "sources/hcp1065/points"},
                           comment=f"HCP-1065 population-average tractography, {nF} streamlines resampled to {P} points (fibre-major), "
                                   "MNI152 → fsaverage (MNI305) by FreeSurfer's affine",
                           producer_json=jdump({"points_per_component": int(P), **{k: v for k, v in h["meta"].items() if k != "mni_to_scs"}}),
                           populate=lambda at: sub.write_array("sources/hcp1065/points", flat, compress=True))
        ends = h["ends"]
        vtx = np.where(ends["hemi"] == 1, ends["vertex"] + n_lh, ends["vertex"])        # combined index
        ok = ends["vertex"] >= 0
        fib = np.repeat(np.arange(nF)[:, None], 2, 1)
        col = np.where(np.arange(2)[None, :] == 0, fib * P, fib * P + P - 1)
        W.csr("hcp1065_ends", V, nF * P, vtx[ok], col[ok], kind="fibre ends", from_manifold_id=fid, to_manifold_id=pial_id,
              description=f"each streamline's ends → the default cortex vertex they land on (ends on ICBM152 white ≤ {gate_mm} mm, "
                          "carried through the registered sphere); off-cortex ends omitted")
        for l in connectome_levels:
            if l > finest:
                continue
            rid = W.regions("sources/cortex_pial", l, level_ids[l])
            C = _hcp.connectome(ends["code"], finest, l).toarray().astype(np.float32)
            sub.field(name=f"hcp1065_connectome_L{l}", path=f"sources/hcp1065_connectome_L{l}", kind="connectome",
                      manifold_id=sub.product([rid, rid], session=None), data=C, unit="streamlines",
                      description=f"HCP-1065 streamline counts between group-tree tiles of level {l} (both ends on the cortex); "
                                  "any coarser level is this one rolled up", producer_json=jdump({"measure": "streamline_count", "level": l}))
        hcp_info = {k: h["meta"][k] for k in ("fibres", "cortical_fibres", "end_to_white_median_mm")}
        log(f"  HCP-1065: {hcp_info['cortical_fibres']}/{hcp_info['fibres']} fibres cortex-to-cortex; connectomes at {list(connectome_levels)}")
    return {"vertices": V, "finest": finest, "atlases": atlases, "options": list(options), "singular": int(sing.sum()), "hcp": hcp_info}


def load_default_template(ds, default_row: dict | None = None) -> Template:
    """The group tree (Template) from the dataset's default subject: its sphere and its finest tile codes, by its rows."""
    from .atlas_store import default_subject
    from .nxr_store import SubjectStore
    d = default_row or default_subject(ds)
    if d is None:
        raise FileNotFoundError(f"{ds.row['name']}: no default subject (nxr-convert atlas default-subject)")
    st = SubjectStore(ds, d, surface="cortex_pial")
    sphere = st.sphere
    structures, _ = st.labels("Structures")
    members = ds.db.all("SELECT m.member_ordinal AS l, m.array_path FROM selection m JOIN selection s ON s.id = m.member_of_id "
                        "WHERE s.subject_id = ? AND s.path = 'sources/cortex_pial_tiles' ORDER BY m.member_ordinal", d["id"])
    finest = max(m["l"] for m in members)
    codes = st.array(members[-1]["array_path"])
    lh = structures == 1
    return Template(DEFAULT_SUBJECT, finest, {"lh": _unit(sphere[lh]), "rh": _unit(sphere[~lh])},
                    {"lh": codes[lh], "rh": codes[~lh] - 2 ** finest}, {"store": str(st.path), "subject": d["id"]})
