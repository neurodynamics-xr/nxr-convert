"""The TEMPLATES-AND-REFERENCES datastore: fsaverage (ico7) as the template subject, reference maps as Fields on it.

Input is the folder nsp builds (``references-fsaverage-ico7``): ``fsaverage/<fsaverage|fsaverage5|fsaverage6>/{surf,label}``
(FreeSurfer 7.4.1) and ``maps/<id>/<name>_space-fsaverage_den-164k_hemi-{L,R}.func.gii`` + ``provenance.json``.

    sources/pial (primary) · white · inflated · pial_sphere     ico7, 163 842 V/hemisphere, metres, MNI305 (fsaverage)
    sources/pial_atlases/{Structures,Desikan-Killiany,Destrieux} the parcellations; sources/pial_cortex the cortex mask
    sources/<surf>_fsaverage5 | _fsaverage6                      lower densities; sphere vertices = the first N of ico7's
    sources/<map id>                                             one Field per reference map on `pial` [L then R], NaN kept
    sources/hcp1065 · hcp1065_ends                               HCP-1065 streamlines (optional, --trk) and their ends

Re-runnable: every step skips what the subject already holds, so adding maps later is running it again. The Hodge-Dirac
package is added afterwards with ``nxr-convert hodge`` (the primary ``pial`` is in metres, as nsp's solves are).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .atlas.default_subject import MNI305_TO_MNI152, DefaultSubjectWriter, _read_hemis
from .crud import Subject, create_dataset, jdump, open_dataset
from .naming import sanitize_node_name
from .subject import stamp

SUBJECT = "fsaverage"
DENSITIES = {"fsaverage5": 10242, "fsaverage6": 40962}


def _gii(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.float32)


def _surface(src: Path, res: str, surf: str, sphere: bool):
    parts = _read_hemis(src / res, surf)
    pos = np.vstack([p for p, _ in parts]) / 1000.0                      # mm → m
    n_lh = len(parts[0][0])
    faces = np.vstack([parts[0][1], parts[1][1] + n_lh])
    sph = None
    if sphere:
        sph = np.vstack([p for p, _ in _read_hemis(src / res, "sphere.reg")]) / 1000.0
    return pos, faces, sph, n_lh


def _map_files(maps: Path):
    """(name, id, provenance, L file, R file) for every hemisphere pair under maps/<id>/."""
    for d in sorted(p for p in maps.iterdir() if p.is_dir()):
        prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").is_file() else {}
        for L in sorted(d.glob("*_hemi-L.func.gii")):
            R = L.with_name(L.name.replace("_hemi-L", "_hemi-R"))
            if R.is_file():
                yield L.name.split("_space-")[0], d.name, prov, L, R


def _fibres(W: DefaultSubjectWriter, trk: Path, pial_id: str, verts: np.ndarray, white: np.ndarray, *, points=64, gate_mm=5.0):
    from scipy.spatial import cKDTree
    from .atlas.hcp import read_trk
    sub = W.sub
    pts, off = read_trk(trk)                                           # MNI152 mm (DSI Studio -> Brainstorm MNI)
    pts = (np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(MNI305_TO_MNI152).T)[:, :3] / 1000.0   # -> fsaverage m
    nF = len(off) - 1
    out = np.empty((nF, points, 3), np.float32)
    for i in range(nF):
        s = pts[off[i]:off[i + 1]]
        if len(s) < 2:
            s = np.vstack([s, s])
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))]
        q = np.linspace(0, t[-1], points)
        out[i] = np.stack([np.interp(q, t, s[:, k]) for k in range(3)], 1)
    flat = out.reshape(-1, 3)
    fid = sub.manifold(name="hcp1065", path="sources/hcp1065", type="curve", n_vertices=int(nF * points),
                       n_edges=int(nF * (points - 1)), topology={"kind": "path", "rank": 1, "n_components": int(nF)},
                       geometry={"form": "stored", "cs_key": "fsaverage", "positions_path": "sources/hcp1065/points"},
                       comment=f"HCP-1065 population-average tractography (DSI Studio), {nF} streamlines resampled to {points} points "
                               "(fibre-major), MNI152 -> fsaverage (MNI305) by FreeSurfer's affine",
                       source=trk.name, producer_json=jdump({"points_per_component": points, "tracking": "one realisation (random thresholds)"}),
                       populate=lambda at: sub.write_array("sources/hcp1065/points", flat, compress=True))
    ends = np.concatenate([out[:, 0], out[:, -1]]).astype(np.float64)
    d, vtx = cKDTree(white).query(ends)
    ok = d * 1000 <= gate_mm
    col = np.concatenate([np.arange(nF) * points, np.arange(nF) * points + points - 1])
    W.csr("hcp1065_ends", len(verts), nF * points, vtx[ok], col[ok], kind="fibre ends", from_manifold_id=fid, to_manifold_id=pial_id,
          description=f"each streamline end -> the nearest white-surface vertex within {gate_mm} mm; farther ends omitted")
    return {"fibres": nF, "ends_on_cortex": int(ok.sum()), "of": int(len(ok))}


def write_readme(source: Path, datastore: Path, dataset: str, subject: str, info: dict, fields: list) -> None:
    rows = "\n".join(f"| `{n}` | {p.get('map')} | {p.get('original_space')} {p.get('original_density')} | {p.get('method_class')} | "
                     f"{p.get('source_paper')} | {p.get('licence')} |" for n, _, p, _, _ in fields)
    (datastore / "README.md").write_text(f"""# Templates and references (nxr datastore)

Built by `nxr-convert templates` (schema 49) from `{source.name}`; opened by Cortical Flow like any datastore
(dataset `{dataset}`, template subject `{subject}`). Re-run the command to add maps; Hodge-Dirac modes go on later with
`nxr-convert hodge --dataset {datastore / dataset} --subject {subject} --from <dir>`.

**Template:** FreeSurfer 7.4.1 fsaverage (fir), ico7, 163 842 vertices per hemisphere, left then right (327 684), metres.
**Manifolds** (`sources/`): `pial` (primary), `white`, `inflated`, `pial_sphere` (sphere.reg); `pial_fsaverage5|6`,
`pial_fsaverage5|6_sphere`, `inflated_fsaverage5|6`. fsaverage5/6 spheres are the first 10 242 / 40 962 vertices of
ico7's *per hemisphere*; their pial/inflated coordinates are not ico7 subsets (use index correspondence on spheres only).
**Labels:** `pial_atlases/{{Structures, Desikan-Killiany, Destrieux}}`; `pial_cortex` (cortex mask, indices).
**Fibres:** `hcp1065` (curves, 64 points, fsaverage metres) and `hcp1065_ends` (ends -> nearest white vertex within 5 mm),
when built with `--trk`: HCP-1065 template tracking by DSI Studio (Yeh et al.), one random-threshold realisation.
**Fields** (`sources/<id>`, on `pial`, NaN = not given; provenance JSON in each field's producer record):

| id | map | original | method | source | licence |
|---|---|---|---|---|---|
{rows}

MNI152 volume maps are APPROXIMATE (registration fusion); their volumes are not stored here (see `references/neuromaps-receptors`).
Resampling to ico7 adds no resolution (README of `{source.name}`).
""")


def build(source: str | Path, dataset: str | Path, *, subject: str = SUBJECT, trk: str | Path | None = None, log=print) -> dict:
    """Build or extend the templates dataset at ``<datastore>/<dataset>`` from ``source``."""
    import nibabel.freesurfer as fs
    src = Path(source)
    ds_dir = Path(dataset)
    ds = open_dataset(ds_dir) if (ds_dir / "dataset.sqlite").is_file() else create_dataset(
        ds_dir.parent, ds_dir.name, source_tool="freesurfer+neuromaps", source_path=str(src), created_by=stamp())
    try:
        row = ds.db.one("SELECT * FROM subject WHERE dataset_id = ? AND name = ?", ds.id, subject)
        sub = (Subject.open(ds, subject, created_by=stamp()) if row else
               Subject.create(ds, subject, kind="template", folder=f"{subject}.nxr.zarr", source_format="freesurfer",
                              source_path=str(src / "fsaverage"), template_id=None, created_by=stamp()))
        info = {"surfaces": [], "fields": [], "skipped": []}
        with sub.composition():
            W = DefaultSubjectWriter(sub, log)
            fsdir = src / "fsaverage" / "fsaverage"
            prod = {"source": "FreeSurfer 7.4.1 fsaverage (fir)"}
            pial = sub.find("manifold", "sources/pial")
            if pial is None:
                for name, surf, primary in (("pial", "pial", True), ("white", "white", False), ("inflated", "inflated", False)):
                    pos, faces, sph, n_lh = _surface(src / "fsaverage", "fsaverage", surf, sphere=primary)
                    _, sid = W.surface(name, pos, faces, sph, n_lh, {**prod, "surface": surf, "units": "m"}, primary=primary)
                    info["surfaces"].append(name)
                    if primary:
                        pial_id, n, nl = sid, len(pos), n_lh
                pial = sub.find("manifold", "sources/pial")
                W.parcellations(fsdir, "sources/pial", pial_id, nl, n)
                idx = np.concatenate([fs.read_label(str(fsdir / "label" / f"{h}.cortex.label")) + off
                                      for h, off in (("lh", 0), ("rh", nl))]).astype(np.int32)
                W.indices("sources/pial_cortex", pial_id, np.sort(idx), "the cortex mask (FreeSurfer cortex.label): vertices that are not the medial wall")
                for res, nv in DENSITIES.items():          # nested by the sphere: vertex i of fsaverage5/6 is vertex i of ico7
                    for name, surf, primary in ((f"pial_{res}", "pial", True), (f"inflated_{res}", "inflated", False)):
                        pos, faces, sph, n_lh = _surface(src / "fsaverage", res, surf, sphere=primary)
                        W.surface(name, pos, faces, sph, n_lh, {**prod, "surface": surf, "units": "m", "density": res, "n_per_hemisphere": nv,
                                  "note": "sphere vertices are the first N of ico7's; pial/inflated coordinates are NOT ico7 subsets"})
                        info["surfaces"].append(name)
            pial_id, n = pial["id"], int(pial["n_vertices"])
            for name, mid, prov, L, R in _map_files(src / "maps"):
                path = f"sources/{sanitize_node_name(name)}"
                if sub.find("field", path):
                    info["skipped"].append(name)
                    continue
                v = np.concatenate([_gii(L), _gii(R)])
                if v.size != n:
                    raise ValueError(f"{name}: {v.size} values for a {n}-vertex pial")
                flag = " APPROXIMATE (volume projected by registration fusion)" if "APPROX" in str(prov.get("method_class", "")).upper() else ""
                sub.field(name=sanitize_node_name(name), path=path, kind=prov.get("map") or name, manifold_id=pial_id, data=v, compress=True,
                          source=str(prov.get("source_paper") or ""), comment=prov.get("map"),
                          description=f"{prov.get('map')} - from {prov.get('original_space')} {prov.get('original_density')}; "
                                      f"{prov.get('method_class')}{flag}. Licence: {prov.get('licence')}. NaN = not given (e.g. medial wall).",
                          producer_json=jdump(prov))
                info["fields"].append(name)
            if trk and not sub.find("manifold", "sources/hcp1065"):
                white = np.vstack([p for p, _ in _read_hemis(src / "fsaverage" / "fsaverage", "white")]) / 1000.0
                info["hcp1065"] = _fibres(W, Path(trk), pial_id, white, white)
        write_readme(src, ds_dir.parent, ds_dir.name, subject, info, list(_map_files(src / "maps")))
        return {"store": str(sub.store), "id": sub.id, **info}
    finally:
        ds.close()
