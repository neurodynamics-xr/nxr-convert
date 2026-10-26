"""A SOURCE MAP — a full ``results_*.mat`` that is not a kernel — as a Field on the
cortex (register D124).

Brainstorm writes a volume projected onto a surface (``mri_interp_vol2tess``: a PET
SUVR, an fMRI map) as ``results_surface_<…>.mat``: ``ImageGridAmp`` [nVertices × nTime]
on the surface its ``SurfaceFile`` names, ``DataType: results``, no ``ImagingKernel``.
Nothing read these before; the kernel path skips them by design.

    sources/<stem>    Field  [nVertices] float32 on <surface> — one value per vertex (its kind the map's Comment)

A STATIC map is one value per vertex: Brainstorm stores it as one time column, or as
two identical columns (its ``Time = [0 1]`` convention for a map with no time). A map
over time (``Time`` of more than two points — a BOLD series, 0.2.12) is a Field on the
surface × a time Line at its sampling (the TR), stored as a recording is (sharded, its
chunk grid and envelope); its confounds (``matrix_*.mat`` over time, ``export_matrix_series``)
sit on a points manifold of their rows × the SAME Line:

    sources/<stem>          Field  [nVertices, nTime] float32 on <surface> × sources/<stem>_time
    sources/<conf>          Field  [nRows, nTime] on sources/<conf>_rows × sources/<stem>_time (rows named by <conf>_rows_names)

The surface is the map's own: if the store does not hold it yet it is exported
beside the primary (never as the primary — the inverse's surface is the primary).
A map on fsLR32k or fsaverage is a map on THAT surface (a ``tess_*.mat`` in ``anat/``).

A REGION MAP — a ``matrix_*.mat`` Brainstorm's scout extraction writes (``Value`` [nScouts × 1|2],
``Description`` the scout labels, ``Atlas`` the atlas they come from, ``SurfaceFile``) — is a Field on
that atlas's REGIONS manifold, the one the fibres' connectomes sit on (D47/D126):

    sources/<stem>    Field  [nCodes] float32 on <surface>_<atlas>_regions — element k is code k, NaN where not given
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .entities import labelling
from .matio import load_mat, load_mat_vars, struct_rows
from .naming import sanitize_node_name, surface_node_name
from .surface import export_surface, folder_for_surface


def is_source_map(mat: dict) -> bool:
    """A results file that holds VALUES on its surface: a non-empty ``ImageGridAmp`` and no
    ``ImagingKernel``. Recognised by content, not by ``DataType`` — Brainstorm writes that
    field on timefreq files but not on a projected map (measured on a research cohort's projected PET)."""
    dt = str(mat.get("DataType", "") or "")
    return (dt in ("", "results")
            and mat.get("ImageGridAmp") is not None and bool(np.size(mat.get("ImageGridAmp")))
            and not np.size(mat.get("ImagingKernel") if mat.get("ImagingKernel") is not None else []))


def find_source_maps(cond_dir: str | Path) -> list[Path]:
    """Every non-kernel ``results_*.mat`` in a condition folder."""
    d = Path(cond_dir)
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("results_*.mat"))
            if "KERNEL" not in p.name and not p.name.startswith("._")]


def export_source_map(sub: Subject, results_file: str | Path, *, bst_root: str | Path, session: str | None) -> dict[str, Any]:
    p = Path(results_file)
    m = load_mat(p)          # ponytail: a time-resolved map is loaded whole (15k vertices × 150 TRs = 18 MB); stream it if maps grow to 10⁵ × 10³
    if not is_source_map(m):
        raise ValueError(f"{p.name}: not a source map (needs a non-empty ImageGridAmp and no ImagingKernel)")
    t = np.asarray(m.get("Time") if m.get("Time") is not None else [], dtype=np.float64).ravel()
    amp = np.asarray(m["ImageGridAmp"], dtype=np.float64)
    amp = amp[:, None] if amp.ndim == 1 else amp
    timed = t.size > 2
    if not timed:
        amp = _static_column(amp, p.name)[:, None]
    surface_rel = str(m.get("SurfaceFile", ""))
    surf, spath, _ = _surface_of(sub, surface_rel, bst_root=bst_root, what=p.name)
    sname = spath.rsplit("/", 1)[-1]
    if surf["n_vertices"] != amp.shape[0]:
        raise ValueError(f"{p.name}: {amp.shape[0]} values for a {surf['n_vertices']}-vertex surface ({sname})")
    comment = str(m.get("Comment", "")) or p.stem
    stem = sanitize_node_name(p.stem.removeprefix("results_"))
    node = f"sources/{stem}"
    if sub.find("field", node):
        raise ValueError(f"{p.name}: the subject already holds {node}")
    text = f"{stem} {comment}".lower()
    unit = "SUVR" if "suvr" in text else None
    producer = {"surface": surface_rel, **({"display_units": str(m["DisplayUnits"])} if m.get("DisplayUnits") else {})}
    if timed:                                                        # a BOLD series: on the surface × its own time Line, stored as a recording is
        if amp.shape[1] != t.size:
            raise ValueError(f"{p.name}: {amp.shape[1]} columns for {t.size} time points")
        info = _series(sub, stem, amp, t, rows_id=surf["id"], session=session, comment=comment, source=p.name, unit=unit, producer=producer,
                       description=f"{comment} — one series per vertex of {sname}", what=p.name)
        return {"name": stem, "id": info["id"], "surface": sname, "n_vertices": int(amp.shape[0]), "kind": comment,
                "n_samples": int(t.size), "spacing_s": info["spacing_s"]}
    fid = sub.field(name=stem, path=node, kind=comment, manifold_id=surf["id"], data=amp[:, 0].astype(np.float32),
                    description=f"{comment} — one value per vertex of {sname}", comment=comment, session=session, source=p.name,
                    unit=unit, producer_json=jdump(producer))
    return {"name": stem, "id": fid, "surface": sname, "n_vertices": int(amp.shape[0]), "kind": comment}


def _series(sub: Subject, stem: str, x: np.ndarray, t: np.ndarray, *, rows_id: str, session: str | None, comment: str, source: str,
            unit: str | None, producer: dict, description: str, what: str) -> dict:
    """[rows × T] values on rows × a time Line — the Line another field of the session already has at the same sampling (a BOLD
    map's, for its confounds), else its own — written by the recording writer (chunk grid, envelope)."""
    from .grid import BLOCK
    from .timeseries import _recording, _write_blocks
    dt = np.diff(t)
    if not (dt[0] > 0 and np.allclose(dt, dt[0], rtol=1e-6, atol=0)):
        raise ValueError(f"{what}: its Time is not a regular sampling — a time Line needs one")
    t0, step = float(t[0]), float(dt[0])
    line = sub.db.one("SELECT m.id FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.subject_id = ? "
                      "AND m.type = 'line' AND m.session IS ? AND d.n_vertices = ? AND abs(d.spacing - ?) <= 1e-9 * ? "
                      "AND abs(coalesce(d.origin, 0) - ?) <= 1e-9 * ? AND m.path LIKE 'sources/%'",
                      sub.id, session, int(t.size), step, step, t0, step)
    xs = np.ascontiguousarray(x, dtype=np.float32)
    info = _recording(sub, stem, n_chan=xs.shape[0], n_samples=xs.shape[1], sfreq=1.0 / step, t0=t0, session=session, channels_id=rows_id,
                      write=lambda arr: _write_blocks(arr, xs, BLOCK), flags=None, events=None, channel_names=None, comment=comment,
                      kind="map", source=source, producer=producer, folder="sources", field_kind=comment, description=description,
                      unit=unit, time_id=line["id"] if line else None)
    return {**info, "spacing_s": step}


def is_matrix_series(path: str | Path) -> bool:
    """A ``matrix_*.mat`` over time (``Time`` of more than two points) that is not a region map — an fMRI confounds matrix."""
    return np.size(load_mat_vars(path, ["Time"]).get("Time")) > 2 and not is_region_map(path)


def export_matrix_series(sub: Subject, matrix_file: str | Path, *, session: str | None) -> dict[str, Any]:
    """A matrix over time (``Value`` [nRows × nTime], ``Description`` the row names): a Field on a ``points`` manifold of its rows
    (named by a labelling) × the session's time Line at its sampling — so confounds share time with their BOLD map."""
    p = Path(matrix_file)
    m = load_mat(p)
    v = np.asarray(m.get("Value"), dtype=np.float64)
    t = np.asarray(m.get("Time"), dtype=np.float64).ravel()
    names = [str(d).strip() for d in np.atleast_1d(np.asarray(m.get("Description"), dtype=object)).ravel()]
    if v.ndim != 2 or v.shape != (len(names), t.size):
        raise ValueError(f"{p.name}: Value {v.shape} for {len(names)} names × {t.size} time points")
    comment = str(m.get("Comment", "")) or p.stem
    stem = sanitize_node_name(p.stem.removeprefix("matrix_"))
    if sub.find("field", f"sources/{stem}"):
        raise ValueError(f"{p.name}: the subject already holds sources/{stem}")
    rows = f"sources/{stem}_rows"
    rid = sub.manifold(name=rows.rsplit("/", 1)[-1], path=rows, type="points", n_vertices=len(names), topology={"kind": "none", "rank": 0},
                       session=session, comment=f"the {len(names)} rows of {comment} — element k is row k")
    labelling(sub, f"{rows}_names", np.arange(len(names)), names, manifold_id=rid, dictionary=f"{stem}_rows", session=session,
              description=f"the names of the rows of {comment}")
    info = _series(sub, stem, v, t, rows_id=rid, session=session, comment=comment, source=p.name, unit=None, producer={},
                   description=f"{comment} — {len(names)} rows over time", what=p.name)
    return {"name": stem, "id": info["id"], "rows": len(names), "kind": comment, "n_samples": int(t.size), "spacing_s": info["spacing_s"]}


def _static_column(values: Any, name: str) -> np.ndarray:
    """One value per row: one column, or Brainstorm's two identical ones (``Time = [0 1]``); anything else refused."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 2 and np.array_equal(a[:, 0], a[:, 1], equal_nan=True):
        a = a[:, :1]
    if a.shape[1] != 1:
        raise ValueError(f"{name}: a map over {a.shape[1]} time points needs a time Line — "
                         "only static maps (one value per row) are converted")
    return a[:, 0]


def _surface_of(sub: Subject, surface_rel: str, *, bst_root: str | Path, what: str) -> tuple[dict, str, Path]:
    """The subject's manifold for a Brainstorm ``SurfaceFile`` (exported beside the primary if absent): (row, store path, file)."""
    if not surface_rel:
        raise ValueError(f"{what}: no SurfaceFile — a map with no surface has nowhere to live")
    surface_path = Path(bst_root) / "anat" / surface_rel
    sname = surface_node_name(surface_path.name)
    spath = f"{folder_for_surface(sname)}/{sname}"
    surf = sub.find("manifold", spath)
    if surf is None:
        if not surface_path.is_file():
            raise FileNotFoundError(f"{what}: its surface {surface_rel} is not in the protocol")
        export_surface(sub, surface_path, bst_root=str(bst_root), primary=False)
        surf = sub.find("manifold", spath)
    return surf, spath, surface_path


def atlas_regions(sub: Subject, cortex_path: str, cortex_file: str | Path, atlas: str) -> tuple[list[str], np.ndarray, str] | None:
    """An atlas of the cortex at ``cortex_path`` as REGIONS: (its levels — ``unassigned`` then the scouts in Brainstorm's
    order —, the code of each level, the regions ``points`` manifold whose element k is code k, made if absent).
    None when the cortex holds no such atlas. Its size is the cortex FILE's atlas (every scout), so every reader agrees."""
    fam = sanitize_node_name(atlas)
    atlases = sub.find("selection", f"{cortex_path}_atlases")
    member = sub.db.one("SELECT * FROM selection WHERE member_of_id = ? AND name = ?", atlases["id"], fam) if atlases else None
    entry = next((e for e in struct_rows(load_mat(cortex_file).get("Atlas"), "Name")
                  if sanitize_node_name(str(e.get("Name", ""))) == fam), None)
    if member is None or entry is None:
        return None
    levels = ["unassigned", *(str(x["Label"]) for x in struct_rows(entry.get("Scouts"), "Label"))]
    code = {r["name"]: r["code"] for r in sub.db.all("SELECT name, code FROM dictionary_entry WHERE dictionary_id = ?",
                                                     member["dictionary_id"])}
    code_of = np.asarray([code[nm] for nm in levels], dtype=np.int64)
    n_codes = int(code_of.max()) + 1
    reg = f"{cortex_path}_{fam}_regions"
    have = sub.find("manifold", reg)
    rid = have["id"] if have else sub.manifold(
        name=reg.rsplit("/", 1)[-1], path=reg, type="points", n_vertices=n_codes, topology={"kind": "none", "rank": 0},
        partition_id=member["id"], comment=f"the {fam} regions of {cortex_path.split('/')[-1]} — element k is code k (0 = unassigned)")
    return levels, code_of, rid


def is_region_map(path: str | Path) -> bool:
    """A ``matrix_*.mat`` that names the atlas and the surface its rows are regions of (scout extraction's output)."""
    v = load_mat_vars(path, ["Atlas", "SurfaceFile"])
    return bool(v.get("SurfaceFile")) and bool(struct_rows(v.get("Atlas"), "Name"))


def export_region_map(sub: Subject, matrix_file: str | Path, *, bst_root: str | Path, session: str | None) -> dict[str, Any]:
    """A per-region static map (one value per scout) as a Field on its atlas's regions manifold."""
    p = Path(matrix_file)
    m = load_mat(p)
    atlas = next(iter(struct_rows(m.get("Atlas"), "Name")), None)
    if atlas is None:
        raise ValueError(f"{p.name}: no Atlas — a region map names the atlas its rows are regions of")
    values = _static_column(m.get("Value"), p.name)
    labels = [str(d).split(" @ ")[0].strip() for d in np.atleast_1d(np.asarray(m.get("Description"), dtype=object)).ravel()]
    if len(labels) != values.size:
        raise ValueError(f"{p.name}: {values.size} values for {len(labels)} region labels")
    surf, spath, surface_file = _surface_of(sub, str(m.get("SurfaceFile", "")), bst_root=bst_root, what=p.name)
    aname = str(atlas.get("Name", ""))
    regions = atlas_regions(sub, spath, surface_file, aname)
    if regions is None:
        raise ValueError(f"{p.name}: no atlas '{aname}' on {spath}")
    levels, code_of, rid = regions
    missing = [lab for lab in labels if lab not in levels]
    if missing:
        raise ValueError(f"{p.name}: regions not in the atlas '{aname}': {missing}")
    out = np.full(int(code_of.max()) + 1, np.nan, dtype=np.float32)
    out[[int(code_of[levels.index(lab)]) for lab in labels]] = values
    comment = str(m.get("Comment", "")) or p.stem
    stem = sanitize_node_name(p.stem.removeprefix("matrix_"))
    node = f"sources/{stem}"
    if sub.find("field", node):
        raise ValueError(f"{p.name}: the subject already holds {node}")
    fam = sanitize_node_name(aname)
    fid = sub.field(name=stem, path=node, kind=comment, manifold_id=rid, data=out, session=session, source=p.name, comment=comment,
                    description=f"{comment} — one value per {fam} region of {spath.rsplit('/', 1)[-1]} (NaN: not given)",
                    unit="SUVR" if "suvr" in f"{stem} {comment}".lower() else None,
                    producer_json=jdump({"surface": str(m.get("SurfaceFile")), "atlas": aname, "regions": len(labels)}))
    return {"name": stem, "id": fid, "surface": spath.rsplit("/", 1)[-1], "atlas": fam, "regions": len(labels), "kind": comment}


def export_map(sub: Subject, path: str | Path, *, bst_root: str | Path, session: str | None) -> dict[str, Any]:
    """A map file of any kind: ``matrix_*`` a matrix over time or a region map, else a source map (static or over time)."""
    if Path(path).name.startswith("matrix_") and is_matrix_series(path):
        return export_matrix_series(sub, path, session=session)
    f = export_region_map if Path(path).name.startswith("matrix_") else export_source_map
    return f(sub, path, bst_root=bst_root, session=session)


def template_map_files(root: str | Path) -> list[Path]:
    """The static maps on the DEFAULT ANATOMY: every source map and region map in the protocol's shared folders
    (``@default_study``, ``@inter``, every ``Group_analysis`` condition) whose ``SurfaceFile`` is a ``@default_subject``
    surface. A patient's own folders are never read — their maps are that patient's, even on the default anatomy."""
    data = Path(root) / "data"
    homes = [data / "@default_study", data / "@inter",
             *(sorted(d for d in (data / "Group_analysis").iterdir() if d.is_dir()) if (data / "Group_analysis").is_dir() else [])]
    out = []
    for h in homes:
        cands = [*find_source_maps(h), *(sorted(q for q in h.glob("matrix_*.mat") if not q.name.startswith("._")) if h.is_dir() else [])]
        for q in cands:
            sf = str(load_mat_vars(q, ["SurfaceFile"]).get("SurfaceFile") or "").replace("\\", "/")
            if sf.startswith("@default_subject/") and (q.name.startswith("results_") or is_region_map(q)):
                out.append(q)
    return out
