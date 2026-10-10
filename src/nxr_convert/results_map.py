"""A SOURCE MAP — a full ``results_*.mat`` that is not a kernel — as a Field on the
cortex (register D124).

Brainstorm writes a volume projected onto a surface (``mri_interp_vol2tess``: a PET
SUVR, an fMRI map) as ``results_surface_<…>.mat``: ``ImageGridAmp`` [nVertices × nTime]
on the surface its ``SurfaceFile`` names, ``DataType: results``, no ``ImagingKernel``.
Nothing read these before; the kernel path skips them by design.

    sources/<stem>    Field  [nVertices] float32 on <surface> — one value per vertex (its kind the map's Comment)

A STATIC map is one value per vertex: Brainstorm stores it as one time column, or as
two identical columns (its ``Time = [0 1]`` convention for a map with no time). A map
that genuinely varies over time needs a time Line to live on, which a source map in
this store does not have yet — it is REFUSED with that reason, not flattened.

The surface is the map's own: if the store does not hold it yet it is exported
beside the primary (never as the primary — the inverse's surface is the primary).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, jdump
from .matio import load_mat, load_mat_vars
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
    nt = np.size(load_mat_vars(p, ["Time"]).get("Time"))          # PROBED: a time-resolved map is V × T, never loaded
    if nt > 2:
        raise ValueError(f"{p.name}: a map over {nt} time points needs a time Line — "
                         "only static maps (one value per vertex) are converted")
    m = load_mat(p)
    if not is_source_map(m):
        raise ValueError(f"{p.name}: not a source map (needs a non-empty ImageGridAmp and no ImagingKernel)")
    amp = np.asarray(m["ImageGridAmp"], dtype=np.float64)
    if amp.ndim == 1:
        amp = amp[:, None]
    if amp.shape[1] == 2 and np.array_equal(amp[:, 0], amp[:, 1]):
        amp = amp[:, :1]                      # Brainstorm's Time = [0 1] for a static map
    if amp.shape[1] != 1:
        raise ValueError(f"{p.name}: a map over {amp.shape[1]} time points needs a time Line — "
                         "only static maps (one value per vertex) are converted")
    surface_rel = str(m.get("SurfaceFile", ""))
    if not surface_rel:
        raise ValueError(f"{p.name}: no SurfaceFile — a map with no surface has nowhere to live")
    surface_path = Path(bst_root) / "anat" / surface_rel
    sname = surface_node_name(surface_path.name)
    spath = f"{folder_for_surface(sname)}/{sname}"
    surf = sub.find("manifold", spath)
    if surf is None:
        if not surface_path.is_file():
            raise FileNotFoundError(f"{p.name}: its surface {surface_rel} is not in the protocol")
        export_surface(sub, surface_path, bst_root=str(bst_root), primary=False)
        surf = sub.find("manifold", spath)
    if surf["n_vertices"] != amp.shape[0]:
        raise ValueError(f"{p.name}: {amp.shape[0]} values for a {surf['n_vertices']}-vertex surface ({sname})")
    comment = str(m.get("Comment", "")) or p.stem
    stem = sanitize_node_name(p.stem.removeprefix("results_"))
    node = f"sources/{stem}"
    text = f"{stem} {comment}".lower()
    fid = sub.field(name=stem, path=node, kind=comment, manifold_id=surf["id"], data=amp[:, 0].astype(np.float32),
                    description=f"{comment} — one value per vertex of {sname}", comment=comment, session=session, source=p.name,
                    unit="SUVR" if "suvr" in text else None,
                    producer_json=jdump({"surface": surface_rel, **({"display_units": str(m["DisplayUnits"])} if m.get("DisplayUnits") else {})}))
    return {"name": stem, "id": fid, "surface": sname, "n_vertices": int(amp.shape[0]), "kind": comment}
