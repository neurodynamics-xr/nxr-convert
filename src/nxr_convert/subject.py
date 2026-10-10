"""ONE COMMAND, THE WHOLE SUBJECT — every part of a Brainstorm subject as one COMPOSITION (D141/D142, D121–D128).

    nxr-convert subject <protocol> --subject S --dataset <datastore>/<dataset>

The subject row and every node row are written ``pending``, every array laid out from its row and populated, then the
composition COMPLETES once and the store is drained once. In this order:

    1. every condition with recordings or a raw link, in SORTED order — the first brings the anatomy, each a SESSION
       (session-scoped names never collide: D121)
    2. every subject image — the T1, every atlas as codes (D123), every float map (D124)
    3. every source map (``results_*`` that is not a kernel) in every condition, including map-only ones such as PET (D124)
    4. every fibres file, its ends on the cortex and its connectomes (D125, D126)

Any failure removes what the composition made (``Subject.abort``): a subject is complete or absent; a crash mid-way is
rolled back at the next open (``crud.recover``). Everything skipped is reported with its reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crud import Dataset, Subject

Emit = Callable[[dict], None]


def stamp() -> str:
    import sys
    from . import __version__
    return f"nxr-convert {__version__} / Python {sys.version.split()[0]}"


def _conditions(root: Path, subject: str) -> tuple[list[str], list[str]]:
    """(convertible conditions, every condition folder name) for the subject."""
    from .protocol import list_subjects
    convertible: list[str] = []
    for s in list_subjects(root):
        if s["subject"] == subject:
            convertible = [c["name"] for c in s["conditions"] if c["recordings"] or c["raw_links"]]
    data = root / "data" / subject
    folders = sorted({(d.name[4:] if d.name.startswith("@raw") else d.name)
                      for d in data.iterdir() if d.is_dir()
                      and not (d.name.startswith("@") and not d.name.startswith("@raw"))}) if data.is_dir() else []
    return sorted(convertible), folders


def index_cortex(root: Path, subject: str) -> str | None:
    """The subject's cortex as Brainstorm's index (``iCortex``) names it, or None — a FALLBACK, never what stops an import."""
    try:
        from .sources import read_source
        view = read_source(root, prefer="protocol.mat")
    except Exception:  # noqa: BLE001
        return None
    for s in view.subjects:
        if s.name == subject:
            return s.cortex
    return None


def anat_dir(root: Path, subject: str) -> tuple[Path, bool]:
    """The subject's anatomy folder, and whether it is INHERITED (``UseDefaultAnat``: it borrows ``@default_subject``)."""
    from .matio import load_mat_vars
    own = root / "anat" / subject
    bs = own / "brainstormsubject.mat"
    if bs.is_file():
        v = load_mat_vars(bs, ["UseDefaultAnat"]).get("UseDefaultAnat")
        if v is not None and np.size(v) and int(np.ravel(np.asarray(v))[0]) == 1:
            return root / "anat" / "@default_subject", True
    return own, False


def kernel_surface(cond_dir: Path, raw_dir: Path, anat: Path) -> str | None:
    """The surface this condition's kernels are defined on (their own ``SurfaceFile``) — None when there is no kernel or
    they DISAGREE (picking one silently would index a second inverse by the wrong manifold). Both condition homes."""
    from .matio import load_mat_vars
    found: set[str] = set()
    for home in (cond_dir, raw_dir):
        if not home.is_dir():
            continue
        for kf in sorted(home.glob("results_*KERNEL*.mat")):
            if kf.name.startswith("._"):
                continue
            sf = load_mat_vars(kf, ["SurfaceFile"]).get("SurfaceFile")
            if sf and (anat / Path(str(sf)).name).is_file():
                found.add(Path(str(sf)).name)
    return found.pop() if len(found) == 1 else None


def condition_plan(root: Path, subject: str, condition: str, *, surface: str | None = None, extra_surfaces: list[str] | None = None,
                   recording: str | None = None, channel_file: str | None = None, include_imported: bool = False,
                   no_raw: bool = False) -> dict[str, Any]:
    """What ``convert_condition`` reads for one condition: THE PRIMARY SURFACE IS THE INVERSE'S (its kernels' own
    ``SurfaceFile``; else the index's ``iCortex``; else the pial literal), THE CONTINUOUS IS THE DEFAULT (the raw link —
    the treated ``.bst``; excerpts opt-in, or all there is without a link), channels and kernels in either home."""
    cond = root / "data" / subject / condition
    raw_dir = root / "data" / subject / f"@raw{condition}"
    anat, _ = anat_dir(root, subject)
    surf = anat / (surface or kernel_surface(cond, raw_dir, anat) or index_cortex(root, subject) or "tess_cortex_pial_low.mat")
    extra = [anat / e for e in (extra_surfaces or []) if e != surf.name]
    has_raw = raw_dir.is_dir() and any(p for p in raw_dir.glob("data_0raw_*.mat") if not p.name.startswith("._"))
    include_raw = has_raw and not no_raw
    if recording:
        data_files = [cond / recording]
    elif include_imported or not has_raw:
        data_files = sorted(p for p in cond.glob("data_*.mat") if not p.name.startswith("._")) if cond.is_dir() else []
    else:
        data_files = []
    if not data_files and not include_raw:
        raise ValueError(f"nothing to import in {condition}: no raw link and no data_*.mat")
    chans = (sorted(p for p in cond.glob("channel*.mat") if not p.name.startswith("._")) if cond.is_dir() else []) \
        or (sorted(p for p in raw_dir.glob("channel*.mat") if not p.name.startswith("._")) if raw_dir.is_dir() else [])
    if channel_file:
        cfile = (cond if (cond / channel_file).exists() else raw_dir) / channel_file
    elif chans:
        cfile = chans[0]
    else:
        raise ValueError(f"no channel*.mat in {condition} (either home)")
    kernels = sorted({*(p for p in (cond.glob("results_*KERNEL*.mat") if cond.is_dir() else []) if not p.name.startswith("._")),
                      *(p for p in (raw_dir.glob("results_*KERNEL*.mat") if raw_dir.is_dir() else []) if not p.name.startswith("._"))})
    raw_file = None
    if include_raw:
        raws = sorted(p for p in raw_dir.glob("data_0raw_*.mat") if not p.name.startswith("._"))
        if len(raws) != 1:
            raise ValueError(f"{len(raws)} raw link(s) in @raw{condition}")
        raw_file = raws[0]
    return {"surface_file": surf, "extra_surfaces": extra, "data_files": data_files, "channel_file": cfile,
            "kernel_files": list(kernels), "raw_data_file": raw_file}


def _subject_cortex(root: Path, subject: str, sub: Subject) -> str:
    """The cortex the fibres were assigned on: Brainstorm's ``iCortex``, else ``brainstormsubject.mat``'s ``Cortex``, else
    the subject's primary surface."""
    from .matio import load_mat_vars
    found = index_cortex(root, subject)
    if found:
        return Path(found).name
    bs = root / "anat" / subject / "brainstormsubject.mat"
    if bs.is_file():
        c = load_mat_vars(bs, ["Cortex"]).get("Cortex")
        if c:
            return Path(str(c)).name
    prim = sub.db.one("SELECT name FROM manifold WHERE subject_id = ? AND type = 'surface' AND is_primary = 1", sub.id)
    if prim:
        return f"tess_{prim['name']}.mat"
    raise ValueError(f"{subject}: cannot tell which cortex the fibres were assigned on")


def compose_subject(sub: Subject, root: str | Path, subject: str, *, emit: Emit) -> dict[str, Any]:
    """Every part of the Brainstorm subject into the Subject ``sub`` (rows pending; the caller completes)."""
    from .convert import convert_condition
    from .fibers import export_fibers, is_fibers_file
    from .mri import import_subjectimages
    from .naming import surface_node_name
    from .results_map import export_source_map, find_source_maps
    from .surface import export_surface, folder_for_surface

    root = Path(root)
    summary: dict[str, Any] = {"subject": subject, "sessions": [], "volumes": [], "maps": [], "fibers": [], "connectomes": [], "skipped": []}

    def skip(what: str, reason: str) -> None:
        summary["skipped"].append({"what": what, "reason": reason})
        emit({"stage": "skipped", "what": what, "reason": reason})

    convertible, folders = _conditions(root, subject)
    if not convertible:
        raise ValueError(f"{subject}: no condition with recordings or a raw link — nothing to anchor a subject")
    # 1. sessions — a condition that cannot convert (an unsupported raw link, no channels …) is rolled back and REPORTED
    for cond in convertible:
        mark = sub.mark()
        try:
            plan = condition_plan(root, subject, cond)
            convert_condition(sub, bst_root=root, condition=cond, progress=emit, **plan)
            summary["sessions"].append(cond)
        except (ValueError, FileNotFoundError, NotImplementedError) as e:
            sub.rollback_to(mark)
            skip(f"condition {cond}", f"{type(e).__name__}: {e}")
    if not summary["sessions"]:
        raise ValueError(f"{subject}: no condition converted — {summary['skipped']}")
    # 2. subject images
    anat, _ = anat_dir(root, subject)
    try:
        summary["volumes"] = [v["name"] for v in import_subjectimages(sub, anat, progress=emit)]
    except FileNotFoundError as e:
        skip("subject images", str(e))
    # 3. source maps, in every condition folder (map-only ones included)
    for cond in folders:
        for home in (root / "data" / subject / cond, root / "data" / subject / f"@raw{cond}"):
            for mf in find_source_maps(home):
                try:
                    info = export_source_map(sub, mf, bst_root=root, session=cond)
                    summary["maps"].append(info["name"])
                    emit({"stage": "map", **{k: v for k, v in info.items() if k != "id"}})
                except (ValueError, FileNotFoundError) as e:
                    skip(mf.name, str(e))
    # 4. fibres — ends on the subject's cortex (iCortex), connectomes per assigned atlas
    fibre_files = [p for p in sorted(anat.glob("tess_*.mat")) if is_fibers_file(p) and not p.name.startswith("._")]
    if fibre_files:
        cortex_file = _subject_cortex(root, subject, sub)
        cname = surface_node_name(cortex_file)
        cpath = f"{folder_for_surface(cname)}/{cname}"
        if sub.find("manifold", cpath) is None:
            export_surface(sub, anat / cortex_file, bst_root=str(root), primary=False)
        for ff in fibre_files:
            info = export_fibers(sub, ff, subject=subject, cortex_path=cpath, cortex_file=anat / cortex_file, progress=emit)
            summary["fibers"].append(info["name"])
            summary["connectomes"] += [f"{info['name']}_connectome_{a}" for a in info["connectomes"]]
    return summary


def convert_subject(ds: Dataset, root: str | Path, subject: str, *, emit: Emit) -> dict[str, Any]:
    """The whole subject as ONE composition: created pending, completed once (one drain), or removed whole."""
    root = Path(root)
    anat, inherited = anat_dir(root, subject)
    sub = Subject.create(ds, subject, source_format="brainstorm", source_path=str(anat), created_by=stamp())
    with sub.composition():
        summary = compose_subject(sub, root, subject, emit=emit)
    n = {t: ds.db.value(f"SELECT count(*) FROM {t} WHERE subject_id = ?", sub.id) for t in ("manifold", "field", "operator", "selection")}
    emit({"stage": "subject-done", "subject": subject, "id": sub.id, "store": str(sub.store), "rows": n,
          "anatomy": "@default_subject" if inherited else "own", **{k: v for k, v in summary.items() if k != "subject"}})
    return {**summary, "id": sub.id, "rows": n}


def convert_template(ds: Dataset, root: str | Path, *, surfaces: list[str] | None = None, emit: Emit) -> dict[str, Any]:
    """The protocol's default anatomy (``anat/@default_subject``) as the dataset's TEMPLATE subject (D55/D127): every
    low-resolution cortex (pial first, the primary — a group result or a subject's kernel may sit on any) and its images."""
    from .mri import import_subjectimages
    from .surface import export_surface
    root = Path(root)
    src = root / "anat" / "@default_subject"
    if not src.is_dir():
        raise FileNotFoundError(f"{src}: not in this protocol")
    files = surfaces or sorted((f.name for f in src.glob("tess_cortex_*_low.mat") if not f.name.startswith("._")),
                               key=lambda n: (0 if n == "tess_cortex_pial_low.mat" else 1, n))
    if not files:
        raise FileNotFoundError(f"{src}: no tess_cortex_*_low.mat to export as the template's cortex")
    sub = Subject.create(ds, "template", kind="template", source_format="brainstorm", source_path=str(src), created_by=stamp())
    with sub.composition():
        names = []
        for k, sf in enumerate(files):
            info = export_surface(sub, src / sf, bst_root=str(root), primary=(k == 0))
            names.append(info["name"])
            emit({"stage": "surface", **{x: y for x, y in info.items() if x != "id"}})
        vols = import_subjectimages(sub, src, None, progress=emit)
    return {"id": sub.id, "surfaces": names, "volumes": [v["name"] for v in vols]}


def convert_group(ds: Dataset, root: str | Path, *, emit: Emit) -> dict[str, Any]:
    """The protocol's group results (``data/Group_analysis/@intra``) as the dataset's GROUP subject (D55): each average a
    Field whose contributions are the members' fields."""
    from .timefreq import export_timefreq
    root = Path(root)
    src = root / "data" / "Group_analysis" / "@intra"
    if not src.is_dir():
        raise FileNotFoundError(f"{src}: not in this protocol")
    sub = Subject.create(ds, "group", kind="group", source_format="brainstorm", source_path=str(src), created_by=stamp())
    with sub.composition():
        wrote = []
        for f in sorted(p for p in src.glob("timefreq_*.mat") if not p.name.startswith("._")):
            info = export_timefreq(sub, f, session=None, group=True)
            if info:
                wrote.append(info["name"])
                emit({"stage": "field", **{x: y for x, y in info.items() if x != "id"}})
    return {"id": sub.id, "fields": wrote}
