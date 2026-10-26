"""The ``nxr-convert`` CLI — CRUD of a dataset and of each entity, by the procedure (D141/D142).

    nxr-convert --version                                                             the version and the database schema it writes
    nxr-convert dataset create <datastore> <name> [--source-tool T --source-protocol P --source-path S]
    nxr-convert dataset delete <datastore> <name>
    nxr-convert dataset recover <datastore>/<dataset>
    nxr-convert subject  <protocol> --subject S --dataset <datastore>/<dataset>      the whole subject, one composition
    nxr-convert convert  <protocol> --subject S --condition C --dataset D [...]       one session (the subject made if absent)
    nxr-convert template <protocol> --dataset D [--surface tess_*.mat …]             the default anatomy → the TEMPLATE
    nxr-convert group    <protocol> --dataset D                                      Group_analysis → the GROUP subject
    nxr-convert surface  --dataset D --subject S --file tess_*.mat … [--bst-root P]  add surfaces
    nxr-convert mri      --dataset D --subject S --anat <anat dir> [--only N …]      add MRI volumes
    nxr-convert inverse  --dataset D --subject S --file results_*KERNEL*.mat --session C --bst-root P [--channels-id …]
    nxr-convert carry-inverse --dataset D --subject S --from <dataset> --kernel NAME --session C   an inverse kernel from another store of the subject
    nxr-convert resample --dataset D --subject S --recording NAME --to HZ [--name N]  a stored recording resampled, beside it (``timeseries.py``)
    nxr-convert surface  --dataset D --subject S --fs-dir <FreeSurfer subject dir> --name N [--scs-from tess_*.mat]
                                                                                      add FreeSurfer's lh/rh.pial as one surface
    nxr-convert modes    --dataset D --subject S --work DIR [--modes K] [--surface NAME] [--replace]   SOLVE the Hodge–Dirac basis from the store
                         (the Editor's glue by anatomy, intrinsic Delaunay, d + δ; ``eigen/``) and store it on the glued surface;
                         --export: only the solve's input (to run elsewhere); --input FILE: solve that input (no dataset), store with `hodge`
    nxr-convert hodge    --dataset D --subject S --from <nsp rheome-hodge-dirac dir> [--nsp-subject S] [--replace] [--surface NAME]
                                                                                      add the Hodge–Dirac eigenmodes of the primary
                                                                                      (or of the named surface, suffixed) (``hodge.py``)
    nxr-convert interp   --dataset D --subject S --from SRC --to DST [--replace]     the operator DST ← SRC through the spheres (``interp.py``)
    nxr-convert templates <references-fsaverage-ico7 dir> --dataset D [--subject fsaverage] [--trk hcp1065.trk]
                                                                                      the templates-and-references dataset (``templates.py``)
    nxr-convert map      --dataset D --subject S --file results_*.mat|matrix_*.mat … --bst-root P [--session C]
                                                                                      add static maps (surface or region)
    nxr-convert remove   --dataset D --subject S [--node PATH]                      delete a subject, or one node of it
    nxr-convert list <protocol> | scan <protocol>                                     what a protocol holds
    nxr-convert atlas default-subject | build | reduce | info                         the dyadic atlas (``atlas/cli.py``)

Every write goes through the dataset's ``dataset.sqlite`` (``<datastore>/<dataset>/``): the rows first, the store their
image. Every command that writes a dataset takes its WRITER LOCK first (``<dataset>/.writer.lock``, #26,
``writer_lock.py``) and is refused while another live writer — the app, another conversion — holds it; run BY the app, it
accepts the app's lock through ``NXR_WRITER_LOCK_HELD=<lock file>:<app pid>`` (the handshake). Progress is JSON-LINES on stdout (one object per event, ``stage`` keyed) — the contract Electron main parses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .protocol import list_subjects


def _print(obj: dict) -> None:
    print(json.dumps(obj, default=str), flush=True)


def version_string() -> str:
    """``nxr-convert <version> (database schema <N>)`` — N is the ``PRAGMA user_version`` every database it writes carries."""
    from . import __version__
    from .db import schema_version
    return f"nxr-convert {__version__} (database schema {schema_version()})"


def _dataset(path: str):
    from .crud import open_dataset
    return open_dataset(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nxr-convert",
                                description="Convert Brainstorm protocols into an nxr datastore the Cortical Flow app opens.")
    p.add_argument("--version", action="version", version=version_string())
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dataset", help="create, delete or recover a dataset (its dataset.sqlite and folder; the catalog)")
    pda = pd.add_subparsers(dest="action", required=True)
    c = pda.add_parser("create")
    c.add_argument("datastore")
    c.add_argument("name")
    c.add_argument("--source-tool", default=None)
    c.add_argument("--source-protocol", default=None)
    c.add_argument("--source-path", default=None)
    d = pda.add_parser("delete")
    d.add_argument("datastore")
    d.add_argument("name")
    r = pda.add_parser("recover", help="roll back unfinished writes, finish deletions, drain")
    r.add_argument("dataset")

    pl = sub.add_parser("list", help="list a protocol's subjects/conditions as JSON")
    pl.add_argument("root")

    ps = sub.add_parser("scan", help="read the protocol's database index as a source view")
    ps.add_argument("root")
    ps.add_argument("--prefer", choices=["protocol.mat", "walk"], default=None)
    ps.add_argument("--out", default=None, help="write the view here as JSON (default: stdout)")
    ps.add_argument("--summary", action="store_true", help="print counts instead of the whole view")

    psub = sub.add_parser("subject", help="convert EVERY part of one Brainstorm subject as one composition: sessions, volumes, "
                          "source maps, fibres and connectomes (D121–D126)")
    psub.add_argument("root")
    psub.add_argument("--subject", required=True)
    psub.add_argument("--dataset", required=True, help="<datastore>/<dataset> (created with `dataset create`)")
    psub.add_argument("--resample", type=float, default=None, metavar="HZ", help="store the recordings resampled to HZ (resample_poly)")

    pc = sub.add_parser("convert", help="convert one condition as a session of a subject (the subject made if absent)")
    pc.add_argument("root")
    pc.add_argument("--subject", required=True)
    pc.add_argument("--condition", required=True)
    pc.add_argument("--dataset", required=True)
    pc.add_argument("--surface", default=None, help="tess_*.mat to import as the PRIMARY (default: the kernels' own SurfaceFile)")
    pc.add_argument("--extra-surface", action="append", default=None, metavar="tess_*.mat")
    pc.add_argument("--recording", default=None, help="one data_*.mat (default: the raw link, else every excerpt)")
    pc.add_argument("--channel-file", default=None)
    pc.add_argument("--no-gain", action="store_true")
    pc.add_argument("--all-surfaces", action="store_true")
    pc.add_argument("--include-imported", action="store_true")
    pc.add_argument("--no-raw", action="store_true")
    pc.add_argument("--resample", type=float, default=None, metavar="HZ", help="store the recording resampled to HZ (resample_poly)")

    pt = sub.add_parser("template", help="the protocol's default anatomy (anat/@default_subject) as the TEMPLATE subject (D55)")
    pt.add_argument("root")
    pt.add_argument("--dataset", required=True)
    pt.add_argument("--surface", action="append", default=None)
    pg = sub.add_parser("group", help="the protocol's group results (data/Group_analysis/@intra) as the GROUP subject (D55)")
    pg.add_argument("root")
    pg.add_argument("--dataset", required=True)

    pf = sub.add_parser("surface", help="add Brainstorm tess_*.mat surfaces to a subject")
    pf.add_argument("--dataset", required=True)
    pf.add_argument("--subject", required=True)
    pf.add_argument("--file", nargs="+", help="tess_*.mat files")
    pf.add_argument("--bst-root", default="")
    pf.add_argument("--fs-dir", help="a FreeSurfer subject dir: surf/lh|rh.pial + sphere.reg + label/*.annot as ONE surface (--name)")
    pf.add_argument("--name", default="cortex_pial", help="with --fs-dir: the surface's name")
    pf.add_argument("--scs-from", default=None, help="with --fs-dir: a tess_*.mat of the same vertices; stores the fitted tkr -> SCS affine")

    pm = sub.add_parser("mri", help="add a subject's MRI volumes (subjectimage_*.mat)")
    pm.add_argument("--dataset", required=True)
    pm.add_argument("--subject", required=True)
    pm.add_argument("--anat", required=True, help="the subject's Brainstorm anatomy folder (…/anat/<subject>)")
    pm.add_argument("--only", nargs="*", default=None)

    pi = sub.add_parser("inverse", help="add ONE inverse kernel (results_*KERNEL*.mat) to a subject; never overwrites")
    pi.add_argument("--dataset", required=True)
    pi.add_argument("--subject", required=True)
    pi.add_argument("--file", required=True)
    pi.add_argument("--session", required=True)
    pi.add_argument("--bst-root", required=True)
    pi.add_argument("--channels-id", default=None)
    pi.add_argument("--surface-id", default=None)
    pi.add_argument("--forward-id", default=None)
    pi.add_argument("--name", default=None)
    pi.add_argument("--channel-file", default=None)

    ph = sub.add_parser("hodge", help="add the Hodge-Dirac eigenmode package (nsp rheome-hodge-dirac outputs) to a subject's "
                        "primary cortex; refused if present unless --replace")
    ph.add_argument("--dataset", required=True)
    ph.add_argument("--subject", required=True)
    ph.add_argument("--from", dest="src", required=True, help="the folder holding <sub>_hodge_mu/modes/mass.npy, _glued_*, _idt_F, _hodge.json")
    ph.add_argument("--nsp-subject", default=None, help="the file prefix there, when it is not the subject's name")
    ph.add_argument("--replace", action="store_true")
    ph.add_argument("--surface", default=None, help="the surface the solve was on (default: the primary); others get suffixed names")
    pmo = sub.add_parser("modes", help="SOLVE a subject's Hodge-Dirac basis from its stored surface with the Editor's own methods (glue, "
                         "intrinsic Delaunay, d + delta) and store it as `hodge` does")
    pmo.add_argument("--dataset", default=None)
    pmo.add_argument("--subject", default=None)
    pmo.add_argument("--modes", type=int, default=3000, help="about this many positive modes, half exact and half coexact (default 3000)")
    pmo.add_argument("--surface", default=None, help="a surface by name (default: the primary)")
    pmo.add_argument("--work", required=True, help="where the solve's derivative is written (kept: what was stored, and why)")
    pmo.add_argument("--replace", action="store_true", help="replace the surface's stored basis (its arrays moved aside, never deleted)")
    pmo.add_argument("--export", action="store_true", help="only write the solve's INPUT (glue-ready surface + medial wall) to --work, for a solve elsewhere")
    pmo.add_argument("--input", default=None, help="solve from an exported input (no dataset): the derivative to --work; store it with `hodge --from`")

    pn = sub.add_parser("interp", help="write the interpolation operator DST <- SRC (barycentric on the registration spheres)")
    pn.add_argument("--dataset", required=True)
    pn.add_argument("--subject", required=True)
    pn.add_argument("--from", dest="src", required=True)
    pn.add_argument("--to", dest="dst", required=True)
    pn.add_argument("--replace", action="store_true")

    pmap = sub.add_parser("map", help="add static maps to a subject: a source map (results_*.mat, one value per vertex of its "
                          "SurfaceFile) or a region map (matrix_*.mat from scout extraction, one value per region of its Atlas)")
    pmap.add_argument("--dataset", required=True)
    pmap.add_argument("--subject", required=True)
    pmap.add_argument("--file", required=True, nargs="+")
    pmap.add_argument("--bst-root", required=True)
    pmap.add_argument("--session", default=None)

    pr = sub.add_parser("remove", help="delete a subject, or one node of it (by its store-relative path)")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--subject", required=True)
    pr.add_argument("--node", default=None)

    pci = sub.add_parser("carry-inverse", help="an inverse kernel CARRIED from another store of the same subject (an earlier import): its channels by name, the same cortex checked")
    pci.add_argument("--dataset", required=True)
    pci.add_argument("--subject", required=True)
    pci.add_argument("--from", dest="src", required=True, help="the other dataset folder (<datastore>/<dataset>)")
    pci.add_argument("--from-subject", default=None, help="the subject's name there (default: the same)")
    pci.add_argument("--kernel", required=True, help="the kernel's name there")
    pci.add_argument("--session", required=True, help="this subject's session it maps (its channels, its head model)")
    pci.add_argument("--name", default=None)
    prs = sub.add_parser("resample", help="a STORED recording resampled to HZ (resample_poly) as a recording beside it — no protocol read")
    prs.add_argument("--dataset", required=True)
    prs.add_argument("--subject", required=True)
    prs.add_argument("--recording", required=True, help="the recording's field name")
    prs.add_argument("--to", type=float, required=True, metavar="HZ")
    prs.add_argument("--name", default=None, help="the new recording's name (default: <recording>_<HZ>Hz)")
    pq = sub.add_parser("qc", help="sensor QC of a subject's recordings: the PSD Field, its measures and qc_flag, meg-qc's metrics")
    pq.add_argument("--dataset", required=True)
    pq.add_argument("--subject", required=True)
    pq.add_argument("--megqc", default=None, help="meg-qc's <sub>.tar.gz (its session metrics become measurements of the rest runs)")
    pq.add_argument("--replace", action="store_true")
    pmg = sub.add_parser("migrate", help="bring a schema-48 shared database into a dataset folder at this schema, every node's "
                         "zarr.json rewritten from its row (48 → 49 only; an exception to D139)")
    pmg.add_argument("--from", dest="src", required=True, help="the old database (nxr.sqlite)")
    pmg.add_argument("--dataset", required=True, help="<datastore>/<dataset>: its subject stores already extracted here")
    pfr = sub.add_parser("fragment", help="one subject's writable copy of a shared dataset, for a parallel job (merge folds it back)")
    pfr.add_argument("--dataset", required=True)
    pfr.add_argument("--subject", default=None, help="the subject (one already in the dataset)")
    pfr.add_argument("--new", action="store_true", help="an EMPTY fragment, for a subject not yet in the dataset (converted into it, merged in)")
    pfr.add_argument("--to", dest="dst", required=True, help="a folder; the fragment is <to>/<dataset>, the command's --dataset")
    pme = sub.add_parser("merge", help="fold fragments back into their shared dataset, serially, their nodes resynced")
    pme.add_argument("--dataset", required=True)
    pme.add_argument("fragments", nargs="+")
    pme.add_argument("--replace", action="store_true", help="a new fragment's subject replaces one of the same name (its store moved aside)")
    pt2 = sub.add_parser("templates", help="build or extend the templates-and-references dataset: fsaverage (ico7) as the template, "
                                           "reference maps (nsp's references-fsaverage-ico7 folder) as Fields; re-run to add maps")
    pt2.add_argument("source", help="the references-fsaverage-ico7 folder")
    pt2.add_argument("--dataset", required=True, help="<datastore>/<dataset>, made if absent")
    pt2.add_argument("--subject", default="fsaverage")
    pt2.add_argument("--trk", default=None, help="hcp1065.trk (DSI Studio): add the streamlines and their ends")
    pt2.add_argument("--surface", default=None, help="the surface the maps go on (default: the subject's surface with each map's vertex count)")

    from .atlas.cli import add_parser as _add_atlas_parser
    _add_atlas_parser(sub)

    a = p.parse_args(argv)
    if a.cmd == "atlas":
        from .atlas.cli import run as _run_atlas
        return _run_atlas(a)
    if a.cmd == "list":
        _print({"stage": "protocol", "root": a.root, "subjects": list_subjects(a.root)})
        return 0
    if a.cmd == "scan":
        from .sources import read_source
        view = read_source(a.root, prefer=a.prefer)
        payload = view.to_json()
        # `--out` WRITES, whatever --summary says: --summary chooses what is PRINTED, --out whether the view is persisted
        wrote = None
        if a.out:
            wrote = Path(a.out)
            wrote.parent.mkdir(parents=True, exist_ok=True)
            wrote.write_text(json.dumps(payload, indent=2) + "\n")
        if a.summary or wrote is not None:
            _print({"stage": "scan", "kind": view.kind, "root": view.root, **({"wrote": str(wrote)} if wrote else {}), **view.counts})
        else:
            print(json.dumps(payload, indent=2))
        return 0
    if a.cmd == "modes" and a.input:                       # a solve where the store is not (nsp on fir): input file in, derivative out
        from .eigen.hodge_dirac import load_input, solve_input
        try:
            _print({"stage": "result", **solve_input(load_input(Path(a.input)), n_modes=a.modes, work=Path(a.work), progress=_print)})
            return 0
        except Exception as e:  # noqa: BLE001
            _print({"stage": "error", "message": f"{type(e).__name__}: {e}"})
            return 1
    if a.cmd == "modes" and not (a.dataset and a.subject):
        _print({"stage": "error", "message": "modes: --dataset and --subject (or --input)"})
        return 2
    if a.cmd in ("migrate", "fragment", "merge"):
        from . import shared_dataset as sd
        try:
            out = (sd.migrate(a.src, a.dataset) if a.cmd == "migrate" else sd.fragment(a.dataset, a.subject, a.dst, new=a.new)
                   if a.cmd == "fragment" else sd.merge(a.dataset, a.fragments, replace=a.replace))
        except Exception as e:  # noqa: BLE001
            _print({"stage": "error", "message": f"{type(e).__name__}: {e}"})
            return 1
        _print({"stage": "result", **out})
        return 0
    if a.cmd == "templates":
        from .templates import build
        try:
            _print({"stage": "result", **build(a.source, a.dataset, subject=a.subject, trk=a.trk, surface=a.surface, log=lambda *m: None)})
            return 0
        except Exception as e:  # noqa: BLE001
            _print({"stage": "error", "message": f"{type(e).__name__}: {e}"})
            return 1
    try:
        return _run(a)
    except Exception as e:  # noqa: BLE001 — the UI reads stage:error lines
        from .writer_lock import DatasetNotWritable, WriterLockHeld
        if not isinstance(e, (ValueError, FileNotFoundError, FileExistsError, KeyError, NotImplementedError, WriterLockHeld,
                              DatasetNotWritable)):
            import traceback
            traceback.print_exc()
        _print({"stage": "error", "message": f"{type(e).__name__}: {e}"})
        return 1


def _run(a) -> int:
    from .crud import Subject, create_dataset, delete_dataset, recover, remove_node, write_catalog
    from .db import drain
    from .subject import stamp

    if a.cmd == "dataset":
        if a.action == "create":
            ds = create_dataset(a.datastore, a.name, source_tool=a.source_tool, source_protocol=a.source_protocol,
                                source_path=a.source_path, created_by=stamp())
            _print({"stage": "result", "dataset": ds.row["name"], "id": ds.id, "database": str(ds.folder / "dataset.sqlite")})
            ds.close()
        elif a.action == "delete":
            delete_dataset(a.datastore, a.name)
            _print({"stage": "result", "deleted": a.name})
        else:
            from .crud import open_dataset
            ds = open_dataset(a.dataset, recover_now=False)
            healed = recover(ds.db, ds.root)
            drain(ds.db, ds.root)
            write_catalog(ds.root)
            _print({"stage": "result", **healed})
            ds.close()
        return 0

    with _dataset(a.dataset) as ds:
        if a.cmd == "subject":
            from .subject import convert_subject
            r = convert_subject(ds, a.root, a.subject, emit=_print, resample=a.resample)
            _print({"stage": "result", **r})
            return 0
        if a.cmd == "template":
            from .subject import convert_template
            _print({"stage": "result", **convert_template(ds, a.root, surfaces=a.surface, emit=_print)})
            return 0
        if a.cmd == "group":
            from .subject import convert_group
            _print({"stage": "result", **convert_group(ds, a.root, emit=_print)})
            return 0
        if a.cmd == "convert":
            from .convert import convert_condition
            from .subject import anat_dir, condition_plan
            root = Path(a.root)
            plan = condition_plan(root, a.subject, a.condition, surface=a.surface, extra_surfaces=a.extra_surface,
                                  recording=a.recording, channel_file=a.channel_file, include_imported=a.include_imported,
                                  no_raw=a.no_raw)
            sub = (Subject.open(ds, a.subject, created_by=stamp()) if ds.subject(a.subject)
                   else Subject.create(ds, a.subject, source_format="brainstorm", source_path=str(anat_dir(root, a.subject)[0]),
                                       created_by=stamp()))
            with sub.composition():
                r = convert_condition(sub, bst_root=root, condition=a.condition, write_gain=not a.no_gain,
                                      all_surfaces=a.all_surfaces, progress=_print, resample=a.resample, **plan)
            write_catalog(ds.root)
            _print({"stage": "result", **r})
            return 0
        sub = Subject.open(ds, a.subject, created_by=stamp())
        if a.cmd == "surface":
            from .surface import export_surface
            if bool(a.fs_dir) == bool(a.file):
                raise SystemExit("surface: give --file tess_*.mat … or --fs-dir")
            with sub.composition():
                wrote = []
                if a.fs_dir:
                    from .surface import export_freesurfer
                    info = export_freesurfer(sub, a.fs_dir, name=a.name, scs_from=a.scs_from)
                    wrote.append(info["name"])
                    _print({"stage": "surface", **{k: v for k, v in info.items() if k != "id"}})
                for f in a.file or []:
                    if not Path(f).is_file():
                        raise FileNotFoundError(f"not a file: {f}")
                    info = export_surface(sub, f, bst_root=a.bst_root, primary=False)
                    wrote.append(info["name"])
                    _print({"stage": "surface", **{k: v for k, v in info.items() if k != "id"}})
            _print({"stage": "result", "surfaces": wrote})
        elif a.cmd == "mri":
            from .mri import import_subjectimages
            with sub.composition():
                wrote = import_subjectimages(sub, a.anat, a.only, progress=_print)
            _print({"stage": "result", "volumes": [w["name"] for w in wrote]})
        elif a.cmd == "inverse":
            from .convert import add_inverse
            with sub.composition():
                info = add_inverse(sub, kernel_file=a.file, session=a.session, bst_root=a.bst_root, channels_id=a.channels_id,
                                   surface_id=a.surface_id, forward_id=a.forward_id, name=a.name, channel_file=a.channel_file,
                                   progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "hodge":
            from .hodge import add_hodge
            with sub.composition():
                info = add_hodge(sub, a.src, sub=a.nsp_subject, replace=a.replace, surface=a.surface)
            _print({"stage": "result", **info})
        elif a.cmd == "modes" and a.export:
            from .eigen.hodge_dirac import modes_input, save_input
            Path(a.work).mkdir(parents=True, exist_ok=True)
            out = save_input(modes_input(sub, a.surface), Path(a.work) / f"{sub.row['name']}_{a.surface or 'primary'}_modes_input.npz")
            _print({"stage": "result", "input": str(out)})
        elif a.cmd == "modes":
            from .eigen.hodge_dirac import prepare_hodge
            from .hodge import add_hodge
            got = prepare_hodge(sub, n_modes=a.modes, work=Path(a.work), surface=a.surface, progress=_print)
            with sub.composition():
                info = add_hodge(sub, a.work, sub=got["prefix"], replace=a.replace, surface=a.surface)
            _print({"stage": "result", **info, "mesh": got["mesh"], "solve": {k: got[k] for k in ("exact", "coexact", "seconds")}})
        elif a.cmd == "carry-inverse":
            from .convert import carry_inverse
            with sub.composition():
                info = carry_inverse(sub, src=a.src, kernel=a.kernel, session=a.session, src_subject=a.from_subject, name=a.name, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "resample":
            from .timeseries import add_resampled
            with sub.composition():
                info = add_resampled(sub, a.recording, a.to, name=a.name, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "qc":
            from .qc import add_qc
            with sub.composition():
                info = add_qc(sub, megqc=a.megqc, replace=a.replace, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "interp":
            from .interp import add_interpolation
            with sub.composition():
                info = add_interpolation(sub, a.src, a.dst, replace=a.replace)
            _print({"stage": "result", **info})
        elif a.cmd == "map":
            from .results_map import export_map
            with sub.composition():
                wrote = []
                for f in a.file:
                    if not Path(f).is_file():
                        raise FileNotFoundError(f"not a file: {f}")
                    info = export_map(sub, f, bst_root=a.bst_root, session=a.session)
                    wrote.append(info["name"])
                    _print({"stage": "map", **{k: v for k, v in info.items() if k != "id"}})
            _print({"stage": "result", "maps": wrote})
        elif a.cmd == "remove":
            if a.node is None:
                remove_node(ds.db, ds.root, "subject", sub.id)
                write_catalog(ds.root)
                _print({"stage": "result", "removed": a.subject})
            else:
                hit = next(((t, r) for t in ("manifold", "field", "operator", "selection") if (r := sub.find(t, a.node))), None)
                if hit is None:
                    raise KeyError(f"{a.node}: no node of {a.subject}")
                remove_node(ds.db, ds.root, hit[0], hit[1]["id"])
                _print({"stage": "result", "removed": a.node, "table": hit[0]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
