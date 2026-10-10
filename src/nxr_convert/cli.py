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
    pf.add_argument("--file", required=True, nargs="+")
    pf.add_argument("--bst-root", default="")

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

    pr = sub.add_parser("remove", help="delete a subject, or one node of it (by its store-relative path)")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--subject", required=True)
    pr.add_argument("--node", default=None)

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
            r = convert_subject(ds, a.root, a.subject, emit=_print)
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
                                      all_surfaces=a.all_surfaces, progress=_print, **plan)
            write_catalog(ds.root)
            _print({"stage": "result", **r})
            return 0
        sub = Subject.open(ds, a.subject, created_by=stamp())
        if a.cmd == "surface":
            from .surface import export_surface
            with sub.composition():
                wrote = []
                for f in a.file:
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
