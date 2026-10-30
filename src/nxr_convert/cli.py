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
    nxr-convert carry-hodge --dataset D --subject S --from <dataset> [--replace]   the primary cortex's Hodge-Dirac basis from another store of the subject
    nxr-convert resample --dataset D --subject S --recording NAME --to HZ [--name N]  a stored recording resampled, beside it (``timeseries.py``)
    nxr-convert surface  --dataset D --subject S --fs-dir <FreeSurfer subject dir> --name N [--scs-from tess_*.mat]
                                                                                      add FreeSurfer's lh/rh.pial as one surface
    nxr-convert modes    --dataset D --subject S --work DIR [--modes K] [--surface NAME] [--replace]   SOLVE the Hodge–Dirac basis from the store
                         (the Editor's glue by anatomy, intrinsic Delaunay, d + δ; ``eigen/``) and store it on the glued surface;
                         --export: only the solve's input (to run elsewhere); --input FILE: solve that input (no dataset), store with `hodge`
    nxr-convert transfermatrix --dataset D --subject S --subject-solve DIR --template-solve DIR --modes K --out DIR [--template T] [--check]
                         REGISTER a subject to its template: T carries its Hodge–Dirac coefficients onto the template's (``eigen/register.py``)
    nxr-convert registerkernel --dataset D --subject S --kernel NAME --subject-solve DIR --template-solve DIR --modes K --out DIR
    nxr-convert rotationplane --dataset D --subject S --kernel K --recording R [--band 8 12]   the kernel's top rotation plane, as two sensor filters
    nxr-convert carry --dataset D --subject S (--field NAME | --selection NAME)   a native map or points carried to the template
                         a kernel REGISTERED onto the template (fuse, transfer, synthesise) and stored on the subject (``eigen/registered.py``)
    nxr-convert hodge    --dataset D --subject S --from <nsp rheome-hodge-dirac dir> [--nsp-subject S] [--replace] [--surface NAME]
                                                                                      add the Hodge–Dirac eigenmodes of the primary
                                                                                      (or of the named surface, suffixed) (``hodge.py``)
    nxr-convert interp   --dataset D --subject S --from SRC --to DST [--replace]     the operator DST ← SRC through the spheres (``interp.py``)
    nxr-convert templates <references-fsaverage-ico7 dir> --dataset D [--subject fsaverage] [--trk hcp1065.trk]
                                                                                      the templates-and-references dataset (``templates.py``)
    nxr-convert map      --dataset D --subject S --file results_*.mat|matrix_*.mat … --bst-root P [--session C]
                                                                                      add static maps (surface or region)
    nxr-convert qc       --dataset D --subject S [--megqc <sub>.tar.gz] [--line-freq HZ] [--replace]  sensor QC of the recordings (``qc.py``)
    nxr-convert migrate  --from nxr.sqlite --dataset D                              a schema-48 shared database into a dataset at this schema
    nxr-convert fragment --dataset D (--subject S | --new) --to DIR                  one subject's writable copy, for a parallel job
    nxr-convert merge    --dataset D FRAGMENT … [--replace]                          fold fragments back, serially (``shared_dataset.py``)
    nxr-convert badsegments --dataset D --subject S --from JSON [--replace]       bad segments detected after the conversion (``events.py``)
    nxr-convert mne-eeg  --dataset D --subject S --from DIR --fsaverage DIR [--nsp-subject P]
                                                                                      MNE's EEG source package (nsp mne-eeg-source): EEG, events,
                                                                                      kernels on fsaverage's ico-5 cortex, band power (``mne_eeg.py``)
    nxr-convert bids-ieeg --dataset D --subject S --bids ROOT [--template-surfaces DIR] [--surface N] [--max-distance-mm D]
                                                                                      a BIDS-iEEG subject; its contacts on the template (``bids_ieeg.py``)
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


def _transfer(a) -> int:
    """`nxr-convert transfermatrix`: the two surfaces and spheres from the store, the two solves from their folders; T written for the Editor"""
    import zarr
    import numpy as np
    from .crud import Subject, open_dataset
    from .eigen.register import Registration, check_transfer, load_solve, transfer_matrix, write_transfer
    def surface(sub, name):
        cx = sub.db.one("SELECT m.*, g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND "
                        + ("m.is_primary = 1" if name is None else "m.type = 'surface' AND m.name = ?"), sub.id, *([] if name is None else [name]))
        if cx is None:
            raise KeyError(f"{sub.row['name']}: no surface {name or '(primary)'}")
        rd = lambda p: np.asarray(zarr.open_array(str(sub.at(p)), mode="r")[...])
        F = rd(sub.db.one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", cx["topology_id"])["path"]).astype(np.int64)
        sph = sub.db.one("SELECT m.*, g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.name = ?",
                         sub.id, f"{cx['name']}_sphere")
        if sph is None:
            raise KeyError(f"{sub.row['name']}: {cx['name']} has no registered sphere ({cx['name']}_sphere)")
        return cx["name"], rd(cx["positions_path"]).astype(np.float32).astype(np.float64), F, rd(sph["positions_path"]).astype(np.float64)
    with open_dataset(Path(a.dataset), lock=False) as ds:
        s = Subject.open(ds, a.subject)
        tname = a.template or (ds.db.one("SELECT name FROM subject WHERE id = ?", s.row["template_id"]) or {}).get("name")
        if not tname:                                     # no template on the subject's row: the dataset's one template subject
            tpl = ds.db.all("SELECT name FROM subject WHERE dataset_id = ? AND kind = 'template'", ds.id)
            if len(tpl) != 1:
                raise KeyError(f"{a.subject}: no template subject on its row and {len(tpl)} in the dataset (name one with --template)")
            tname = tpl[0]["name"]
        t = Subject.open(ds, tname)
        _, Ps, Fs, Ss = surface(s, a.subject_surface)
        tsurf, Pt, Ft, St = surface(t, a.template_surface)
    sub_solve = load_solve(Path(a.subject_solve), a.subject_prefix or a.subject, a.modes)
    tpl_solve = load_solve(Path(a.template_solve), a.template_prefix or tname, a.modes)
    reg = Registration(Ps, Fs, Ss, Pt, Ft, St, subject_left=a.subject_left, template_left=a.template_left, sampling=a.sampling)
    T, R = transfer_matrix(sub_solve, tpl_solve, reg)
    out = write_transfer(Path(a.out), T, sub_solve, tpl_solve, subject_name=a.subject, template_name=tname, template_surface=tsurf)
    res = {"stage": "result", "transfer": str(out), "K": a.modes, "sampling": a.sampling, "R": list(R.shape), "unregistered_rows": int((R.getnnz(1) == 0).sum()),
           "from": sub_solve.sha, "to": tpl_solve.sha}
    if a.check:
        rows = check_transfer(T, R, sub_solve, tpl_solve, reg)
        res["check"] = rows
        res["gate"] = all(r["gate"] for r in rows)
        res["target"] = all(r["target"] for r in rows)
    _print(res)
    return 0


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

    ptr = sub.add_parser("transfermatrix", help="REGISTER a subject to its template by the TRANSFER MATRIX of their Hodge-Dirac modes (nxr_convert.eigen.register)")
    ptr.add_argument("--dataset", required=True)
    ptr.add_argument("--subject", required=True)
    ptr.add_argument("--template", default=None, help="the template subject (default: the subject's own template, the dataset's default subject)")
    ptr.add_argument("--subject-solve", required=True, help="the subject's `modes` solve folder")
    ptr.add_argument("--template-solve", required=True, help="the template's `modes` solve folder")
    ptr.add_argument("--subject-prefix", default=None, help="file prefix in the subject's solve (default: the subject's name)")
    ptr.add_argument("--template-prefix", default=None, help="file prefix in the template's solve (default: the template's name)")
    ptr.add_argument("--subject-surface", default=None, help="the subject's surface (default: its primary); its sphere is <surface>_sphere")
    ptr.add_argument("--template-surface", default=None, help="the template's surface (default: its primary)")
    ptr.add_argument("--subject-left", default="+y", help="the axis the subject's left hemisphere lies on (+y: Brainstorm SCS)")
    ptr.add_argument("--template-left", default="-x", help="the axis the template's left hemisphere lies on (-x: MNI, fsaverage)")
    ptr.add_argument("--sampling", default="barycentric", choices=("barycentric", "nearest"),
                     help="on the registered spheres: barycentric (default) or nearest (FreeSurfer's mri_surf2surf default)")
    ptr.add_argument("--modes", type=int, required=True, help="K: the first K modes of each solve (the Editor loads exactly these)")
    ptr.add_argument("--out", required=True, help="where hodge-to-<template>.f32/.json go")
    ptr.add_argument("--check", action="store_true", help="also the spherical-harmonic self-check (gate err_reg <= 0.05)")

    pca = sub.add_parser("carry", help="CARRY a subject's measurement to its template: a scalar map (sampled through the registered spheres) or points "
                         "(each to the nearest template vertex), measured on the subject's own cortex; stored on the template (nxr_convert.eigen.carry)")
    pca.add_argument("--dataset", required=True)
    pca.add_argument("--subject", required=True)
    what = pca.add_mutually_exclusive_group(required=True)
    what.add_argument("--field", help="a scalar map on the subject's surface, by name")
    what.add_argument("--selection", help="a point selection (vertex indices) on the subject's surface, by name")
    pca.add_argument("--template", default=None, help="the template subject (default: the subject's template, else the dataset's one)")
    pca.add_argument("--subject-surface", default=None)
    pca.add_argument("--template-surface", default=None)
    pca.add_argument("--subject-left", default="+y")
    pca.add_argument("--template-left", default="-x")
    pca.add_argument("--sampling", default="barycentric", choices=("barycentric", "nearest"))
    pca.add_argument("--replace", action="store_true")
    prp = sub.add_parser("rotationplane", help="a kernel's top rotation plane in a band, as two SENSOR filters (the Editor's Virtual sensors read it); "
                         "stored on the subject as <kernel>_rotationPlane (nxr_convert.eigen.plane)")
    prp.add_argument("--dataset", required=True)
    prp.add_argument("--subject", required=True)
    prp.add_argument("--kernel", required=True)
    prp.add_argument("--recording", required=True, help="the recording's name")
    prp.add_argument("--band", type=float, nargs=2, default=(8.0, 12.0))
    prp.add_argument("--replace", action="store_true")
    prk = sub.add_parser("registerkernel", help="REGISTER a subject's inverse kernel onto its template: fused into the Hodge-Dirac modes, carried by "
                         "the transfer matrix, synthesised on the template; stored on the subject (nxr_convert.eigen.registered)")
    prk.add_argument("--dataset", required=True)
    prk.add_argument("--subject", required=True)
    prk.add_argument("--kernel", required=True, help="the subject's inverse kernel, by name")
    prk.add_argument("--template", default=None, help="the template subject (default: the subject's template, else the dataset's one)")
    prk.add_argument("--subject-solve", required=True, help="the subject's solve folder, or `store`: its stored basis (the mass rebuilt from the glued mesh; no solve)")
    prk.add_argument("--template-solve", required=True)
    prk.add_argument("--subject-prefix", default=None)
    prk.add_argument("--template-prefix", default=None)
    prk.add_argument("--subject-surface", default=None)
    prk.add_argument("--template-surface", default=None)
    prk.add_argument("--subject-left", default="+y")
    prk.add_argument("--template-left", default="-x")
    prk.add_argument("--sampling", default="barycentric", choices=("barycentric", "nearest"))
    prk.add_argument("--flow", default="fields", choices=("fields", "potentials"),
                     help="carry the flow modes as unit flow fields (default: the template's size does not steepen them) or as potentials")
    prk.add_argument("--modes", type=int, required=True)
    prk.add_argument("--out", required=True, help="where the transfer files are written")
    prk.add_argument("--replace", action="store_true")

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
    pmap.add_argument("--replace", action="store_true", help="a source map of the same name is replaced (its record too)")
    pmap.add_argument("--provenance", default=None, help="a JSON file: what made the map(s), stored in each map's producer_json "
                      "(over the file's own Provenance field)")

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
    pch = sub.add_parser("carry-hodge", help="the primary cortex's Hodge-Dirac basis CARRIED from another dataset holding the same subject on the "
                         "same cortex (its stored modes, μ, support, glued surface), stored as `hodge --from` stores a solve")
    pch.add_argument("--dataset", required=True)
    pch.add_argument("--subject", required=True)
    pch.add_argument("--from", dest="src", required=True, help="the other dataset folder (<datastore>/<dataset>)")
    pch.add_argument("--from-subject", default=None, help="the subject's name there (default: the same)")
    pch.add_argument("--replace", action="store_true", help="replace a basis already stored (moved aside, never deleted)")
    prs = sub.add_parser("resample", help="a STORED recording resampled to HZ (resample_poly) as a recording beside it — no protocol read")
    prs.add_argument("--dataset", required=True)
    prs.add_argument("--subject", required=True)
    prs.add_argument("--recording", required=True, help="the recording's field name")
    prs.add_argument("--to", type=float, required=True, metavar="HZ")
    prs.add_argument("--name", default=None, help="the new recording's name (default: <recording>_<HZ>Hz)")
    pbs = sub.add_parser("badsegments", help="BAD SEGMENTS detected after the conversion (a JSON per subject) as <rec>_bad_segments, onto each recording "
                         "and the recordings derived from it (run it after resample)")
    pbs.add_argument("--dataset", required=True)
    pbs.add_argument("--subject", required=True)
    pbs.add_argument("--from", dest="src", required=True, help="{recordings: [{name, segments: [{label, start_s[], stop_s[]}]}]}")
    pbs.add_argument("--replace", action="store_true")
    pmn = sub.add_parser("mne-eeg", help="MNE-Python's EEG source package (nsp mne-eeg-source) into a subject (made if absent): the fsaverage "
                         "ico-5 cortex its kernels index, the EEG channels and recording, its annotations as events, the MNE / dSPM / eLORETA "
                         "kernels, the per-stage band power — no source time series (needs nxr-convert[mne])")
    pmn.add_argument("--dataset", required=True)
    pmn.add_argument("--subject", required=True)
    pmn.add_argument("--from", dest="src", required=True, help="the package's folder (<p>_task-<t>_desc-preproc_eeg.fif, <p>_kernel.json, …)")
    pmn.add_argument("--fsaverage", required=True, help="the fsaverage subject folder MNE ships (surf/, label/, bem/fsaverage-ico-5-src.fif)")
    pmn.add_argument("--nsp-subject", default=None, help="the package's file prefix (default: the subject)")
    pie = sub.add_parser("bids-ieeg", help="a BIDS-iEEG subject (BrainVision runs, channels, MNI electrodes, events) into a subject (made if "
                         "absent); with a template surface, each contact's nearest vertex, its distance and its parcel")
    pie.add_argument("--dataset", required=True)
    pie.add_argument("--subject", required=True, help="the BIDS subject (sub-<label>); also the subject's name")
    pie.add_argument("--bids", required=True, help="the BIDS root")
    pie.add_argument("--space", default="MNI152NLin2009aSym", help="the electrodes' space-<label> (default MNI152NLin2009aSym)")
    pie.add_argument("--template-surfaces", default=None, help="the template derivative's folder (tpl-<space>_hemi-{L,R}_<surface>.surf.gii, "
                     "hemi-{L,R}_atlas-<atlas>_dseg.label.gii + .tsv): imported once as the dataset's template (needs nibabel)")
    pie.add_argument("--surface", default="midthickness", help="the template surface the contacts are mapped onto (default midthickness)")
    pie.add_argument("--max-distance-mm", type=float, default=5.0, help="a contact farther from that surface is not on the cortex (default 5)")
    pq = sub.add_parser("qc", help="sensor QC of a subject's recordings: the PSD Field, its measures and qc_flag, meg-qc's metrics")
    pq.add_argument("--dataset", required=True)
    pq.add_argument("--subject", required=True)
    pq.add_argument("--megqc", default=None, help="meg-qc's <sub>.tar.gz (its session metrics become measurements of the rest runs)")
    pq.add_argument("--line-freq", type=float, default=60.0, help="the mains frequency in Hz (line_residual_db at its harmonics; default 60)")
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
    if a.cmd == "transfermatrix":
        try:
            return _transfer(a)
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
        if a.cmd in ("mne-eeg", "bids-ieeg"):
            if a.cmd == "bids-ieeg" and a.template_surfaces:
                from .bids_ieeg import ensure_template
                _print({"stage": "template", **ensure_template(ds, a.template_surfaces, space=a.space, emit=_print)})
            fmt = "mne" if a.cmd == "mne-eeg" else "bids-ieeg"
            sub = (Subject.open(ds, a.subject, created_by=stamp()) if ds.subject(a.subject)
                   else Subject.create(ds, a.subject, source_format=fmt, source_path=str(Path(a.src if fmt == "mne" else a.bids).resolve()),
                                       created_by=stamp()))
            with sub.composition():
                if a.cmd == "mne-eeg":
                    from .mne_eeg import import_mne_eeg
                    r = import_mne_eeg(sub, a.src, fsaverage=a.fsaverage, prefix=a.nsp_subject, progress=_print)
                else:
                    from .bids_ieeg import import_bids_ieeg
                    r = import_bids_ieeg(sub, a.bids, space=a.space, surface=a.surface, max_distance_mm=a.max_distance_mm, progress=_print)
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
        elif a.cmd in ("registerkernel", "carry"):
            from .eigen.registered import register_kernel
            from .eigen.carry import carry
            tname = a.template or (ds.db.one("SELECT name FROM subject WHERE id = ?", sub.row["template_id"]) or {}).get("name")
            if not tname:
                tpl_rows = ds.db.all("SELECT name FROM subject WHERE dataset_id = ? AND kind = 'template'", ds.id)
                if len(tpl_rows) != 1:
                    raise KeyError(f"{a.subject}: no template on its row and {len(tpl_rows)} in the dataset (--template)")
                tname = tpl_rows[0]["name"]
            tpl = Subject.open(ds, tname, created_by=stamp())
            if a.cmd == "carry":
                with tpl.composition():
                    info = carry(sub, tpl, field=a.field, selection=a.selection, subject_surface=a.subject_surface, template_surface=a.template_surface,
                                 subject_left=a.subject_left, template_left=a.template_left, sampling=a.sampling, replace=a.replace)
            else:
                with sub.composition():
                    info = register_kernel(sub, tpl, kernel=a.kernel, subject_solve=Path(a.subject_solve), template_solve=Path(a.template_solve), modes=a.modes,
                                           out=Path(a.out), subject_prefix=a.subject_prefix, template_prefix=a.template_prefix, subject_surface=a.subject_surface,
                                           template_surface=a.template_surface, subject_left=a.subject_left, template_left=a.template_left,
                                           sampling=a.sampling, flow=a.flow, replace=a.replace, progress=_print)
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
        elif a.cmd == "rotationplane":
            from .eigen.plane import store_rotation_plane
            with sub.composition():
                info = store_rotation_plane(sub, kernel=a.kernel, recording=a.recording, band=a.band, replace=a.replace)
            _print({"stage": "result", **info})
        elif a.cmd == "carry-inverse":
            from .convert import carry_inverse
            with sub.composition():
                info = carry_inverse(sub, src=a.src, kernel=a.kernel, session=a.session, src_subject=a.from_subject, name=a.name, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "carry-hodge":
            from .hodge import carry_hodge
            with sub.composition():
                info = carry_hodge(sub, a.src, src_subject=a.from_subject, replace=a.replace)
            _print({"stage": "result", **info})
        elif a.cmd == "resample":
            from .timeseries import add_resampled
            with sub.composition():
                info = add_resampled(sub, a.recording, a.to, name=a.name, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "badsegments":
            from .events import import_bad_segments
            with sub.composition():
                info = import_bad_segments(sub, json.loads(Path(a.src).read_text()), replace=a.replace)
            _print({"stage": "result", **info})
        elif a.cmd == "qc":
            from .qc import add_qc
            with sub.composition():
                info = add_qc(sub, megqc=a.megqc, replace=a.replace, line_freq=a.line_freq, progress=_print)
            _print({"stage": "result", **info})
        elif a.cmd == "interp":
            from .interp import add_interpolation
            with sub.composition():
                info = add_interpolation(sub, a.src, a.dst, replace=a.replace)
            _print({"stage": "result", **info})
        elif a.cmd == "map":
            from .results_map import export_map
            prov = json.loads(Path(a.provenance).read_text()) if a.provenance else None
            with sub.composition():
                wrote = []
                for f in a.file:
                    if not Path(f).is_file():
                        raise FileNotFoundError(f"not a file: {f}")
                    info = export_map(sub, f, bst_root=a.bst_root, session=a.session, replace=a.replace, provenance=prov)
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
