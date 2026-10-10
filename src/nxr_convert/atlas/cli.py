"""``nxr-convert atlas …`` — the dyadic atlas producer's subcommands (wired into ``nxr_convert.cli``).

    nxr-convert atlas default-subject DATASET --templates DIR [--default fsaverage5] [--options fsaverage6,fsaverage]
        [--hcp-trk hcp1065.trk --hcp-bst-dir <brainstorm @default_subject>] [--depth 8] [--frame-level -18]
    nxr-convert atlas build DATASET --subject S [--replace] [--trees subject,group] [--kernels all|constrained|free|latest|<method|stamp|name|glob>,…]
        [--frames-domain sphere|cortex] [--lc-levels 4,6] [--trajectory 4,2] [--recordings N] [--no-maps] [--no-meg]
    nxr-convert atlas reduce DATASET [--from OTHER_DATASET …] [--kernels …]
    nxr-convert atlas info DATASET

DATASET is ``<datastore>/<dataset>`` (its ``dataset.sqlite``): every atlas is rows of it and plain arrays (D129–D134).

Progress is JSON-lines on stdout (``stage`` keyed), as every other subcommand.
"""
from __future__ import annotations

import json


def _print(obj: dict) -> None:
    print(json.dumps(obj, default=str), flush=True)


def add_parser(sub) -> None:
    p = sub.add_parser("atlas", help="the dyadic atlas: the default subject, a subject's atlases, the group sums (D127–D137)")
    s = p.add_subparsers(dest="atlas_cmd", required=True)

    d = s.add_parser("default-subject", help="write the dataset's default subject (fsaverage + group tree + frames + HCP-1065)")
    d.add_argument("dataset", help="<datastore>/<dataset> (holding its dataset.sqlite)")
    d.add_argument("--templates", required=True, help="folder holding fsaverage5 / fsaverage6 / fsaverage (FreeSurfer subjects)")
    d.add_argument("--default", default="fsaverage5", help="the default resolution (the subjects' resolution)")
    d.add_argument("--options", default="fsaverage6,fsaverage", help="option resolutions ('' for none)")
    d.add_argument("--hcp-trk", default=None, help="DSI Studio HCP-1065 tractography (.trk)")
    d.add_argument("--hcp-bst-dir", default=None,
                   help="Brainstorm @default_subject holding tess_fibers_tess_hcp1065.mat and tess_cortex_white_high.mat")
    d.add_argument("--gate-mm", type=float, default=5.0)
    d.add_argument("--lc-levels", default="4,6", help="levels of the Levi-Civita connection")
    d.add_argument("--depth", type=int, default=8, help="the default leaf depth of the dataset's atlases")
    d.add_argument("--frame-level", type=int, default=-18, help="the base frame: a cycle of this tower level (86 400·2^L s; −18 ≈ 0.33 s)")

    b = s.add_parser("build", help="write one subject's atlases (rows of its dataset, arrays in its store)")
    b.add_argument("dataset", help="<datastore>/<dataset>")
    b.add_argument("--subject", required=True)
    b.add_argument("--replace", action="store_true", help="rebuild: remove the subject's previous atlas first")
    b.add_argument("--trees", default="subject,group")
    b.add_argument("--kernels", default="all", help="comma tokens, kinds AND: all | constrained | free | latest (newest stamp a method) | kernel method, "
                        "stamp, name or name glob (e.g. dSPM-unscaled_MEG,constrained,latest or 261005_2149)")
    b.add_argument("--frames-domain", choices=["sphere", "cortex"], default="sphere")
    b.add_argument("--lc-levels", default=None, help="levels with explicit Levi-Civita reductions (default: the dataset's)")
    b.add_argument("--trajectory", default=None, help="SPACE,TIME levels of the peak trajectories (default: the dataset's)")
    b.add_argument("--recordings", type=int, default=None, help="only the first N recordings (testing)")
    b.add_argument("--chunk-frames", type=int, default=4, help="frames projected at a time (memory only)")
    b.add_argument("--no-maps", action="store_true")
    b.add_argument("--no-meg", action="store_true")

    r = s.add_parser("reduce", help="sum the subjects' group-tree atlases into the default subject's atlas (D131)")
    r.add_argument("dataset", help="<datastore>/<dataset> whose default subject receives the group sums")
    r.add_argument("--from", dest="sources", nargs="+", default=[], metavar="DATASET",
                   help="other datasets whose subjects are summed too (read only; same default subject's group tree)")
    r.add_argument("--kernels", default="all", help="the members' kernels summed, as build's --kernels; a recording left with two "
                                                    "kernels of one method and orientation is refused")

    i = s.add_parser("info", help="the dataset's default atlas definitions and its atlas rows")
    i.add_argument("dataset")


def run(a) -> int:
    log = lambda msg: _print({"stage": "log", "message": msg})
    try:
        if a.atlas_cmd == "default-subject":
            from .default_subject import build_default_subject
            r = build_default_subject(a.dataset, a.templates, default=a.default,
                                      options=tuple(x for x in a.options.split(",") if x),
                                      hcp_trk=a.hcp_trk, hcp_bst_dir=a.hcp_bst_dir, gate_mm=a.gate_mm,
                                      lc_levels=tuple(int(x) for x in a.lc_levels.split(",") if x.strip()),
                                      depth=a.depth, frame_level=a.frame_level, log=log)
        elif a.atlas_cmd == "build":
            from .build import build_subject
            r = build_subject(a.dataset, a.subject, trees=tuple(a.trees.split(",")), replace=a.replace,
                              maps=not a.no_maps, meg=not a.no_meg, kernels=a.kernels, frames_domain=a.frames_domain,
                              lc_levels=tuple(int(x) for x in a.lc_levels.split(",") if x.strip()) if a.lc_levels is not None else None,
                              trajectory=tuple(int(x) for x in a.trajectory.split(",")) if a.trajectory else None,
                              recordings=a.recordings, chunk_frames=a.chunk_frames, log=log)
        elif a.atlas_cmd == "reduce":
            from .reduce import reduce_dataset
            r = reduce_dataset(a.dataset, sources=tuple(a.sources), kernels=a.kernels, log=log)
        elif a.atlas_cmd == "info":
            from ..crud import open_dataset
            from .atlas_store import definition
            with open_dataset(a.dataset, lock=False) as ds:      # a reader: no writer lock, no recovery (#26)
                rows = ds.db.all("SELECT s.name AS subject, count(*) AS partitions, sum(json_extract(x.params_json, '$.joint') IS NOT NULL) AS joint "
                                 "FROM selection x JOIN subject s ON s.id = x.subject_id WHERE json_extract(x.params_json, '$.atlas') IS NOT NULL "
                                 "GROUP BY s.name")
                r = {"definition": definition(ds), "subjects": rows}
        else:                                                   # argparse refuses anything else
            return 2
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        _print({"stage": "error", "message": f"{type(e).__name__}: {e}"})
        return 1
    _print({"stage": "result", **r})
    return 0
