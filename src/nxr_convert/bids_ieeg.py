"""A BIDS-iEEG SUBJECT (BrainVision runs, electrodes in a template space) → a subject, as entities (D141).

    sub-X/[ses-Y/]ieeg/sub-X_task-<t>[_acq-<a>]_run-<r>_ieeg.{vhdr,eeg,vmrk}   the recording (BrainVision, read here with numpy)
                       …_channels.tsv · …_events.tsv · …_ieeg.json            its channels (type, status), events, sampling
    sub-X_space-<space>_electrodes.tsv + _coordsystem.json                     the contacts' positions (mm or m) in the template space

A RUN is a session (its BIDS entities, ``naming.session_label``: ``task-sleep_acq-N2_run-01`` — a vigilance stage is the
``acq``), written as every recording is:

    timeseries/channel_ieeg      Manifold  the channels at their contacts (metres, the space's coordinate system; a bipolar
                                           ``A1-A2`` at its two contacts' midpoint, ``sensors.export_sensors``); shared by the
                                           runs whose channels and electrodes are byte-identical, qualified otherwise (D121)
    timeseries/<run>             Field     float32 [channels × samples] in volts (``timeseries._recording``), its events.tsv as
                                           events (a span per row, code = trial_type), channels.tsv's status as its flags

THE CONTACTS ON THE CORTEX, when the dataset has a TEMPLATE (``ensure_template``: the template derivative's GIFTI surfaces
and parcellation, imported once as the dataset's ``template`` subject) — every contact to its nearest vertex of one template
surface (default ``midthickness``), as fibre ends are (``fibers.py``):

    <channels>_to_<surface>             Operator  csr [V × channels], contacts → the template surface's vertices: a 1 per
                                                  contact within ``max_distance_mm`` (push a channel field through it and it
                                                  lands on the cortex); a farther contact has no entry
    timeseries/<channels>_cortex_distance   Field  of it, on the channels: the distance to that vertex, m (NaN: no position)
    timeseries/<channels>_cortex_vertex     Field  of it, integer: the nearest vertex (-1: no position), on or off the cortex
    timeseries/<channels>_cortex_fsaverage / _fsaverage5 / _fsaverage_distance     Fields, with ``fsaverage_map`` only: the
                                                  contact's nearest vertex looked up in a template-to-fsaverage map (fsaverage
                                                  ico7 index · fsaverage5 index, integer, -1: no position; the map's distance
                                                  from the template vertex to its registration surface, m, NaN: no position)
    timeseries/<channels>_cortex        Selection  a set: ``on_cortex`` (on cortex · off cortex · no position) and, per
                                                  parcellation of that surface, the contact's PARCEL (its nearest vertex's
                                                  code in the template's dictionary; an off-cortex contact ``unassigned``)

Not read: the ``.vmrk`` markers (events.tsv is BIDS' record of them), ``space-Talairach`` electrodes unless ``space``
names it, an individual MRI (there is none).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Dataset, Subject, jdump
from .grid import BLOCK
from .naming import claim_node_name, sanitize_node_name, session_label, source_fingerprint

CHANNELS = "channel_ieeg"
_FORMATS = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "INT_32": "<i4", "UINT_16": "<u2"}
_SCALE = {"µv": 1e-6, "μv": 1e-6, "uv": 1e-6, "mv": 1e-3, "v": 1.0, "nv": 1e-9}
_UNITS_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


# ══ BRAINVISION ═══════════════════════════════════════════════════════════════════════════════════════════════════════

def read_vhdr(path: str | Path) -> dict[str, Any]:
    """A BrainVision header: its sections as dicts (``[Comment]`` and after not read), the channels (name, resolution,
    unit) and what reading the binary needs."""
    p = Path(path)
    b = p.read_bytes()
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        text = b.decode("latin-1")                               # BrainVision's own files are ANSI: µ is 0xB5
    sec: dict[str, dict[str, str]] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            cur = m.group(1).strip()
            if cur.lower() == "comment":
                break
            sec[cur] = {}
        elif cur and "=" in line:
            k, v = line.split("=", 1)
            sec[cur][k.strip()] = v.strip()
    common, binary = sec.get("Common Infos", {}), sec.get("Binary Infos", {})
    if common.get("DataFormat", "BINARY").upper() != "BINARY":
        raise ValueError(f"{p.name}: DataFormat {common.get('DataFormat')} — only BINARY is read")
    fmt = binary.get("BinaryFormat", "").upper()
    if fmt not in _FORMATS:
        raise ValueError(f"{p.name}: BinaryFormat {fmt or '(none)'} — read: {', '.join(_FORMATS)}")
    n = int(common["NumberOfChannels"])
    names, res, units = [], [], []
    for k in range(1, n + 1):
        parts = [x.replace(r"\1", ",") for x in sec.get("Channel Infos", {}).get(f"Ch{k}", f"Ch{k}").split(",")]
        names.append(parts[0])
        res.append(float(parts[2]) if len(parts) > 2 and parts[2] else 1.0)
        units.append(parts[3] if len(parts) > 3 and parts[3] else "µV")
    data = p.parent / common["DataFile"]
    item = np.dtype(_FORMATS[fmt]).itemsize
    n_t = data.stat().st_size // (item * n)
    return {"file": data, "dtype": _FORMATS[fmt], "format": fmt, "orientation": common.get("DataOrientation", "MULTIPLEXED").upper(),
            "n_channels": n, "n_samples": int(n_t), "sfreq": 1e6 / float(common["SamplingInterval"]), "names": names,
            "resolution": np.asarray(res), "units": units, "marker_file": common.get("MarkerFile")}


def volts(units: list[str]) -> np.ndarray:
    """Each channel's unit → its factor to volts (1 for a unit that is not a voltage — it stays in its own)."""
    return np.array([_SCALE.get(u.strip().lower(), 1.0) for u in units])


def read_block(h: dict, s0: int, n: int) -> np.ndarray:
    """Samples ``[s0, s0 + n)`` of every channel, float32 [C × n], in volts (resolution × unit)."""
    mm = np.memmap(h["file"], dtype=h["dtype"], mode="r")
    C, T = h["n_channels"], h["n_samples"]
    x = mm[: C * T].reshape(T, C)[s0:s0 + n].T if h["orientation"] == "MULTIPLEXED" else mm[: C * T].reshape(C, T)[:, s0:s0 + n]
    return (x.astype(np.float64) * (h["resolution"] * volts(h["units"]))[:, None]).astype(np.float32)


# ══ BIDS ══════════════════════════════════════════════════════════════════════════════════════════════════════════════

def tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def sidecar(vhdr: Path, suffix: str) -> Path | None:
    """``…_ieeg.vhdr`` → ``…_<suffix>`` beside it (BIDS' one inheritance level used here: the run's own file)."""
    f = vhdr.with_name(vhdr.name.replace("_ieeg.vhdr", f"_{suffix}"))
    return f if f.is_file() else None


def electrodes(subject_dir: Path, subject: str, space: str) -> tuple[dict[str, np.ndarray], dict[str, str], Path | None, dict]:
    """The contacts' positions in metres by name, their groups, the file, its coordinate-system record — empty without one."""
    hits = sorted(subject_dir.rglob(f"{subject}*_space-{space}_electrodes.tsv"))
    if not hits:
        return {}, {}, None, {}
    if len(hits) > 1:
        raise ValueError(f"{subject}: {len(hits)} electrode files in space {space} — one is read")
    cs_file = hits[0].with_name(hits[0].name.replace("_electrodes.tsv", "_coordsystem.json"))
    cs = json.loads(cs_file.read_text()) if cs_file.is_file() else {}
    unit = str(cs.get("iEEGCoordinateUnits", "mm")).lower()
    if unit not in _UNITS_M:
        raise ValueError(f"{hits[0].name}: coordinates in {unit!r} — read: {', '.join(_UNITS_M)}")
    pos, group = {}, {}
    for r in tsv(hits[0]):
        xyz = np.array([_num(r.get(k)) for k in ("x", "y", "z")]) * _UNITS_M[unit]
        if np.isfinite(xyz).all():
            pos[r["name"]] = xyz
        if r.get("group") not in (None, "", "n/a"):
            group[r["name"]] = r["group"]
    return pos, group, hits[0], cs


def _loc(name: str, pos: dict[str, np.ndarray]) -> np.ndarray | None:
    """A channel's ``Loc``: its contact [3 × 1], or a bipolar ``A-B``'s two [3 × 2]; None without a position."""
    if name in pos:
        return pos[name].reshape(3, 1)
    a, sep, b = name.partition("-")
    if sep and a in pos and b in pos:
        return np.stack([pos[a], pos[b]], axis=1)
    return None


def _events(path: Path | None) -> list[dict]:
    """events.tsv in Brainstorm's ``Events`` shape (``events.parse_events``): one entry a ``trial_type`` (else ``value``)."""
    if path is None:
        return []
    by: dict[str, list[tuple[float, float]]] = {}
    for r in tsv(path):
        on = _num(r.get("onset"))
        if not np.isfinite(on):
            continue
        du = _num(r.get("duration"))
        label = next((r[k] for k in ("trial_type", "value") if r.get(k) not in (None, "", "n/a")), "event")
        by.setdefault(str(label), []).append((on, on + (du if np.isfinite(du) and du > 0 else 0.0)))
    return [{"label": k, "times": np.array(v, dtype=np.float64).T, "epochs": np.ones(len(v))} for k, v in by.items()]


def runs(bids: str | Path, subject: str) -> list[Path]:
    root = Path(bids) / subject
    if not root.is_dir():
        raise FileNotFoundError(f"{root}: no such BIDS subject")
    return sorted(f for f in root.rglob("*_ieeg.vhdr") if f.parent.name == "ieeg")


# ══ THE TEMPLATE ══════════════════════════════════════════════════════════════════════════════════════════════════════

def _gifti():
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("reading the template's GIFTI files needs nibabel: pip install 'nxr-convert[ieeg]'") from e
    return nib


def template_files(folder: str | Path, space: str) -> dict[str, Any]:
    """The template derivative's surfaces (``{name: (L, R)}``, midthickness first) and parcellations (``{atlas: (L, R, tsv)}``)."""
    d = Path(folder)
    surfs: dict[str, tuple[Path, Path]] = {}
    for L in sorted(d.rglob(f"tpl-{space}*_hemi-L_*.surf.gii")):
        R = L.with_name(L.name.replace("_hemi-L_", "_hemi-R_"))
        if R.is_file():
            surfs[L.name.split("_hemi-L_", 1)[1].removesuffix(".surf.gii")] = (L, R)
    atlases = {}
    for L in sorted(d.rglob("*hemi-L_atlas-*_dseg.label.gii")):
        R = L.with_name(L.name.replace("hemi-L_", "hemi-R_"))
        name = re.search(r"_atlas-([A-Za-z0-9]+)_", L.name).group(1)
        t = [f for f in (L.with_suffix("").with_suffix(".tsv"), L.parent / L.name.replace("hemi-L_", "").replace(".label.gii", ".tsv"))
             if f.is_file()]
        if R.is_file():
            atlases[name] = (L, R, t[0] if t else None)
    if not surfs:
        raise FileNotFoundError(f"{d}: no tpl-{space}_hemi-L/R_<surface>.surf.gii pair")
    order = sorted(surfs, key=lambda s: (s != "midthickness", s))
    return {"surfaces": {k: surfs[k] for k in order}, "atlases": atlases}


def _parcellation(nib, L: Path, R: Path, table: Path | None, n_l: int, n_r: int):
    """(codes [V], level names, colours) — code 0 ``unassigned``, then each entry owning a vertex, ``<name> L|R`` as
    ``freesurfer.atlas_members`` names them."""
    names_tsv = {}
    if table is not None:
        for r in tsv(table):
            k = r.get("index", r.get("label"))
            if k not in (None, "", "n/a"):
                names_tsv[int(float(k))] = r.get("name") or r.get("label_name") or str(k)
    codes, levels, colors = np.zeros(n_l + n_r, np.int32), ["unassigned"], [[0, 0, 0]]
    for f, off, n, side in ((L, 0, n_l, "L"), (R, n_l, n_r, "R")):
        img = nib.load(str(f))
        keys = np.asarray(img.darrays[0].data).astype(np.int64).ravel()
        if keys.size != n:
            raise ValueError(f"{f.name}: {keys.size} vertices, the hemisphere has {n}")
        lab = {int(l.key): l for l in img.labeltable.labels}
        for k in np.unique(keys[keys > 0]):
            nm = names_tsv.get(int(k)) or (lab[int(k)].label if int(k) in lab else str(k))
            rgba = lab[int(k)].rgba if int(k) in lab and lab[int(k)].rgba is not None else (0.5, 0.5, 0.5, 1)
            codes[off + np.flatnonzero(keys == k)] = len(levels)
            levels.append(f"{nm} {side}")
            colors.append([int(round(float(c) * 255)) for c in rgba[:3]])
    return codes, levels, np.asarray(colors, np.uint8)


def ensure_template(ds: Dataset, folder: str | Path, *, space: str, emit=None) -> dict:
    """The dataset's TEMPLATE subject (``tpl-<space>``) from the template derivative, once: each surface pair one surface
    (lh then rh, mm → m, the space's coordinate system; midthickness the primary), every parcellation on each. A dataset
    that has a template keeps it."""
    from .subject import stamp
    from .surface import add_surface
    have = ds.template()
    if have is not None:
        return {"template": have["name"], "existing": True}
    nib = _gifti()
    got = template_files(folder, space)
    sub = Subject.create(ds, f"tpl-{space}", kind="template", source_format="gifti", source_path=str(Path(folder).resolve()),
                         template_id=None, created_by=stamp())
    wrote = []
    with sub.composition():
        for k, (name, (L, R)) in enumerate(got["surfaces"].items()):
            (pl, fl), (pr, fr) = [nib.load(str(f)).agg_data(("pointset", "triangle")) for f in (L, R)]
            n_l, n_r = len(pl), len(pr)
            members = [(fam, *_parcellation(nib, aL, aR, t, n_l, n_r)) for fam, (aL, aR, t) in got["atlases"].items()]
            info = add_surface(sub, name=sanitize_node_name(name), folder="sources",
                               vertices=np.concatenate([pl, pr]).astype(np.float64) / 1000.0,
                               faces=np.concatenate([fl, fr + n_l]).astype(np.int64), source=L.name.replace("_hemi-L_", "_hemi-{L,R}_"),
                               comment=f"{space} {name}", hemis=[("left", 0, n_l), ("right", n_l, n_r)], atlas_members=members or None,
                               primary=k == 0, cs_key=space, producer={"template": {"space": space, "units": "m", "nV_lh": n_l, "nV_rh": n_r}})
            wrote.append(info["name"])
            if emit:
                emit({"stage": "surface", "subject": sub.name, "name": info["name"], "n_vertices": info["n_vertices"], "atlases": info["atlases"]})
    return {"template": sub.name, "existing": False, "surfaces": wrote, "atlases": list(got["atlases"])}


# ══ THE SUBJECT ═══════════════════════════════════════════════════════════════════════════════════════════════════════

def _channels(sub: Subject, vhdr: Path, h: dict, chans: list[dict], pos, group, efile: Path | None, space: str, *, session: str) -> dict:
    from .sensors import export_sensors
    by = {r["name"]: r for r in chans}
    missing = [n for n in h["names"] if n not in by]
    if missing:
        raise ValueError(f"{vhdr.name}: channels not in its channels.tsv: {missing[:5]}")
    flags = np.array([-1 if by[n].get("status", "good").lower() == "bad" else 1 for n in h["names"]])
    fp = source_fingerprint([f for f in (sidecar(vhdr, "channels.tsv"), efile) if f], f"{space}|{h['names']}".encode())
    name, shared = claim_node_name(sub, "timeseries", CHANNELS, session, fp)
    if shared:
        return {"name": name, "id": sub.find("manifold", f"timeseries/{name}")["id"], "names": h["names"], "flags": flags, "shared": True}
    rows = [{"Name": n, "Type": by[n].get("type", "MISC").upper(), "Loc": _loc(n, pos), "Group": by[n].get("group") or group.get(n, "")}
            for n in h["names"]]
    rows = [{**r, "Group": "" if r["Group"] == "n/a" else r["Group"]} for r in rows]
    si = export_sensors(sub, {"Channel": rows}, flags, name=name, session=session, source_sha1=fp, cs_key=space,
                        producer={"space": space, **({"electrodes": efile.name} if efile else {}),
                                  "contacts": f"in {space} (metres); a channel without one NaN"})
    return {**si, "flags": flags, "shared": False}


def fsaverage_map(folder: str | Path, space: str, n_l: int, n_r: int) -> dict[str, Any]:
    """A template-to-fsaverage vertex lookup: ``tpl-<space>_to_fsaverage_hemi-{L,R}.npz`` in ``folder``, each indexed by the
    template hemisphere's vertex (``fsaverage_vertex``, ``fsaverage5_vertex``, ``distance_mm``), joined lh then rh as the template
    surface's vertices are; ``files`` names them with their sha256."""
    import hashlib
    d, parts, files = Path(folder), {"fsaverage_vertex": [], "fsaverage5_vertex": [], "distance_mm": []}, {}
    for side, n in (("L", n_l), ("R", n_r)):
        f = d / f"tpl-{space}_to_fsaverage_hemi-{side}.npz"
        if not f.is_file():
            raise FileNotFoundError(f"{f}: no template-to-fsaverage map for hemisphere {side}")
        with np.load(f) as z:
            for k in parts:
                if k not in z.files:
                    raise ValueError(f"{f.name}: no '{k}' array")
                if z[k].shape != (n,):
                    raise ValueError(f"{f.name}: '{k}' has {z[k].size} entries, the template's hemisphere {side} has {n} vertices")
                parts[k].append(np.asarray(z[k]))
        files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {**{k: np.concatenate(v) for k, v in parts.items()}, "files": files}


def contacts_on_cortex(sub: Subject, channels: dict, *, surface: str = "midthickness", max_distance_mm: float = 5.0,
                       fsaverage_dir: str | Path | None = None, space: str = "MNI152NLin2009aSym") -> dict | None:
    """Each contact to its nearest vertex of the dataset template's ``surface``: the operator, the distance and vertex
    Fields and the ``_cortex`` set (module docstring); with ``fsaverage_dir``, also that vertex's fsaverage lookup. None
    without a template holding that surface."""
    import zarr
    from scipy.spatial import cKDTree
    from .entities import label_set
    t = sub.ds.template()
    cx = t and sub.db.one("SELECT * FROM manifold WHERE subject_id = ? AND path = ?", t["id"], f"sources/{surface}")
    if not cx:
        return None
    tpl_store = sub.root / sub.ds.row["path"] / t["path"]
    V = np.asarray(zarr.open_array(str(tpl_store / f"sources/{surface}/vertices"), mode="r")[...], dtype=np.float64)
    fsa = None
    if fsaverage_dir is not None:
        nv = json.loads(cx["producer_json"])["template"]
        fsa = fsaverage_map(fsaverage_dir, space, int(nv["nV_lh"]), int(nv["nV_rh"]))
    P = np.asarray(zarr.open_array(str(sub.at(f"timeseries/{channels['name']}")), mode="r")[...], dtype=np.float64)
    has = np.isfinite(P).all(axis=1)
    dist = np.full(len(P), np.nan)
    vtx = np.full(len(P), -1, dtype=np.int64)
    if has.any():
        dist[has], vtx[has] = cKDTree(V).query(P[has])
    on = has & (dist * 1000 <= max_distance_mm)
    cols = np.flatnonzero(on)
    order = np.argsort(vtx[cols], kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(vtx[cols], minlength=len(V)))]
    ch, session = channels["name"], sub.db.read("manifold", channels["id"])["session"]
    op = f"{ch}_to_{surface}"
    oid = sub.operator(name=op, path=op, kind="contact to cortex", from_manifold_id=channels["id"], to_manifold_id=cx["id"], layout="csr",
                       n_rows=len(V), n_cols=len(P), nnz=int(cols.size), session=session,
                       sparse={"indptr": indptr.astype(np.int32), "indices": cols[order].astype(np.int32), "data": np.ones(cols.size, np.float32)},
                       description=f"each contact -> its nearest vertex of {t['name']}'s {surface} within {max_distance_mm:g} mm (1 per contact); "
                                   "farther contacts omitted",
                       producer_json=jdump({"method": "nearest vertex (scipy cKDTree)", "template": t["name"], "surface": surface,
                                            "max_distance_mm": max_distance_mm, **({"fsaverage_map": fsa["files"]} if fsa else {})}))
    base = f"timeseries/{ch}"
    sub.field(name=f"{ch}_cortex_distance", path=f"{base}_cortex_distance", kind="contact distance", manifold_id=channels["id"],
              data=dist.astype(np.float32), unit="m", of_operator_id=oid, session=session,
              description=f"each contact's distance to its nearest vertex of {surface} (NaN: no position)")
    sub.field(name=f"{ch}_cortex_vertex", path=f"{base}_cortex_vertex", kind="contact vertex", manifold_id=channels["id"], value_type="integer",
              data=vtx.astype(np.int32), of_operator_id=oid, session=session,
              description=f"each contact's nearest vertex of {surface}, on or off the cortex (-1: no position)")
    if fsa:                                      # looked up at the nearest vertex, on or off the cortex like _cortex_vertex
        for key, nm, what in (("fsaverage_vertex", "fsaverage", "fsaverage (ico7) vertex"), ("fsaverage5_vertex", "fsaverage5", "fsaverage5 vertex")):
            sub.field(name=f"{ch}_cortex_{nm}", path=f"{base}_cortex_{nm}", kind="contact vertex", manifold_id=channels["id"], value_type="integer",
                      data=np.where(has, fsa[key][np.where(has, vtx, 0)], -1).astype(np.int32), of_operator_id=oid, session=session,
                      description=f"the {what} of each contact's nearest template vertex, from the template-to-fsaverage map (-1: no position)")
        sub.field(name=f"{ch}_cortex_fsaverage_distance", path=f"{base}_cortex_fsaverage_distance", kind="contact distance",
                  manifold_id=channels["id"], unit="m", of_operator_id=oid, session=session,
                  data=np.where(has, fsa["distance_mm"][np.where(has, vtx, 0)] / 1000.0, np.nan).astype(np.float32),
                  description="the template-to-fsaverage map's distance from each contact's nearest template vertex to the registration surface (NaN: no position)")
    state = np.where(on, 0, np.where(has, 1, 2))
    present = sorted(set(state.tolist()))
    words = ["on cortex", "off cortex", "no position"]
    members = [("on_cortex", np.array([present.index(s) for s in state]), [words[s] for s in present], None)]
    for a in sub.db.all("SELECT s.* FROM selection s JOIN selection p ON p.id = s.member_of_id WHERE p.subject_id = ? AND p.path = ?",
                        t["id"], f"sources/{surface}_atlases"):
        codes = np.asarray(zarr.open_array(str(tpl_store / a["array_path"]), mode="r")[...]).ravel()
        names = {r["code"]: r["name"] for r in sub.db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", a["dictionary_id"])}
        got = [names.get(int(codes[v]), "unassigned") if ok else "unassigned" for v, ok in zip(vtx, on)]
        levels = list(dict.fromkeys(got))
        members.append((a["name"], np.array([levels.index(g) for g in got]), levels, None))
    label_set(sub, f"{base}_cortex", members, manifold_id=channels["id"], session=session,
              description=f"each contact on {t['name']}'s {surface}: on_cortex, and its parcel in " + (", ".join(m[0] for m in members[1:]) or "no parcellation"))
    return {"operator": op, "on_cortex": int(on.sum()), "off_cortex": int((has & ~on).sum()), "no_position": int((~has).sum()),
            "parcellations": [m[0] for m in members[1:]], **({"fsaverage_map": fsa["files"]} if fsa else {})}


def import_bids_ieeg(sub: Subject, bids: str | Path, *, space: str = "MNI152NLin2009aSym", surface: str = "midthickness",
                     max_distance_mm: float = 5.0, fsaverage_dir: str | Path | None = None, progress=None) -> dict:
    """``nxr-convert bids-ieeg``: every run of the BIDS subject named as ``sub`` into it — a run already there (its session)
    is skipped, said so — and each new channel set's contacts on the template."""
    from .convert import session_rows
    from .timeseries import _recording
    emit = progress or (lambda e: None)
    subject = sub.row["name"] if sub.row["name"].startswith("sub-") else f"sub-{sub.row['name']}"
    pos, group, efile, cs = electrodes(Path(bids) / subject, subject, space)
    if efile is None:
        emit({"stage": "no-electrodes", "space": space, "message": "no electrodes in this space: channels without positions, no contacts on the cortex"})
    done, mapped = [], []
    for vhdr in runs(bids, subject):
        rec = sanitize_node_name(vhdr.stem)
        session = session_label(vhdr.stem.removeprefix(subject))
        if session_rows(sub, session):
            emit({"stage": "skip-run", "run": rec, "reason": f"session {session} already in {sub.name}"})
            continue
        h = read_vhdr(vhdr)
        side = json.loads(sidecar(vhdr, "ieeg.json").read_text()) if sidecar(vhdr, "ieeg.json") else {}
        fs = side.get("SamplingFrequency")
        if fs:                                   # the header's interval is µs, often rounded (512 Hz: 1953.125 → 1953.12): the sidecar's rate
            if abs(float(fs) - h["sfreq"]) > 1e-4 * h["sfreq"]:
                raise ValueError(f"{vhdr.name}: the header samples at {h['sfreq']:g} Hz, its ieeg.json says {fs}")
            h["sfreq"] = float(fs)
        chans = tsv(sidecar(vhdr, "channels.tsv")) if sidecar(vhdr, "channels.tsv") else [{"name": n} for n in h["names"]]
        si = _channels(sub, vhdr, h, chans, pos, group, efile, space, session=session)
        emit({"stage": "sensors-shared" if si["shared"] else "sensors", "name": si["name"], "n_channels": len(si["names"])})
        n_t = h["n_samples"]

        def write(arr, h=h, n_t=n_t) -> None:
            for s0 in range(0, n_t, BLOCK):
                arr[:, s0:s0 + BLOCK] = read_block(h, s0, min(BLOCK, n_t - s0))
        unit = "V" if all(u.strip().lower() in _SCALE for u in h["units"]) else None
        ri = _recording(sub, rec, n_chan=h["n_channels"], n_samples=n_t, sfreq=h["sfreq"], t0=0.0, session=session, channels_id=si["id"],
                        write=write, flags=si["flags"], events=_events(sidecar(vhdr, "events.tsv")), channel_names=si["names"], comment=None,
                        kind="sensor", source=vhdr.name, unit=unit,
                        producer={"format": "BrainVision", "binary_format": h["format"], "orientation": h["orientation"],
                                  "units": sorted(set(h["units"])), **{k: side[k] for k in ("TaskName", "iEEGReference", "PowerLineFrequency",
                                  "SEEGChannelCount", "ECOGChannelCount", "RecordingDuration") if k in side}})
        done.append(rec)
        emit({"stage": "timeseries", **{k: ri[k] for k in ("name", "n_samples", "sfreq", "events")}})
        if not si["shared"] and efile is not None:
            m = contacts_on_cortex(sub, si, surface=surface, max_distance_mm=max_distance_mm, fsaverage_dir=fsaverage_dir, space=space)
            if m is None:
                emit({"stage": "no-template", "message": f"the dataset has no template with sources/{surface}: contacts not mapped (--template-surfaces)"})
            else:
                mapped.append(m)
                emit({"stage": "contacts", "channels": si["name"], **m})
    return {"subject": sub.name, "runs": done, "space": space, "electrodes": efile.name if efile else None,
            "coordinate_system": cs.get("iEEGCoordinateSystem"), "contacts": mapped}
