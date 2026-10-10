"""THE SUBJECT'S MRI — Brainstorm ``subjectimage_*.mat`` → ``mri/<name>`` entities (``write_mri``).

WHAT MAKES IT WORTH IMPORTING is not the voxels — it is that the registration
arrives WITH them. ``SCS`` is the voxel→SCS transform, and the cortex surfaces
are already stored in SCS, so importing the pair puts the slices and the cortex
in ONE frame. Without it this is a standalone image stack; with it, it is this
subject's brain.

THE DEFECT THIS MODULE EXISTS TO PREVENT is a left-right mirror. A transposed or
sign-flipped axis renders a perfectly plausible brain, and radiological vs
neurological orientation is a real convention people argue about — so a reviewer
looking at a mirrored volume has no reason to call it wrong. Nothing downstream
can catch it either. The fiducials are in the same file as the cube, so there is
no reason to take left-vs-right on trust: ``verify_scs`` checks it and the import
REFUSES rather than warning. A mirrored brain that imported successfully is worse
than no import.

The convention, measured against the OMEGA T1 on 2026-08-20 and asserted here:

    scs_mm = R @ (voxel * voxsize) + T

    Origin -> (0, 0, 0)          exact
    NAS    -> (+103.8,  0,   0)   +x is ANTERIOR
    LPA    -> (  0, +75.1, 0)     +y is LEFT
    RPA    -> (  0, -75.1, 0)     and all three fiducials sit at z = 0
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .crud import Subject, array_meta, jdump, lattice_counts
from .matio import load_mat

#: Cubic CHUNKS, not the byte-budget policy's slabs along axis 0.
#:
#: `default_chunks` chunks axis 0, which for a volume means a SAGITTAL slice
#: reads the entire array. Cubic chunks make any orthogonal slice touch a plane
#: of 4x4 = 16 chunks (~4 MB at uint8). It changes nothing locally — the viewer
#: uploads the whole volume once — and it is the difference between a usable and
#: an unusable remote root.
CHUNK = 64

#: How far a fiducial may miss its expected axis before the import is refused.
#:
#: 0.01 mm — a hundredth of a voxel, far below anything anatomically meaningful
#: and far above the rounding in a transform stored at reduced precision.
#:
#: It was 1e-6 first, which is not a tolerance on ORIENTATION at all but an audit
#: of float precision: the OMEGA T1's own matrices satisfy it exactly, and the
#: same matrices rounded to 8 decimals miss by 5e-3. A gate that rejects a
#: legitimately-rounded registration fails for the wrong reason, and the failures
#: this exists to catch — a mirror, a transposed axis — are off by tens of
#: millimetres or carry the wrong sign outright.
TOL_MM = 1e-2


@dataclass(frozen=True)
class Mri:
    """One Brainstorm subject image, in the shape this module writes."""
    name: str
    kind: str                       # 'anatomical' | 'atlas' | 'scalar'
    #: anatomical: uint8 intensities. atlas: compact CODES 0..N-1 (uint8 when N <= 256,
    #: else uint16) — the row of the label table, which is how the database numbers a
    #: dictionary (D123). scalar: float32 values (a PET mean or SUVR resliced to the T1).
    cube: np.ndarray                # [ni, nj, nk]
    voxsize: np.ndarray             # [3] float, mm
    scs: dict[str, np.ndarray]
    ncs: dict[str, np.ndarray]
    comment: str
    source: str
    #: Atlas only: ids [N] int32 (= the codes 0..N-1), source_ids [N] int32 (Brainstorm's
    #: label values), names [N] str, colors [N,3] uint8.
    labels: dict[str, Any] | None = None


def mri_name(path: str | Path, comment: str = "") -> str:
    """``subjectimage_MRI_T1.mat`` -> ``T1``; ``subjectimage_ASEG_volatlas.mat``
    -> ``ASEG``.

    Derived from the FILENAME rather than the ``Comment``, so a re-import
    overwrites the node it wrote last time instead of accumulating a second one
    beside it under a comment somebody edited in the Brainstorm GUI.
    """
    stem = Path(path).stem
    stem = re.sub(r"^subjectimage_", "", stem)
    stem = re.sub(r"_?volatlas$", "", stem)
    stem = re.sub(r"^MRI_", "", stem)
    from .naming import sanitize_node_name
    # A volume Brainstorm saved from a script is named only by its creation time
    # (`subjectimage_260928_1516.mat`); its identity is then the Comment
    # (`PET 18FNAV4694_suvr`), which is the one name that says what it is.
    if re.fullmatch(r"\d{6}_\d{4}(_\d+)?", stem) and comment.strip():
        return sanitize_node_name(comment)
    return sanitize_node_name(stem) if stem else "MRI"


def _vec(obj: Any, field: str) -> np.ndarray:
    return np.asarray(getattr(obj, field) if hasattr(obj, field) else obj[field], dtype=float).ravel()


def _mat(obj: Any, field: str) -> np.ndarray:
    return np.asarray(getattr(obj, field) if hasattr(obj, field) else obj[field], dtype=float).reshape(3, 3)


def to_scs(voxel: np.ndarray, scs: dict[str, np.ndarray], voxsize: np.ndarray) -> np.ndarray:
    """Voxel index -> SCS millimetres. The one definition; the reader mirrors it."""
    v = np.asarray(voxel, dtype=float).reshape(-1, 3)
    return v * np.asarray(voxsize, dtype=float) @ np.asarray(scs["R"], dtype=float).T + np.asarray(scs["T"], dtype=float)


def verify_scs(scs: dict[str, np.ndarray], voxsize: np.ndarray) -> None:
    """The four gates of the design's §3.2. Raises; never warns.

    Gate 2 is the one that matters and the reason the others are cheap company:
    LPA must land on +y and RPA on -y. Everything else about a mirrored volume
    looks right.
    """
    # The FIDUCIALS are required and `Origin` is not: Brainstorm writes an
    # `Origin` on a T1 and omits it on a volume atlas sharing the same grid. It
    # is a redundant convenience (it is R^-1 · -T), and refusing an import for
    # want of a derivable value would have blocked every volatlas — while the
    # gate that actually matters, the mirror check below, needs only the three
    # fiducials.
    missing = [k for k in ("R", "T", "NAS", "LPA", "RPA") if k not in scs]
    if missing:
        raise ValueError(f"SCS is missing {missing} — cannot verify orientation, so cannot import")

    if "Origin" in scs:
        o = to_scs(scs["Origin"], scs, voxsize)[0]
        if np.linalg.norm(o) > TOL_MM:
            raise ValueError(f"SCS origin does not map to the SCS origin: {np.round(o, 4).tolist()} mm")

    nas, lpa, rpa = (to_scs(scs[k], scs, voxsize)[0] for k in ("NAS", "LPA", "RPA"))

    if nas[0] <= 0:
        raise ValueError(f"NAS is not anterior (+x): {np.round(nas, 3).tolist()} mm")
    # THE MIRROR GATE.
    if lpa[1] <= 0 or rpa[1] >= 0:
        raise ValueError(
            "LEFT-RIGHT ORIENTATION IS WRONG — LPA must map to +y and RPA to -y. "
            f"Got LPA {np.round(lpa, 3).tolist()}, RPA {np.round(rpa, 3).tolist()} mm. "
            "Importing this would produce a mirrored brain that looks entirely correct."
        )
    worst = max(abs(nas[2]), abs(lpa[2]), abs(rpa[2]))
    if worst > TOL_MM:
        raise ValueError(f"the fiducials do not define the z=0 plane (worst |z| = {worst:.4g} mm)")


def read_subjectimage(path: str | Path) -> Mri:
    """Read one ``subjectimage_*.mat`` and verify its orientation."""
    p = Path(path)
    m = load_mat(p)
    if "Cube" not in m:
        raise ValueError(f"{p.name}: no Cube — not a Brainstorm subject image")

    cube = np.asarray(m["Cube"])
    if cube.ndim == 4 and cube.shape[3] == 1:
        cube = cube[..., 0]
    if cube.ndim != 3:
        raise ValueError(f"{p.name}: Cube is {cube.ndim}-D, expected 3")
    labels = _read_labels(m.get("Labels"))
    if labels is not None:
        # A LABEL volume needs no windowing, whatever its integer width: DKT,
        # Desikan-Killiany and Destrieux come as uint16 (ids up to ~15000).
        if not np.issubdtype(cube.dtype, np.integer):
            raise ValueError(f"{p.name}: an atlas Cube must hold integers, not {cube.dtype}")
        cube, labels = _to_codes(cube, labels)
        kind = "atlas"
    elif cube.dtype == np.uint8:
        kind = "anatomical"
    elif np.issubdtype(cube.dtype, np.floating):
        # A float volume on the T1's grid — PET mean/SUVR, a resliced map. Its VALUES
        # are the data: a Field on the one Volume, never a second Volume (D124).
        cube = cube.astype(np.float32)
        kind = "scalar"
    else:
        # An int16 CT or intensity volume needs a windowing decision this module is
        # in no position to make, so it refuses rather than truncating.
        raise ValueError(f"{p.name}: Cube is {cube.dtype}; only uint8 anatomy, integer atlases "
                         "and float maps are supported")

    voxsize = np.asarray(m.get("Voxsize", [1, 1, 1]), dtype=float).ravel()
    if voxsize.size != 3:
        raise ValueError(f"{p.name}: Voxsize has {voxsize.size} entries, expected 3")

    scs_raw, ncs_raw = m.get("SCS"), m.get("NCS")
    if scs_raw is None:
        raise ValueError(f"{p.name}: no SCS — the registration is the reason to import this")
    scs = {}
    for k in ("R", "T", "NAS", "LPA", "RPA", "Origin"):
        try:
            v = _mat(scs_raw, k) if k == "R" else _vec(scs_raw, k)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if v.size:
            scs[k] = v
    verify_scs(scs, voxsize)

    ncs: dict[str, np.ndarray] = {}
    if ncs_raw is not None:
        for k in ("R", "T", "AC", "PC", "IH"):
            try:
                ncs[k] = _mat(ncs_raw, k) if k == "R" else _vec(ncs_raw, k)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass   # NCS is optional in whole and in part — a store may be un-normalised

    return Mri(
        name=mri_name(p, m["Comment"] if isinstance(m.get("Comment"), str) else ""),
        kind=kind,
        cube=cube, voxsize=voxsize, scs=scs, ncs=ncs,
        comment=str(m.get("Comment", "")), source=p.name, labels=labels,
    )


def _read_labels(raw: Any) -> dict[str, Any] | None:
    """Brainstorm's ``Labels`` is an [N x 3] cell array of ``[id, name, rgb]`` —
    which is `CategoricalScheme`'s ids, names and colours, already in that shape.

    An anatomical volume carries a 0x0 ``Labels``, which is how a volume says it
    is not an atlas; that is the ONLY thing distinguishing the two on disk.
    """
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=object)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 3:
        return None
    ids, names, colors = [], [], []
    for row in arr:
        ids.append(int(np.ravel(row[0])[0]))
        names.append(str(np.ravel(row[1])[0]))
        colors.append(np.asarray(row[2], dtype=np.uint8).ravel()[:3])
    return {
        "ids": np.asarray(ids, dtype=np.int32),
        "names": names,
        "colors": np.vstack(colors).astype(np.uint8),
    }


def _to_codes(cube: np.ndarray, labels: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rewrite an atlas's voxels as CODES — the row of its label table (D123).

    Brainstorm stores label VALUES (ASEG 0-255, Destrieux up to ~15000); the database
    numbers a dictionary's entries 0..N-1 in table order, and the viewer uploads a
    uint8 texture. Codes make all three agree: voxel code k is table row k. The source
    values are kept (``source_ids``). A value in the volume that the table does not
    name gets its own ``unlabelled <v>`` row — reported by its name, never dropped.
    """
    src = np.asarray(labels["ids"], dtype=np.int64)
    names = list(labels["names"])
    colors = np.asarray(labels["colors"], dtype=np.uint8).reshape(-1, 3)
    present = np.unique(cube)
    missing = np.setdiff1d(present, src).astype(np.int64).tolist()
    if missing:
        src = np.concatenate([src, np.asarray(missing, dtype=np.int64)])
        names += [f"unlabelled {v}" for v in missing]
        colors = np.vstack([colors, np.tile(np.array([[128, 128, 128]], dtype=np.uint8), (len(missing), 1))])
    if len(set(src.tolist())) != len(src):
        raise ValueError(f"the label table repeats a value: {sorted(src.tolist())}")
    order = np.argsort(src)
    codes = order[np.searchsorted(src[order], cube)]
    dtype = np.uint8 if len(src) <= 256 else np.uint16
    return codes.astype(dtype), {
        "ids": np.arange(len(src), dtype=np.int32),
        "source_ids": src.astype(np.int32),
        "names": names,
        "colors": colors,
        "unlabelled": missing,
    }


def _frame_key(scs: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """The voxel→SCS transform as JSON — what decides whether two volumes share a grid."""
    return {"r": np.asarray(scs["R"], dtype=float).ravel().tolist(),
            "t": np.asarray(scs["T"], dtype=float).ravel().tolist()}


def _affine(scs: dict[str, np.ndarray], voxsize: np.ndarray) -> np.ndarray:
    """``scs_mm = R @ (voxel * voxsize) + T`` as ONE 3x4 matrix over voxel INDEX —
    the operator's own numbers, so a reader applies it without knowing the convention."""
    R = np.asarray(scs["R"], dtype=float).reshape(3, 3) * np.asarray(voxsize, dtype=float)[None, :]
    T = np.asarray(scs["T"], dtype=float).reshape(3)
    return np.hstack([R, T[:, None]])


def _anatomical_volume(sub: Subject) -> dict | None:
    """The ``mri/<x>_volume`` manifold already in the subject (its row)."""
    return next((r for r in sub.db.all("SELECT * FROM manifold WHERE subject_id = ? AND type = 'volume' ORDER BY path", sub.id)
                 if (r["path"] or "").startswith("mri/") and r["path"].endswith("_volume")), None)


def write_mri(sub: Subject, img: Mri) -> str:
    """One volume into ``mri/``, as entities:

        mri/<name>_volume    Manifold  volume (lattice, rank 3) in closed form, in its own voxel chart (``voxel:<name>_volume``)
        mri/<name>           Field     the intensities (anatomical) or the values of a map (PET, D124) on that volume
        mri/<name>           Selection an atlas: one CODE per voxel on the T1's volume (D123), read with the app-wide dictionary
                                       of that atlas — Brainstorm's label value kept per entry (``source_id``)
        <name>_scs, _ncs     Operator  the transitions out of voxel space (3x4 affines), between coordinate systems

    ONE VOLUME (decision 2): an atlas or a map is placed on the anatomical grid, so its frame must equal the T1's — refused
    otherwise, and refused when no anatomical volume has been imported yet."""
    node = f"mri/{img.name}"
    for table in ("manifold", "field", "operator", "selection"):
        if sub.find(table, node) is not None:
            raise ValueError(f"{img.name}: {node} is already in the subject — remove it first")
    chunks = tuple(min(CHUNK, n) for n in img.cube.shape)
    if img.kind == "anatomical":
        vol = f"{node}_volume"
        shape = list(img.cube.shape)
        vid = sub.manifold(name=vol.rsplit("/", 1)[-1], path=vol, type="volume", n_vertices=int(np.prod(shape)),
                           topology={"kind": "lattice", "rank": 3}, **lattice_counts(shape),
                           geometry={"form": "closed_form", "cs_key": f"voxel:{vol.rsplit('/', 1)[-1]}",
                                     "dimensions": [{"name": d, "n_vertices": int(n), "spacing": float(v), "unit": "mm"}
                                                    for d, n, v in zip("ijk", shape, img.voxsize)]},
                           source=img.source, producer_json=jdump({"scs": _frame_key(img.scs),
                                                                   "convention": "scs_mm = R @ (voxel * voxsize) + T",
                                                                   "axes": "+x anterior, +y left, +z superior; fiducials at z=0"}))
        sub.field(name=img.name, path=node, kind="T1-weighted", manifold_id=vid, data=img.cube, chunks=chunks, compress=True,
                  dense=True, description="T1-weighted intensity per voxel", comment=img.comment or None, source=img.source)
        for frame, tf, extra in (("scs", img.scs, ("NAS", "LPA", "RPA", "Origin")), ("ncs", img.ncs, ("AC", "PC", "IH"))):
            if not tf or "R" not in tf or "T" not in tf:
                continue
            sub.operator(name=f"{img.name}_{frame}", path=f"{img.name}_{frame}", kind="transition", from_manifold_id=vid,
                         from_coordinate_system_id=sub.cs(f"voxel:{vol.rsplit('/', 1)[-1]}"), to_coordinate_system_id=sub.cs(frame),
                         layout="dense", n_rows=3, n_cols=4, data=_affine(tf, img.voxsize),
                         description=f"voxel index → {frame.upper()} millimetres (3x4 affine)",
                         producer_json=jdump({"convention": f"{frame}_mm = R @ (voxel * voxsize) + T",
                                              **({"axes": "+x anterior, +y left, +z superior; fiducials at z=0"} if frame == "scs" else {}),
                                              **{k.lower(): np.asarray(tf[k], dtype=float).ravel().tolist() for k in extra if k in tf}}))
        return node
    va = _anatomical_volume(sub)
    if va is None:
        raise ValueError(f"{img.name}: an atlas or map lives on the anatomical volume — import the T1 first")
    dims = sub.db.all("SELECT d.n_vertices FROM geometry_dimension d WHERE d.geometry_id = ? ORDER BY d.ordinal", va["geometry_id"])
    stored = json.loads(va["producer_json"] or "{}").get("scs")
    if stored != json.loads(jdump(_frame_key(img.scs))) or [d["n_vertices"] for d in dims] != list(img.cube.shape):
        raise ValueError(f"{img.name}: its voxel grid or SCS frame differs from {va['path']}'s — one Volume per subject "
                         "(decision 2), and this volume is not on it")
    if img.kind == "atlas":
        lab = img.labels
        entries = [{"code": int(c), "name": n, "ordinal": int(c), "color_json": [int(x) for x in col],
                    "attributes_json": {"source_id": int(src)}}
                   for c, n, col, src in zip(lab["ids"], lab["names"], lab["colors"], lab["source_ids"])]
        did, code_of = sub.dictionary(img.name, scope="app", entries=entries)
        code_of = np.asarray(code_of)
        cube = code_of[img.cube.astype(np.int64)].astype(np.uint8 if code_of.max() < 256 else np.uint16)
        sub.selection(name=img.name, path=node, type="set", manifold_id=va["id"], data=cube, elements=False, dense=True,
                      meta=array_meta(cube.dtype, cube.shape, chunks=chunks, compress=True),
                      dictionary_id=did, n_elements=int(cube.size), n_members=int(len(np.unique(cube))),
                      description="one label per voxel — the CODE, i.e. the dictionary entry; its source_id is Brainstorm's "
                                  "label value (D123)" + (f"; unlabelled values {lab['unlabelled']}" if lab.get("unlabelled") else ""))
        return node
    text = f"{img.name} {img.comment}".lower()
    sub.field(name=img.name, path=node, kind=img.comment or img.name, manifold_id=va["id"], data=img.cube.astype(np.float32),
              chunks=chunks, compress=True, dense=True, description="a value per voxel of the anatomical volume (resliced to its grid)",
              comment=img.comment or None, source=img.source, unit="SUVR" if "suvr" in text else None)
    return node


def find_subjectimages(anat_dir: str | Path) -> list[Path]:
    """Every ``subjectimage_*.mat`` in a subject's Brainstorm anatomy folder,
    anatomical first so a store's first volume is its T1."""
    d = Path(anat_dir)
    found = [p for p in sorted(d.glob("subjectimage_*.mat")) if not p.name.startswith("._")]
    return sorted(found, key=lambda p: ("volatlas" in p.name, p.name))


def import_subjectimages(
    sub: Subject,
    anat_dir: str | Path,
    only: list[str] | None = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Import every subject image in ``anat_dir`` into the subject ``sub``: the T1 first (atlases and maps are placed on
    its grid), every atlas as codes (D123), every float map as a Field (D124). An unreadable volume is reported and
    skipped, never silent."""
    out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    candidates = find_subjectimages(anat_dir)
    images: list[Mri] = []
    for p in candidates:
        # FILTER BY NAME BEFORE READING. `--only T1 ASEG` on sub-0003 used to read
        # every subjectimage first and filter after, so an unrelated float32 SPM
        # volume raised and took the whole run down before ASEG was reached — a
        # file the caller had explicitly not asked for.
        if only and mri_name(p) not in only:
            continue
        try:
            img = read_subjectimage(p)
        except (ValueError, KeyError, TypeError) as e:
            # REPORTED AND SKIPPED, never silent. One exotic volume (a float32
            # SPM reslice, an int16 CT) must not block the T1, and the caller has
            # to be able to see that it did not come in.
            #
            # An ORIENTATION failure is NOT in this class and never reaches here
            # as a skip: `verify_scs` raises the same ValueError, and a mirrored
            # volume being quietly left out is exactly the right outcome for it.
            rec = {"stage": "skipped", "file": Path(p).name, "reason": str(e)}
            skipped.append(rec)
            if progress:
                progress(rec)
            continue
        images.append(img)
    # The anatomical volume FIRST: atlases and maps are placed on its grid.
    rank = {"anatomical": 0, "atlas": 1, "scalar": 2}
    for img in sorted(images, key=lambda i: (rank[i.kind], i.source)):
        try:
            base = write_mri(sub, img)
        except ValueError as e:
            rec = {"stage": "skipped", "file": img.source, "reason": str(e)}
            skipped.append(rec)
            if progress:
                progress(rec)
            continue
        rec = {"stage": "mri", "name": img.name, "kind": img.kind,
               "shape": list(img.cube.shape), "path": base,
               **({"labels": len(img.labels["ids"])} if img.labels else {})}
        out.append(rec)
        if progress:
            progress(rec)
    if not out:
        why = "; ".join(f"{r['file']}: {r['reason']}" for r in skipped) or "none found"
        raise FileNotFoundError(f"no subject image imported from {anat_dir} ({why})")
    return out
