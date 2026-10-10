"""STATUS: a data service over the datastore (one HTTP service, the only reader and writer) is planned; the converter
will become its client. Until then the Brainstorm readers write through this module, which is kept as it is and not
extended.

THE CONVERTER IS CRUD (D141/D142) — create · read · update · delete of a dataset and of each entity, by the procedure.

    1 CREATE    the row(s), one transaction, ``status = 'pending'``
    2 LAY OUT   the node's zarr.json from the schema's view (``db.drain``), its arrays' metadata from the row
    3 POPULATE  the bytes — at once, or chunk by chunk
    4 COMPLETE  ``status = 'complete'``, zarr.json refreshed; readers see only complete rows
    UPDATE      the row, then zarr.json        DELETE   'deleting' → the folder removed → the row removed
    RECOVER     on open: a 'pending' row is rolled back (its node and row removed), a 'deleting' one finished

``backend/src/db/write.ts`` is the TypeScript side of the same procedure. A whole Brainstorm subject is a COMPOSITION
(``Subject``): every row pending, every array laid out from its row and populated, then ONE completion and ONE drain.

THE DATASET (D140): ``<datastore>/<dataset>/dataset.sqlite``, created from the schema with the dataset, holding one
``dataset`` row; the datastore's ``catalog.json`` indexes the datasets' databases and is rewritten whole, atomically,
whenever a dataset is created or deleted or its subjects change.

ONE WRITER A DATASET (OPEN-DEFECTS #26, ``writer_lock.py``): a writing open (``open_dataset``, ``create_dataset``,
``delete_dataset``) takes ``<dataset>/.writer.lock`` BEFORE it recovers or writes, and ``Dataset.close`` releases it. A lock a
live process holds (the app, another conversion) refuses the open — nothing is recovered, so that writer's pending rows are
never rolled back. The app hands its lock to the converter it runs (``NXR_WRITER_LOCK_HELD=<lock file>:<app pid>``).
"""
from __future__ import annotations

from . import _zarr_compat  # noqa: F401  (hardlink-less filesystems)

import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .writer_lock import DatasetNotWritable, WriterLock, acquire_writer_lock, database_not_writable
from .db import (CATALOG_FILE, DATASET_FILE, STATUS_TABLES, Database, attributes_text, drain, ensure_groups, js_json,
                 location_of, now_utc, open_database, remove_folder, schema_version, uuid7, zarr_json_text)

Populate = Callable[[Path], None]


def jdump(v: Any) -> str | None:
    """A ``*_json`` column as ``JSON.stringify`` writes it (compact, numbers JavaScript's way); None stays NULL."""
    return None if v is None else js_json(_plain(v))


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


# ══ THE PROCEDURE ═════════════════════════════════════════════════════════════════════════════════════════════════════

def set_status(db: Database, table: str, id: str, status: str) -> None:
    db.execute(f"UPDATE {table} SET status = ? WHERE id = ?", status, id)


def create_node(db: Database, root: str | Path, table: str, id: str, rows: Callable[[], None],
                populate: Populate | None = None) -> None:
    """CREATE a node by the procedure: ``rows`` writes its row(s) inside the transaction; the node is laid out from them;
    ``populate`` writes its bytes at the node's location; then it is complete. A failure in ``populate`` rolls the node
    back — nothing half-written survives."""
    with db.tx():
        rows()
        set_status(db, table, id, "pending")
    drain(db, root)
    try:
        if populate is not None:
            at = location_of(db, table, id)
            populate(Path(root) / at if at else Path(root))
    except BaseException:
        remove_node(db, root, table, id)
        raise
    with db.tx():
        set_status(db, table, id, "complete")
    drain(db, root)


def update_node(db: Database, root: str | Path, rows: Callable[[], None]) -> None:
    """UPDATE: the row(s), then the node's attributes from them."""
    with db.tx():
        rows()
    drain(db, root)


def remove_node(db: Database, root: str | Path, table: str, id: str) -> None:
    """DELETE: marked 'deleting' (so a crash finishes it), the folder removed, then the row (its subject's root refreshed)."""
    if db.read(table, id) is None:
        return
    at = location_of(db, table, id)
    with db.tx():
        set_status(db, table, id, "deleting")
    if at:
        remove_folder(Path(root) / at)
    with db.tx():
        if table == "subject":
            # the SELECTIONS first: a dictionary in use is RESTRICT, and the cascade may reach the subject's own first
            db.execute("DELETE FROM selection WHERE subject_id = ?", id)
        db.execute(f"DELETE FROM {table} WHERE id = ?", id)
    drain(db, root)


def recover(db: Database, root: str | Path) -> dict[str, int]:
    """RECOVER at open: every 'pending' row rolled back, every 'deleting' row finished."""
    out = {"rolled_back": 0, "finished": 0}
    for table in STATUS_TABLES:
        for r in db.all(f"SELECT id, status FROM {table} WHERE status <> 'complete'"):
            if db.read(table, r["id"]) is None:          # gone with a parent removed before it
                continue
            remove_node(db, root, table, r["id"])
            out["rolled_back" if r["status"] == "pending" else "finished"] += 1
    return out


# ══ DATASETS AND THE CATALOG (D140) ═══════════════════════════════════════════════════════════════════════════════════

def read_catalog(datastore: str | Path) -> dict | None:
    f = Path(datastore) / CATALOG_FILE
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_catalog(datastore: str | Path, *, id: str | None = None) -> dict:
    """``catalog.json`` — the index of the datastore's datasets' databases, rewritten whole and atomically. Its ``id`` is
    the datastore's (the ``datastore`` row every dataset's database carries)."""
    datastore = Path(datastore)
    old = read_catalog(datastore) or {}
    cid = old.get("id") or id or uuid7()
    rows = []
    for f in sorted(datastore.glob(f"*/{DATASET_FILE}")):
        folder = f.parent.name
        db = open_database(f)
        try:
            ds = db.one("SELECT * FROM dataset ORDER BY name LIMIT 1")
            if ds is None:
                continue
            n = db.value("SELECT count(*) FROM subject WHERE dataset_id = ? AND status = 'complete'", ds["id"])
            rows.append({"id": ds["id"], "name": ds["name"], "path": folder, "database": f"{folder}/{DATASET_FILE}",
                         "subjects": int(n), "modified_utc": now_utc()})
        finally:
            db.close()
    cat = {"schema_version": schema_version(), "id": cid, "datasets": sorted(rows, key=lambda r: r["path"])}
    tmp = datastore / f"{CATALOG_FILE}.tmp"
    tmp.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(datastore / CATALOG_FILE)
    return cat


@dataclass
class Dataset:
    """An open dataset: its database, the datastore root its paths are under, and its row."""
    db: Database
    root: Path
    row: dict[str, Any]
    lock: WriterLock | None = None       # the writer lock this open holds (#26); None for a reader


    @property
    def id(self) -> str:
        return self.row["id"]

    @property
    def folder(self) -> Path:
        return self.root / self.row["path"]

    def subject(self, name: str) -> dict | None:
        return self.db.one("SELECT * FROM subject WHERE dataset_id = ? AND name = ?", self.id, name)

    def template(self) -> dict | None:
        return self.db.one("SELECT * FROM subject WHERE dataset_id = ? AND kind = 'template' AND status = 'complete' "
                           "ORDER BY created_utc LIMIT 1", self.id)

    def close(self) -> None:
        self.db.close()
        if self.lock is not None:
            self.lock.release()

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


BAND_DICTIONARY = "frequency bands"
DEFAULT_BANDS = [(1, "delta", "δ", 1, 4), (2, "theta", "θ", 4, 8), (3, "alpha", "α", 8, 13), (4, "beta", "β", 13, 30),
                 (5, "gamma", "γ", 30, 80)]


def ensure_band_dictionary(db: Database) -> str:
    """The app-wide ``frequency bands`` dictionary (D109) — each dataset carries its own copy (D140); ``bands.ts``."""
    have = db.one("SELECT id FROM dictionary WHERE scope = 'app' AND name = ?", BAND_DICTIONARY)
    if have:
        return have["id"]
    did = uuid7()
    with db.tx():
        db.insert("dictionary", {"id": did, "name": BAND_DICTIONARY, "scope": "app",
                                 "description": "the named frequency bands (D109): each entry a band's edges in Hz — a labelling of the "
                                                "frequency space, looked up on any Line"})
        for i, (code, name, sym, lo, hi) in enumerate(DEFAULT_BANDS):
            db.insert("dictionary_entry", {"dictionary_id": did, "code": code, "name": name, "measurement_unit": "Hz", "ordinal": i,
                                           "attributes_json": jdump({"lo_hz": lo, "hi_hz": hi, "symbol": sym,
                                                                     "label": f"{sym} {lo}–{hi} Hz", "coordinate_system": "hz"})})
    return did


def create_dataset(datastore: str | Path, name: str, *, source_tool: str | None = None, source_protocol: str | None = None,
                   source_path: str | None = None, created_by: str | None = None) -> Dataset:
    """CREATE a dataset: its database from the schema, its row, its folder (laid out from the row), the catalog."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid dataset name: {name!r}")
    datastore = Path(datastore).resolve()
    folder = datastore / name
    file = folder / DATASET_FILE
    if file.exists():
        raise FileExistsError(f"{folder}: already a dataset")
    made = not folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        lock = acquire_writer_lock(folder)                     # another live writer: refused, naming it (#26)
    except BaseException:
        if made:
            remove_folder(folder)
        raise
    cat = read_catalog(datastore) or {}
    cid = cat.get("id") or uuid7()
    try:
        db = open_database(file, create=True, built_by=created_by or "nxr-convert")
    except BaseException:
        lock.release()
        raise
    try:
        now = now_utc()
        row = {"id": uuid7(), "datastore_id": cid, "name": name, "path": name, "source_tool": source_tool,
               "source_protocol": source_protocol, "source_path": source_path, "created_utc": now, "created_by": created_by,
               "modified_utc": now, "modified_by": created_by}
        with db.tx():
            db.insert("datastore", {"id": cid, "url": str(datastore), "driver": "file", "name": datastore.name})
            db.insert("dataset", row)
        ensure_band_dictionary(db)
        drain(db, datastore)
    except BaseException:
        db.close()
        lock.release()
        if made:
            remove_folder(folder)
        else:
            file.unlink(missing_ok=True)
        raise
    write_catalog(datastore, id=cid)
    return Dataset(db, datastore, db.read("dataset", row["id"]), lock)


def open_dataset(folder: str | Path, *, recover_now: bool | None = None, lock: bool = True) -> Dataset:
    """OPEN a dataset by its folder (``<datastore>/<dataset>``): its database at this schema version, recovered.

    ``lock`` (the default — every writer): the dataset's WRITER LOCK is taken first (#26) and held until ``close``; a live
    holder raises ``WriterLockHeld`` before anything is recovered. ``lock=False`` is a READER: no lock, and no recovery.
    RECOVERY ONLY BY THE TAKER: ``recover_now`` defaults to whether THIS process took the lock — a lock BORROWED through the
    app's handshake recovers nothing (the app recovered when it took it; a row pending since is its live write)."""
    folder = Path(folder).resolve()
    held = acquire_writer_lock(folder) if lock else None
    # RECOVERY ONLY BY THE TAKER (#26): a BORROWED lock (the app's, handed through NXR_WRITER_LOCK_HELD) recovers nothing — the
    # app recovered when it took the lock, and a row pending since is its live write (an import, a save)
    took = held is not None and not held.borrowed
    if recover_now is None:
        recover_now = took
    if recover_now and not took:
        if held is not None:
            held.release()
        raise ValueError(f"{folder}: recovery rolls back pending rows — only the process that took the writer lock may run it"
                         f"{' (this one borrowed it)' if held is not None else ''}")
    # A WRITABLE FOLDER, A READ-ONLY DATABASE (0444, or a -wal/-shm that cannot be written): SQLite would open it and fail at
    # the first write, inside recovery — refused here, clearly, the lock let go (the app opens such a dataset read-only)
    nw = database_not_writable(folder / DATASET_FILE) if lock else None
    if nw is not None:
        if held is not None:
            held.release()
        raise DatasetNotWritable(nw[1], nw[0])
    try:
        db = open_database(folder / DATASET_FILE)
    except BaseException:
        if held is not None:
            held.release()
        raise
    ds = db.one("SELECT * FROM dataset ORDER BY name LIMIT 1")
    if ds is None:
        db.close()
        if held is not None:
            held.release()
        raise RuntimeError(f"{folder}: its database holds no dataset row")
    root = folder.parent
    if recover_now:
        try:
            recover(db, root)
            drain(db, root)
        except BaseException:                    # a failed recovery leaks neither the database nor the lock (F5)
            db.close()
            if held is not None:
                held.release()
            raise
    return Dataset(db, root, ds, held)


def delete_dataset(datastore: str | Path, name: str) -> None:
    """DELETE a dataset: its folder (its database with it), then the catalog — refused while another writer holds it (#26)."""
    folder = Path(datastore) / name
    held = acquire_writer_lock(folder) if folder.exists() else None
    try:
        remove_folder(folder)
    finally:
        if held is not None:
            held.release()
    write_catalog(datastore)


# ══ ARRAYS — laid out FROM THE ROW, then populated ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArrayMeta:
    """An array's stored layout, as the row states it (``data_type`` · ``chunk_shape_json`` · ``codecs_json`` ·
    ``fill_value_json``) plus the chunk GRID (the shard, for a sharded array — the row's chunk is the inner one)."""
    shape: tuple[int, ...]
    data_type: str
    chunks: tuple[int, ...]
    grid: tuple[int, ...]
    codecs: list
    fill_value: Any

    def row(self) -> dict[str, Any]:
        return {"data_type": self.data_type, "chunk_shape_json": jdump(list(self.chunks)),
                "codecs_json": jdump(self.codecs), "fill_value_json": jdump(self.fill_value)}

    def doc(self) -> dict[str, Any]:
        """The zarr.json document of the array, canonical order (§5)."""
        return {"zarr_format": 3, "node_type": "array", "shape": list(self.shape), "data_type": self.data_type,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(self.grid)}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": self.fill_value, "codecs": self.codecs}


BIG_BYTES = 4 * 1024 * 1024
CHUNK_BYTES = 2 * 1024 * 1024


def default_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """The byte-budget policy: chunk axis 0 to ~2 MiB blocks once the array exceeds 4 MiB; otherwise one chunk (every
    extent floored at 1 — Zarr v3 needs positive chunks, and an empty array is legitimate)."""
    total = int(np.prod(shape)) * itemsize
    if total <= BIG_BYTES or len(shape) == 0 or shape[0] <= 1:
        return tuple(max(1, s) for s in shape)
    row = int(np.prod(shape[1:])) * itemsize
    rows = max(1, CHUNK_BYTES // max(row, 1))
    n = shape[0]
    nchunks = max(1, -(-n // rows))
    rows = -(-n // nchunks)
    return (max(1, min(rows, n)), *tuple(max(1, s) for s in shape[1:]))


@lru_cache(maxsize=None)
def _codecs(dtype: str, compress: bool, inner: tuple[int, ...] | None, grid: tuple[int, ...]) -> tuple[str, str]:
    """The codec pipeline zarr spells for these choices, and the fill value — read off an in-memory array once."""
    import zarr
    from zarr.storage import MemoryStore
    st = MemoryStore()
    kw: dict[str, Any] = {"compressors": [{"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"}}]
                          if compress else None, "filters": None}
    if inner is not None:
        zarr.create_array(st, name="x", shape=grid, dtype=dtype, chunks=inner, shards=grid, fill_value=0, **kw)
    else:
        zarr.create_array(st, name="x", shape=grid, dtype=dtype, chunks=grid, fill_value=0, **kw)
    doc = json.loads(st._store_dict["x/zarr.json"].to_bytes())
    return json.dumps(doc["codecs"]), json.dumps(doc["fill_value"])


def array_meta(dtype: Any, shape: Sequence[int], *, chunks: Sequence[int] | None = None, compress: bool | None = None,
               shards: Sequence[int] | None = None) -> ArrayMeta:
    """What an array's row says about its layout. ``compress=None``: blosc for an array over 4 MiB. ``shards``: the
    chunk grid is the shard and ``chunks`` the inner chunk (the row's ``chunk_shape_json``, the unit the app loads)."""
    dt = np.dtype(dtype)
    shape = tuple(int(s) for s in shape)
    if chunks is None:
        chunks = default_chunks(shape, dt.itemsize)
    chunks = tuple(max(1, int(c)) for c in chunks)
    if compress is None:
        compress = int(np.prod(shape)) * dt.itemsize > BIG_BYTES
    grid = tuple(max(1, int(s)) for s in shards) if shards is not None else chunks
    codecs, fill = _codecs(dt.name, bool(compress), chunks if shards is not None else None, grid)
    return ArrayMeta(shape, dt.name, chunks, grid, json.loads(codecs), json.loads(fill))


def layout_array(at: Path, meta: ArrayMeta, attributes: str | None = None) -> None:
    """LAY OUT an array from its row: the canonical zarr.json (its attributes the view's text when the node has a row)."""
    at.mkdir(parents=True, exist_ok=True)
    stale = at / "c"
    if stale.exists():
        remove_folder(stale)
    (at / "zarr.json").write_text(zarr_json_text(meta.doc(), attributes or "{}"), encoding="utf-8")


def populate_array(at: Path, data: np.ndarray | None = None, *, dense: bool = False):
    """POPULATE the laid-out array at ``at`` (whole, when ``data`` is given); returns the zarr array for chunk writes."""
    import zarr
    # dense: EVERY CHUNK ON DISK (a missing chunk reaches the renderer as a 404 it cannot predict); a 3-D grid is
    # c/<i>/<j>/<k>, and on exFAT concurrent directory creation races — serialised
    with zarr.config.set({"async.concurrency": 1, "array.write_empty_chunks": True}) if dense else nullcontext():
        a = zarr.open_array(str(at), mode="r+")
        if data is not None and (dense or a.size):
            a[...] = data
        return a


# ══ COORDINATE SYSTEMS (spacesFor, D25–D28) ═══════════════════════════════════════════════════════════════════════════

def _space_spec(key: str) -> tuple[str, str, str, Any, list[dict]]:
    """(space name, geometry, description, parameters, the systems it is created with — reference first) for a key, and
    the system the key names among them: ``spacesFor`` of ``database.ts``."""
    if key in ("scs", "ncs", "mni") or key.startswith("voxel:"):
        systems = [{"key": "scs", "name": "scs", "unit": "m", "reference": True,
                    "description": "Brainstorm's subject coordinate system: origin between the ears, x toward the nasion"}]
        if key == "ncs":
            systems.append({"key": "ncs", "name": "ncs", "unit": "mm", "transition": "affine",
                            "description": "Brainstorm's normalised coordinate system (AC · PC · IH)"})
        if key == "mni":
            systems.append({"key": "mni", "name": "mni", "unit": "mm", "transition": "affine", "description": "MNI template space"})
        if key.startswith("voxel:"):
            systems.append({"key": key, "name": key, "unit": "voxel", "transition": "affine",
                            "description": f"{key[6:]}'s voxel indices; the transition to scs is its voxel→SCS affine"})
        return "head", "volume", "R³ around the subject's head", None, systems
    if key == "freesurfer-sphere":
        return ("sphere", "volume", "the R³ FreeSurfer's registration sphere sits in (its positions are stored as three numbers each)",
                None, [{"key": key, "name": "freesurfer", "unit": None, "reference": True, "description": "radius 100"}])
    if key in ("frequency", "octave"):
        systems = [{"key": "frequency", "name": "hz", "unit": "Hz", "reference": True, "description": "cycles per second"}]
        if key == "octave":
            systems.append({"key": "octave", "name": "octave", "unit": "log2 Hz", "transition": "log", "params": {"base": 2},
                            "description": "log2 of the frequency in hertz"})
        return "frequency", "line", "the frequency line, R¹ — what a time → frequency operator maps the time line onto", None, systems
    if key == "clock":
        return "time", "line", "the time line, R¹", None, [{"key": key, "name": "clock", "unit": "s", "reference": True,
                                                             "description": "the recording's clock"}]
    if key.startswith("eigenvalue:"):
        op = key[len("eigenvalue:"):]
        dirac = "dirac" in op.lower()
        return (f"spectrum:{op}", "line", f"the spectral line of the {op} operator — its eigenvalues, in {'1/m' if dirac else '1/m²'}", None,
                [{"key": key, "name": "eigenvalue", "unit": "1/m" if dirac else "1/m²", "reference": True, "description": "λ"},
                 {"key": f"wavelength:{op}", "name": "wavelength", "unit": "m", "transition": "power",
                  "params": {"scale": 2 * math.pi, "exponent": -1, "abs": True} if dirac else {"scale": 2 * math.pi, "exponent": -0.5},
                  "description": "2π / |λ| — the spatial wavelength of a mode" if dirac else "2π / √λ — the spatial wavelength of a mode"}])
    return key, "volume", f"the {key} space", None, [{"key": key, "name": key, "unit": None, "reference": True}]


_RANK = {"line": 1, "circle": 1, "plane": 2, "sphere": 2, "torus": 2, "volume": 3}


def coordinate_system(db: Database, subject_id: str, key: str) -> str:
    """The id of the coordinate system ``key`` names for a subject — its space and systems created on demand."""
    space, geometry, description, params, systems = _space_spec(key)
    want = next(s for s in systems if s["key"] == key)
    sp = db.one("SELECT id FROM space WHERE subject_id = ? AND name = ?", subject_id, space)
    with db.tx():
        if sp is None:
            sid = uuid7()
            db.insert("space", {"id": sid, "subject_id": subject_id, "name": space, "geometry": geometry, "rank": _RANK[geometry],
                                "parameters_json": jdump(params), "description": description})
        else:
            sid = sp["id"]
        for s in systems if sp is None else [systems[0], want]:
            if db.one("SELECT id FROM coordinate_system WHERE space_id = ? AND name = ?", sid, s["name"]) is None:
                ref = bool(s.get("reference"))
                db.insert("coordinate_system", {"id": uuid7(), "space_id": sid, "name": s["name"], "unit": s.get("unit"),
                                                "is_reference": int(ref), "transition": "identity" if ref else s.get("transition", "identity"),
                                                "transition_json": jdump(s.get("params")), "description": s.get("description")})
    return db.value("SELECT id FROM coordinate_system WHERE space_id = ? AND name = ?", sid, want["name"])


# ══ THE f-VECTOR ══════════════════════════════════════════════════════════════════════════════════════════════════════

def lattice_counts(ns: Sequence[int]) -> dict[str, int | None]:
    """Edges, faces and top cells of a regular grid whose vertices are the sample sites (``latticeCounts``)."""
    ns = [int(n) for n in ns]
    idx = range(len(ns))
    prod = lambda ks: int(np.prod([ns[k] for k in ks])) if ks else 1
    edges = sum((ns[i] - 1) * prod([k for k in idx if k != i]) for i in idx)
    faces = sum((ns[i] - 1) * (ns[j] - 1) * prod([k for k in idx if k not in (i, j)]) for i in idx for j in idx if i < j)
    vols = int(np.prod([n - 1 for n in ns])) if len(ns) >= 3 else None
    return {"n_edges": edges, "n_faces": faces if len(ns) >= 2 else None, "n_volumes": vols}


def product_counts(factors: Sequence[dict]) -> dict[str, int | None]:
    """The f-vector of a product of cell complexes — the f-polynomials multiply (``productCounts``)."""
    poly: list[int | None] = [1]
    rank = 0
    for f in factors:
        g = [f["n_vertices"], f["n_edges"] if f["rank"] >= 1 else 0, f["n_faces"] if f["rank"] >= 2 else 0,
             f["n_volumes"] if f["rank"] >= 3 else 0]
        out: list[int | None] = [0, 0, 0, 0]
        for i in range(len(poly)):
            for j in range(4):
                if i + j >= 4:
                    continue
                a, b, cur = poly[i], g[j], out[i + j]
                out[i + j] = (cur if (a == 0 or b == 0) else None) if (cur is None or a is None or b is None) else cur + a * b
        poly, rank = out, rank + f["rank"]
    at = lambda k: None if k > rank else poly[k]
    return {"n_vertices": poly[0], "n_edges": at(1), "n_faces": at(2), "n_volumes": at(3), "rank": rank}


# ══ A SUBJECT — the composition ═══════════════════════════════════════════════════════════════════════════════════════

CHANNEL_FAMILY_COLORS = {
    "hemisphere": {"L": [76, 120, 168], "R": [228, 87, 86], "Z": [157, 157, 157], "-": [107, 107, 107]},
    "lobe": {"F": [242, 142, 43], "C": [89, 161, 79], "P": [76, 120, 168], "O": [176, 122, 161], "T": [237, 201, 72], "-": [107, 107, 107]},
    "type": {"MEG": [143, 211, 244], "MEG REF": [157, 117, 93], "MEG MAG": [143, 211, 244], "MEG GRAD": [118, 183, 178],
             "EEG": [118, 183, 178], "ECG": [228, 87, 86], "EOG": [255, 157, 167], "VEOG": [255, 157, 167], "HEOG": [255, 190, 120],
             "EMG": [186, 176, 172], "SYSCLOCK": [107, 107, 107], "STIM": [157, 157, 157], "MISC": [157, 157, 157]},
    "quality": {"good": [90, 170, 110], "bad": [228, 87, 86]},
}


class Subject:
    """One subject's entities, written by the procedure.

    A COMPOSITION (the default): every node row goes in ``pending``, its arrays are laid out from the row and populated at
    once, and ``complete()`` marks them all complete and drains ONCE; ``abort()`` removes what it made. ``immediate=True``
    runs the whole procedure per node (create → lay out → populate → complete), for a single entity added to a subject.
    """

    def __init__(self, ds: Dataset, row: dict, *, immediate: bool = False, owns_subject: bool = False, created_by: str | None = None):
        self.ds, self.db, self.root = ds, ds.db, ds.root
        self.row = row
        self.id = row["id"]
        self.name = row["name"]
        self.rel = f"{ds.row['path']}/{row['path']}"
        self.store = self.root / self.rel
        self.immediate = immediate
        self.owns_subject = owns_subject
        self.created: list[tuple[str, str]] = []
        self.created_by = created_by
        self._products: dict[tuple[str, ...], str] = {}

    # ── the subject itself ──────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, ds: Dataset, name: str, *, kind: str = "subject", folder: str | None = None, source_format: str | None = None,
               source_path: str | None = None, created_by: str | None = None, template_id: str | None | bool = True,
               immediate: bool = False) -> "Subject":
        """CREATE a subject (pending) — its store folder follows its row. A ``subject`` points at the dataset's default
        subject (its template, D127) unless ``template_id=None``."""
        folder = folder or f"{name}.nxr.zarr"
        if ds.db.one("SELECT 1 FROM subject WHERE dataset_id = ? AND (path = ? OR name = ?)", ds.id, folder, name):
            raise FileExistsError(f"{name}: already a subject of {ds.row['name']} — delete it first (a subject is converted whole)")
        if (ds.root / ds.row["path"] / folder).exists():
            remove_folder(ds.root / ds.row["path"] / folder)              # a folder no row describes: debris of a crash
        if template_id is True:
            t = ds.template() if kind == "subject" else None
            template_id = t["id"] if t else None
        now = now_utc(ms=False)
        row = {"id": uuid7(), "dataset_id": ds.id, "name": name, "path": folder, "kind": kind, "template_id": template_id or None,
               "source_format": source_format, "source_path": source_path, "created_utc": now, "created_by": created_by,
               "modified_utc": now, "modified_by": created_by}
        if immediate:
            create_node(ds.db, ds.root, "subject", row["id"], lambda: ds.db.insert("subject", row))
            return cls(ds, ds.db.read("subject", row["id"]), immediate=True, owns_subject=True, created_by=created_by)
        with ds.db.tx():
            ds.db.insert("subject", {**row, "status": "pending"})
        sub = cls(ds, ds.db.read("subject", row["id"]), owns_subject=True, created_by=created_by)
        ensure_groups(ds.root, sub.rel)
        sub.created.append(("subject", row["id"]))
        return sub

    @classmethod
    def open(cls, ds: Dataset, name: str, *, immediate: bool = False, created_by: str | None = None) -> "Subject":
        """An existing (complete) subject, to ADD entities to."""
        row = ds.db.one("SELECT * FROM subject WHERE dataset_id = ? AND (name = ? OR path = ? OR id = ?) AND status = 'complete'",
                        ds.id, name, name, name)
        if row is None:
            raise KeyError(f"{name}: no subject of {ds.row['name']}")
        return cls(ds, row, immediate=immediate, created_by=created_by)

    def complete(self) -> dict:
        """COMPLETE the composition: every node it made → 'complete', then ONE drain."""
        with self.db.tx():
            for table, id in self.created:
                if table in STATUS_TABLES:
                    set_status(self.db, table, id, "complete")
        out = drain(self.db, self.root)
        self.created.clear()
        if self.owns_subject:
            write_catalog(self.root)
        return out

    def abort(self) -> None:
        """Remove everything this composition made — the subject, when it made it — and the dictionaries it made that
        nothing reads any more."""
        dicts = [id for t, id in self.created if t == "dictionary"]
        if self.owns_subject:
            remove_node(self.db, self.root, "subject", self.id)
        else:
            for table, id in reversed(self.created):
                if table in STATUS_TABLES:
                    remove_node(self.db, self.root, table, id)
        self._drop_unused(dicts)
        self.created.clear()
        if self.owns_subject:
            write_catalog(self.root)

    def _drop_unused(self, dictionary_ids: Iterable[str]) -> None:
        with self.db.tx():
            for did in dictionary_ids:
                if self.db.one("SELECT 1 FROM selection WHERE dictionary_id = ? LIMIT 1", did) is None:
                    self.db.execute("DELETE FROM dictionary WHERE id = ?", did)

    def mark(self) -> int:
        """A point to roll the composition back to (``rollback_to``) — a part that fails is removed, the rest kept."""
        return len(self.created)

    def rollback_to(self, mark: int) -> int:
        """Remove every node made since ``mark`` (newest first). Returns how many."""
        undone = self.created[mark:]
        for table, id in reversed(undone):
            if table in STATUS_TABLES and (table, id) != ("subject", self.id):
                remove_node(self.db, self.root, table, id)
        self._drop_unused([id for t, id in undone if t == "dictionary"])
        del self.created[mark:]
        return len(undone)

    @contextmanager
    def composition(self):
        """``with sub.composition():`` — complete on success, abort on any failure."""
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        self.complete()

    # ── the procedure, per node ─────────────────────────────────────────────────────────────────────────────────────
    def node(self, table: str, id: str, rows: Callable[[], None], populate: Populate | None = None) -> str:
        """One node by the procedure: its rows (``pending``), its layout, its bytes."""
        if self.immediate:
            create_node(self.db, self.root, table, id, rows, populate)
            return id
        with self.db.tx():
            rows()
            set_status(self.db, table, id, "pending")
        self.created.append((table, id))
        if populate is not None:
            at = location_of(self.db, table, id)
            try:
                populate(self.root / at if at else self.root)
            except BaseException:
                remove_node(self.db, self.root, table, id)
                self.created.remove((table, id))
                raise
        return id

    def at(self, path: str) -> Path:
        """A store-relative path's folder on disk."""
        return self.store / path

    def write_array(self, path: str, data: np.ndarray, *, chunks=None, compress: bool | None = None, shards=None,
                    dense: bool = False, meta: ArrayMeta | None = None, table: str | None = None, id: str | None = None) -> ArrayMeta:
        """An array at a store-relative path, laid out (canonical zarr.json) and populated; the groups above it plain groups.
        With ``table``/``id`` the node's attributes ride in its document from the start."""
        data = np.ascontiguousarray(data)
        meta = meta or array_meta(data.dtype, data.shape, chunks=chunks, compress=compress, shards=shards)
        ensure_groups(self.store, path, self_too=False)
        text = attributes_text(self.db, table, id) if table and id else None
        layout_array(self.at(path), meta, text)
        populate_array(self.at(path), data, dense=dense)
        return meta

    # ── rows ────────────────────────────────────────────────────────────────────────────────────────────────────────
    def cs(self, key: str) -> str:
        return coordinate_system(self.db, self.id, key)

    def _prov(self, created_utc: str | None = None) -> dict:
        return {"created_utc": created_utc, "created_by": self.created_by, "modified_utc": created_utc, "modified_by": self.created_by}

    def topology(self, kind: str, rank: int, *, n_components: int = 1, winding: str | None = None,
                 cells: Iterable[tuple[int, str]] = ()) -> str:
        tid = uuid7()
        self.db.insert("topology", {"id": tid, "subject_id": self.id, "kind": kind, "rank": rank, "n_components": n_components,
                                    "winding": winding})
        for rank_, path in cells:
            self.db.insert("topology_cell", {"topology_id": tid, "rank": rank_, "path": path})
        return tid

    def geometry(self, form: str, *, cs_key: str | None = None, positions_path: str | None = None,
                 dimensions: Sequence[dict] = (), direction: Any = None) -> str:
        cs = self.cs(cs_key) if cs_key else None
        gid = uuid7()
        self.db.insert("geometry", {"id": gid, "subject_id": self.id, "form": form, "coordinate_system_id": cs,
                                    "positions_path": positions_path, "direction_json": jdump(direction)})
        for k, d in enumerate(dimensions):
            self.db.insert("geometry_dimension", {"geometry_id": gid, "ordinal": k, **d})
        return gid

    def manifold(self, *, name: str, path: str | None, type: str, n_vertices: int, topology: dict | str,
                 geometry: dict | None = None, populate: Populate | None = None, **cols) -> str:
        """A MANIFOLD node: its topology (a dict for its own, or the id of one it shares), its geometry (``form`` +
        ``cs_key`` + positions or dimensions), its row; ``populate`` writes its arrays (vertices · faces …)."""
        mid = cols.pop("id", None) or uuid7()

        def rows():
            tid = topology if isinstance(topology, str) else self.topology(**topology)
            gid = self.geometry(**geometry) if geometry else None
            self.db.insert("manifold", {"id": mid, "subject_id": self.id, "name": name, "path": path, "type": type,
                                        "n_vertices": int(n_vertices), "topology_id": tid, "geometry_id": gid,
                                        **self._prov(cols.pop("created_utc", None)), **cols})
        return self.node("manifold", mid, rows, populate)

    def product(self, factors: Sequence[str], *, session: str | None | bool = True, name: str | None = None) -> str:
        """The PRODUCT of manifolds, in array-dimension order (D37) — no node, a row its subject's root carries."""
        key = tuple(factors)
        have = self._products.get(key) or self._find_product(factors)
        if have:
            self._products[key] = have
            return have
        rows = [self.db.one("SELECT m.*, t.kind AS tkind, t.rank AS trank, t.n_components AS tcomp FROM manifold m "
                            "JOIN topology t ON t.id = m.topology_id WHERE m.id = ?", f) for f in factors]
        counts = product_counts([{**r, "rank": r["trank"]} for r in rows])
        reduces = all(r["tkind"] in ("none", "lattice", "path") for r in rows)
        ncomp = int(np.prod([r["n_vertices"] if r["tkind"] == "none" else r["tcomp"] for r in rows]))
        kind = "none" if counts["rank"] == 0 else ("lattice" if reduces else "product")
        if session is True:
            session = next((r["session"] for r in rows if r["session"] is not None), None)
        pid = self.manifold(name=name or " × ".join(r["name"] for r in rows), path=None, type="product", n_vertices=counts["n_vertices"],
                            topology={"kind": kind, "rank": counts["rank"], "n_components": ncomp},
                            geometry={"form": "product", "dimensions": [{"name": r["name"], "n_vertices": r["n_vertices"],
                                                                         "manifold_id": r["id"]} for r in rows]},
                            n_edges=counts["n_edges"], n_faces=counts["n_faces"], n_volumes=counts["n_volumes"], session=session)
        self._products[key] = pid
        return pid

    def _find_product(self, factors: Sequence[str]) -> str | None:
        sql = ("SELECT m.id FROM manifold m WHERE m.subject_id = ? AND m.type = 'product' AND "
               "(SELECT count(*) FROM geometry_dimension d WHERE d.geometry_id = m.geometry_id) = ?")
        for r in self.db.all(sql, self.id, len(factors)):
            dims = self.db.all("SELECT d.manifold_id FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id "
                               "WHERE m.id = ? ORDER BY d.ordinal", r["id"])
            if [d["manifold_id"] for d in dims] == list(factors):
                return r["id"]
        return None

    def field(self, *, name: str, path: str | None, kind: str, manifold_id: str, value_type: str = "scalar",
              meta: ArrayMeta | None = None, data: np.ndarray | None = None, populate: Populate | None = None,
              dense: bool = False, contributions: Sequence[dict] = (), **cols) -> str:
        """A FIELD node: its row (the array's layout FROM its meta), laid out, then populated — ``data`` at once, or
        ``populate(location)`` chunk by chunk."""
        fid = cols.pop("id", None) or uuid7()
        n = {"scalar": 1, "integer": 1, "complex": 2, "vector2": 2, "vector3": 3, "matrix3": 9, "quaternion": 4}[value_type]
        if data is not None and meta is None:
            data = np.ascontiguousarray(data)
            meta = array_meta(data.dtype, data.shape, chunks=cols.pop("chunks", None), compress=cols.pop("compress", None))
        cols.pop("chunks", None), cols.pop("compress", None)

        def rows():
            self.db.insert("field", {"id": fid, "subject_id": self.id, "name": name, "path": path, "kind": kind,
                                     "manifold_id": manifold_id, "value_type": value_type, "n_components": n,
                                     **(meta.row() if meta else {"data_type": cols.pop("data_type", "float32")}),
                                     **self._prov(cols.pop("created_utc", None)), **cols})
            for k, c in enumerate(contributions):
                self.db.insert("field_contribution", {"field_id": fid, "ordinal": k, "weight": c.get("weight", 1),
                                                      "subject_id": c["subject_id"], "source_field_id": c.get("source_field_id")})
        return self.node("field", fid, rows, self._array_populate("field", fid, meta, data, populate, dense) if path else None)

    def _array_populate(self, table: str, id: str, meta: ArrayMeta | None, data, populate: Populate | None, dense: bool):
        if meta is None:
            return populate

        def go(at: Path) -> None:
            ensure_groups(self.root, str(at.relative_to(self.root)), self_too=False)
            layout_array(at, meta, attributes_text(self.db, table, id))
            if data is not None:
                populate_array(at, data, dense=dense)
            if populate is not None:
                populate(at)
        return go

    def operator(self, *, name: str, path: str | None, kind: str, from_manifold_id: str, layout: str = "dense",
                 n_rows: int, n_cols: int, meta: ArrayMeta | None = None, data: np.ndarray | None = None,
                 populate: Populate | None = None, sparse: dict[str, np.ndarray] | None = None, **cols) -> str:
        """An OPERATOR node: dense (an array, laid out from the row), sparse (a group of ``indptr`` · ``indices`` ·
        ``data``), analytic (no bytes) or declared without its bytes."""
        oid = cols.pop("id", None) or uuid7()
        if data is not None and meta is None:
            data = np.ascontiguousarray(data)
            meta = array_meta(data.dtype, data.shape, chunks=cols.pop("chunks", None), compress=cols.pop("compress", None))
        cols.pop("chunks", None), cols.pop("compress", None)

        def rows():
            self.db.insert("operator", {"id": oid, "subject_id": self.id, "name": name, "path": path, "kind": kind,
                                        "from_manifold_id": from_manifold_id, "layout": layout, "n_rows": int(n_rows), "n_cols": int(n_cols),
                                        **(meta.row() if meta else {}), **self._prov(cols.pop("created_utc", None)), **cols})
        def write_sparse(at: Path) -> None:
            for part, arr in sparse.items():
                self.write_array(f"{path}/{part}", arr)
        pop = None if not path else write_sparse if sparse is not None else \
            self._array_populate("operator", oid, meta, data, populate, False)
        return self.node("operator", oid, rows, pop)

    def selection(self, *, name: str, path: str | None, type: str, manifold_id: str | None = None, data: np.ndarray | None = None,
                  meta: ArrayMeta | None = None, elements: np.ndarray | None = None, picked: bool = False,
                  spans: Sequence[tuple[float, float, int | None]] | None = None, extents: Sequence[tuple[int, float, float]] = (),
                  measurements: Sequence[dict] = (), populate: Populate | None = None, dense: bool = False, **cols) -> str:
        """A SELECTION node. ``data``: its per-element array (codes of a partition, indices of a subset) at its path —
        ``elements`` rows from it unless ``elements=False``; ``spans``: (start, stop, code) member rows (D87);
        ``measurements``: its own (array-backed or valued) rows."""
        sid = cols.pop("id", None) or uuid7()
        if data is not None and meta is None:
            data = np.ascontiguousarray(data)
            meta = array_meta(data.dtype, data.shape)
        el = data if elements is None else (None if elements is False else elements)

        def rows():
            row = {"id": sid, "subject_id": self.id, "name": name, "path": path, "type": type, "manifold_id": manifold_id,
                   **({"array_path": path} if data is not None and path else {}), **self._prov(cols.pop("created_utc", None)), **cols}
            self.db.insert("selection", row)
            if el is not None:
                codes = np.asarray(el).ravel()
                if picked:
                    self.db.conn.executemany("INSERT INTO selection_element (selection_id, element, code) VALUES (?, ?, NULL)",
                                             ((sid, c) for c in codes.astype(np.int64, copy=False).tolist()))
                    self.db.execute("UPDATE selection SET n_elements = ? WHERE id = ?", int(codes.size), sid)
                else:
                    self.db.conn.executemany("INSERT INTO selection_element (selection_id, element, code) VALUES (?, ?, ?)",
                                             ((sid, i, c) for i, c in enumerate(codes.astype(np.int64, copy=False).tolist())))
                    self.db.execute("UPDATE selection SET n_elements = ?, n_members = (SELECT count(DISTINCT code) FROM "
                                    "selection_element WHERE selection_id = ?) WHERE id = ?", int(codes.size), sid, sid)
            if spans is not None:
                self.db.conn.executemany("INSERT INTO selection_span (selection_id, ordinal, start, stop, code) VALUES (?, ?, ?, ?, ?)",
                                         ((sid, k, float(a), float(b), None if c is None else int(c)) for k, (a, b, c) in enumerate(spans)))
                self.db.execute("UPDATE selection SET n_members = ? WHERE id = ?", len(spans), sid)
            for axis, a, b in extents:
                self.db.insert("selection_extent", {"selection_id": sid, "axis": axis, "start": a, "stop": b})
            for m in measurements:
                self.db.insert("selection_measurement", {"selection_id": sid, **m})
        pop = self._array_populate("selection", sid, meta, data, populate, dense) if (path and (meta is not None or populate)) else None
        return self.node("selection", sid, rows, pop)

    def dictionary(self, name: str, names: Sequence[str] | None = None, *, scope: str = "subject", colors=None,
                   entries: Sequence[dict] | None = None, description: str | None = None) -> tuple[str, list[int]]:
        """A DICTIONARY by name in its scope, and the CODE each requested name has in it. An existing dictionary keeps its
        codes and gains the names it lacks (so one app-wide 'Desikan-Killiany' codes every subject alike); a new one takes
        the requested codes (``entries``' own, else 0…n−1). ``entries``: full rows (code · name · color_json ·
        measurement · attributes_json …) instead of ``names`` + ``colors``."""
        scope_cols = {"dataset_id": self.ds.id} if scope == "dataset" else {"subject_id": self.id} if scope == "subject" else {}
        where = {"app": "scope = 'app'", "dataset": "scope = 'dataset' AND dataset_id = ?", "subject": "scope = 'subject' AND subject_id = ?"}[scope]
        args = [] if scope == "app" else [self.ds.id if scope == "dataset" else self.id]
        if entries is None:
            fam = CHANNEL_FAMILY_COLORS.get(name) if scope == "app" else None
            entries = []
            for k, n in enumerate(names or []):
                c = None if colors is None else colors[k]
                if c is None and fam is not None:
                    c = fam.get(n)
                entries.append({"code": k, "name": n, "ordinal": k, "color_json": None if c is None else [int(x) for x in c]})
        with self.db.tx():
            have = self.db.one(f"SELECT * FROM dictionary WHERE {where} AND name = ?", *args, name)
            if have is None:
                did = uuid7()
                self.db.insert("dictionary", {"id": did, "name": name, "scope": scope, "description": description, **scope_cols,
                                              **self._prov()})
                if not self.immediate:
                    self.created.append(("dictionary", did))
            else:
                did = have["id"]
            known = {r["name"]: r["code"] for r in self.db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", did)}
            taken = set(known.values())
            nxt = max(taken, default=-1) + 1
            codes = []
            for e in entries:
                if e["name"] in known:
                    codes.append(known[e["name"]])
                    continue
                code = int(e["code"])
                if code in taken:
                    code, nxt = nxt, nxt + 1
                nxt = max(nxt, code + 1)
                row = {k: (jdump(v) if k.endswith("_json") and v is not None and not isinstance(v, str) else v) for k, v in e.items()}
                row.update({"dictionary_id": did, "code": code})
                if "ordinal" in e and have is not None:
                    row["ordinal"] = code
                self.db.insert("dictionary_entry", row)
                known[e["name"]] = code
                taken.add(code)
                codes.append(code)
        return did, codes

    # ── reads ───────────────────────────────────────────────────────────────────────────────────────────────────────
    def find(self, table: str, path: str) -> dict | None:
        return self.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND path = ?", self.id, path)

    def by_name(self, table: str, name: str) -> dict | None:
        return self.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND name = ?", self.id, name)

