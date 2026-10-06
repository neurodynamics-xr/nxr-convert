"""A DATASET'S DATABASE from Python — the one schema (``model.sql``, shipped in this package: a byte copy of the monorepo's
``backend/schema/model.sql``, written by ``backend/scripts/schema-module.mjs`` and held equal by ``tests/test_schema_copy.py``
and ``backend/src/db/schema.test.ts``), the one version, the one serializer.

    <datastore>/<dataset>/dataset.sqlite      created from model.sql with the dataset (D140); another version is REFUSED (D139)

THE ROW FIRST, THE STORE ITS IMAGE (D141/D145). Every writer writes rows; the schema's triggers mark the node of every changed
row DIRTY (``sync_job``); ``drain`` then writes each dirty node's ``zarr.json`` as the canonical document — the node's own
keys, compact, then ``"attributes":`` and the TEXT of ``SELECT attributes FROM <table>_attributes WHERE id = ?`` verbatim — and
removes the folder of a deleted node. It is ``syncPending`` / ``zarrJsonText`` of ``backend/src/db/sync.ts`` line for line, so a
store drained here and one drained by the app are byte-identical (the gate: ``tests/test_byte_identity.py``).

Nothing here knows a column by heart: the column sets are read from the schema (``PRAGMA table_info``).

STATUS: a data service over the datastore (one HTTP service, the only reader and writer) is planned; the converter
will become its client. Until then the Brainstorm readers write through this module, which is kept as it is and not
extended.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import uuid as _uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

MODEL_SQL = "model.sql"                                     # package data of ``nxr_convert`` (``importlib.resources``)
DATASET_FILE = "dataset.sqlite"
CATALOG_FILE = "catalog.json"
NODE_TABLES = ("dataset", "subject", "manifold", "selection", "field", "operator")
STATUS_TABLES = ("manifold", "selection", "field", "operator", "subject")      # recovery's order, as write.ts


def model_sql() -> str:
    """The DDL — the ONE place Python reads the schema."""
    return resources.files("nxr_convert").joinpath(MODEL_SQL).read_text(encoding="utf-8")


def schema_version() -> int:
    m = re.search(r"^PRAGMA user_version = (\d+);", model_sql(), flags=re.M)
    if not m:
        raise RuntimeError(f"nxr_convert/{MODEL_SQL}: no `PRAGMA user_version = N;` statement")
    return int(m.group(1))


def uuid7(now_ms: int | None = None) -> str:
    """A version-7 UUID — the primary key of every row (D16): 48-bit ms timestamp, version 7, variant 10, random."""
    ms = (int(time.time() * 1000) if now_ms is None else int(now_ms)) & ((1 << 48) - 1)
    b = bytearray(os.urandom(16))
    b[0:6] = ms.to_bytes(6, "big")
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(_uuid.UUID(bytes=bytes(b)))


def now_utc(ms: bool = True) -> str:
    """ISO 8601 UTC as JavaScript's ``toISOString`` writes it (``2026-10-02T12:34:56.789Z``)."""
    t = datetime.now(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + (f".{t.microsecond // 1000:03d}Z" if ms else "Z")


# ── the database ──────────────────────────────────────────────────────────────────────────────────────────────────────

class Database:
    """One dataset's SQLite file, foreign keys on, transactions explicit (``tx``)."""

    def __init__(self, conn: sqlite3.Connection, path: str):
        self.conn = conn
        self.path = path
        self._depth = 0
        self._cols: dict[str, list[str]] = {}
        self._pk: dict[str, list[str]] = {}

    # transactions — nested ones are savepoints, so a helper can open one inside a caller's
    @contextmanager
    def tx(self) -> Iterator["Database"]:
        if self._depth == 0:
            self.conn.execute("BEGIN")
        else:
            self.conn.execute(f"SAVEPOINT sp{self._depth}")
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO sp{self._depth}")
                self.conn.execute(f"RELEASE sp{self._depth}")
            raise
        self._depth -= 1
        if self._depth == 0:
            try:
                self.conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                # a DEFERRED key fails at COMMIT with no row named — name it before rolling back
                bad = self.conn.execute("PRAGMA foreign_key_check").fetchall()[:5]
                self.conn.execute("ROLLBACK")
                raise sqlite3.IntegrityError(f"{e}: " + "; ".join(f"{b[0]} rowid {b[1]} → {b[2]}" for b in bad)) from e
        else:
            self.conn.execute(f"RELEASE sp{self._depth}")

    def columns(self, table: str) -> list[str]:
        """The table's columns, FROM THE SCHEMA — never a hand list that can drift."""
        if table not in self._cols:
            info = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            if not info:
                raise KeyError(f"no table {table!r} in the schema")
            self._cols[table] = [r[1] for r in info]
            self._pk[table] = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
        return self._cols[table]

    def primary_key(self, table: str) -> list[str]:
        self.columns(table)
        return self._pk[table]

    def all(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, args)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def one(self, sql: str, *args: Any) -> dict[str, Any] | None:
        cur = self.conn.execute(sql, args)
        r = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], r)) if r is not None else None

    def value(self, sql: str, *args: Any) -> Any:
        r = self.conn.execute(sql, args).fetchone()
        return r[0] if r is not None else None

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        return self.conn.execute(sql, args)

    def read(self, table: str, id: str) -> dict[str, Any] | None:
        return self.one(f"SELECT * FROM {table} WHERE id = ?", id)

    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = [c for c in self.columns(table) if c in row]
        unknown = set(row) - set(self.columns(table))
        if unknown:
            raise KeyError(f"{table}: no column(s) {sorted(unknown)}")
        self.conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                          [_sql_value(row[c]) for c in cols])

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        """INSERT, or on the primary key UPDATE the given columns (never a REPLACE: a delete would cascade)."""
        cols = [c for c in self.columns(table) if c in row]
        unknown = set(row) - set(self.columns(table))
        if unknown:
            raise KeyError(f"{table}: no column(s) {sorted(unknown)}")
        pk = self.primary_key(table)
        rest = [c for c in cols if c not in pk]
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        sql += f" ON CONFLICT ({', '.join(pk)}) DO " + (f"UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in rest)}" if rest else "NOTHING")
        self.conn.execute(sql, [_sql_value(row[c]) for c in cols])

    def close(self) -> None:
        self.conn.close()


def _sql_value(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):     # a numpy scalar
        return v.item()
    return v


def open_database(path: str | Path, *, create: bool = False, built_by: str = "nxr-convert") -> Database:
    """Open a dataset's database. ``create``: an empty file gets the schema (tables, version and meta in ONE transaction).
    A database at another ``user_version`` is REFUSED (D139) — it is re-converted, never migrated."""
    p = str(path)
    if p != ":memory:" and not create and not Path(p).is_file():
        raise FileNotFoundError(f"{p}: no dataset database here — create the dataset first (nxr-convert dataset create)")
    conn = sqlite3.connect(p, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    has_meta = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()[0] > 0
    if not has_meta:
        if not create:
            conn.close()
            raise RuntimeError(f"{p}: not a dataset database (no schema)")
        conn.executescript("BEGIN;\n" + model_sql()
                           + f"\nINSERT INTO meta (key, value) VALUES ('built_utc', '{now_utc()}');"
                           + f"\nINSERT INTO meta (key, value) VALUES ('built_by', '{built_by}');\nCOMMIT;")
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    want = schema_version()
    if v != want:
        conn.close()
        raise RuntimeError(f"{p}: user_version {v}, this build reads {want} — re-convert it from its source (D139/D144)")
    return Database(conn, p)


# ── the canonical zarr.json (D145) ────────────────────────────────────────────────────────────────────────────────────

def js_number(x: float) -> str:
    """A number as JavaScript's ``JSON.stringify`` writes it (ECMAScript Number::toString): ``0.0`` → ``0``, ``1e-07`` →
    ``1e-7``, ``1e21`` → ``1e+21``. Both languages print the SHORTEST digits that round-trip; only the layout differs."""
    if x != x or x in (float("inf"), float("-inf")):
        return "null"
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    t = Decimal(repr(abs(x))).as_tuple()
    # value = int(digits) × 10^exponent; trailing zeros go into the exponent
    full = "".join(map(str, t.digits)).lstrip("0") or "0"
    exp = int(t.exponent)
    while len(full) > 1 and full.endswith("0"):
        full, exp = full[:-1], exp + 1
    digits, k = full, len(full)
    n = k + exp                                     # value = 0.digits × 10^n
    if k <= n <= 21:
        s = digits + "0" * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        es = ("+" if e > 0 else "-") + str(abs(e))
        s = (digits + "e" + es) if k == 1 else (digits[0] + "." + digits[1:] + "e" + es)
    return sign + s


def js_json(v: Any) -> str:
    """``JSON.stringify`` of a value parsed from JSON: compact, keys in their order, non-ASCII as is."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return js_number(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + js_json(x) for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(js_json(x) for x in v) + "]"
    raise TypeError(f"not JSON: {type(v).__name__}")


def zarr_json_text(doc: dict[str, Any], attributes: str) -> str:
    """THE CANONICAL zarr.json (D145): the document's own keys as they are, then ``"attributes":`` and the view's TEXT
    verbatim — ``zarrJsonText`` of ``sync.ts``, so two writers write the same bytes."""
    rest = {k: v for k, v in doc.items() if k != "attributes"}
    head = js_json(rest)
    return f"{head[:-1]}{',' if len(head) > 2 else ''}\"attributes\":{attributes}}}"


#: A PLAIN group (a layout folder, no row) as the app's writer creates one (zarrita: two-space indent) — so a folder
#: either language lays out is the same bytes. A NODE's document is rewritten canonically by the drain.
EMPTY_GROUP = '{\n  "zarr_format": 3,\n  "node_type": "group",\n  "attributes": {}\n}'


def ensure_groups(root: str | Path, location: str, *, self_too: bool = True) -> None:
    """Every MISSING group on the way to ``location`` (and it, with ``self_too``) as a plain group — the layout folders
    (``sources/``, ``timeseries/``) are Zarr groups so a walk descends them. Existing documents are never touched."""
    parts = [p for p in location.split("/") if p]
    upto = len(parts) if self_too else len(parts) - 1
    for i in range(1, upto + 1):
        d = Path(root, *parts[:i])
        f = d / "zarr.json"
        if not f.exists():
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(EMPTY_GROUP, encoding="utf-8")


def remove_folder(path: str | Path) -> None:
    """Remove a node's folder, tolerating entries that vanish while it goes (exFAT's AppleDouble ``._*`` companions are
    deleted with their files, which makes a plain rmtree fail half-way)."""
    p = Path(path)
    if not p.exists():
        return

    def gone_is_fine(_f, _p, exc):
        if not isinstance(exc, FileNotFoundError):
            raise exc
    shutil.rmtree(p, onexc=gone_is_fine)
    if p.exists():
        shutil.rmtree(p, onexc=gone_is_fine)


# ── where a row's node is ─────────────────────────────────────────────────────────────────────────────────────────────

def store_path_of(db: Database, subject_id: str) -> str:
    """``<dataset.path>/<subject.path>`` — a subject's store under the datastore root."""
    r = db.one("SELECT ds.path AS d, s.path AS s FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = ?", subject_id)
    if r is None:
        raise KeyError(f"subject {subject_id}: no such subject")
    return f"{r['d']}/{r['s']}"


def location_of_row(db: Database, table: str, attrs: dict[str, Any]) -> str:
    """A row's location by table: a dataset is its folder, a subject its store, a node inside its subject's store."""
    path = attrs["path"]
    if table == "dataset":
        return path
    if table == "subject":
        ds = db.read("dataset", attrs["dataset_id"])
        if ds is None:
            raise KeyError(f"subject {attrs.get('id')}: no dataset")
        return f"{ds['path']}/{path}"
    return f"{store_path_of(db, attrs['subject_id'])}/{path}"


def location_of(db: Database, table: str, id: str) -> str | None:
    """Where a node is (datastore-relative); None for a row with no node of its own (it rides in its subject's root)."""
    r = db.read(table, id)
    if r is None or not r.get("path"):
        return None
    if table == "subject":
        return store_path_of(db, id)
    if table == "dataset":
        return r["path"]
    return f"{store_path_of(db, r['subject_id'])}/{r['path']}"


def attributes_text(db: Database, table: str, id: str) -> str | None:
    """The node's attributes as the SCHEMA writes them — the text of its view (D145)."""
    return db.value(f"SELECT attributes FROM {table}_attributes WHERE id = ?", id)


# ── the drain (D141's dirty-tracker) ──────────────────────────────────────────────────────────────────────────────────

def pending_jobs(db: Database) -> list[dict[str, Any]]:
    return db.all("SELECT * FROM sync_job WHERE done_utc IS NULL ORDER BY id")


def _finish(db: Database, job_id: int, error: str | None) -> None:
    db.execute("UPDATE sync_job SET done_utc = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), error = ? WHERE id = ?", error, job_id)


def write_zarr_json(file: Path, attributes: str) -> None:
    doc = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {"zarr_format": 3, "node_type": "group"}
    file.write_text(zarr_json_text(doc, attributes), encoding="utf-8")


def drain(db: Database, root: str | Path, *, only=None) -> dict[str, list[str]]:
    """Drain ``sync_job`` against the datastore at ``root`` — ``syncPending`` exactly: pending jobs in id order; an upsert
    writes the node's canonical zarr.json from its view's text (creating the group, and the groups above it, when absent);
    a delete removes the folder at the job's path; a row with no path re-syncs its subject, whose root carries it."""
    out: dict[str, list[str]] = {"written": [], "removed": [], "skipped": []}
    root = Path(root)

    def nxt():
        jobs = pending_jobs(db)
        if only is not None:
            jobs = [j for j in jobs if only(j)]
        return jobs[0] if jobs else None

    job = nxt()
    while job is not None:
        try:
            if job["op"] == "delete":
                if job["path"] and (root / job["path"]).exists():
                    remove_folder(root / job["path"])
                if job["path"]:
                    out["removed"].append(job["path"])
                _finish(db, job["id"], None)
            else:
                text = attributes_text(db, job["table_name"], job["row_id"])
                if text is None:
                    _finish(db, job["id"], "row gone before sync")
                    out["skipped"].append(job["row_id"])
                else:
                    attrs = json.loads(text)
                    if not attrs.get("path"):
                        if attrs.get("subject_id"):
                            db.execute("INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', ?, 'upsert' WHERE NOT EXISTS "
                                       "(SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = ? AND j.op = 'upsert' "
                                       "AND j.done_utc IS NULL)", attrs["subject_id"], attrs["subject_id"])
                        _finish(db, job["id"], None)
                        out["skipped"].append(job["row_id"])
                    else:
                        location = location_of_row(db, job["table_name"], attrs)
                        file = root / location / "zarr.json"
                        if not file.exists():
                            ensure_groups(root, location)
                        write_zarr_json(file, text)        # WHOLESALE: the attributes ARE the row
                        out["written"].append(location)
                        _finish(db, job["id"], None)
        except Exception as e:
            _finish(db, job["id"], str(e))
            raise
        job = nxt()
    return out


def resync_all(db: Database) -> int:
    """Enqueue EVERY node row (``resyncAll``)."""
    n = 0
    for table in NODE_TABLES:
        for r in db.all(f"SELECT id FROM {table} WHERE path IS NOT NULL"):
            db.execute("INSERT OR IGNORE INTO sync_job (table_name, row_id, op) VALUES (?, ?, 'upsert')", table, r["id"])
            n += 1
    return n


def verify(db: Database, root: str | Path) -> list[dict[str, str]]:
    """Every node row whose zarr.json is missing or whose attributes are not its view's — enqueued again (``verify``)."""
    stale = []
    for table in NODE_TABLES:
        for r in db.all(f"SELECT * FROM {table} WHERE path IS NOT NULL"):
            file = Path(root) / location_of_row(db, table, r) / "zarr.json"
            reason = None
            if not file.exists():
                reason = "missing"
            else:
                a = json.loads(file.read_text(encoding="utf-8")).get("attributes") or {}
                if a.get("id") != r["id"]:
                    reason = "another id"
                elif a != json.loads(attributes_text(db, table, r["id"]) or "null"):
                    reason = "stale"
            if reason:
                stale.append({"table": table, "id": r["id"], "path": r["path"], "reason": reason})
                db.execute("INSERT OR IGNORE INTO sync_job (table_name, row_id, op) VALUES (?, ?, 'upsert')", table, r["id"])
    return stale


def text_parity(db: Database, root: str | Path) -> dict[str, Any]:
    """EVERY node row's zarr.json against its view, as TEXT: the bytes after ``"attributes":`` must be the view's text
    exactly (D145), and the whole file the canonical document. Returns the counts and the first mismatches."""
    n, bad = 0, []
    for table in NODE_TABLES:
        for r in db.all(f"SELECT id FROM {table} WHERE path IS NOT NULL"):
            row = db.read(table, r["id"])
            file = Path(root) / location_of_row(db, table, row) / "zarr.json"
            n += 1
            if not file.exists():
                bad.append({"table": table, "path": row["path"], "reason": "missing"})
                continue
            raw = file.read_text(encoding="utf-8")
            text = attributes_text(db, table, r["id"])
            want = zarr_json_text(json.loads(raw), text)
            if raw != want:
                bad.append({"table": table, "path": row["path"], "reason": "not the view's text" if raw.find(text) < 0 else "not canonical"})
    return {"nodes": n, "mismatched": len(bad), "first": bad[:5]}
