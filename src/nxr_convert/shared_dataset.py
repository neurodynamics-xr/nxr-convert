"""A SHARED dataset written by many parallel jobs: ``migrate`` · ``fragment`` · ``merge``.

MIGRATE (a dated exception to D139, 2026-10-25, for a shared store built at schema 48 — pending Diellor's review):
``nxr-convert migrate --from <old nxr.sqlite> --dataset <datastore>/<dataset>`` copies the database into the dataset folder as
``dataset.sqlite`` and brings it to this build's schema IN PLACE. Only a step this module knows is taken: 48 → 49 differs in
no data, only in the ``status`` column of the five node tables (default 'complete') and in views, indexes and triggers, which
are recreated from ``model.sql``. Then EVERY node's ``zarr.json`` is rewritten from its row (``resync_all`` + ``drain``), so a
store written in the old attribute dialect carries the rows' — and the catalog is written. A node row whose folder is missing
is reported and left alone (never a group made in its place).

FRAGMENT / MERGE (SQLite on Lustre takes ONE writer): a parallel job never opens the shared ``dataset.sqlite`` for writing.
``nxr-convert fragment --dataset <ds> --subject X --to <root>`` makes ``<root>/<dataset>/dataset.sqlite`` — the subject's rows,
the shared rows (the dataset's dictionaries, every subject row so references hold), the TEMPLATE subjects' rows (the
``@default_subject``, read there and never merged back) — and links the subject's and the templates' real stores into it, so every per-subject command (surface · hodge · interp · atlas build · qc) runs with ``--dataset <root>/<dataset>``: arrays
land in the real store, rows in the fragment. ``nxr-convert merge --dataset <ds> <root>/<dataset> …`` (ONE job, serially)
replaces each fragment's subject in the shared database by the fragment's copy, adds shared rows it does not have (a
dictionary the dataset already holds under the same name is the dataset's: the fragment's references are moved to it), and
rewrites that subject's nodes' ``zarr.json`` from the merged rows. A merge never removes a folder: the command that ran in the
fragment did whatever its own rows say on disk.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

from .db import (DATASET_FILE, NODE_TABLES, _finish, drain, location_of_row, model_sql, open_database, pending_jobs,
                 resync_all, schema_version)

MIGRATES_FROM = 48
FRAGMENT_FILE = "fragment.json"
# a table with no subject_id whose rows belong to a parent's subject (FK column, parent table)
CHILD = {"dictionary_entry": ("dictionary_id", "dictionary"), "geometry_dimension": ("geometry_id", "geometry"),
         "operator_factor": ("operator_id", "operator"), "selection_element": ("selection_id", "selection"),
         "selection_extent": ("selection_id", "selection"), "selection_span": ("selection_id", "selection"),
         "selection_measurement": ("selection_id", "selection"), "topology_cell": ("topology_id", "topology"),
         "coordinate_system": ("space_id", "space"), "field_contribution": ("field_id", "field")}
# shared rows a fragment may ADD, matched to the dataset's by their unique key (and the columns that reference them)
SHARED_KEYED = {"dictionary": (("scope", "dataset_id", "subject_id", "name"), (("selection", "dictionary_id"),)),
                "variable_dictionary": (("dataset_id", "name"), (("subject_variable", "variable_dictionary_id"),))}
NEVER = ("sync_job",)


def _tables(conn: sqlite3.Connection, schema: str = "main") -> list[str]:
    return [r[0] for r in conn.execute(f"SELECT name FROM {schema}.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]


def _cols(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA {schema}.table_info("{table}")')]


def _owners(conn: sqlite3.Connection) -> list[str]:
    return [t for t in _tables(conn) if "subject_id" in _cols(conn, t) and t not in CHILD]


# ── migrate ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _column_def(table_sql: str, col: str) -> str:
    m = re.search(rf"^\s*{col}\s+(.*?),?\s*(--.*)?$", table_sql, flags=re.M)
    if not m:
        raise RuntimeError(f"model.sql: no definition of column {col}")
    return f"{col} {m.group(1).rstrip(',')}"


def upgrade_schema(conn: sqlite3.Connection) -> dict:
    """48 → this build's schema, in one transaction: missing columns added, every view, index and trigger made model.sql's."""
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    want = schema_version()
    if v == want:
        return {"from": v, "to": v, "added": [], "objects": 0}
    if v != MIGRATES_FROM:
        raise RuntimeError(f"user_version {v}: migrate knows {MIGRATES_FROM} → {want} only — re-convert it from its source (D139)")
    ref = sqlite3.connect(":memory:")
    ref.executescript(model_sql())
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"--[^\n]*", "", s or "")).strip()
    added = []
    conn.execute("BEGIN")
    try:
        for (t, tsql) in ref.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"):
            have = _cols(conn, t)
            if not have:
                conn.execute(tsql)
                added.append(t)
                continue
            for c in _cols(ref, t):
                if c not in have:
                    conn.execute(f'ALTER TABLE "{t}" ADD COLUMN {_column_def(tsql, c)}')
                    added.append(f"{t}.{c}")
        old = {n: (k, s) for n, k, s in conn.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('view', 'index', 'trigger') AND sql IS NOT NULL")}
        for n, (k, _) in old.items():                       # every view goes (they hold no data); a changed index or trigger too
            if k == "view":
                conn.execute(f'DROP VIEW "{n}"')
        n_obj = 0
        for n, k, s in ref.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('view', 'index', 'trigger') AND sql IS NOT NULL ORDER BY rowid"):
            if k != "view" and n in old:
                if norm(old[n][1]) == norm(s):
                    continue
                conn.execute(f'DROP {k.upper()} "{n}"')
            conn.execute(s)
            n_obj += 1
        conn.execute(f"PRAGMA user_version = {want}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {"from": v, "to": want, "added": added, "objects": n_obj}


def _skip_missing(db, root: Path) -> list[str]:
    """pending upserts whose node folder is not on disk: finished with an error instead of a group made in its place"""
    missing = []
    with db.tx():
        for job in pending_jobs(db):
            if job["op"] != "upsert":
                continue
            r = db.read(job["table_name"], job["row_id"])
            if r is None or not r.get("path"):
                continue
            loc = location_of_row(db, job["table_name"], r)
            if not (root / loc).is_dir():
                _finish(db, job["id"], "node missing on disk")
                missing.append(loc)
    return missing


def migrate(source: str | Path, folder: str | Path) -> dict:
    from .crud import write_catalog
    folder = Path(folder).resolve()
    dst = folder / DATASET_FILE
    if not dst.exists():
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{DATASET_FILE}.migrating"
        shutil.copyfile(source, tmp)
        conn = sqlite3.connect(tmp, isolation_level=None)
        up = upgrade_schema(conn)
        conn.close()
        tmp.replace(dst)
    else:
        conn = sqlite3.connect(dst, isolation_level=None)
        up = upgrade_schema(conn)
        conn.close()
    db = open_database(dst)
    try:
        ds = db.one("SELECT * FROM dataset ORDER BY name LIMIT 1")
        if ds is None or ds["path"] != folder.name:
            raise RuntimeError(f"{dst}: its dataset row's path is {ds and ds['path']!r}, the folder is {folder.name!r}")
        gone = [s["path"] for s in db.all("SELECT path FROM subject WHERE dataset_id = ?", ds["id"]) if not (folder / s["path"]).is_dir()]
        if gone:
            raise RuntimeError(f"{len(gone)} subject stores are not in {folder} (e.g. {gone[:3]}): extract them first")
        with db.tx():
            n = resync_all(db) + len(db.all("SELECT id FROM dataset"))
            db.execute("INSERT OR IGNORE INTO sync_job (table_name, row_id, op) SELECT 'dataset', id, 'upsert' FROM dataset")
        missing = _skip_missing(db, folder.parent)
        out = drain(db, folder.parent)
        fk = db.conn.execute("PRAGMA foreign_key_check").fetchall()
        store = db.one("SELECT id FROM datastore LIMIT 1")
    finally:
        db.close()
    cat = write_catalog(folder.parent, id=store["id"] if store else None)
    return {**up, "database": str(dst), "nodes": n, "written": len(out["written"]), "missing": len(missing),
            "missing_first": missing[:20], "foreign_key_violations": len(fk), "catalog_datasets": len(cat["datasets"])}


# ── fragment / merge ─────────────────────────────────────────────────────────────────────────────────────────────────

def _insert(conn, table: str, where: str, args=(), mode: str = "OR IGNORE") -> int:
    c = [x for x in _cols(conn, table, "f") if x in set(_cols(conn, table))]
    q = ", ".join(f'"{x}"' for x in c)
    return conn.execute(f'INSERT {mode} INTO main."{table}" ({q}) SELECT {q} FROM f."{table}" WHERE {where}', args).rowcount


def fragment(folder: str | Path, subject: str, out_root: str | Path) -> dict:
    folder = Path(folder).resolve()
    frag = Path(out_root).resolve() / folder.name
    if (frag / DATASET_FILE).exists():
        raise FileExistsError(f"{frag}: a fragment is already here (merge it, or remove it)")
    frag.mkdir(parents=True, exist_ok=True)
    open_database(frag / DATASET_FILE, create=True, built_by="nxr-convert fragment").close()
    conn = sqlite3.connect(f"file:{frag / DATASET_FILE}", uri=True, isolation_level=None)    # a URI connection: the source opens IMMUTABLE
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ATTACH DATABASE ? AS f", (f"file:{folder / DATASET_FILE}?mode=ro&immutable=1",))
    try:
        row = conn.execute("SELECT id, path FROM f.subject WHERE name = ?", (subject,)).fetchone()
        if row is None:
            raise KeyError(f"{folder}: no subject {subject!r}")
        sid, spath = row
        conn.execute("BEGIN")
        owners = _owners(conn)
        for t in _tables(conn):
            if t in NEVER or t in CHILD or t in owners:
                continue
            _insert(conn, t, "1", mode="OR REPLACE" if t == "meta" else "OR IGNORE")    # dataset · datastore · subject · meta · …
        tpl = conn.execute("SELECT id, path FROM f.subject WHERE kind = 'template'").fetchall()   # read, never merged back
        for t in owners:
            _insert(conn, t, "subject_id = ? OR subject_id IS NULL OR subject_id IN (SELECT id FROM f.subject WHERE kind = 'template')", (sid,))
        for t, (col, parent) in CHILD.items():
            _insert(conn, t, f'"{col}" IN (SELECT id FROM main."{parent}")')
        conn.execute("DELETE FROM sync_job")
        conn.execute("COMMIT")
    finally:
        conn.execute("DETACH DATABASE f")
        conn.close()
    for p in [spath] + [p for _, p in tpl]:
        if (folder / p).exists() and not (frag / p).exists():
            os.symlink(folder / p, frag / p, target_is_directory=True)
    link = frag / spath
    (frag / FRAGMENT_FILE).write_text(json.dumps({"subject": subject, "subject_id": sid, "dataset": str(folder),
                                                  "templates": [i for i, _ in tpl]}, indent=2))
    return {"fragment": str(frag), "subject": subject, "store": str(link)}


def _merge_one(db, frag: Path) -> dict:
    info = json.loads((frag / FRAGMENT_FILE).read_text())
    sid = info["subject_id"]
    conn = db.conn
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ATTACH DATABASE ? AS f", (str(frag / DATASET_FILE),))
    try:
        if conn.execute("PRAGMA f.user_version").fetchone()[0] != schema_version():
            raise RuntimeError(f"{frag}: not at schema {schema_version()}")
        owners = _owners(conn)
        conn.execute("BEGIN")
        j0 = conn.execute("SELECT coalesce(max(id), 0) FROM sync_job").fetchone()[0]
        # the subject's rows in the shared database go (children first), then the fragment's copy comes in whole
        for t, (col, parent) in CHILD.items():
            conn.execute(f'DELETE FROM main."{t}" WHERE "{col}" IN (SELECT id FROM main."{parent}" WHERE subject_id = ?)', (sid,))
        for t in owners:
            conn.execute(f'DELETE FROM main."{t}" WHERE subject_id = ?', (sid,))
        conn.execute("DELETE FROM main.subject WHERE id = ?", (sid,))
        _insert(conn, "subject", "id = ?", (sid,), mode="")
        n = {t: _insert(conn, t, "subject_id = ?", (sid,), mode="") for t in owners}
        for t, (col, parent) in CHILD.items():
            n[t] = _insert(conn, t, f'"{col}" IN (SELECT id FROM f."{parent}" WHERE subject_id = ?)', (sid,), mode="")
        # shared rows the fragment added: kept when new; when the dataset has one under the same key, the fragment's
        # references are moved to the dataset's
        remap = 0
        for t, (key, refs) in SHARED_KEYED.items():
            on = " AND ".join(f"m.{k} IS g.{k}" for k in key)
            for old, new in conn.execute(f'SELECT g.id, m.id FROM f."{t}" g JOIN main."{t}" m ON {on} WHERE g.subject_id IS NULL AND g.id != m.id'
                                         if "subject_id" in key else f'SELECT g.id, m.id FROM f."{t}" g JOIN main."{t}" m ON {on} WHERE g.id != m.id').fetchall():
                for rt, rc in refs:
                    remap += conn.execute(f'UPDATE main."{rt}" SET "{rc}" = ? WHERE "{rc}" = ? AND subject_id = ?', (new, old, sid)).rowcount
        for t in owners:
            _insert(conn, t, "subject_id IS NULL")
        _insert(conn, "variable_dictionary", "1")
        for t, (col, parent) in CHILD.items():
            if "subject_id" in _cols(conn, parent):
                _insert(conn, t, f'"{col}" IN (SELECT id FROM f."{parent}" WHERE subject_id IS NULL) AND "{col}" IN (SELECT id FROM main."{parent}")')
        # the merge's own triggers would delete folders the fragment's command already handled: only upserts, of this subject
        conn.execute("DELETE FROM sync_job WHERE id > ?", (j0,))
        conn.execute("INSERT OR IGNORE INTO sync_job (table_name, row_id, op) VALUES ('subject', ?, 'upsert')", (sid,))
        for t in NODE_TABLES:
            if t in ("subject", "dataset"):
                continue
            conn.execute(f"INSERT OR IGNORE INTO sync_job (table_name, row_id, op) SELECT ?, id, 'upsert' FROM {t} WHERE subject_id = ? AND path IS NOT NULL", (t, sid))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("DETACH DATABASE f")
        conn.execute("PRAGMA foreign_keys = ON")
    return {"subject": info["subject"], "rows": sum(n.values()), "remapped": remap}


def merge(folder: str | Path, fragments: list[str | Path]) -> dict:
    from .crud import open_dataset
    out = []
    with open_dataset(folder) as ds:
        for f in fragments:
            out.append(_merge_one(ds.db, Path(f).resolve()))
            w = drain(ds.db, ds.root)
            out[-1]["written"] = len(w["written"])
        fk = ds.db.conn.execute("PRAGMA foreign_key_check").fetchall()
    return {"merged": out, "foreign_key_violations": len(fk)}
