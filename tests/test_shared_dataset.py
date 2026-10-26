"""``migrate`` (48 → this schema, every node's zarr.json from its row) and ``fragment`` / ``merge`` (parallel per-subject writers,
one serial merge) on a small synthetic dataset. ``fixtures/schema-48.sql`` is a schema-48 store's own DDL (2026-10-25)."""
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from nxr_convert import shared_dataset as sd
from nxr_convert.crud import Subject, create_dataset, open_dataset, remove_node
from nxr_convert.db import attributes_text, uuid7

S48 = Path(__file__).parent / "fixtures" / "schema-48.sql"


def subject(ds, name, **kw):
    sub = Subject.create(ds, name, **{"source_format": "brainstorm", **kw}, immediate=True)
    V, F = np.random.default_rng(0).random((4, 3)), np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def arrays(at):
        sub.write_array("sources/c/vertices", V)
        sub.write_array("sources/c/faces", F)
    mid = uuid7()
    sub.manifold(id=mid, name="c", path="sources/c", type="surface", n_vertices=4, n_faces=2, is_primary=1,
                 topology={"kind": "simplicial", "rank": 2, "cells": [(2, "sources/c/faces")]},
                 geometry={"form": "stored", "cs_key": "scs", "positions_path": "sources/c/vertices"}, populate=arrays)
    fid = sub.field(name="pet", path="sources/pet", kind="PET", manifold_id=mid, data=np.ones(4, np.float32))
    return sub, mid, fid


def on_disk(root, loc):
    t = (root / loc / "zarr.json").read_text()
    return t[t.index('"attributes":') + len('"attributes":'):-1]


@pytest.fixture
def shared(tmp_path):
    ds = create_dataset(tmp_path / "store", "ds")
    a, ma, fa = subject(ds, "sub-01")
    b, mb, fb = subject(ds, "sub-02")
    ids = dict(a=a.id, b=b.id, ma=ma, fa=fa, fb=fb)
    ds.close()
    return tmp_path / "store" / "ds", ids


def test_fragment_then_merge_replaces_one_subject_and_keeps_the_rest(shared, tmp_path):
    folder, ids = shared
    fr = sd.fragment(folder, "sub-01", tmp_path / "frag")
    with open_dataset(folder) as ds:                                  # another job's merge gave the dataset 'qc flag' meanwhile
        did, _ = Subject.open(ds, "sub-02", immediate=True).dictionary("qc flag", ["pass", "check", "fail"], scope="dataset")
        before_b = {t: ds.db.all(f"SELECT * FROM {t} WHERE subject_id = ? ORDER BY id", ids["b"]) for t in ("manifold", "field")}
    frag = Path(fr["fragment"])
    with open_dataset(frag) as ds:                                    # a per-subject command, run in the fragment
        s = Subject.open(ds, "sub-01", immediate=True)
        assert ds.db.value("SELECT count(*) FROM field WHERE subject_id = ?", ids["b"]) == 0      # only its own rows
        new = s.field(name="psd", path="frequency/psd", kind="psd", manifold_id=ids["ma"], data=np.arange(4, dtype=np.float32))
        remove_node(ds.db, ds.root, "field", ids["fa"])
        fd, _ = s.dictionary("qc flag", ["pass", "check", "fail"], scope="dataset")   # the fragment's own copy, another id
        assert fd != did
        sel = s.selection(name="flag", path="sources/flag", type="set", manifold_id=ids["ma"], dictionary_id=fd, data=np.zeros(4, np.int32))
    assert (folder / "sub-01.nxr.zarr" / "frequency/psd/zarr.json").exists()                     # the array is in the REAL store
    out = sd.merge(folder, [frag])
    assert out["foreign_key_violations"] == 0
    with open_dataset(folder) as ds:
        db = ds.db
        assert db.read("field", new) and db.read("field", ids["fa"]) is None and db.read("selection", sel)["dictionary_id"] == did
        assert {t: db.all(f"SELECT * FROM {t} WHERE subject_id = ? ORDER BY id", ids["b"]) for t in ("manifold", "field")} == before_b
        assert db.value("SELECT count(*) FROM dictionary WHERE name = 'qc flag'") == 1
        assert db.value("SELECT count(*) FROM sync_job WHERE done_utc IS NULL") == 0
        assert db.value("SELECT count(*) FROM sync_job WHERE op = 'delete' AND id > (SELECT max(id) - 50 FROM sync_job)") == 0
        for t, i, loc in (("field", new, "ds/sub-01.nxr.zarr/frequency/psd"), ("selection", sel, "ds/sub-01.nxr.zarr/sources/flag"),
                          ("subject", ids["a"], "ds/sub-01.nxr.zarr")):
            assert on_disk(ds.root, loc) == attributes_text(db, t, i), t
    assert (folder / "sub-01.nxr.zarr/sources/c/vertices/zarr.json").exists()                  # the merge removed no folder


def test_a_merged_selection_keeps_its_own_new_dataset_dictionary(shared, tmp_path):
    folder, ids = shared
    frag = Path(sd.fragment(folder, "sub-01", tmp_path / "frag")["fragment"])
    with open_dataset(frag) as ds:
        s = Subject.open(ds, "sub-01", immediate=True)
        fd, _ = s.dictionary("channel status", ["ok", "flat", "noisy"], scope="dataset")
        s.selection(name="st", path="sources/st", type="set", manifold_id=ids["ma"], dictionary_id=fd, data=np.zeros(4, np.int32))
    sd.merge(folder, [frag])
    with open_dataset(folder) as ds:
        assert ds.db.value("SELECT count(*) FROM dictionary_entry WHERE dictionary_id = ?", fd) == 3


def test_migrate_48(shared, tmp_path):
    folder, ids = shared
    old = tmp_path / "nxr.sqlite"                                      # the same rows in a schema-48 database
    o = sqlite3.connect(old)
    o.executescript(S48.read_text())
    o.execute("PRAGMA foreign_keys = OFF")
    o.execute("ATTACH DATABASE ? AS n", (str(folder / "dataset.sqlite"),))
    for (t,) in o.execute("SELECT name FROM main.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name != 'sync_job'").fetchall():
        c = [r[1] for r in o.execute(f'PRAGMA main.table_info("{t}")')]
        o.execute(f'INSERT INTO main."{t}" ({", ".join(c)}) SELECT {", ".join(c)} FROM n."{t}"')
    o.execute("DELETE FROM sync_job")
    o.commit()
    o.execute("DETACH DATABASE n")
    o.execute("PRAGMA user_version = 48")
    o.close()
    new = tmp_path / "new" / "ds"                                     # the stores extracted there, in the OLD attribute dialect
    new.mkdir(parents=True)
    import shutil
    for s in ("sub-01.nxr.zarr", "sub-02.nxr.zarr"):
        shutil.copytree(folder / s, new / s)
    f = new / "sub-01.nxr.zarr/sources/pet/zarr.json"
    d = json.loads(f.read_text())
    d["attributes"] = {"schema": "nxr.field@5.0", "nodeKind": "field"}
    f.write_text(json.dumps(d))
    shutil.rmtree(new / "sub-02.nxr.zarr/sources/pet")                # a node the store lost
    out = sd.migrate(old, new)
    assert out["from"] == 48 and out["to"] == 49 and "field.status" in out["added"] and out["missing"] == 1
    assert out["foreign_key_violations"] == 0 and out["catalog_datasets"] == 1
    with open_dataset(new) as ds:
        assert ds.db.value("SELECT count(*) FROM field WHERE status = 'complete'") == 2
        assert on_disk(ds.root, "ds/sub-01.nxr.zarr/sources/pet") == attributes_text(ds.db, "field", ids["fa"])
    assert not (new / "sub-02.nxr.zarr/sources/pet").exists()          # never a group made where a node is missing
    with pytest.raises(RuntimeError, match="user_version 47"):
        c = sqlite3.connect(tmp_path / "x.sqlite"); c.execute("PRAGMA user_version = 47"); c.close()
        sd.migrate(tmp_path / "x.sqlite", tmp_path / "y" / "ds")


def test_a_fragment_carries_the_template_read_only(shared, tmp_path):
    folder, ids = shared
    with open_dataset(folder) as ds:
        t, tm, _ = subject(ds, "@default_subject", kind="template", folder="@default_subject.nxr.zarr")
        before = ds.db.all("SELECT * FROM manifold WHERE subject_id = ? ORDER BY id", t.id)
    frag = Path(sd.fragment(folder, "sub-01", tmp_path / "frag")["fragment"])
    assert (frag / "@default_subject.nxr.zarr").is_symlink() and json.loads((frag / "fragment.json").read_text())["templates"] == [t.id]
    with open_dataset(frag) as ds:
        assert ds.db.all("SELECT * FROM manifold WHERE subject_id = ? ORDER BY id", t.id) == before
        assert ds.db.value("SELECT count(*) FROM topology_cell WHERE topology_id = (SELECT topology_id FROM manifold WHERE id = ?)", tm) > 0
        ds.db.conn.execute("UPDATE manifold SET name = 'changed' WHERE id = ?", (tm,))
        ds.db.conn.commit()
    sd.merge(folder, [frag])
    with open_dataset(folder) as ds:
        assert ds.db.all("SELECT * FROM manifold WHERE subject_id = ? ORDER BY id", t.id) == before     # never merged back


# ── a NEW subject converted in parallel (fragment --new, merge) — 0.2.12 ─────────────────────────────────────────────────────────

import re
import shutil

from nxr_convert.crud import remove_node

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?")


def convert(ds, name):
    """what a job does to a subject: its surface and map (``subject``), a dataset dictionary and a labelling read with it"""
    sub, mid, _ = subject(ds, name)
    did, _ = sub.dictionary("qc flag", ["pass", "check", "fail"], scope="dataset")
    sub.selection(name="flag", path="sources/flag", type="set", manifold_id=mid, dictionary_id=did, data=np.array([0, 1, 2, 1], np.int32))


def canon(folder: Path) -> dict:
    """a dataset's CONTENT with ids and stamps aside: every table's rows, every store file (a zarr.json's ids and stamps masked)"""
    conn = sqlite3.connect(folder / "dataset.sqlite")
    rows = {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT IN ('sync_job', 'meta')"):
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")') if r[1] != "id" and not r[1].endswith("_id") and not r[1].endswith(("_utc", "_by"))]
        rows[t] = sorted(json.dumps([UUID.sub("<id>", STAMP.sub("<t>", str(v))) for v in r]) for r in conn.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{t}"')) if cols else []
    conn.close()
    files = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != "dataset.sqlite" and ".replaced-" not in str(p):
            b = p.read_bytes()
            files[str(p.relative_to(folder))] = UUID.sub("<id>", STAMP.sub("<t>", b.decode())) if p.name == "zarr.json" else b
    return {"rows": rows, "files": files}


def two_copies(shared, tmp_path):
    folder, _ = shared
    a, b = tmp_path / "serial", tmp_path / "parallel"
    shutil.copytree(folder.parent, a); shutil.copytree(folder.parent, b)
    return a / folder.name, b / folder.name


def test_new_fragments_merged_equal_a_serial_build(shared, tmp_path):
    serial, parallel = two_copies(shared, tmp_path)
    with open_dataset(serial) as ds:                                   # the serial build: one writer, one subject after another
        convert(ds, "sub-03"); convert(ds, "sub-04")
    frags = []
    for name in ("sub-03", "sub-04"):                                  # the parallel build: an empty fragment a job, then one merge
        fr = sd.fragment(parallel, None, tmp_path / f"job-{name}", new=True)
        assert fr["new"] and "store" not in fr
        with open_dataset(fr["fragment"]) as ds:
            assert ds.db.value("SELECT count(*) FROM field") == 0     # the shared rows only — no subject's
            convert(ds, name)
        frags.append(fr["fragment"])
    out = sd.merge(parallel, frags)
    assert [m["subject"] for m in out["merged"]] == ["sub-03", "sub-04"] and out["stores"] == {"renamed": 2} and out["foreign_key_violations"] == 0
    assert not any((Path(f) / f"{n}.nxr.zarr").exists() for f, n in zip(frags, ("sub-03", "sub-04")))   # moved, not left behind
    a, b = canon(serial), canon(parallel)
    assert a["rows"] == b["rows"]                                       # the same rows, ids and stamps aside
    assert a["files"].keys() == b["files"].keys() and a["files"] == b["files"]   # the same store tree, the same bytes
    cat = json.loads((parallel.parent / "catalog.json").read_text())
    assert cat["datasets"][0]["subjects"] == 4


def test_a_store_on_another_filesystem_is_copied_then_the_fragments_copy_removed(shared, tmp_path, monkeypatch):
    folder, _ = shared
    fr = sd.fragment(folder, None, tmp_path / "job", new=True)
    with open_dataset(fr["fragment"]) as ds:
        convert(ds, "sub-03")
    real = os.rename
    monkeypatch.setattr(sd.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError(18, "cross-device")) if str(a).endswith("sub-03.nxr.zarr") else real(a, b))
    out = sd.merge(folder, [fr["fragment"]])
    assert out["merged"][0]["store"] == "copied" and (folder / "sub-03.nxr.zarr/sources/flag/zarr.json").exists()
    assert not (Path(fr["fragment"]) / "sub-03.nxr.zarr").exists()


def test_a_name_the_dataset_holds_is_refused_and_replace_moves_the_old_store_aside(shared, tmp_path):
    folder, ids = shared
    fr = sd.fragment(folder, None, tmp_path / "job", new=True)
    with open_dataset(fr["fragment"]) as ds:                           # a re-conversion: the name removed in the fragment, then converted
        remove_node(ds.db, ds.root, "subject", ds.db.value("SELECT id FROM subject WHERE name = 'sub-02'"))
        convert(ds, "sub-02")
    with pytest.raises(RuntimeError, match="already holds a subject 'sub-02'"):
        sd.merge(folder, [fr["fragment"]])
    assert (Path(fr["fragment"]) / "sub-02.nxr.zarr").is_dir()         # refused: nothing moved
    out = sd.merge(folder, [fr["fragment"]], replace=True)
    aside = Path(out["merged"][0]["replaced_store_moved_to"])
    assert aside.is_dir() and aside.parent == folder                   # the old store kept, beside the dataset's
    with open_dataset(folder) as ds:
        new = ds.db.one("SELECT * FROM subject WHERE name = 'sub-02'")
        assert new["id"] != ids["b"] and ds.db.value("SELECT count(*) FROM field WHERE subject_id = ?", ids["b"]) == 0
        assert ds.db.value("SELECT count(*) FROM sync_job WHERE done_utc IS NULL") == 0
    assert (folder / "sub-02.nxr.zarr/sources/flag/zarr.json").exists()   # the new store in place: the old one's delete never ran on it


def test_a_failed_merge_puts_the_store_back(shared, tmp_path, monkeypatch):
    folder, _ = shared
    fr = sd.fragment(folder, None, tmp_path / "job", new=True)
    with open_dataset(fr["fragment"]) as ds:
        convert(ds, "sub-03")
    real = sd._insert
    monkeypatch.setattr(sd, "_insert", lambda conn, t, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")) if t == "field" else real(conn, t, *a, **k))
    with pytest.raises(RuntimeError, match="boom"):
        sd.merge(folder, [fr["fragment"]])
    assert (Path(fr["fragment"]) / "sub-03.nxr.zarr").is_dir() and not (folder / "sub-03.nxr.zarr").exists()
    with open_dataset(folder) as ds:
        assert ds.db.one("SELECT 1 FROM subject WHERE name = 'sub-03'") is None


def test_fragment_takes_a_subject_or_new_not_both(shared, tmp_path):
    folder, _ = shared
    with pytest.raises(ValueError):
        sd.fragment(folder, "sub-01", tmp_path / "x", new=True)
    with pytest.raises(ValueError):
        sd.fragment(folder, None, tmp_path / "y")
