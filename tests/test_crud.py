"""THE PROCEDURE IN PYTHON (D141/D142/D145): per entity — create → lay out → populate → complete, update, delete — and
recovery from a crash injected after each step. What lands in zarr.json is the schema view's TEXT, byte for byte."""
import json
import sqlite3

import numpy as np
import pytest
import zarr

from nxr_convert.crud import (Subject, create_dataset, create_node, open_dataset, recover, remove_node, update_node,
                              write_catalog)
from nxr_convert.db import (attributes_text, drain, js_json, js_number, open_database, schema_version, text_parity, uuid7,
                            zarr_json_text)


@pytest.fixture
def ds(tmp_path):
    d = create_dataset(tmp_path / "store", "ds", source_tool="brainstorm")
    yield d
    d.close()


def raw(ds, rel):
    return (ds.root / rel / "zarr.json").read_text(encoding="utf-8")


def attrs_text_on_disk(ds, rel):
    t = raw(ds, rel)
    return t[t.index('"attributes":') + len('"attributes":'):-1]


def status(ds, table, id):
    r = ds.db.read(table, id)
    return r and r["status"]


# ── the database and the dataset ─────────────────────────────────────────────────────────────────────────────────────

def test_a_dataset_is_its_database_its_row_its_folder_and_the_catalog(ds, tmp_path):
    assert (ds.folder / "dataset.sqlite").is_file()
    assert ds.db.value("PRAGMA user_version") == schema_version() == 49
    a = json.loads(raw(ds, "ds"))["attributes"]
    assert a["class"] == "dataset" and a["schema_version"] == 49 and a["id"] == ds.id
    assert attrs_text_on_disk(ds, "ds") == attributes_text(ds.db, "dataset", ds.id)
    cat = json.loads((tmp_path / "store" / "catalog.json").read_text())
    assert cat["schema_version"] == 49 and cat["datasets"][0]["database"] == "ds/dataset.sqlite"
    assert cat["id"] == ds.db.value("SELECT id FROM datastore")               # the datastore's id, in every database
    assert ds.db.value("SELECT count(*) FROM dictionary WHERE name = 'frequency bands' AND scope = 'app'") == 1
    with pytest.raises(FileExistsError):
        create_dataset(tmp_path / "store", "ds")


def test_another_version_is_refused(tmp_path):
    p = tmp_path / "x.sqlite"
    open_database(p, create=True).close()
    c = sqlite3.connect(p); c.execute("PRAGMA user_version = 48"); c.commit(); c.close()
    with pytest.raises(RuntimeError, match="user_version 48"):
        open_database(p)
    with pytest.raises(FileNotFoundError):
        open_database(tmp_path / "absent.sqlite")


def test_js_numbers_are_javascripts():
    for x, want in [(0.0, "0"), (-0.0, "0"), (1.0, "1"), (1e-7, "1e-7"), (1e21, "1e+21"), (1e20, "100000000000000000000"),
                    (0.1, "0.1"), (123.456, "123.456"), (2.5e-5, "0.000025"), (5e-324, "5e-324"), (1.5e300, "1.5e+300")]:
        assert js_number(x) == want, (x, js_number(x))
    assert js_json({"a": [1, 0.5, None, True, "é"]}) == '{"a":[1,0.5,null,true,"é"]}'
    assert zarr_json_text({"zarr_format": 3, "node_type": "group", "attributes": {"x": 1}}, '{"y":2}') == \
        '{"zarr_format":3,"node_type":"group","attributes":{"y":2}}'


# ── per entity: create → lay out → populate → complete ───────────────────────────────────────────────────────────────

def world(ds):
    """A subject with a surface (manifold + its arrays) — written by the procedure, node by node."""
    sub = Subject.create(ds, "sub-01", source_format="brainstorm", immediate=True)
    seen = {}
    V, F = np.random.default_rng(0).random((4, 3)), np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def arrays(at):
        seen["status"] = status(ds, "manifold", mid)
        seen["laid_out"] = json.loads((at / "zarr.json").read_text())["attributes"]["status"]
        sub.write_array("sources/c/vertices", V)
        sub.write_array("sources/c/faces", F)
    mid = uuid7()
    sub.manifold(id=mid, name="c", path="sources/c", type="surface", n_vertices=4, n_faces=2, is_primary=1,
                 topology={"kind": "simplicial", "rank": 2, "cells": [(2, "sources/c/faces")]},
                 geometry={"form": "stored", "cs_key": "scs", "positions_path": "sources/c/vertices"}, populate=arrays)
    return sub, mid, seen


def test_create_lays_out_then_populates_then_completes_every_entity(ds):
    sub, mid, seen = world(ds)
    assert seen == {"status": "pending", "laid_out": "pending"}                # laid out from the row BEFORE the bytes
    assert status(ds, "manifold", mid) == "complete" and status(ds, "subject", sub.id) == "complete"
    store = f"ds/{sub.row['path']}"
    fid = sub.field(name="pet", path="sources/pet", kind="PET suvr", manifold_id=mid, data=np.arange(4, dtype=np.float32), unit="SUVR")
    oid = sub.operator(name="m", path="m_mass", kind="mass", from_manifold_id=mid, to_manifold_id=mid, layout="csr", n_rows=4, n_cols=4,
                       nnz=4, sparse={"indptr": np.arange(5, dtype=np.int32), "indices": np.arange(4, dtype=np.int32),
                                      "data": np.ones(4, np.float32)})
    did, codes = sub.dictionary("Structures", ["unassigned", "Cortex L", "Cortex R"], scope="app")
    sid = sub.selection(name="Structures", path="sources/c_Structures", type="set", manifold_id=mid, dictionary_id=did,
                        data=np.array([1, 1, 2, 2], np.int32))
    sp = sub.selection(name="ev", path="sources/ev", type="spans", manifold_id=mid, cell="1", spans=[(0.0, 1.0, 0), (2.0, 2.0, 1)])
    for table, id, rel in (("subject", sub.id, store), ("manifold", mid, f"{store}/sources/c"), ("field", fid, f"{store}/sources/pet"),
                           ("operator", oid, f"{store}/m_mass"), ("selection", sid, f"{store}/sources/c_Structures"),
                           ("selection", sp, f"{store}/sources/ev")):
        assert status(ds, table, id) == "complete"
        assert attrs_text_on_disk(ds, rel) == attributes_text(ds.db, table, id), table          # THE VIEW'S TEXT
    doc = json.loads(raw(ds, f"{store}/sources/pet"))
    assert list(doc)[:8] == ["zarr_format", "node_type", "shape", "data_type", "chunk_grid", "chunk_key_encoding", "fill_value", "codecs"]
    row = ds.db.read("field", fid)                                                                # the array FROM the row
    assert (json.loads(row["chunk_shape_json"]), json.loads(row["codecs_json"]), row["fill_value_json"]) == \
        (doc["chunk_grid"]["configuration"]["chunk_shape"], doc["codecs"], "0")
    assert np.array_equal(zarr.open_array(str(ds.root / store / "sources/pet"))[:], np.arange(4))
    assert ds.db.value("SELECT count(*) FROM selection_element WHERE selection_id = ?", sid) == 4
    assert ds.db.read("selection", sid)["n_members"] == 2 and ds.db.read("selection", sp)["n_members"] == 2
    assert text_parity(ds.db, ds.root)["mismatched"] == 0


def test_a_failure_while_populating_rolls_the_node_back(ds):
    sub, mid, _ = world(ds)

    def dies(at):
        raise RuntimeError("the converter died")
    with pytest.raises(RuntimeError, match="died"):
        sub.field(name="f", path="sources/f", kind="x", manifold_id=mid, data=np.zeros(4, np.float32), populate=dies)
    assert ds.db.one("SELECT 1 FROM field WHERE name = 'f'") is None
    assert not (sub.store / "sources/f").exists()


def test_update_rewrites_the_attributes_from_the_row_and_delete_removes_node_then_row(ds):
    sub, mid, _ = world(ds)
    fid = sub.field(name="pet", path="sources/pet", kind="PET", manifold_id=mid, data=np.ones(4, np.float32))
    update_node(ds.db, ds.root, lambda: ds.db.execute("UPDATE field SET unit = 'Bq/mL' WHERE id = ?", fid))
    assert json.loads(raw(ds, f"ds/{sub.row['path']}/sources/pet"))["attributes"]["unit"] == "Bq/mL"
    remove_node(ds.db, ds.root, "field", fid)
    assert ds.db.read("field", fid) is None and not (sub.store / "sources/pet").exists()
    # a manifold's delete takes what is on it (the cascade), their folders with them (the triggers)
    fid2 = sub.field(name="pet2", path="sources/pet2", kind="PET", manifold_id=mid, data=np.ones(4, np.float32))
    remove_node(ds.db, ds.root, "manifold", mid)
    assert ds.db.read("field", fid2) is None and not (sub.store / "sources/pet2").exists() and not (sub.store / "sources/c").exists()
    assert text_parity(ds.db, ds.root)["mismatched"] == 0


@pytest.mark.parametrize("table", ["manifold", "field", "operator", "selection", "subject"])
def test_a_crash_after_each_step_is_recovered_at_open(ds, table):
    """Laid out but never completed (pending) → rolled back; marked for deletion (deleting) → finished. Idempotent."""
    sub, mid, _ = world(ds)

    def make(s):
        return {
            "manifold": lambda: s.manifold(name="p", path="sources/p", type="points", n_vertices=2, topology={"kind": "none", "rank": 0}),
            "field": lambda: s.field(name="f", path="sources/f", kind="x", manifold_id=mid, data=np.ones(4, np.float32)),
            "operator": lambda: s.operator(name="o", path="o", kind="x", from_manifold_id=mid, layout="dense", n_rows=4, n_cols=4,
                                           data=np.eye(4, dtype=np.float32)),
            "selection": lambda: s.selection(name="s", path="sources/s", type="indices", manifold_id=mid, data=np.array([0, 2], np.int32), picked=True),
        }[table]()
    if table == "subject":
        gone = Subject.create(ds, "sub-02", immediate=True).id
        half = Subject.create(ds, "sub-03")                       # pending: laid out (below), never completed
        half_id = half.id
    else:
        gone = make(sub)                                           # complete
        ds.db.execute(f"UPDATE {table} SET path = path || '_gone' WHERE id = ?", gone)
        drain(ds.db, ds.root)
        comp = Subject.open(ds, "sub-01")                          # a composition: rows pending, populated, not completed
        half_id = make(comp)
    drain(ds.db, ds.root)                                          # step 2 done …
    half_at = ds.root / (f"ds/{ds.db.read('subject', half_id)['path']}" if table == "subject"
                         else f"ds/{sub.row['path']}/{ds.db.read(table, half_id)['path']}")
    assert status(ds, table, half_id) == "pending" and (half_at / "zarr.json").exists()
    ds.db.execute(f"UPDATE {table} SET status = 'deleting' WHERE id = ?", gone)        # … and another dies mid-delete
    assert recover(ds.db, ds.root) == {"rolled_back": 1, "finished": 1}
    assert ds.db.read(table, half_id) is None and ds.db.read(table, gone) is None and not half_at.exists()
    assert recover(ds.db, ds.root) == {"rolled_back": 0, "finished": 0}                # idempotent
    assert text_parity(ds.db, ds.root)["mismatched"] == 0


def test_a_composition_completes_once_or_is_removed_whole(ds, tmp_path):
    sub = Subject.create(ds, "sub-01")
    with sub.composition():
        mid = sub.manifold(name="p", path="timeseries/ch", type="points", n_vertices=3, topology={"kind": "none", "rank": 0})
        sub.dictionary("ch names", ["a", "b", "c"])
        assert status(ds, "manifold", mid) == "pending" and status(ds, "subject", sub.id) == "pending"
    assert status(ds, "manifold", mid) == "complete" and status(ds, "subject", sub.id) == "complete"
    assert json.loads((tmp_path / "store" / "catalog.json").read_text())["datasets"][0]["subjects"] == 1
    bad = Subject.create(ds, "sub-02")
    with pytest.raises(ValueError):
        with bad.composition():
            bad.manifold(name="p", path="timeseries/ch", type="points", n_vertices=3, topology={"kind": "none", "rank": 0})
            sub_dict, _ = bad.dictionary("my names", ["x"], scope="app")
            raise ValueError("a reader refused")
    assert ds.subject("sub-02") is None and not bad.store.exists()
    assert ds.db.one("SELECT 1 FROM dictionary WHERE name = 'my names'") is None        # what it made and nothing reads
    # a crash mid-composition (the process gone, no abort): the next open rolls it back
    c = Subject.create(ds, "sub-03")
    c.manifold(name="p", path="timeseries/ch", type="points", n_vertices=3, topology={"kind": "none", "rank": 0})
    drain(ds.db, ds.root)
    ds.db.close()
    ds.lock.release()          # the process gone: its writer lock goes stale (taken over by the next writer — test_writer_lock)
    again = open_dataset(ds.folder)
    assert again.subject("sub-03") is None and not c.store.exists() and again.subject("sub-01")["status"] == "complete"
    ds.db, ds.lock = again.db, again.lock


def test_a_dictionary_codes_every_subject_alike(ds):
    a = Subject.create(ds, "a", immediate=True)
    b = Subject.create(ds, "b", immediate=True)
    _, ca = a.dictionary("Desikan-Killiany", ["unassigned", "x L", "y L"], scope="app")
    _, cb = b.dictionary("Desikan-Killiany", ["unassigned", "y L", "z R"], scope="app")
    assert ca == [0, 1, 2] and cb == [0, 2, 3]                              # existing names keep their codes; new ones appended
    _, cc = a.dictionary("type", ["MEG", "EEG"], scope="app")
    colours = [json.loads(r["color_json"]) for r in ds.db.all("SELECT color_json FROM dictionary_entry e JOIN dictionary d ON d.id = e.dictionary_id WHERE d.name = 'type' ORDER BY code")]
    assert colours == [[143, 211, 244], [118, 183, 178]]                     # the app's family colours (D111)


def test_the_catalog_is_rewritten_atomically(ds, tmp_path):
    create_dataset(tmp_path / "store", "other").close()
    cat = write_catalog(tmp_path / "store")
    assert [d["path"] for d in cat["datasets"]] == ["ds", "other"] and not (tmp_path / "store" / "catalog.json.tmp").exists()
