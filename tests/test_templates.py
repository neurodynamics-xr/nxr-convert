"""templates: fsaverage ico7 + reference maps as Fields, re-runnable. Skips unless NXR_TEST_REFS points at the nsp folder."""
import os, sqlite3, pytest
from nxr_convert.templates import build

SRC = os.environ.get("NXR_TEST_REFS")


@pytest.mark.skipif(not SRC, reason="NXR_TEST_REFS (references-fsaverage-ico7) not set")
def test_build_and_rerun(tmp_path):
    ds = tmp_path / "ds" / "templates"
    out = build(SRC, ds)
    assert out["fields"] and not out["skipped"]
    c = sqlite3.connect(ds / "dataset.sqlite")
    assert c.execute("select n_vertices from manifold where path='sources/pial'").fetchone() == (327684,)
    assert c.execute("select count(*) from field").fetchone()[0] == len(out["fields"])
    again = build(SRC, ds)
    assert not again["fields"] and len(again["skipped"]) == len(out["fields"])


def test_the_maps_go_on_a_default_subject_s_own_surfaces(tmp_path):
    """no NXR_TEST_REFS needed: a subject that already has its surfaces takes the folder's maps, however many, each on the surface
    with its vertex count; no surface is written; a map no surface fits is reported, the rest still go on"""
    import json
    nib = pytest.importorskip("nibabel")
    import numpy as np
    from nxr_convert.crud import create_dataset
    from test_shared_dataset import subject

    ds = create_dataset(tmp_path / "store", "cohort")
    subject(ds, "@default_subject", kind="template", template_id=None)      # its primary surface: sources/c, 4 vertices
    ds.close()
    maps = tmp_path / "refs" / "maps"
    for mid, name, n in (("m1", "myelin", 2), ("m2", "thickness", 2), ("m3", "odd", 3)):
        (maps / mid).mkdir(parents=True)
        (maps / mid / "provenance.json").write_text(json.dumps({"map": name}))
        for h, off in (("L", 0), ("R", n)):
            img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.arange(off, off + n, dtype=np.float32))])
            nib.save(img, str(maps / mid / f"{name}_space-fsaverage_den-164k_hemi-{h}.func.gii"))
    out = build(tmp_path / "refs", tmp_path / "store" / "cohort", subject="@default_subject")
    assert out["fields"] == ["myelin", "thickness"] and out["surfaces"] == []
    assert [u["map"] for u in out["unplaced"]] == ["odd"] and "0 surfaces with 6 vertices" in out["unplaced"][0]["reason"]
    c = sqlite3.connect(tmp_path / "store" / "cohort" / "dataset.sqlite")
    assert c.execute("select count(*) from manifold where type = 'surface'").fetchone() == (1,)
    assert c.execute("select m.path from field f join manifold m on m.id = f.manifold_id where f.name = 'myelin'").fetchone() == ("sources/c",)
    assert not (tmp_path / "store" / "README.md").exists()                # another datastore's README never written
    assert build(tmp_path / "refs", tmp_path / "store" / "cohort", subject="@default_subject")["skipped"] == ["myelin", "thickness"]
