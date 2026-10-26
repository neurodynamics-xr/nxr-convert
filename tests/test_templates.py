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
