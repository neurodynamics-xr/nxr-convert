"""Put tests/ on the path so `from eeg_fixture import ...` resolves without
making the fixture part of the installed package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402


@pytest.fixture
def sub(tmp_path):
    """A subject being composed in a fresh dataset (``crud.Subject``): ``sub.complete()`` drains it."""
    from nxr_convert.crud import Subject, create_dataset
    ds = create_dataset(tmp_path / "store", "d")
    s = Subject.create(ds, "s")
    yield s
    ds.close()


def row_at(sub, table, path):
    """The row of ``table`` at a store path of the subject."""
    return sub.db.one(f"SELECT * FROM {table} WHERE subject_id = ? AND path = ?", sub.id, path)
