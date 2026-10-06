"""The packaged DDL (``nxr_convert/model.sql``) is the monorepo's ``backend/schema/model.sql`` byte for byte — checked when
the package sits in the monorepo (heal it with ``node backend/scripts/schema-module.mjs``); standalone, only that the
packaged copy is readable and declares its version."""
from importlib import resources
from pathlib import Path

import pytest

from nxr_convert.db import model_sql, schema_version

MONOREPO_SQL = Path(__file__).resolve().parents[3] / "schema" / "model.sql"


def test_the_packaged_ddl_is_readable_and_versioned():
    assert model_sql().startswith("--") or "CREATE TABLE" in model_sql()
    assert schema_version() > 0


@pytest.mark.skipif(not MONOREPO_SQL.is_file(), reason="standalone: no monorepo backend/schema/model.sql to compare")
def test_the_packaged_ddl_is_the_monorepo_schema_verbatim():
    packaged = resources.files("nxr_convert").joinpath("model.sql").read_bytes()
    assert packaged == MONOREPO_SQL.read_bytes(), "nxr_convert/model.sql is stale — run `node backend/scripts/schema-module.mjs`"
