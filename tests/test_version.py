"""``nxr-convert --version`` names the package version and the database schema it writes."""
import pytest

from nxr_convert import __version__
from nxr_convert.cli import main
from nxr_convert.db import schema_version


def test_version_names_the_package_and_the_schema(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"nxr-convert {__version__} (database schema {schema_version()})"
