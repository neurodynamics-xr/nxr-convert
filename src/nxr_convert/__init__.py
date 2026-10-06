"""nxr-convert — Brainstorm protocols into nxr datastores (the Cortical Flow desktop app's importer)."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("nxr-convert")
except PackageNotFoundError:  # run from a source tree that was never installed
    __version__ = "0+unknown"
