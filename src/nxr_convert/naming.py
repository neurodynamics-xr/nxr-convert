"""Node naming — Brainstorm's own file stems (decision 4, 2026-09-21).

A node is named by the Brainstorm file it came from, with the file's TYPE prefix
removed where the prefix is a kind marker rather than part of the identity::

    tess_cortex_pial_low.mat        -> cortex_pial_low
    data_0raw_sub-0002_..._high.mat -> sub-0002_..._high      (a raw link)
    data_block001_02_band.mat       -> block001_02_band       (an imported excerpt)
    results_MN_MEG_KERNEL_..._1117  -> MN_MEG_KERNEL_..._1117
    channel_ctf_acc1.mat            -> channel_ctf_acc1       (prefix kept: it IS the name)
    headmodel_surf_os_meg.mat       -> headmodel_surf_os_meg  (likewise)
    subjectimage_MRI_T1.mat         -> T1                     (``mri.mri_name``)

Brainstorm's ``Comment`` (``Raw | notch(60Hz) | high(0.3Hz)``, ``MN: MEG(Unconstr)
2018``) is processing HISTORY and goes in the node's ``comment`` attribute, never
in its name. The converter's old literal ``"continuous"`` was never a Brainstorm
name and is gone. Nodes with no Brainstorm file are named after the node they
belong to: ``<recording>_time``, ``<recording>_flags``, ``<channels>_names``.

SESSION-SCOPED NODES — the channels, head model and kernels a condition fits — are
named by the same stems, and Brainstorm reuses a stem in every condition
(``channel_ctf_acc1``, ``headmodel_surf_os_meg``, ``results_dSPM…_KERNEL_…``). A
recording's stem carries its task and run, so recordings never collide; these do.
``claim_node_name`` keeps the stem for the FIRST session, SHARES the node when a later
session's source is byte-identical, and otherwise qualifies the name with the later
session's label (``channel_ctf_acc1__task-rest_run-02``) — never overwriting what an
earlier session wrote (register D121, amending decision 4).

``sanitize_node_name`` is the ``sanitize_node_name.m`` port, held byte-for-byte
by the oracle store's names: a run of non-alphanumeric characters (hyphen, dot
and underscore are safe) collapses to one underscore; ends are trimmed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def sanitize_node_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(name))
    out = re.sub(r"^[_.\-]+|[_.\-]+$", "", out)
    if not out:
        raise ValueError(f"name sanitizes to nothing: {name!r}")
    return out


def _stem_minus(path: str | Path, prefix: str) -> str:
    return sanitize_node_name(re.sub(prefix, "", Path(path).stem))


def surface_node_name(surface_file: str | Path) -> str:
    """``tess_cortex_pial_low.mat`` -> ``cortex_pial_low``."""
    return _stem_minus(surface_file, r"^tess_")


def recording_node_name(data_file: str | Path) -> str:
    """``data_0raw_<x>.mat`` / ``data_<x>.mat`` -> ``<x>``."""
    return _stem_minus(data_file, r"^data_(0raw_)?")


def kernel_node_name(results_file: str | Path) -> str:
    """``results_<x>.mat`` -> ``<x>``."""
    return _stem_minus(results_file, r"^results_")


def channel_node_name(channel_file: str | Path) -> str:
    """``channel_ctf_acc1.mat`` -> ``channel_ctf_acc1`` — the prefix stays."""
    return sanitize_node_name(Path(channel_file).stem)


def headmodel_node_name(headmodel_file: str | Path) -> str:
    """``headmodel_surf_os_meg.mat`` -> ``headmodel_surf_os_meg`` — the prefix stays."""
    return sanitize_node_name(Path(headmodel_file).stem)


# BIDS entities that tell two conditions of one subject apart, in BIDS order.
_SESSION_ENTITIES = ("ses", "task", "acq", "run", "rec")


def session_label(condition: str) -> str:
    """The short, human part of a condition that tells it from its siblings:
    ``sub-X_ses-02_task-rest_run-02_meg_resample`` -> ``ses-02_task-rest_run-02``.
    A condition with no BIDS entities is its own label."""
    found = [m.group(0) for e in _SESSION_ENTITIES
             for m in [re.search(rf"(?<![A-Za-z0-9]){e}-[A-Za-z0-9]+", condition)] if m]
    return sanitize_node_name("_".join(found) if found else condition)


def source_fingerprint(files: Iterable[str | Path], extra: bytes = b"") -> str:
    """sha1 over the source files' bytes (+ ``extra``, e.g. a session's channel
    flags) — what makes two sessions' nodes the SAME node."""
    h = hashlib.sha1()
    for f in files:
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    h.update(extra)
    return h.hexdigest()


def claim_node_name(sub, folder: str, stem: str, session: str, fingerprint: str) -> tuple[str, bool]:
    """The name a session-scoped node gets in the subject ``sub``, and whether it already EXISTS as-is.

    Returns ``(stem, False)`` when no row has that path yet; ``(name, True)`` when a row of that path holds
    byte-identical source (its ``producer_json.source_sha1`` — share it, write nothing); otherwise the stem qualified by
    ``session_label(session)`` — and, if even that is taken by other content, a numbered suffix. Deterministic for a fixed
    conversion order.
    """
    label = session_label(session)
    candidates = [stem, f"{stem}__{label}"] + [f"{stem}__{label}_{k}" for k in range(2, 100)]
    for name in candidates:
        path = f"{folder}/{name}" if folder else name
        row = next((r for t in ("manifold", "operator", "field", "selection") if (r := sub.find(t, path)) is not None), None)
        if row is None:
            return name, False
        if json.loads(row.get("producer_json") or "{}").get("source_sha1") == fingerprint:
            return name, True
    raise ValueError(f"{stem}: 100 different sources claim this name in {sub.name}")
