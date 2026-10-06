"""Brainstorm protocol traversal — what the Data page's Import stage lists.

A protocol is ``<root>/{anat,data}/<SubjectName>/…``: anatomy holds
``tess_*.mat`` surfaces; each data subdirectory is a CONDITION holding
``channel_*.mat``, ``data_*.mat`` recordings, ``results_*KERNEL*.mat``
inverse kernels and ``headmodel_*.mat`` head models. Listing reads NAMES and
small headers only — it must stay fast enough to answer a UI click.
"""
from __future__ import annotations

from pathlib import Path


def _raw_format(raw_dir: Path, names: list[str]) -> str | None:
    """The first raw link's sFile format ('BST-BIN' imports; others are
    listed but not importable). One small .mat header read."""
    if not names:
        return None
    try:
        from .matio import load_mat
        f = load_mat(raw_dir / names[0]).get("F")
        return str(f.get("format")) if isinstance(f, dict) else None
    except Exception:  # noqa: BLE001 — an unreadable link lists as unknown
        return None


def list_subjects(root: str | Path) -> list[dict]:
    root = Path(root)
    data = root / "data"
    anat = root / "anat"
    if not data.is_dir():
        raise FileNotFoundError(f"not a Brainstorm protocol (no data/): {root}")
    out = []
    for subj in sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith("@")):
        # A CONDITION is a base name with up to two homes (2026-08-14, the
        # raw-first workflow): the plain dir holds "Import in database" excerpts
        # (data_*.mat) and the @raw<base> dir holds the raw LINK — plus,
        # in the modern Brainstorm workflow, THE KERNEL and head model, which
        # are computed directly ON the raw link and never leave that folder.
        # Scanning only plain dirs is how omega's kernels read as "0 kernels".
        # A raw-only condition (nothing imported in Brainstorm's database at
        # all) is fully importable from the treated .bst — no duplication.
        bases: dict[str, dict] = {}
        for cond in sorted(p for p in subj.iterdir() if p.is_dir()):
            is_raw = cond.name.startswith("@raw")
            if cond.name.startswith("@") and not is_raw:
                continue                       # @default_study / @intra
            base = cond.name[4:] if is_raw else cond.name
            e = bases.setdefault(base, {
                "name": base, "recordings": [], "kernels": [], "channel_files": [],
                "raw_links": [], "raw_format": None,
            })
            kerns = sorted(p.name for p in cond.glob("results_*KERNEL*.mat"))
            chans = sorted(p.name for p in cond.glob("channel_*.mat"))
            if is_raw:
                raws = sorted(p.name for p in cond.glob("data_0raw_*.mat"))
                e["raw_links"] = raws
                e["raw_format"] = _raw_format(cond, raws)
                # Kernels computed on the raw link — full-fidelity inverses.
                e["kernels"] = sorted({*e["kernels"], *kerns})
                if not e["channel_files"]:
                    e["channel_files"] = chans
            else:
                e["recordings"] = sorted(p.name for p in cond.glob("data_*.mat"))
                e["kernels"] = sorted({*e["kernels"], *kerns})
                if chans:
                    e["channel_files"] = chans
        conditions = [e for e in bases.values()
                      if e["recordings"] or e["kernels"] or e["raw_links"]]
        conditions.sort(key=lambda e: e["name"])
        surfaces = sorted(p.name for p in (anat / subj.name).glob("tess_*.mat")) if (anat / subj.name).is_dir() else []
        # A subject with NO importable conditions (Brainstorm's Group_analysis
        # pseudo-subject is the live case) has nothing to tick — omit it from
        # the import list rather than rendering an empty row.
        if not conditions:
            continue
        out.append({"subject": subj.name, "conditions": conditions, "surfaces": surfaces})
    return out
