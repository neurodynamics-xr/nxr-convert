"""The filesystem traversal, as a SourceView.

Kept, not replaced. It is the answer for a protocol with no index — half-copied,
hand-assembled, or pruned outside Brainstorm — and it is the other side of the
cross-check: the index says what Brainstorm believes, this says what is on disk.

It is a thin adapter over `protocol.list_subjects`, which already encodes the hard
-won knowledge about `@raw` conditions (the kernel and head model live on the raw
side in the modern workflow, so scanning only plain dirs reads as "0 kernels").
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..protocol import list_subjects
from .model import SourceView, SourceSubject, SourceCondition, SourceRecording
from .protocol_mat import _importability, _mtime


def read_walk(root: str | Path) -> SourceView:
    root = Path(root)
    anat = root / "anat"
    subjects = []
    for s in list_subjects(root):
        # `list_subjects` keys the subject as 'subject', not 'name'. Reading the
        # wrong key and falling back to `str(s)` stringified the entire record into
        # a PATH — which surfaced as "File name too long" from `is_dir()` rather
        # than as a missing key, i.e. a filesystem error for a data-shape mistake.
        name = s["subject"]
        conds = s.get("conditions", [])
        sdir = anat / name
        out_conds = []
        for c in conds:
            recs = [SourceRecording(file=r, comment=Path(r).stem) for r in c.get("recordings", [])]
            recs += [SourceRecording(file=r, comment=Path(r).stem, is_raw_link=True,
                                     raw_format=c.get("raw_format"))
                     for r in c.get("raw_links", [])]
            cond = SourceCondition(
                name=c.get("name", ""),
                recordings=recs,
                kernels=list(c.get("kernels", [])),
                channel_file=(c.get("channel_files") or [None])[0],
                # The walk cannot know these without opening files, and opening files
                # is the cost this whole design exists to avoid. Absent, not zero.
                n_channels=None,
            )
            # SAME rule as the index reader. Two readers of one protocol reporting
            # different importability for the same condition is a difference in the
            # READER, not in the data, and it would read as the latter.
            cond.importability = _importability(cond, name)
            out_conds.append(cond)
        subjects.append(SourceSubject(
            name=name, conditions=out_conds, anatomy=sdir.is_dir(),
            surfaces=list(s.get("surfaces", [])) or (sorted(p.name for p in sdir.glob("tess_*.mat")) if sdir.is_dir() else []),
        ))
    return SourceView(
        root=str(root), kind="walk", subjects=subjects,
        scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The WALK reads the filesystem, so it has no index to go stale against
        # — but it records the mtime anyway when there IS one, because the
        # database can still move under a walk and a caller comparing views
        # should not have to know which reader produced this one.
        index_mtime=_mtime(root / "data" / "protocol.mat"),
    )
