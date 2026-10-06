"""The SOURCE VIEW model — one shape, whatever read it.

A view is a CACHE of an external database (a Brainstorm protocol), not data and not
authority. It answers "what could I import, and would it work?" without opening a
single recording.

It is deliberately NOT shaped like `nxr.subject@2.0`. A metadata-only store in the
subject schema would be indistinguishable from real data to every reader in the app,
which is the "structurally perfect and empty" failure the repo already has one of
(root CLAUDE.md §5b — a K=400 basis of 4,096,800 zeros that read as a cache hit).
`schema: nxr.source@1.0` is a different node type on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

SCHEMA = "nxr.source@1.0"

# Whether a condition can be converted, and if not why — stated up front rather than
# discovered by running a conversion and reading the traceback.
#
# `ready-raw` IS IMPORTABLE. It was called `raw-only`, which reads as a barrier, and
# on a real 111-subject cohort that mislabelled 228 of 343 conditions — every resting
# recording held as a raw link. A raw condition converts fine from the treated .bst
# (the filesystem walk's own comment has said so all along); the provenance is worth
# stating, the exclusion would have been wrong. `IMPORTABLE` is the set, so no caller
# has to remember which strings mean yes.
Importability = Literal[
    "ready", "ready-raw", "missing-kernel", "unsupported-raw", "no-channels",
    # A GROUP RESULT, not a source recording. Brainstorm's `Group_analysis`
    # subject holds cross-subject outputs; it is a legitimate database object and
    # is LISTED rather than hidden — hiding data is how a reader learns not to
    # trust the list — but it can never become an `nxr.subject@2.0` session.
    "group-analysis",
]
IMPORTABLE: frozenset[str] = frozenset({"ready", "ready-raw"})

# Brainstorm's own name for the cross-subject workspace. Not configurable: it is
# a fixed identifier in `bst_get`, not a user-chosen subject name.
GROUP_ANALYSIS_SUBJECT = "Group_analysis"


@dataclass
class SourceRecording:
    """One `data_*.mat` (or raw link) as the database describes it."""
    file: str                      # Brainstorm-relative FileName
    comment: str                   # what a human called it — the useful name
    data_type: str = ""
    bad_trial: bool = False
    is_raw_link: bool = False
    raw_format: str | None = None  # 'BST-BIN' imports; others list but do not

    # THE PRIMARY FACT ABOUT A RESTING-STATE RUN, and it was absent everywhere.
    # The index does not carry any of these — Brainstorm keeps them in the raw
    # LINK, a ~8 KB file beside the data — so they arrive with `raw_format`
    # from the same read rather than from a second pass.
    duration_s: float | None = None
    sfreq: float | None = None
    n_samples: int | None = None


@dataclass
class SourceCondition:
    name: str
    recordings: list[SourceRecording] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)      # results_*KERNEL* Comments
    head_models: list[str] = field(default_factory=list)
    channel_file: str | None = None
    n_channels: int | None = None
    modalities: list[str] = field(default_factory=list)
    # THE SURFACE THE INVERSE IS DEFINED ON — the kernel's own `SurfaceFile`.
    #
    # Not a preference and never a choice: every operand the analysis builds
    # (the mass, the eigenbases, the Dirac fusion) is indexed by this surface's
    # vertices, so a store whose manifold is a different tess_*.mat has a kernel
    # whose rows do not correspond to it. It loads, it renders, it is wrong.
    #
    # None when the condition has no kernel, and when its kernels DISAGREE —
    # two inverses on two surfaces is a real protocol and picking one silently
    # would index the second by the wrong manifold.
    primary_surface: str | None = None
    importability: Importability = "ready"
    # Set when the index and the disk disagree — a protocol copied without its
    # protocol.mat, or edited outside Brainstorm. Reported, never silently repaired.
    discrepancy: str | None = None


@dataclass
class SourceSurface:
    """One entry of a subject's `Surface` list, as the INDEX states it.

    Richer than the filename the scan used to glob for: `comment` is
    Brainstorm's own label (`cortex_20484V`), which is what a surface picker
    should show, and `type` is the classification (`Cortex`, `Scalp`, `Other`)
    rather than something inferred from the name.

    `present` is the one DISK touch, and it is VERIFICATION, not discovery: a
    surface can be listed and absent — a protocol pruned outside Brainstorm,
    which is exactly what a `_clean` copy is — and that fact is reported, never
    silently repaired.
    """

    file: str                    # `tess_cortex_pial_low.mat`
    comment: str = ""            # `cortex_20484V`
    type: str = ""               # `Cortex` | `Scalp` | `Other` | …
    present: bool = True


@dataclass
class SourceSubject:
    name: str
    conditions: list[SourceCondition] = field(default_factory=list)
    surfaces: list[SourceSurface] = field(default_factory=list)
    anatomy: bool = False

    # THE CORTEX, as `iCortex` NAMES it — not a guess. Brainstorm records which
    # surface is the cortex; the scan hardcoded `tess_cortex_pial_low.mat`,
    # which is right for this protocol by coincidence rather than by fact.
    cortex: str | None = None
    scalp: str | None = None

    # `UseDefaultAnat` — the subject borrows `@default_subject`'s anatomy. A
    # subject with no `anat/<name>/` of its own is NOT anatomy-less when this is
    # set; it inherits. Ignoring it makes a perfectly importable subject read as
    # "no anatomy", which is a confident wrong answer rather than a gap.
    uses_default_anat: bool = False
    # Set when the index and the disk disagree — listed surfaces that are not
    # there. Reported, never silently repaired (the `SourceCondition` precedent).
    discrepancy: str | None = None


@dataclass
class SourceView:
    root: str
    kind: Literal["protocol.mat", "sqlite", "walk"]
    subjects: list[SourceSubject] = field(default_factory=list)
    db_version: float | None = None
    scanned_at: str = ""
    # `data/protocol.mat`'s mtime WHEN THIS VIEW WAS TAKEN.
    #
    # A scan is a SNAPSHOT of a live database, and this one moved under us:
    # measured 2026-08-17, the same command read 6 subjects / 10 conditions and,
    # ten minutes later, 7 / 11 — because a MATLAB job in another session had
    # rewritten the index and created `data/Group_analysis/`. Importing off a
    # view that no longer matches the database is a silent-wrong-data path, so
    # the mtime travels with the view and a caller can tell.
    index_mtime: float | None = None
    protocol_comment: str | None = None
    schema: str = SCHEMA

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def counts(self) -> dict[str, int]:
        conds = [c for s in self.subjects for c in s.conditions]
        return {
            "subjects": len(self.subjects),
            "conditions": len(conds),
            "recordings": sum(len(c.recordings) for c in conds),
            "kernels": sum(len(c.kernels) for c in conds),
            "importable": sum(1 for c in conds if c.importability in IMPORTABLE),
        }
