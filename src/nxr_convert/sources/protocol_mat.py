"""Read `<root>/data/protocol.mat` — Brainstorm's own database index.

ONE READ instead of a traversal. Measured on the tutorial protocol: 78 KB holding
every subject, study, recording Comment, channel count and modality list. The
filesystem walk recovers a subset of this from globs plus a `.mat` header open per
raw link.

WHAT THE INDEX GIVES THAT THE DISK DOES NOT: `Comment` — what a human named the
recording. A user picking what to import reads "Right / Left / Avg: deviant", not
`data_block001_02.mat`.

WHAT THE DISK GIVES THAT THE INDEX DOES NOT: the truth. The index is Brainstorm's
belief, written when Brainstorm last saved; a protocol copied, pruned or edited
outside it will disagree. `read_source(..., prefer='walk')` reads the other side and
`cross_check` diffs them — nothing here silently reconciles the two.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import (
    SourceView, SourceSubject, SourceSurface, SourceCondition, SourceRecording,
    Importability, GROUP_ANALYSIS_SUBJECT,
)


def _records(st: Any, by: str = "FileName") -> list[dict]:
    """A MATLAB struct ARRAY → a list of records.

    `pymatreader` returns a struct array as a STRUCT-OF-ARRAYS — one key per field,
    each holding N parallel values — not as an array of structs. So
    `ProtocolStudies.Study` is `{'Name': [7 names], 'Data': [7 entries], …}`, and
    reading it as though it were a list of 7 dicts yields ONE pseudo-record whose
    every field is the whole column. That is not a crash: it produces a subject
    literally named after a joined list, which is how this was caught.

    A 1-element struct array comes back with scalars rather than 1-lists, so the
    length is the longest field and scalars broadcast.
    """
    if not isinstance(st, dict) or not st:
        return []
    # COUNT ON A FIELD THAT IS SCALAR PER RECORD. Taking the longest list is wrong
    # whenever a field is legitimately list-valued: a single Channel record carries
    # `Modalities: ['ADC A','ADC V',…,'MEG',…]`, thirteen entries, and inferring the
    # count from it transposed one channel file into thirteen phantom records — each
    # keeping one modality, which is why the first read reported `['ADC A']` for CTF
    # MEG data. `FileName`/`Name` are strings, one per record, so they cannot lie.
    key = next((k for k in (by, "FileName", "Name") if k in st), None)
    if key is None:
        return [st]
    col = st[key]
    n = len(col) if isinstance(col, list) else 1
    # A 1-ELEMENT STRUCT ARRAY IS THE RECORD ITSELF — do not index into it. Taking
    # `v[0]` of every list-valued field turned one channel record's thirteen
    # modalities into the single value 'ADC A', which reads as a plausible answer
    # (it IS a modality) rather than as a parse error. The same shape would silently
    # keep only the first of anything else list-valued.
    if n == 1:
        return [dict(st)]
    out: list[dict] = []
    for i in range(n):
        rec = {}
        for k, v in st.items():
            rec[k] = v[i] if isinstance(v, list) and i < len(v) else (None if isinstance(v, list) else v)
        out.append(rec)
    return out


def _as_list(v: Any) -> list:
    """A field that may be one value or many (Modalities, and the like).

    NUMPY ARRAYS COUNT AS MANY. `pymatreader` hands most cell/char arrays back
    as Python lists, but not all of them and not on every writer — a
    `scipy.io.savemat` round trip gives `ndarray` for the same field. Treating
    one as a single value collapsed a nine-entry surface list into ONE bogus
    entry whose name was the array's repr, and nothing about that looks like a
    parse failure: the subject simply appeared to have one oddly-named surface.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "tolist") and getattr(v, "ndim", 0) > 0:
        return list(v.tolist())
    return [v]


def _s(v: Any) -> str:
    """A MATLAB value as a string, TRIMMED.

    A MATLAB char matrix pads its rows to equal width, so a `SurfaceType`
    column reads `'Other '` beside `'Cortex'` — equal-length, not equal-meaning.
    An untrimmed compare against `'Other'` then fails for a value that is
    correct.
    """
    return "" if v is None else str(v).strip()


def _mtime(p: Path) -> float | None:
    """A file's mtime, or None when it is not there — never an exception.

    Used to date a SNAPSHOT, so its absence is information rather than a
    failure: a protocol with no index simply cannot go stale in this sense.
    """
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _first(v: Any) -> Any:
    """The scalar behind a field pymatreader may hand back as a 1-element array.

    An UNSET MATLAB index comes back as an empty `uint8` array — falsy, but not
    `None`, so a bare truth test on it is fine while `int()` on it raises. This
    normalises both shapes to something a caller can test.
    """
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.item() if v.size == 1 else (None if v.size == 0 else v.flat[0])
    except Exception:
        pass
    return v


def _kernel_surface(root: Path, kernel_files: list[str]) -> str | None:
    """The surface this condition's kernels are defined on.

    The INDEX does not carry it — `Result` holds Comment/FileName/DataFile/
    HeadModelType and no SurfaceFile — so this is the one place the scan opens
    a file, and it opens it for ONE field (`load_mat_vars`). It is what lets the
    listing name the manifold an import will actually build, rather than
    describing it in the abstract.

    None when the kernels disagree: two inverses on two surfaces is a real
    protocol, and naming one would be a guess presented as a fact.
    """
    from ..matio import load_mat_vars
    found: set[str] = set()
    for rel in kernel_files:
        p = root / "data" / rel
        if not p.is_file():
            continue
        try:
            sf = load_mat_vars(p, ["SurfaceFile"]).get("SurfaceFile")
        except Exception:
            continue                      # unreadable kernel → simply unknown
        if sf:
            found.add(Path(str(sf)).name)
    return found.pop() if len(found) == 1 else None


def _surfaces_of(rec: dict, fallback_dir: Path) -> tuple[list[SourceSurface], str | None, str | None, str | None]:
    """A subject's surfaces FROM THE INDEX — list, cortex, scalp, discrepancy.

    Brainstorm's `Subject.Surface` is a struct-of-arrays
    (`{Comment: [...], FileName: [...], SurfaceType: [...]}`), and `iCortex` /
    `iScalp` are 1-BASED indices into it. That is strictly richer than the glob
    this replaces, which yielded filenames only — and it is the difference
    between the database STATING which surface is the cortex and us guessing
    `tess_cortex_pial_low.mat`, correct here by coincidence.

    ONE DISK TOUCH, as VERIFICATION rather than discovery: `is_file` per listed
    surface. A protocol pruned outside Brainstorm — which is exactly what a
    `_clean` copy is — lists surfaces it no longer has, and that is reported
    through `discrepancy`, never silently repaired.

    `fallback_dir` is the subject's own `anat/<name>/`. When the index carries
    no list at all (a subject inheriting the default anatomy, or a DbVersion
    that predates the field), the glob is still there as the fallback — absence
    of a list is not evidence of absence of surfaces.
    """
    surf = rec.get("Surface")
    files = [_s(x) for x in _as_list((surf or {}).get("FileName"))] if isinstance(surf, dict) else []
    files = [f for f in files if f]
    if not files:
        if not fallback_dir.is_dir():
            return [], None, None, None
        names = sorted(p.name for p in fallback_dir.glob("tess_*.mat"))
        return [SourceSurface(file=n) for n in names], None, None, None

    comments = [_s(x) for x in _as_list((surf or {}).get("Comment"))]
    types = [_s(x) for x in _as_list((surf or {}).get("SurfaceType"))]

    out: list[SourceSurface] = []
    missing: list[str] = []
    for i, rel in enumerate(files):
        name = Path(rel).name
        present = (fallback_dir / name).is_file()
        if not present:
            missing.append(name)
        out.append(SourceSurface(
            file=name,
            comment=comments[i] if i < len(comments) else "",
            type=types[i] if i < len(types) else "",
            present=present,
        ))

    def at(key: str) -> str | None:
        """`iCortex`/`iScalp` — 1-based, and EMPTY when the subject has none.

        pymatreader gives an empty uint8 array for an unset index, which is
        falsy but not None, so an `is not None` test would index with it.
        """
        v = rec.get(key)
        try:
            idx = int(v)                      # empty array / None both raise
        except (TypeError, ValueError):
            return None
        return out[idx - 1].file if 1 <= idx <= len(out) else None

    disc = (f"{len(missing)} surface(s) listed by the index are not on disk: "
            + ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "")) if missing else None
    return out, at("iCortex"), at("iScalp"), disc


def _raw_link_facts(root: Path, rel: str) -> dict:
    """`format`, `sfreq`, `duration_s`, `n_samples` — from the raw LINK itself.

    THE INDEX CARRIES NONE OF THEM, and their absence was two separate defects:
    every omega row claimed an unknown raw format (so the `BST-BIN` branch never
    fired and the tooltip said the excerpts would import instead — while excerpts
    = 0, because omega is all raw links), and no row anywhere showed a duration,
    a sample rate or a sample count, which is the primary fact about a
    resting-state run.

    CHEAP, and measured rather than assumed: a raw link is METADATA, ~8 KB, not
    the recording — the samples live in a sibling `.bst`. `F.prop.times` is
    `[t0, t1]` in seconds and `F.prop.sfreq` is the rate, so the sample count is
    arithmetic rather than a second read.

    Same discipline as `_kernel_surface`: the scan opens a file only where the
    database does not answer, and for named fields only. Failure is an empty
    dict — a listing must not die because one link is unreadable.
    """
    from ..matio import load_mat_vars
    p = root / "data" / rel
    if not p.is_file():
        return {}
    try:
        F = load_mat_vars(p, ["F"]).get("F")
    except Exception:
        return {}
    if F is None:
        return {}
    out: dict = {}
    fmt = _s(getattr(F, "format", "") or "")
    if fmt:
        out["raw_format"] = fmt
    prop = getattr(F, "prop", None)
    times = _as_list(getattr(prop, "times", None)) if prop is not None else []
    sfreq = getattr(prop, "sfreq", None) if prop is not None else None
    try:
        sfreq = float(sfreq)
    except (TypeError, ValueError):
        sfreq = None
    if sfreq and sfreq > 0:
        out["sfreq"] = sfreq
    if len(times) >= 2 and sfreq:
        try:
            t0, t1 = float(times[0]), float(times[-1])
        except (TypeError, ValueError):
            return out
        if t1 > t0:
            out["duration_s"] = round(t1 - t0, 6)
            # INCLUSIVE of both endpoints: Brainstorm's `times` are the FIRST and
            # LAST sample's timestamps, not a half-open span, so the count is
            # one more than the intervals between them.
            out["n_samples"] = int(round((t1 - t0) * sfreq)) + 1
    return out


def _kernel_like(comment: str, filename: str) -> bool:
    """Brainstorm marks an inverse KERNEL in the filename, not in a field."""
    return "KERNEL" in filename.upper()


def read_protocol_mat(root: str | Path) -> SourceView:
    from ..matio import load_mat            # local: keeps scipy off the import path

    root = Path(root)
    m = load_mat(root / "data" / "protocol.mat")

    info = m.get("ProtocolInfo") or {}
    subjects_rec = _records((m.get("ProtocolSubjects") or {}).get("Subject"), by="Name")
    studies = _records((m.get("ProtocolStudies") or {}).get("Study"), by="Name")

    # ANATOMY PER SUBJECT, FROM THE INDEX (2026-08-17). This globbed
    # `anat/<name>/tess_*.mat` and said the Surface list "is not present in every
    # DbVersion" — untrue for the official, supported versions this now targets
    # (a deliberate scope decision), and the glob threw away everything that makes the list
    # worth reading: the Comment a picker should show, the SurfaceType, and
    # `iCortex` — the database naming the cortex where the code was guessing it.
    anat = root / "anat"
    default_rec = (m.get("ProtocolSubjects") or {}).get("DefaultSubject") or {}

    by_subject: dict[str, SourceSubject] = {}
    for rec in subjects_rec:
        name = _s(rec.get("Name")) or Path(_s(rec.get("FileName"))).parent.name
        if not name:
            continue
        sdir = anat / name
        use_default = bool(_first(rec.get("UseDefaultAnat")))
        # INHERITED ANATOMY. A subject with `UseDefaultAnat` has no `anat/<name>/`
        # of its own and is NOT anatomy-less — it borrows `@default_subject`'s.
        # Reading only its own directory reports a perfectly importable subject
        # as having no anatomy: a confident wrong answer, not a gap.
        src_rec, src_dir = ((default_rec, anat / "@default_subject") if use_default
                            else (rec, sdir))
        surfaces, cortex, scalp, disc = _surfaces_of(src_rec, src_dir)
        by_subject[name] = SourceSubject(
            name=name,
            anatomy=bool(surfaces),
            surfaces=surfaces,
            cortex=cortex,
            scalp=scalp,
            uses_default_anat=use_default,
            discrepancy=disc,
        )

    for st in studies:
        cond_name = _s(st.get("Condition")) or _s(st.get("Name"))
        subj_file = _s(st.get("BrainStormSubject"))
        subj = Path(subj_file).parent.name if subj_file else ""
        # `@intra`, `@default_study`, `@inter` are Brainstorm's own bookkeeping
        # studies. `@raw<base>` is NOT — it is a real condition whose recording is a
        # raw link, and in the modern workflow it is where the kernel and head model
        # live. Skipping everything starting with '@' dropped the noise condition
        # entirely: the index reported 2 conditions where the filesystem walk found
        # 3. Caught by running both readers against the same protocol and diffing
        # them, which is the reason the walk is kept rather than replaced.
        BOOKKEEPING = ("@default_study", "@intra", "@inter")
        if not cond_name or not subj or cond_name in BOOKKEEPING:
            continue
        is_raw_study = cond_name.startswith("@raw")
        if is_raw_study:
            cond_name = cond_name[4:]        # merge onto the plain condition's row
        elif cond_name.startswith("@"):
            continue                          # any other @study is bookkeeping
        if subj not in by_subject:
            by_subject[subj] = SourceSubject(name=subj)

        chan = _records(st.get("Channel"))
        ch0 = chan[0] if chan else {}
        modalities = ch0.get("Modalities") if ch0 else None
        mods = [_s(x) for x in _as_list(modalities)] if not isinstance(modalities, str) else [modalities]

        recs: list[SourceRecording] = []
        for d in _records(st.get("Data")):
            fn = _s(d.get("FileName"))
            is_raw = "data_0raw" in fn or Path(fn).parent.name.startswith("@raw")
            recs.append(SourceRecording(
                file=fn,
                comment=_s(d.get("Comment")) or Path(fn).name,
                data_type=_s(d.get("DataType")),
                bad_trial=bool(d.get("BadTrial")),
                is_raw_link=is_raw,
                # Only a RAW LINK is opened: an excerpt's samples are in the
                # file itself, so reading one to learn its duration would mean
                # loading the recording during a LISTING.
                **(_raw_link_facts(root, fn) if is_raw else {}),
            ))

        kernel_recs = [r for r in _records(st.get("Result"))
                       if _kernel_like(_s(r.get("Comment")), _s(r.get("FileName")))]
        kernels = [_s(r.get("Comment")) or Path(_s(r.get("FileName"))).name for r in kernel_recs]
        kernel_files = [_s(r.get("FileName")) for r in kernel_recs if r.get("FileName")]
        heads = [_s(h.get("Comment")) or Path(_s(h.get("FileName"))).name
                 for h in _records(st.get("HeadModel"))]

        # A condition may appear twice — the plain study and its `@raw` twin. Merge
        # onto one row, because that is one thing to the user, and because the kernel
        # commonly lives on the raw side while the excerpts live on the plain one
        # (the workflow the filesystem walk had to grow a comment to explain).
        existing = next((c for c in by_subject[subj].conditions if c.name == cond_name), None)
        c = existing or SourceCondition(name=cond_name)
        c.recordings.extend(recs)
        c.kernels = sorted({*c.kernels, *kernels})
        c.head_models = sorted({*c.head_models, *heads})
        c.primary_surface = c.primary_surface or _kernel_surface(root, kernel_files)
        if ch0:
            c.channel_file = c.channel_file or _s(ch0.get("FileName"))
            n = ch0.get("nbChannels")
            c.n_channels = c.n_channels or (int(n) if n else None)
            c.modalities = sorted({*c.modalities, *[x for x in mods if x]})
        if existing is None:
            by_subject[subj].conditions.append(c)

    for s in by_subject.values():
        for c in s.conditions:
            c.importability = _importability(c, s.name)

    return SourceView(
        root=str(root),
        kind="protocol.mat",
        subjects=[by_subject[k] for k in sorted(by_subject)],
        db_version=float(m["DbVersion"]) if m.get("DbVersion") is not None else None,
        protocol_comment=_s(info.get("Comment")) or None,
        scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        index_mtime=_mtime(root / "data" / "protocol.mat"),
    )


def _importability(c: SourceCondition, subject: str = "") -> Importability:
    """Can this condition become an `nxr.subject@2.0` session, and if not, why?

    Stated here so the UI can grey a row WITH A REASON. Today the equivalent is
    running a conversion and reading the error it fails with.
    """
    # FIRST, because it is a fact about WHAT the row is rather than about what
    # it is missing. `Group_analysis` holds cross-subject results; judging it by
    # channels or kernels would report a group average as "no channels", which
    # is true and useless.
    if subject == GROUP_ANALYSIS_SUBJECT:
        return "group-analysis"
    if not c.recordings:
        return "no-channels"
    if not c.channel_file:
        return "no-channels"
    if not c.kernels:
        # Convertible as sensors + anatomy; there is simply no source map to build.
        return "missing-kernel"
    if all(r.is_raw_link for r in c.recordings):
        return "ready-raw"      # importable from the treated .bst — see model.py
    return "ready"
