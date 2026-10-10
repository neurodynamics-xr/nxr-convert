"""The SOURCE VIEW readers, and the two parse traps that cost real time.

`pymatreader` returns a MATLAB struct ARRAY as a STRUCT-OF-ARRAYS — one key per
field holding N parallel values — and both mistakes below produce a plausible,
non-crashing answer, which is why they are pinned rather than left to review.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from nxr_convert.sources import read_source
from nxr_convert.sources.protocol_mat import _records, _importability
from nxr_convert.sources.model import SourceCondition, SourceRecording, SCHEMA, IMPORTABLE

def _protocol() -> Path | None:
    """The Brainstorm TutorialAuditory protocol: ``NXR_TEST_PROTOCOL``, else a local default (not exported), else none."""
    if os.environ.get("NXR_TEST_PROTOCOL"):
        return Path(os.environ["NXR_TEST_PROTOCOL"])
    try:
        from _local_paths import TUTORIAL_AUDITORY
    except ImportError:
        return None
    return Path(TUTORIAL_AUDITORY)


PROTOCOL = _protocol()
has_protocol = PROTOCOL is not None and (PROTOCOL / "data").is_dir()   # a protocol, not just its folder


# ── the transpose ────────────────────────────────────────────────────────────

def test_records_transposes_a_struct_of_arrays():
    """N records from parallel columns — not one record holding whole columns."""
    st = {"Name": ["a", "b", "c"], "FileName": ["x", "y", "z"]}
    recs = _records(st, by="Name")
    assert [r["Name"] for r in recs] == ["a", "b", "c"]


def test_records_does_not_count_by_a_list_valued_field():
    """THE TRAP: a single record whose `Modalities` holds 13 strings must stay ONE
    record. Counting by the longest field split it into 13, each keeping one
    modality — which reported 'ADC A' for CTF MEG data: a real modality, and the
    wrong answer."""
    st = {"FileName": "channel_ctf_acc1.mat", "nbChannels": 340,
          "Modalities": ["ADC A", "ADC V", "MEG", "MEG REF", "Stim"]}
    recs = _records(st)
    assert len(recs) == 1
    assert recs[0]["Modalities"] == ["ADC A", "ADC V", "MEG", "MEG REF", "Stim"]


def test_records_does_not_index_into_a_single_record():
    """A 1-element struct array IS the record. Taking `v[0]` of every list-valued
    field keeps only the first of anything plural, silently."""
    st = {"FileName": "one.mat", "Modalities": ["MEG", "EEG"]}
    assert _records(st)[0]["Modalities"] == ["MEG", "EEG"]


# ── importability, stated rather than discovered by a failing conversion ─────

def test_importability_reasons():
    ready = SourceCondition(name="c", channel_file="ch.mat", kernels=["MN"],
                            recordings=[SourceRecording(file="d.mat", comment="d")])
    assert _importability(ready) == "ready"

    no_kernel = SourceCondition(name="c", channel_file="ch.mat",
                                recordings=[SourceRecording(file="d.mat", comment="d")])
    assert _importability(no_kernel) == "missing-kernel"

    no_chan = SourceCondition(name="c", recordings=[SourceRecording(file="d.mat", comment="d")])
    assert _importability(no_chan) == "no-channels"

    # A raw condition IS importable — from the treated .bst. Calling it `raw-only`
    # mislabelled 228 of 343 conditions on a real cohort as if they were blocked.
    raw = SourceCondition(name="c", channel_file="ch.mat", kernels=["MN"],
                          recordings=[SourceRecording(file="r.mat", comment="r", is_raw_link=True)])
    assert _importability(raw) == "ready-raw"
    assert _importability(raw) in IMPORTABLE


# ── against a real protocol ──────────────────────────────────────────────────

@pytest.mark.skipif(not has_protocol, reason=f"no Brainstorm protocol at {PROTOCOL}")
def test_index_and_walk_agree_on_the_same_protocol():
    """THE CROSS-CHECK, and it has already earned its place: the index reader
    originally skipped every study whose Condition starts with '@', which dropped
    raw-only conditions — 2 conditions where the walk found 3. The disagreement is
    the signal; neither reader alone would have shown it."""
    idx = read_source(PROTOCOL, prefer="protocol.mat")
    walk = read_source(PROTOCOL, prefer="walk")

    names = lambda v: {(s.name, c.name) for s in v.subjects for c in s.conditions}
    assert names(idx) == names(walk)
    assert idx.counts["recordings"] == walk.counts["recordings"]
    assert idx.counts["kernels"] == walk.counts["kernels"]


@pytest.mark.skipif(not has_protocol, reason=f"no Brainstorm protocol at {PROTOCOL}")
def test_the_index_recovers_what_the_walk_cannot():
    """The whole reason for reading the database: MEANING. The walk sees
    `results_MN_MEG_KERNEL_260603_0647.mat`; the index sees
    `MN: MEG(Unconstr) 2018`, and knows the channel count without opening a file."""
    idx = read_source(PROTOCOL, prefer="protocol.mat")
    walk = read_source(PROTOCOL, prefer="walk")
    ic = idx.subjects[0].conditions[0]
    wc = walk.subjects[0].conditions[0]

    assert ic.n_channels and ic.n_channels > 0
    assert wc.n_channels is None                      # the walk would have to open files
    assert "MEG" in ic.modalities
    assert not any(k.endswith(".mat") for k in ic.kernels)   # Comments, not filenames
    assert all(k.endswith(".mat") for k in wc.kernels)       # filenames, as expected
    assert not ic.recordings[0].comment.endswith(".mat")


@pytest.mark.skipif(not has_protocol, reason=f"no Brainstorm protocol at {PROTOCOL}")
def test_view_is_json_serialisable_and_declares_its_schema():
    import json
    v = read_source(PROTOCOL)
    payload = v.to_json()
    assert payload["schema"] == SCHEMA          # NOT nxr.subject@2.0 — see the spec
    assert payload["kind"] == "protocol.mat"
    assert json.loads(json.dumps(payload))["root"] == str(PROTOCOL)


# ── the primary surface is the INVERSE's ─────────────────────────────────────

def _kernel_with_surface(path, surface_name):
    """A minimal `results_*KERNEL*.mat` carrying only what the resolver reads."""
    import scipy.io
    scipy.io.savemat(path, {"SurfaceFile": f"sub-01/{surface_name}",
                            "Comment": "MN: MEG(Unconstr)"})


def test_primary_surface_comes_from_the_kernel(tmp_path):
    """The kernel states the surface it was computed on; we do not guess it.

    Everything downstream — the mass, the bases, the fusions — is indexed by
    that surface's vertices, so a different manifold yields a store that loads,
    renders, and is wrong everywhere.
    """
    from nxr_convert.subject import kernel_surface as _kernel_surface
    cond, raw, anat = tmp_path / "cond", tmp_path / "raw", tmp_path / "anat"
    cond.mkdir(); anat.mkdir()
    (anat / "tess_cortex_mid_low.mat").write_bytes(b"")
    (anat / "tess_cortex_pial_low.mat").write_bytes(b"")
    _kernel_with_surface(cond / "results_MN_MEG_KERNEL_1.mat", "tess_cortex_mid_low.mat")

    assert _kernel_surface(cond, raw, anat) == "tess_cortex_mid_low.mat"


def test_disagreeing_kernels_refuse_to_pick(tmp_path):
    """Two inverses on two surfaces is a REAL protocol, and picking one
    silently would index the second by the wrong manifold. None → the caller
    falls back to an explicit `--surface` or the documented default."""
    from nxr_convert.subject import kernel_surface as _kernel_surface
    cond, raw, anat = tmp_path / "cond", tmp_path / "raw", tmp_path / "anat"
    cond.mkdir(); anat.mkdir()
    (anat / "tess_cortex_mid_low.mat").write_bytes(b"")
    (anat / "tess_cortex_pial_low.mat").write_bytes(b"")
    _kernel_with_surface(cond / "results_A_KERNEL_1.mat", "tess_cortex_mid_low.mat")
    _kernel_with_surface(cond / "results_B_KERNEL_2.mat", "tess_cortex_pial_low.mat")

    assert _kernel_surface(cond, raw, anat) is None


def test_disagreeing_kernels_ACROSS_the_two_homes_still_refuse_to_pick(tmp_path):
    """The same refusal, but with the two kernels split across the two condition
    homes instead of both sitting in the plain dir. `_kernel_surface` unions the
    two globs into one `found` set — that set must stay the reason this
    disagrees, not an accident of both cases so far putting both kernels in
    `cond`. A per-home lookup that silently preferred one home over the other
    would pick a surface here instead of refusing."""
    from nxr_convert.subject import kernel_surface as _kernel_surface
    cond, raw, anat = tmp_path / "cond", tmp_path / "raw", tmp_path / "anat"
    cond.mkdir(); raw.mkdir(); anat.mkdir()
    (anat / "tess_cortex_mid_low.mat").write_bytes(b"")
    (anat / "tess_cortex_pial_low.mat").write_bytes(b"")
    _kernel_with_surface(cond / "results_A_KERNEL_1.mat", "tess_cortex_mid_low.mat")
    _kernel_with_surface(raw / "results_B_KERNEL_2.mat", "tess_cortex_pial_low.mat")

    assert _kernel_surface(cond, raw, anat) is None


def test_a_surface_the_protocol_does_not_have_is_not_returned(tmp_path):
    """A kernel's SurfaceFile can point at anatomy that was never copied — a
    pruned protocol. Naming a file that is not there would fail deep inside the
    surface export instead of at the decision."""
    from nxr_convert.subject import kernel_surface as _kernel_surface
    cond, raw, anat = tmp_path / "cond", tmp_path / "raw", tmp_path / "anat"
    cond.mkdir(); anat.mkdir()
    _kernel_with_surface(cond / "results_MN_KERNEL_1.mat", "tess_cortex_mid_low.mat")

    assert _kernel_surface(cond, raw, anat) is None


# ── the SURFACE LIST, read from the index ────────────────────────────────────
#
# The scan globbed `anat/<name>/tess_*.mat` and threw away everything that makes
# the index worth reading. These pin what the glob could not express, on a
# synthetic protocol so they hold with no data on disk.

def _protocol(tmp_path, subject_rec, *, default_rec=None, files=()):
    """A minimal `protocol.mat` plus whichever `anat/` files the test wants."""
    import scipy.io
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    for rel in files:
        p = tmp_path / "anat" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    subjects = {"Subject": subject_rec}
    if default_rec is not None:
        subjects["DefaultSubject"] = default_rec
    scipy.io.savemat(tmp_path / "data" / "protocol.mat", {
        "ProtocolInfo": {"Comment": "T", "iStudy": 0},
        "ProtocolSubjects": subjects,
        "ProtocolStudies": {"Study": {"Name": [], "Condition": []}},
    })
    return tmp_path


def _surf(files, comments, types):
    return {"FileName": list(files), "Comment": list(comments), "SurfaceType": list(types)}


def test_the_cortex_is_named_by_iCortex_not_guessed(tmp_path):
    """The database STATES which surface is the cortex. The scan hardcoded
    `tess_cortex_pial_low.mat` — right for the protocols to hand, and a
    coincidence rather than a fact. Here the cortex is deliberately NOT that
    file, so a hardcoded default fails and only reading `iCortex` passes."""
    rec = {
        "Name": "S1", "FileName": "S1/brainstormsubject.mat",
        "Surface": _surf(
            ["S1/tess_aseg.mat", "S1/tess_cortex_white_low.mat", "S1/tess_head.mat"],
            ["subcortical", "white_20484V", "head"],
            ["Other", "Cortex", "Scalp"]),
        "iCortex": 2, "iScalp": 3, "UseDefaultAnat": 0,
    }
    root = _protocol(tmp_path, rec, files=[
        "S1/tess_aseg.mat", "S1/tess_cortex_white_low.mat", "S1/tess_head.mat"])
    s = read_source(root, prefer="protocol.mat").subjects[0]
    assert s.cortex == "tess_cortex_white_low.mat"
    assert s.scalp == "tess_head.mat"


def test_the_surface_carries_its_comment_and_type(tmp_path):
    """What the glob could not give: Brainstorm's own label, which is what a
    surface picker should show, and the classification rather than an inference
    from the filename."""
    rec = {
        "Name": "S1", "FileName": "S1/brainstormsubject.mat",
        "Surface": _surf(["S1/tess_cortex_pial_low.mat"], ["cortex_20484V"], ["Cortex"]),
        "iCortex": 1, "UseDefaultAnat": 0,
    }
    root = _protocol(tmp_path, rec, files=["S1/tess_cortex_pial_low.mat"])
    [surf] = read_source(root, prefer="protocol.mat").subjects[0].surfaces
    assert (surf.file, surf.comment, surf.type) == (
        "tess_cortex_pial_low.mat", "cortex_20484V", "Cortex")


def test_a_listed_surface_that_is_absent_is_REPORTED_not_repaired(tmp_path):
    """The one disk touch, and its whole purpose. A protocol pruned outside
    Brainstorm — which is exactly what a `_clean` copy is — lists surfaces it no
    longer has. Dropping them silently would make the index and the disk agree
    by deletion; the entry stays, marked absent, and the subject says so."""
    rec = {
        "Name": "S1", "FileName": "S1/brainstormsubject.mat",
        "Surface": _surf(
            ["S1/tess_cortex_pial_low.mat", "S1/tess_gone.mat"],
            ["cortex_20484V", "pruned"], ["Cortex", "Other"]),
        "iCortex": 1, "UseDefaultAnat": 0,
    }
    root = _protocol(tmp_path, rec, files=["S1/tess_cortex_pial_low.mat"])
    s = read_source(root, prefer="protocol.mat").subjects[0]
    assert [x.present for x in s.surfaces] == [True, False]
    assert s.discrepancy and "tess_gone.mat" in s.discrepancy


def test_UseDefaultAnat_INHERITS_rather_than_reporting_no_anatomy(tmp_path):
    """A subject with `UseDefaultAnat` has no `anat/<name>/` of its own and is
    NOT anatomy-less — it borrows `@default_subject`'s. Reading only its own
    directory reports a perfectly importable subject as having no anatomy: a
    confident wrong answer rather than a gap. (This is why `sub-emptyroom` read
    `anatomy: true` with zero surfaces, contradicting itself.)"""
    rec = {"Name": "S1", "FileName": "S1/brainstormsubject.mat",
           "Surface": _surf([], [], []), "UseDefaultAnat": 1}
    default = {
        "Name": "@default_subject", "FileName": "@default_subject/brainstormsubject.mat",
        "Surface": _surf(["@default_subject/tess_cortex_pial_low.mat"],
                         ["cortex_15002V"], ["Cortex"]),
        "iCortex": 1,
    }
    root = _protocol(tmp_path, rec, default_rec=default,
                     files=["@default_subject/tess_cortex_pial_low.mat"])
    s = read_source(root, prefer="protocol.mat").subjects[0]
    assert s.uses_default_anat is True
    assert s.anatomy is True
    assert [x.file for x in s.surfaces] == ["tess_cortex_pial_low.mat"]
    assert s.cortex == "tess_cortex_pial_low.mat"


def test_no_Surface_list_still_falls_back_to_the_glob(tmp_path):
    """Absence of a list is not evidence of absence of surfaces. A DbVersion
    predating the field, or a hand-assembled protocol, still has files on disk —
    so the glob stays as the FALLBACK, just not as the primary route."""
    rec = {"Name": "S1", "FileName": "S1/brainstormsubject.mat", "UseDefaultAnat": 0}
    root = _protocol(tmp_path, rec, files=["S1/tess_cortex_pial_low.mat", "S1/tess_head.mat"])
    s = read_source(root, prefer="protocol.mat").subjects[0]
    assert [x.file for x in s.surfaces] == ["tess_cortex_pial_low.mat", "tess_head.mat"]
    assert s.cortex is None          # nothing to read it from — absent, not guessed
