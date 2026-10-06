"""The event parser, held to synthetic Brainstorm-shaped structs.

Restated independently of the parser: the expected tables are written out by
hand, because a test that builds its expectation with the code under test can
only catch that code being self-inconsistent.
"""
import numpy as np
import pytest

from nxr_convert.events import EventTable, parse_events, rejoin_cellstr


def bst_events(**types):
    """pymatreader's dict-of-lists shape for a Brainstorm events struct array."""
    labels, colors, times, channels = [], [], [], []
    for label, spec in types.items():
        labels.append(label)
        colors.append(np.asarray(spec.get("color", [0.5, 0.5, 0.5])))
        times.append(np.asarray(spec["times"], dtype=float))
        channels.append(spec.get("channels", [np.array([], dtype=float)] *
                                 np.atleast_2d(np.asarray(spec["times"])).shape[1]))
    return {"label": labels, "color": colors, "times": times, "channels": channels}


def test_simple_events_get_zero_duration():
    ev = bst_events(spike={"times": [[1.0, 2.5, 4.0]], "color": [1.0, 0.0, 0.0]})
    t = parse_events(ev)
    assert t.type_names == ["spike"]
    assert t.type_colors == [[1.0, 0.0, 0.0]]
    assert np.array_equal(t.onset, np.array([1.0, 2.5, 4.0]))
    assert np.array_equal(t.duration, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(t.type, np.array([0, 0, 0], dtype=np.int32))


def test_extended_events_carry_offset_minus_onset():
    """Sleep staging is extended. Onset-only would flatten every stage to a
    zero-width mark, which is the loss this whole table exists to stop."""
    times = [[360.0, 600.0], [420.0, 630.0]]
    t = parse_events(bst_events(N2={"times": times}))
    assert np.array_equal(t.onset, np.array([360.0, 600.0]))
    assert np.array_equal(t.duration, np.array([60.0, 30.0]))
    # ROUND TRIP: the stored pair must reconstruct Brainstorm's own 2xN matrix,
    # so nothing about the source is recoverable only by inference.
    assert np.array_equal(np.vstack([t.onset, t.onset + t.duration]),
                          np.asarray(times, dtype=float))


def test_a_squeezed_single_extended_occurrence_is_recovered_via_epochs():
    """L, the real file (AnphySleep/EPCTL01): a lone extended occurrence's
    2x1 times matrix squeezes through pymatreader to a bare 1-D array of
    length 2 — the exact bit pattern a 1-row/2-column SIMPLE type with two
    onsets also squeezes to. `epochs` (1 x nOccurrences) breaks the tie: its
    count of 1 says this is ONE occurrence, so the reshape recovers the
    (2, 1) split (onset, offset) instead of `atleast_2d`'s wrong (1, 2)
    guess, which read two zero-duration marks at 28560.0 and 28710.0
    instead of one 150 s span."""
    ev = {"label": "L", "color": np.array([0.1, 0.35, 0.8]),
          "times": np.array([28560.0, 28710.0]), "epochs": np.array([1.0]),
          "channels": np.array([])}
    t = parse_events(ev)
    assert t.n_events == 1
    assert np.array_equal(t.onset, np.array([28560.0]))
    assert np.array_equal(t.duration, np.array([150.0]))


def test_two_simple_occurrences_at_the_same_squeezed_shape_do_not_regress():
    """The SAME squeezed shape (2,) as the extended case above, but `epochs`
    says TWO occurrences this time — must still parse as two zero-duration
    marks, not merge into one extended span. This is the case Fix 1 must not
    break while fixing the one above."""
    ev = {"label": "N2", "color": np.array([0.1, 0.35, 0.8]),
          "times": np.array([1.0, 2.0]), "epochs": np.array([1.0, 1.0]),
          "channels": np.array([])}
    t = parse_events(ev)
    assert t.n_events == 2
    assert np.array_equal(t.onset, np.array([1.0, 2.0]))
    assert np.array_equal(t.duration, np.array([0.0, 0.0]))


def test_epochs_size_one_with_a_genuine_2x15_matrix_does_not_silently_truncate():
    """`epochs` of size 1 makes `raw.size % n_occ == 0` true UNCONDITIONALLY —
    that is not a check. Fed a genuine 2x15 extended matrix (30 elements) with
    `epochs` wrongly reporting 1 occurrence, the old code reshaped to (30, 1):
    1 event reported, 14 silently lost, with a duration fabricated from the
    first and last onset. The implied row count here is 30, which is neither
    1 nor 2 — Brainstorm's own contract for `times` — so this must be refused
    rather than truncated. Raising (not falling back to the 15-occurrence
    `atleast_2d` reading) is the chosen behaviour: a `times`/`epochs` mismatch
    this large means the two disagree in a way this parser cannot resolve on
    its own, and reporting count 15 with unverified onset/offset pairing would
    be confident guessing, not recovery."""
    onsets = np.arange(15.0) * 10.0
    offsets = onsets + 5.0
    # A real (2, 15) extended matrix's underlying buffer, C-order-flattened —
    # exactly what `np.asarray(entry['times'])` sees once pymatreader hands
    # back the flat 30-element buffer with `epochs` mis-reporting 1 occurrence.
    times = np.vstack([onsets, offsets]).ravel(order="C")
    ev = {"label": "N2", "color": np.array([0.1, 0.35, 0.8]),
          "times": times, "epochs": np.array([1.0]), "channels": np.array([])}
    with pytest.raises(ValueError, match="N2"):
        parse_events(ev)


def test_a_three_row_implied_times_matrix_raises_and_names_the_label():
    """`epochs` of size 5 against a 15-element `times` implies 3 rows — a
    shape Brainstorm's `times` never legitimately takes (1 row simple, 2 rows
    extended). The old `atleast_2d` fallback would have silently dropped rows
    2+ with no error; this must raise and name both the event label and the
    offending shape."""
    ev = {"label": "triple", "color": np.array([0.2, 0.2, 0.2]),
          "times": np.arange(15.0), "epochs": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
          "channels": np.array([])}
    with pytest.raises(ValueError, match="triple") as e:
        parse_events(ev)
    assert "3" in str(e.value)


def test_a_genuine_2xN_extended_matrix_through_the_epochs_branch():
    """The dominant real shape: sleep stages arrive as a properly-shaped 2xN
    matrix (2x15, 2x41, ...) WITH `epochs` set to match — Brainstorm's normal
    case, not the squeezed edge case the other epochs tests target. Every
    other parser test in this file leaves `epochs` unset (`bst_events` never
    sets it), so this is the only coverage of a real 2-D matrix going through
    the epochs reshape branch rather than the `atleast_2d` fallback."""
    onsets = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    offsets = onsets + 20.0
    times = np.vstack([onsets, offsets])
    ev = {"label": "N2", "color": np.array([0.1, 0.35, 0.8]),
          "times": times, "epochs": np.ones(5), "channels": np.array([])}
    t = parse_events(ev)
    assert t.n_events == 5
    assert np.array_equal(t.onset, onsets)
    assert np.array_equal(t.duration, offsets - onsets)


def test_absent_epochs_falls_back_to_the_old_atleast_2d_guess():
    """No `epochs` field at all — the type's shape has to be guessed the way
    it always was, unchanged. Every pre-existing test in this file relies on
    this fallback (`bst_events` never sets `epochs`); this test states it
    explicitly."""
    ev = {"label": "spike", "color": np.array([1.0, 0.0, 0.0]),
          "times": np.array([1.0, 2.5, 4.0]), "channels": np.array([])}
    t = parse_events(ev)
    assert t.n_events == 3
    assert np.array_equal(t.onset, np.array([1.0, 2.5, 4.0]))
    assert np.array_equal(t.duration, np.array([0.0, 0.0, 0.0]))


def test_rows_are_sorted_by_onset_across_types():
    ev = bst_events(
        late={"times": [[10.0]]},
        early={"times": [[1.0]]},
        middle={"times": [[5.0]]},
    )
    t = parse_events(ev)
    assert np.array_equal(t.onset, np.array([1.0, 5.0, 10.0]))
    assert [t.type_names[i] for i in t.type] == ["early", "middle", "late"]


def test_type_names_are_verbatim_not_sanitized():
    ev = bst_events(**{"Segment: REC START 84": {"times": [[0.0]]}})
    assert parse_events(ev).type_names == ["Segment: REC START 84"]


def test_no_channel_scoping_gives_an_all_zero_csr():
    ev = bst_events(a={"times": [[1.0, 2.0]]})
    t = parse_events(ev)
    assert np.array_equal(t.channel_offsets, np.array([0, 0, 0], dtype=np.int32))
    assert t.channel_index.size == 0


def test_channel_scoping_resolves_names_to_indices_as_csr():
    ev = bst_events(art={
        "times": [[1.0, 2.0]],
        "channels": [["PO7", "PO3"], ["OZ"]],
    })
    t = parse_events(ev, channel_names=["FZ", "OZ", "PO3", "PO7"])
    assert np.array_equal(t.channel_offsets, np.array([0, 2, 3], dtype=np.int32))
    assert np.array_equal(t.channel_index, np.array([3, 2, 1], dtype=np.int32))


def test_an_unresolvable_channel_name_is_FATAL_not_dropped():
    ev = bst_events(art={"times": [[1.0]], "channels": [["NOSUCH"]]})
    with pytest.raises(ValueError, match="NOSUCH"):
        parse_events(ev, channel_names=["FZ", "OZ"])


def test_offset_before_onset_is_FATAL_not_negative():
    """A 2xN times matrix whose offset row precedes its onset row must never
    flow through as a negative duration — refuse, naming the event type."""
    ev = bst_events(N2={"times": [[360.0, 600.0], [420.0, 590.0]]})
    with pytest.raises(ValueError, match="N2"):
        parse_events(ev)


def test_channels_as_object_array_parses_per_occurrence():
    """The h5py/MAT v7.3 path can hand back `channels` as a numpy object
    array rather than a Python list. Each occurrence must still get its OWN
    channels, not the whole array broadcast to every row."""
    ev = bst_events(art={
        "times": [[1.0, 2.0]],
        "channels": np.array([["PO7", "PO3"], ["OZ"]], dtype=object),
    })
    t = parse_events(ev, channel_names=["FZ", "OZ", "PO3", "PO7"])
    assert np.array_equal(t.channel_offsets, np.array([0, 2, 3], dtype=np.int32))
    assert np.array_equal(t.channel_index, np.array([3, 2, 1], dtype=np.int32))


def test_rejoin_cellstr_undoes_pymatreaders_character_split():
    """pymatreader collapses a SINGLE-element cell of strings into a char array,
    so 'P11' reads back as ['P','1','1'] while a genuine two-element cell
    (['PO7','PO3']) reads correctly. Not rejoining these produces confident
    garbage channel names rather than an error."""
    assert rejoin_cellstr(["P", "1", "1"]) == ["P11"]
    assert rejoin_cellstr(["O", "Z"]) == ["OZ"]
    assert rejoin_cellstr(["PO7", "PO3"]) == ["PO7", "PO3"]
    assert rejoin_cellstr("FZ") == ["FZ"]
    assert rejoin_cellstr([]) == []


def test_a_list_of_dicts_parses_the_same_as_a_dict_of_lists():
    """scipy and h5py hand back different shapes for the same struct array."""
    dict_of_lists = bst_events(a={"times": [[1.0]]}, b={"times": [[2.0]]})
    list_of_dicts = [
        {"label": "a", "times": np.array([[1.0]]), "color": np.array([0.5, 0.5, 0.5])},
        {"label": "b", "times": np.array([[2.0]]), "color": np.array([0.5, 0.5, 0.5])},
    ]
    a, b = parse_events(dict_of_lists), parse_events(list_of_dicts)
    assert a.type_names == b.type_names
    assert np.array_equal(a.onset, b.onset)
    assert np.array_equal(a.type, b.type)


def test_absent_events_give_an_empty_table():
    t = parse_events(None)
    assert t.type_names == [] and t.onset.size == 0
    assert np.array_equal(t.channel_offsets, np.array([0], dtype=np.int32))
