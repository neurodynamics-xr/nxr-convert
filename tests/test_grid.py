"""THE TIMESERIES GRID (2026-09-25): the recording one channel × 2^15 samples an inner chunk and every
channel × 2^17 a shard; ONE envelope array of 32-sample windows beside it, sharded on the same block.
The database states the envelope as a closed-form tiling with array-backed min/max (D96)."""
import json

import numpy as np
import pytest
import zarr

import nxr_convert.grid as grid
from nxr_convert.grid import BLOCK, RAW_CHUNK, WINDOW, ENV_CHUNK, raw_grid

def _reference(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """min/max per window by the definition — slow and obvious; the last window may be partial."""
    c, n = x.shape
    nb = -(-n // window)
    out = np.empty((c, nb, 2), dtype=np.float32)
    for b in range(nb):
        seg = x[:, b * window:(b + 1) * window]
        out[:, b, 0] = seg.min(axis=1)
        out[:, b, 1] = seg.max(axis=1)
    return out


def test_the_grid_nests_dyadically():
    assert (WINDOW, RAW_CHUNK, BLOCK, ENV_CHUNK) == (32, 1 << 15, 1 << 17, 4096)
    assert BLOCK % RAW_CHUNK == 0 and RAW_CHUNK % WINDOW == 0 and ENV_CHUNK * WINDOW == BLOCK
    assert raw_grid(300, 1_440_000) == ((1, 32768), (300, 131072))
    assert raw_grid(3, 1000) == ((1, 1000), (3, 1000))                        # shorter than a chunk: one of it


def _recording(tmp_path, x):
    """A recording laid out from its row's layout (``recording_meta``) and populated."""
    from nxr_convert.crud import layout_array, populate_array
    at = tmp_path / "rec"
    layout_array(at, grid.recording_meta(*x.shape))
    return populate_array(at, x)


def test_a_recording_is_sharded_one_channel_per_inner_chunk(tmp_path):
    a = _recording(tmp_path, np.zeros((3, 200_000), np.float32))
    doc = json.loads((tmp_path / "rec" / "zarr.json").read_text())
    assert doc["chunk_grid"]["configuration"]["chunk_shape"] == [3, BLOCK]
    assert doc["codecs"][0]["name"] == "sharding_indexed" and doc["codecs"][0]["configuration"]["chunk_shape"] == [1, RAW_CHUNK]
    assert grid.recording_meta(3, 200_000).chunks == (1, RAW_CHUNK)        # the row states the INNER chunk (D57)


def _teed(tmp_path, x, block=BLOCK):
    """A recording written block by block through the tee, as the converter writes one; returns the envelope."""
    from nxr_convert.crud import layout_array, populate_array
    at = tmp_path / "rec"
    layout_array(at, grid.recording_meta(*x.shape))
    tee = grid.EnvelopeTee(populate_array(at), tmp_path / "env")
    for s0 in range(0, x.shape[1], block):
        tee[:, s0:s0 + block] = x[:, s0:s0 + block]
    np.testing.assert_array_equal(zarr.open_array(str(at))[...], x)
    return zarr.open_array(str(tmp_path / "env"))


def test_the_envelope_is_exact_with_a_partial_tail(tmp_path):
    x = np.random.default_rng(0).standard_normal((3, 1000)).astype(np.float32)
    env = _teed(tmp_path, x)
    assert env.shape == (3, -(-1000 // WINDOW), 2)
    np.testing.assert_array_equal(env[...], _reference(x))


def test_block_boundaries_do_not_change_the_answer(tmp_path):
    x = np.random.default_rng(1).standard_normal((2, 5000)).astype(np.float32)
    np.testing.assert_array_equal(_teed(tmp_path, x, 64 * WINDOW)[...], _reference(x))     # many blocks, a partial last


def test_the_one_pass_envelope_writes_the_bytes_the_two_pass_one_did(tmp_path):
    """The tee replaced a second pass that read the written recording back a BLOCK at a time; same files, byte for byte."""
    from nxr_convert.crud import array_meta, layout_array, populate_array
    x = np.random.default_rng(2).standard_normal((3, BLOCK + 1000)).astype(np.float32)       # a full shard and a partial one
    _teed(tmp_path / "one", x)
    src = zarr.open_array(str(tmp_path / "one" / "rec"))
    n_win = -(-x.shape[1] // WINDOW)                                                         # the old write_envelope, verbatim
    chunks, shards = grid.env_grid(3, n_win)
    layout_array(tmp_path / "two", array_meta(np.float32, (3, n_win, 2), chunks=chunks, shards=shards, compress=False))
    env = populate_array(tmp_path / "two")
    for s0 in range(0, x.shape[1], BLOCK):
        b = np.asarray(src[:, s0:s0 + BLOCK], dtype=np.float32)
        env[:, s0 // WINDOW:s0 // WINDOW + -(-b.shape[1] // WINDOW), :] = grid.reduce_windows(b)
    one = {p.relative_to(tmp_path / "one" / "env"): p.read_bytes() for p in (tmp_path / "one" / "env").rglob("*") if p.is_file()}
    two = {p.relative_to(tmp_path / "two"): p.read_bytes() for p in (tmp_path / "two").rglob("*") if p.is_file()}
    assert one == two and len(one) > 2


def test_a_misaligned_write_is_refused(tmp_path):
    from nxr_convert.crud import layout_array, populate_array
    layout_array(tmp_path / "rec", grid.recording_meta(2, 1000))
    tee = grid.EnvelopeTee(populate_array(tmp_path / "rec"), tmp_path / "env")
    with pytest.raises(ValueError):
        tee[:, 0:100] = np.zeros((2, 100), np.float32)                     # ends mid-window, not at the end
