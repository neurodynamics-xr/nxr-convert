"""The winding canonicalization, held to its geometric contract on synthetic
closed surfaces — an octahedron per 'hemisphere', wound both ways."""
import numpy as np
from nxr_convert.winding import canonicalize_winding


def octahedron(center=(0.0, 0.0, 0.0), ccw_outward=True):
    c = np.asarray(center)
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float) + c
    f = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int32)
    if not ccw_outward:
        f = f[:, [0, 2, 1]]
    return v, f


def signed_volume(v, f):
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.sum(v0 * np.cross(v1, v2)) / 6.0)


def test_flips_an_inward_component_and_keeps_an_outward_one():
    v1, f1 = octahedron((0, 0, 0), ccw_outward=True)
    v2, f2 = octahedron((10, 0, 0), ccw_outward=False)   # Brainstorm-style inward
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    out, info = canonicalize_winding(v, f)
    assert info["n_components"] == 2
    assert info["flipped"] == [1]
    # Both components now enclose positive volume: CCW-outward everywhere.
    assert signed_volume(v, out[:8]) > 0
    assert signed_volume(v, out[8:]) > 0
    # Idempotent.
    again, info2 = canonicalize_winding(v, out)
    assert info2["flipped"] == []
    assert np.array_equal(again, out)
