"""Mesh health, on hand-built minimal meshes.

The AnphySleep defect is reproduced by CONSTRUCTION rather than fixtured from
the real file: a test that reads its expectation from the same source as the
code cannot catch that source being misread. The real mesh is covered by the
opt-in gate instead.

Why this exists at all: driven against the prebuilt addon, an unrepaired
hemisphere does not fail, it ABORTS —
  GC_SAFETY_ASSERT FAILURE ... manifold_surface_mesh.cpp:110 - duplicate edge
  Electron exited with signal SIGABRT
geometry-central rejects a non-manifold edge when it BUILDS the halfedge mesh,
so no try/catch downstream can contain it. This is the only cheap chokepoint.
"""
import numpy as np
import pytest

from nxr_convert.mesh_health import MeshHealth, check_mesh, require_healthy


def octahedron():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)
    return v, f


def test_a_closed_octahedron_is_clean():
    h = check_mesh(*octahedron())
    assert (h.non_manifold_edges, h.boundary_edges, h.isolated_vertices) == (0, 0, 0)
    assert (h.duplicate_faces, h.degenerate_faces) == (0, 0)
    assert not h.fatal


def test_the_anphysleep_flap_is_caught():
    """One extra triangle hanging off an existing edge by a new vertex — the
    exact topology of @default_subject/tess_cortex_mid_low: it makes that edge
    non-manifold AND contributes two boundary edges, all from one face."""
    v, f = octahedron()
    v = np.vstack([v, [[0.5, 0.5, 0.5]]])          # the pendant vertex, index 6
    f = np.vstack([f, [[0, 2, 6]]])                # edge 0-2 now has 3 faces
    h = check_mesh(v, f)
    assert h.non_manifold_edges == 1
    assert h.boundary_edges == 2
    assert h.non_manifold_edge_list == [[0, 2]]
    assert h.fatal


def test_an_isolated_vertex_is_fatal():
    """Zero mass, singular pencil. Deleting the flap FACE alone leaves exactly
    this, which is why the upstream repair must drop the vertex too."""
    v, f = octahedron()
    v = np.vstack([v, [[9.0, 9.0, 9.0]]])
    h = check_mesh(v, f)
    assert h.isolated_vertices == 1
    assert h.isolated_vertex_list == [6]
    assert h.fatal


def test_a_degenerate_face_is_fatal():
    v, f = octahedron()
    v = np.vstack([v, [v[0]]])                     # index 6 duplicates vertex 0
    f = np.vstack([f, [[0, 6, 2]]])                # zero area -> cotan blows up
    assert check_mesh(v, f).degenerate_faces == 1
    assert check_mesh(v, f).fatal


def test_a_duplicate_face_is_fatal():
    v, f = octahedron()
    f = np.vstack([f, f[0:1]])
    h = check_mesh(v, f)
    assert h.duplicate_faces == 1
    assert h.fatal


def test_a_boundary_alone_is_NOT_fatal():
    """ManifoldSurfaceMesh supports meshes with boundary; refusing one would
    reject legitimate surfaces."""
    v, f = octahedron()
    h = check_mesh(v, f[:-1])                      # remove a face -> open hole
    assert h.boundary_edges == 3
    assert h.non_manifold_edges == 0
    assert not h.fatal


def test_require_healthy_only_WARNS_for_an_unhealthy_PRIMARY(monkeypatch):
    monkeypatch.delenv("NXR_STRICT_MESH", raising=False)
    v, f = octahedron(); v = np.vstack([v, [[0.5, 0.5, 0.5]]]); f = np.vstack([f, [[0, 2, 6]]])
    with pytest.warns(UserWarning) as rec:
        require_healthy(check_mesh(v, f), surface="cortex_mid_low", primary=True)
    msg = str(rec[0].message)
    assert "cortex_mid_low" in msg and "non-manifold" in msg and "[0, 2]" in msg and "Prepare" not in msg


def test_require_healthy_refuses_the_PRIMARY_under_NXR_STRICT_MESH(monkeypatch):
    monkeypatch.setenv("NXR_STRICT_MESH", "1")
    v, f = octahedron(); v = np.vstack([v, [[0.5, 0.5, 0.5]]]); f = np.vstack([f, [[0, 2, 6]]])
    with pytest.raises(ValueError) as e:
        require_healthy(check_mesh(v, f), surface="cortex_mid_low", primary=True)
    msg = str(e.value)
    assert "cortex_mid_low" in msg and "non-manifold" in msg and "[0, 2]" in msg


def test_require_healthy_only_WARNS_for_a_display_surface():
    v, f = octahedron()
    v = np.vstack([v, [[0.5, 0.5, 0.5]]])
    f = np.vstack([f, [[0, 2, 6]]])
    with pytest.warns(UserWarning, match="head_mask"):
        require_healthy(check_mesh(v, f), surface="head_mask", primary=False)


def _two_triangles(reversed_second: bool = False):
    """Two triangles sharing edge (1, 2) — an open (boundary) sheet, small
    enough that only the winding of the shared edge is in play."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
    if reversed_second:
        f = f.copy()
        f[1] = f[1][::-1]
    return v, f


def test_two_triangles_with_consistent_winding_are_clean():
    """Consistent winding: the shared edge is traversed in OPPOSITE directions
    by its two faces (directed (1,2) and (2,1)), which is what a correctly
    wound mesh looks like. This must read as clean even though the sheet has
    boundary (open, not fatal)."""
    v, f = _two_triangles(reversed_second=False)
    h = check_mesh(v, f)
    assert h.inconsistent_winding_edges == 0
    assert not h.fatal
    assert h.summary() == "4 boundary edge(s)"


def test_two_triangles_with_one_face_reversed_is_fatal_and_names_the_condition():
    """Reversing the second face makes it traverse the shared edge in the SAME
    direction as the first — a directed-edge duplicate, undetectable by the
    undirected non-manifold-edge check (still exactly 2 incident faces) and
    unrepairable by `canonicalize_winding` (it flips per CONNECTED COMPONENT,
    and this is one component). This is the `duplicate edge in list` shape
    geometry-central aborts on."""
    v, f = _two_triangles(reversed_second=True)
    h = check_mesh(v, f)
    assert h.inconsistent_winding_edges == 1
    assert h.non_manifold_edges == 0, "must not be caught (or miscaught) by the undirected check"
    assert h.fatal
    msg = h.summary()
    assert "inconsistent winding" in msg


def test_as_attrs_is_json_shaped():
    h = check_mesh(*octahedron())
    a = h.as_attrs()
    assert a["non_manifold_edges"] == 0 and a["n_vertices"] == 6
    assert all(isinstance(x, int) for x in a.values())
