"""Is this mesh one geometry-central will accept?

THE FAILURE THIS PREVENTS. Driven against the prebuilt addon on AnphySleep's
default anatomy, an unrepaired hemisphere does not fail — it aborts:

    GC_SAFETY_ASSERT FAILURE ... manifold_surface_mesh.cpp:110
      - duplicate edge in list 4184 -- 4078
    Electron exited with signal SIGABRT

`ManifoldSurfaceMesh` rejects a non-manifold edge when it BUILDS the halfedge
structure, so the exception is a C++ abort that no `try/catch` in
the consumer can contain, and it lands in whatever process builds the halfedge
mesh — arbitrarily far from the import that admitted the mesh. The converter
is the last cheap chokepoint.

The check is TOPOLOGICAL, deliberately. A tolerant assembler (scipy's cotan
Laplacian) accepts the same mesh and returns eigenvalues differing in the sixth
significant figure. The defect is numerically negligible and structurally fatal,
so numerics cannot be the test.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: How many offending indices to name in a message before eliding.
_MAX_LISTED = 8


@dataclass(frozen=True)
class MeshHealth:
    n_vertices: int
    n_faces: int
    non_manifold_edges: int
    duplicate_faces: int
    degenerate_faces: int
    isolated_vertices: int
    boundary_edges: int
    inconsistent_winding_edges: int = 0
    non_manifold_edge_list: list[list[int]] = field(default_factory=list)
    isolated_vertex_list: list[int] = field(default_factory=list)
    inconsistent_winding_edge_list: list[list[int]] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        """Boundary edges are absent here ON PURPOSE: `ManifoldSurfaceMesh`
        supports a mesh with boundary, so refusing one would reject legitimate
        surfaces (a cut medial wall, a head mask)."""
        return bool(self.non_manifold_edges or self.duplicate_faces
                    or self.degenerate_faces or self.isolated_vertices
                    or self.inconsistent_winding_edges)

    def as_attrs(self) -> dict:
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "non_manifold_edges": self.non_manifold_edges,
            "duplicate_faces": self.duplicate_faces,
            "degenerate_faces": self.degenerate_faces,
            "isolated_vertices": self.isolated_vertices,
            "boundary_edges": self.boundary_edges,
            "inconsistent_winding_edges": self.inconsistent_winding_edges,
        }

    def summary(self) -> str:
        bits = []
        if self.non_manifold_edges:
            shown = self.non_manifold_edge_list[:_MAX_LISTED]
            bits.append(f"{self.non_manifold_edges} non-manifold edge(s) "
                        f"(>2 incident faces): {shown}")
        if self.inconsistent_winding_edges:
            shown = self.inconsistent_winding_edge_list[:_MAX_LISTED]
            bits.append(f"{self.inconsistent_winding_edges} edge(s) with "
                        f"inconsistent winding (two faces traverse the same "
                        f"directed edge — a directed-edge duplicate that "
                        f"geometry-central aborts on as 'duplicate edge in "
                        f"list'): {shown}")
        if self.duplicate_faces:
            bits.append(f"{self.duplicate_faces} duplicate face(s)")
        if self.degenerate_faces:
            bits.append(f"{self.degenerate_faces} degenerate face(s) (zero area)")
        if self.isolated_vertices:
            bits.append(f"{self.isolated_vertices} isolated vertex/vertices: "
                        f"{self.isolated_vertex_list[:_MAX_LISTED]}")
        if self.boundary_edges:
            bits.append(f"{self.boundary_edges} boundary edge(s)")
        return "; ".join(bits) or "clean"


def check_mesh(vertices: np.ndarray, faces: np.ndarray, *, area_eps: float = 1e-14) -> MeshHealth:
    """Topological health of a 0-BASED triangle mesh.

    Checked GLOBALLY, on the mesh as stored — not per connected component.
    Slicing by connectivity silently drops an orphan vertex, while the app
    slices by the `structures_field` LABEL and carries it into a context
    instead; checking what is actually on disk is what catches that difference.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    n_v, n_f = int(v.shape[0]), int(f.shape[0])

    sorted_f = np.sort(f, axis=1)
    _, counts = np.unique(sorted_f, axis=0, return_counts=True)
    duplicate = int((counts - 1).sum())

    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    degenerate = int((area <= area_eps).sum())

    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, ecounts = np.unique(e, axis=0, return_counts=True)
    nm = uniq[ecounts > 2]
    boundary = int((ecounts == 1).sum())

    # DIRECTED edges, NOT sorted: two faces sharing an edge with CONSISTENT
    # winding traverse it in OPPOSITE directions (e.g. (a,b) and (b,a)), so a
    # correctly-wound closed/open mesh has no directed-edge duplicate. Two
    # faces traversing the SAME edge in the SAME direction — inconsistent
    # per-face winding — shows up here as a repeated directed pair even though
    # the UNDIRECTED edge above still counts only 2 incident faces, so it
    # slips past `non_manifold_edges` entirely. This is exactly the
    # `duplicate edge in list` shape geometry-central aborts on, and
    # `canonicalize_winding` cannot fix it: it flips a whole connected
    # component at once, not individual faces within one.
    de = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    duniq, dcounts = np.unique(de, axis=0, return_counts=True)
    winding_dup = duniq[dcounts > 1]

    used = np.zeros(n_v, dtype=bool)
    if n_f:
        used[f.ravel()] = True
    isolated = np.where(~used)[0]

    return MeshHealth(
        n_vertices=n_v, n_faces=n_f,
        non_manifold_edges=int(nm.shape[0]),
        duplicate_faces=duplicate,
        degenerate_faces=degenerate,
        isolated_vertices=int(isolated.size),
        boundary_edges=boundary,
        inconsistent_winding_edges=int(winding_dup.shape[0]),
        non_manifold_edge_list=[[int(x), int(y)] for x, y in nm[:_MAX_LISTED]],
        isolated_vertex_list=[int(i) for i in isolated[:_MAX_LISTED]],
        inconsistent_winding_edge_list=[[int(x), int(y)] for x, y in winding_dup[:_MAX_LISTED]],
    )


def health_attrs(h: MeshHealth) -> dict:
    """The verdict as the surface node carries it (`mesh_health`), so a CONSUMER that builds
    operators from the mesh can decide before it builds them — and an unclean mesh is still viewable."""
    return {"non_manifold_edges": h.non_manifold_edges, "duplicate_faces": h.duplicate_faces, "degenerate_faces": h.degenerate_faces,
            "isolated_vertices": h.isolated_vertices, "boundary_edges": h.boundary_edges, "inconsistent_winding_edges": h.inconsistent_winding_edges,
            "fatal": bool(h.fatal), "summary": h.summary() if h.fatal else "clean"}


def require_healthy(h: MeshHealth, *, surface: str, primary: bool) -> None:
    """WARN about a fatal surface and let the import proceed (2026-09-23, register D56): the
    verdict travels with the node as `mesh_health`, so a consumer that builds operators from
    the mesh can read it first; refusing at import only made the subject unviewable.
    `NXR_STRICT_MESH=1` restores the refusal for a pipeline that wants it."""
    import os
    if not h.fatal:
        return
    detail = h.summary()
    if not primary or os.environ.get("NXR_STRICT_MESH") != "1":
        warnings.warn(
            f"{'primary' if primary else 'display'} surface {surface!r} is not a clean manifold ({detail}) — "
            f"exported anyway; operators built from this mesh may be wrong until it is repaired upstream.",
            UserWarning, stacklevel=2)
        return
    raise ValueError(
        f"primary surface {surface!r} is not a clean manifold: {detail}. "
        f"geometry-central refuses a non-manifold mesh when it builds the halfedge "
        f"structure and ABORTS THE PROCESS (SIGABRT) rather than raising, so this "
        f"store would abort any consumer that builds its halfedge mesh. Repair the surface in Brainstorm "
        f"(remove the offending faces, then any vertex left isolated, renumbering "
        f"faces and scouts) and recompute the head model and inverse on the repaired "
        f"mesh; the converter mirrors anatomy and does not transform it.")
