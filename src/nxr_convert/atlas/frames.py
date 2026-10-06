"""
Canonical tangent frames on the cortex — "latitude and longitude" for a curved hemisphere.

Per hemisphere (a closed surface, χ = 2), with FreeSurfer's registered sphere (sphere.reg) sharing
its vertices and faces:

  1. POLES. The north and south poles of the registered sphere (+z, −z). On fsaverage they are the
     vertices at the sphere's extreme z; on a subject, the vertices whose registered-sphere points are
     nearest fsaverage's poles — the same anatomical place in every subject (precentral / fusiform–
     inferior temporal on fsaverage).
  2. FIELD. nxr-compute's trivial connection (Crane et al. 2010) with index +1 at each pole: the
     connection closest to Levi-Civita whose only holonomy sits at the poles, and the unit field
     parallel for it — the optimal (smoothest) direction field with exactly those singularities,
     defined up to ONE global rotation.
  3. GAUGE. The rotation is fixed with the vector heat method: the log map from the north pole gives
     the geodesic distance r, and the field is turned (one angle, everywhere) to align with −∇r.
  4. FRAME. e1 is that field ("north"), e2 = n × e1 (the orthogonal field; "west"), (e1, e2, n)
     right-handed.

WHERE the field is solved (``domain``):

  "sphere"  (default) on the registered sphere, where steps 2–3 give the meridians exactly (checked:
            ≤ 0.2° from the analytic meridian), then PUSHED FORWARD to the cortex by the sphere→cortex
            correspondence (the same mesh: at each vertex the 2×2 Jacobian fitted on its 1-ring edges).
            North means the same thing in every subject and on the average cortex: the frames are
            interpretable across subjects, which group analysis needs. Measured on a research cohort
            (pulled back to the sphere): median 1.4–1.7° from the shared meridian, p90 11–13°.
  "cortex"  on the folded cortex itself. Smooth for that cortex, but the connection twists with the
            folding, so its "north" drifts from the shared one (median 10–12°, p90 24–29°) and
            differs between subjects. Kept for within-subject work and comparison.

Why frame components make vector reductions closed-form: e1 is parallel for a connection that is flat
away from the poles, so summing (v·e1, v·e2) over a tile transports every vector to one point along
that connection, path-independently — the sums roll up exactly like scalars. Levi-Civita transport
(the vector heat method) differs from it by the curvature enclosed; reductions that need it are stored
explicitly per level (``vectors.levi_civita``).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).resolve().parent / "compute.mjs"


# ── nxr-compute ─────────────────────────────────────────────────────────────────────────────

def nxr_compute_dir() -> Path:
    """The directory holding the nxr-compute Node binding's ``index.mjs``.

    ``NXR_COMPUTE`` when set (a release payload, or an nxr-compute checkout); else the addon the app pins and
    fetched (``app/vendor/nxr-compute-addon/``) or an ``nxr-compute`` checkout beside the repository, when this file
    sits in the Cortical Flow monorepo (its root found from this file upward).
    """
    cands = []
    if os.environ.get("NXR_COMPUTE"):
        cands.append(Path(os.environ["NXR_COMPUTE"]))
    for up in Path(__file__).resolve().parents:
        if (up / "backend" / "schema" / "model.sql").is_file():          # the monorepo's root (a source checkout only)
            cands.append(up / "app" / "vendor" / "nxr-compute-addon")
            cands.append(up.parent / "nxr-compute")
            break
    for p in cands:
        if (p / "index.mjs").is_file() and (p / "nxr_compute_addon.node").is_file():
            return p
        if (p / "bindings/node/index.mjs").is_file():
            return p
    raise FileNotFoundError("nxr-compute Node binding not found (tried " + ", ".join(map(str, cands))
                            + "); set NXR_COMPUTE to the directory holding its index.mjs")


def run_compute(mode: str, vertices: np.ndarray, faces: np.ndarray, args: dict, extra: dict | None = None,
                outputs: tuple[str, ...] = ()) -> dict:
    """Run ``compute.mjs`` on a mesh in a temporary directory; returns ``{name: float64 array | dict}`` for
    each requested output (``.f64`` files as arrays, ``.json`` as dicts). The directory is removed after."""
    with tempfile.TemporaryDirectory(prefix="nxr-atlas-") as tmp:
        d = Path(tmp)
        np.ascontiguousarray(vertices, dtype="<f8").tofile(d / "vertices.f64")
        np.ascontiguousarray(faces, dtype="<i4").tofile(d / "faces.i32")
        (d / "args.json").write_text(json.dumps(args))
        for name, arr in (extra or {}).items():
            np.ascontiguousarray(arr).tofile(d / name)
        env = dict(os.environ, NXR_COMPUTE=str(nxr_compute_dir()))
        r = subprocess.run(["node", str(RUNNER), mode, "--dir", str(d)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"nxr-compute {mode} failed:\n{r.stderr[-2000:]}")
        return {n: (json.loads((d / n).read_text()) if n.endswith(".json") else np.fromfile(d / n, dtype="<f8"))
                for n in outputs}


# ── mesh helpers ────────────────────────────────────────────────────────────────────────────

def separate_coincident(p: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Move every vertex that sits exactly on an earlier one to the centroid of its other neighbours.

    fsaverage has one (vertex 40969 on vertex 0, both hemispheres, every surface): two faces of zero
    area, on which no cotan operator is defined. The move is local and sub-millimetre; it is recorded.
    """
    p = p.copy()
    _, first, inv = np.unique(p, axis=0, return_index=True, return_inverse=True)
    dup = np.flatnonzero(first[inv.ravel()] != np.arange(len(p)))
    for v in dup:
        nb = np.unique(f[(f == v).any(1)])
        nb = nb[(nb != v) & ~np.all(p[nb] == p[v], axis=1)]
        p[v] = p[nb].mean(0)
    return p, dup.tolist()


def face_normals_areas(p: np.ndarray, f: np.ndarray):
    c = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    a = np.linalg.norm(c, axis=1)
    return c / np.maximum(a, 1e-300)[:, None], 0.5 * a


def vertex_normals(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    n, a = face_normals_areas(p, f)
    vn = np.zeros_like(p)
    for k in range(3):
        np.add.at(vn, f[:, k], n * a[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


def face_gradient(p: np.ndarray, f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The piecewise-linear gradient of a vertex scalar, per face [nF×3]."""
    n, a = face_normals_areas(p, f)
    g = np.zeros((len(f), 3))
    for k in range(3):
        i, j = f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        g += x[f[:, k]][:, None] * np.cross(n, p[j] - p[i])
    return g / (2 * np.maximum(a, 1e-300))[:, None]


def tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Any orthonormal (b1, b2) spanning each tangent plane."""
    ref = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    b1 = ref - (ref * normals).sum(1, keepdims=True) * normals
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    return b1, np.cross(normals, b1)


def edges_of(f: np.ndarray) -> np.ndarray:
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    return np.unique(np.sort(e, axis=1), axis=0)


def jacobians(src: np.ndarray, src_n: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, f: np.ndarray):
    """Per vertex, the 2×2 least-squares linear map from src's tangent plane to dst's, fitted on the 1-ring edges.

    Returns (J [nV,2,2], src basis (b1,b2), dst basis (b1,b2), condition number [nV]).
    """
    sb1, sb2 = tangent_basis(src_n)
    db1, db2 = tangent_basis(dst_n)
    e = edges_of(f)
    SS = np.zeros((len(src), 2, 2))
    CS = np.zeros((len(src), 2, 2))
    for a, b in ((e[:, 0], e[:, 1]), (e[:, 1], e[:, 0])):
        ds, dd = src[b] - src[a], dst[b] - dst[a]
        s = np.stack([(ds * sb1[a]).sum(1), (ds * sb2[a]).sum(1)], 1)
        c = np.stack([(dd * db1[a]).sum(1), (dd * db2[a]).sum(1)], 1)
        np.add.at(SS, a, s[:, :, None] * s[:, None, :])
        np.add.at(CS, a, c[:, :, None] * s[:, None, :])
    J = CS @ np.linalg.inv(SS)
    sv = np.linalg.svd(J, compute_uv=False)
    return J, (sb1, sb2), (db1, db2), sv[:, 0] / np.maximum(sv[:, 1], 1e-300)


# ── the field ───────────────────────────────────────────────────────────────────────────────

def optimal_field(p: np.ndarray, f: np.ndarray, north: int, south: int):
    """nxr-compute's trivial connection (index +1 at each pole), gauge-fixed towards the north pole.

    Returns (per-face unit field, per-vertex unit field, meta).
    """
    out = run_compute("frames", p, f, {"north": int(north), "south": int(south)},
                      outputs=("field.f64", "logmap.f64", "frames.json"))
    u = out["field.f64"].reshape(-1, 3)
    log = out["logmap.f64"].reshape(-1, 2)
    meta = out["frames.json"]
    if not meta.get("gaussBonnet"):
        raise ValueError(f"Gauss–Bonnet violated: χ = {meta.get('euler'):.3f} ≠ 2 (not a closed sphere)")
    n, a = face_normals_areas(p, f)
    u = u - (u * n).sum(1, keepdims=True) * n
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
    w = np.cross(n, u)
    ref = -face_gradient(p, f, np.linalg.norm(log, axis=1))       # towards the north pole (vector heat log map)
    rn = np.linalg.norm(ref, axis=1)
    ok = rn > 0
    z = ((ref * u).sum(1) + 1j * (ref * w).sum(1))[ok] / rn[ok]
    theta = float(np.angle(np.sum(a[ok] * z)))
    ef = np.cos(theta) * u + np.sin(theta) * w
    align = float(np.sum(a[ok] * (ref[ok] * ef[ok]).sum(1) / rn[ok]) / a[ok].sum())
    vn = vertex_normals(p, f)
    acc = np.zeros_like(p)
    for k in range(3):
        np.add.at(acc, f[:, k], ef * a[:, None])
    acc -= (acc * vn).sum(1, keepdims=True) * vn
    ev = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
    return ef, ev, {"euler": meta["euler"], "gauge_rotation": theta, "alignment_north": align}


@dataclass
class HemisphereFrames:
    e1: np.ndarray          # [nV×3] "north", tangent to the cortex
    e2: np.ndarray          # [nV×3] n × e1 ("west")
    normal: np.ndarray      # [nV×3] outward
    singular: np.ndarray    # [nV] bool — no frame (the poles; an ill-conditioned push-forward)
    north: int
    south: int
    domain: str
    meta: dict


def hemisphere_frames(cortex: np.ndarray, faces: np.ndarray, sphere: np.ndarray, north: int, south: int,
                      domain: str = "sphere", max_condition: float = 50.0) -> HemisphereFrames:
    f = np.asarray(faces, dtype=np.int64)
    cortex, moved_c = separate_coincident(np.asarray(cortex, dtype=np.float64), f)
    sphere, moved_s = separate_coincident(np.asarray(sphere, dtype=np.float64), f)
    su = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    vn = vertex_normals(cortex, f)
    meta = {"coincident_vertices_moved": sorted(set(moved_c) | set(moved_s))}
    if domain == "sphere":
        _, e_s, m = optimal_field(su, f, north, south)
        # the analytic meridian, as a check of the solve
        mer = np.array([0, 0, 1.0]) - su[:, 2:3] * su
        mn = np.linalg.norm(mer, axis=1)
        okm = mn > 1e-6
        dev = np.degrees(np.arccos(np.clip((e_s[okm] * mer[okm]).sum(1) / mn[okm], -1, 1)))
        J, (sb1, sb2), (db1, db2), cond = jacobians(su, su, cortex, vn, f)
        m2 = np.stack([(e_s * sb1).sum(1), (e_s * sb2).sum(1)], 1)
        c2 = (J @ m2[:, :, None])[:, :, 0]
        e1 = c2[:, :1] * db1 + c2[:, 1:] * db2
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-300)
        singular = (cond > max_condition) | ~np.isfinite(cond)
        meta.update(m, meridian_deviation_deg={"median": float(np.median(dev)), "max": float(np.max(dev))},
                    pushforward_condition={"median": float(np.median(cond)), "p99": float(np.percentile(cond, 99))})
    elif domain == "cortex":
        _, e1, m = optimal_field(cortex, f, north, south)
        singular = np.zeros(len(cortex), dtype=bool)
        meta.update(m)
    else:
        raise ValueError(f"domain must be sphere|cortex, not {domain!r}")
    singular[[north, south]] = True
    e1 = e1.copy()
    e1[singular] = np.nan
    e2 = np.cross(vn, e1)
    meta["singular_vertices"] = int(singular.sum())
    return HemisphereFrames(e1, e2, vn, singular, int(north), int(south), domain, meta)


def meridian_angle(fr: HemisphereFrames, cortex: np.ndarray, faces: np.ndarray, sphere: np.ndarray) -> np.ndarray:
    """Per vertex, the angle (degrees) between e1 pulled back to the registered sphere and the sphere's meridian.

    0 everywhere for the sphere domain (by construction); for the cortex domain it measures how far
    that cortex's "north" is from the shared one.
    """
    f = np.asarray(faces, dtype=np.int64)
    cortex, _ = separate_coincident(np.asarray(cortex, dtype=np.float64), f)
    sphere, _ = separate_coincident(np.asarray(sphere, dtype=np.float64), f)
    su = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    J, (cb1, cb2), (sb1, sb2), _ = jacobians(cortex, fr.normal, su, su, f)
    e = np.nan_to_num(fr.e1)
    c2 = np.stack([(e * cb1).sum(1), (e * cb2).sum(1)], 1)
    s2 = (J @ c2[:, :, None])[:, :, 0]
    back = s2[:, :1] * sb1 + s2[:, 1:] * sb2
    mer = np.array([0, 0, 1.0]) - su[:, 2:3] * su
    cos = (back * mer).sum(1) / np.maximum(np.linalg.norm(back, axis=1) * np.linalg.norm(mer, axis=1), 1e-300)
    out = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    out[fr.singular] = np.nan
    return out


def poles_on_sphere(sphere_unit: np.ndarray, north_dir: np.ndarray, south_dir: np.ndarray) -> tuple[int, int]:
    """The vertices whose (unit) sphere points are nearest the given pole directions."""
    return int(np.argmax(sphere_unit @ north_dir)), int(np.argmax(sphere_unit @ south_dir))
