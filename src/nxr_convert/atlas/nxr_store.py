"""What the atlas reads of one subject — FOUND BY ITS ROWS in the dataset's database, read from the arrays the rows name.

  surface       the subject's primary surface (``is_primary``), its positions and faces (its geometry, its topology's cells)
  sphere        ``<surface>_sphere`` — FreeSurfer's registered sphere (the same topology)
  structures    the ``Structures`` member of the surface's parcellations, its codes named by its dictionary
  maps          scalar Fields on the surface, one value per vertex (PET SUVR, …)
  recordings    Fields of kind ``recording`` on channels × time, with their time Line's sampling
  kernels       Operators of kind ``inverse kernel`` [vertices × channels], with the channel Indices they map from
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass
class Kernel:
    name: str
    session: str
    method: str          # the kernel's name up to _KERNEL (e.g. "MN_MEG", "dSPM-unscaled_MEG")
    stamp: str           # Brainstorm's timestamp (e.g. "260930_0516")
    n_vertices: int
    n_channels: int
    id: str = ""
    path: str = ""
    channels_path: str | None = None


@dataclass
class Recording:
    name: str
    session: str
    n_channels: int
    n_samples: int
    sfreq: float
    id: str = ""
    path: str = ""
    time_id: str = ""
    origin: float = 0.0


class SubjectStore:
    """One subject of an open dataset (``crud.Dataset``), read through its rows."""

    def __init__(self, ds, subject: str | dict, surface: str | None = None):
        self.ds, self.db = ds, ds.db
        self.row = subject if isinstance(subject, dict) else ds.db.one(
            "SELECT * FROM subject WHERE dataset_id = ? AND (name = ? OR path = ? OR id = ?) AND status = 'complete'",
            ds.id, subject, subject, subject)
        if self.row is None:
            raise FileNotFoundError(f"{subject}: no subject of {ds.row['name']}")
        self.id = self.row["id"]
        self.subject = self.row["name"]
        self.path = ds.root / ds.row["path"] / self.row["path"]
        if surface is None:
            prim = self.db.one("SELECT name FROM manifold WHERE subject_id = ? AND type = 'surface' AND is_primary = 1", self.id)
            surface = prim["name"] if prim else "cortex_pial_low"
        self.surface = surface

    def array(self, node: str) -> np.ndarray:
        import zarr
        return np.asarray(zarr.open_array(str(self.path / node), mode="r")[...])

    def manifold(self, name: str) -> dict:
        r = self.db.one("SELECT * FROM manifold WHERE subject_id = ? AND name = ? AND component_of_id IS NULL", self.id, name)
        if r is None:
            raise FileNotFoundError(f"{self.subject}: no manifold {name}")
        return r

    def _positions(self, m: dict) -> np.ndarray:
        g = self.db.read("geometry", m["geometry_id"])
        return self.array(g["positions_path"]).astype(np.float64)

    # ── the cortex ────────────────────────────────────────────────────────────────────────
    @cached_property
    def surface_row(self) -> dict:
        return self.manifold(self.surface)

    @cached_property
    def positions(self) -> np.ndarray:
        return self._positions(self.surface_row)

    @cached_property
    def faces(self) -> np.ndarray:
        cell = self.db.one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", self.surface_row["topology_id"])
        return self.array(cell["path"]).astype(np.int64)

    @cached_property
    def sphere(self) -> np.ndarray:
        m = self.manifold(f"{self.surface}_sphere")
        cs = self.db.one("SELECT sp.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id "
                         "JOIN space sp ON sp.id = c.space_id WHERE g.id = ?", m["geometry_id"])
        if cs is not None and "sphere" not in cs["name"].lower():
            raise ValueError(f"{m['path']} is not FreeSurfer's sphere: {cs['name']}")
        return self._positions(m)

    def labels(self, atlas: str) -> tuple[np.ndarray, list[str]]:
        """A parcellation's codes and its names (``names[code]``, from its dictionary)."""
        r = self.db.one("SELECT x.* FROM selection x JOIN selection s ON s.id = x.member_of_id WHERE x.subject_id = ? AND "
                        "s.path = ? AND x.name = ?", self.id, f"sources/{self.surface}_atlases", atlas)
        if r is None:
            raise FileNotFoundError(f"{self.subject}: no {atlas} parcellation on {self.surface}")
        entries = self.db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", r["dictionary_id"])
        names = [""] * (max(e["code"] for e in entries) + 1)
        for e in entries:
            names[e["code"]] = e["name"]
        return self.array(r["array_path"]), names

    @property
    def n_vertices(self) -> int:
        return int(self.surface_row["n_vertices"])

    # ── scalar maps on the surface ────────────────────────────────────────────────────────
    def maps(self) -> dict[str, dict]:
        """Every scalar Field with one value per surface vertex: name → its row."""
        return {r["name"]: r for r in self.db.all(
            "SELECT * FROM field WHERE subject_id = ? AND manifold_id = ? AND value_type = 'scalar' AND path IS NOT NULL "
            "AND status = 'complete' ORDER BY name", self.id, self.surface_row["id"])}

    # ── MEG ───────────────────────────────────────────────────────────────────────────────
    def recordings(self) -> list[Recording]:
        out = []
        for f in self.db.all("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND path IS NOT NULL ORDER BY name", self.id):
            dims = self.db.all("SELECT d.manifold_id, d.n_vertices FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id "
                               "WHERE m.id = ? ORDER BY d.ordinal", f["manifold_id"])
            if len(dims) != 2:
                continue
            line = self.db.read("manifold", dims[1]["manifold_id"])
            d = self.db.one("SELECT spacing, origin FROM geometry_dimension WHERE geometry_id = ? AND ordinal = 0", line["geometry_id"])
            if not d or not d["spacing"]:
                continue
            out.append(Recording(f["name"], f["session"] or "", int(dims[0]["n_vertices"]), int(dims[1]["n_vertices"]),
                                 float(round(1.0 / d["spacing"], 9)), f["id"], f["path"], line["id"], float(d["origin"] or 0.0)))
        return out

    def kernels(self) -> list[Kernel]:
        out = []
        for k in self.db.all("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND path IS NOT NULL ORDER BY name", self.id):
            method, _, rest = k["name"].partition("_KERNEL_")
            ch = self.db.read("selection", k["from_selection_id"]) if k["from_selection_id"] else None
            out.append(Kernel(k["name"], k["session"] or "", method, rest[:11], int(k["n_rows"]), int(k["n_cols"]), k["id"], k["path"],
                              ch["array_path"] if ch else None))
        return out

    def kernel_channels(self, k: Kernel) -> np.ndarray:
        if not k.channels_path:
            return np.arange(k.n_channels)
        return self.array(k.channels_path).astype(np.int64)

    def recording_data(self, r: Recording) -> np.ndarray:
        return self.array(r.path).astype(np.float32, copy=False)

    def bad_spans(self, r: Recording) -> tuple[np.ndarray, np.ndarray]:
        """(start s, stop s) of the recording's bad segments — its ``<rec>_bad_segments`` spans (D143); empty when none."""
        sel = self.db.one("SELECT id FROM selection WHERE subject_id = ? AND of_field_id = ? AND type = 'spans' AND name = ?",
                          self.id, r.id, f"{r.name}_bad_segments")
        if sel is None:
            return np.zeros(0), np.zeros(0)
        rows = self.db.all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", sel["id"])
        return np.array([x["start"] for x in rows], float), np.array([x["stop"] for x in rows], float)
