"""``timeseries/<channels>`` — the channel MANIFOLD (``points`` given by position) and the Selections that describe it.

    <channels>          Manifold  points — its node IS its positions array [M, 3] f32, scs metres
    <channels>_names    Selection a partition, one code per channel, read with the subject's ``<channels>_names`` dictionary
    <channels>_labels   Selection a set of partitions: type · unit · hemisphere · lobe · quality, each read with the app-wide
                                  dictionary of that family (its colours the app's, D111)

Units are a property of the CHANNEL, not the timeseries — hence the ``unit`` family. ``hemisphere`` and ``lobe`` are the
anatomy CTF puts in a channel name (``M<L|R|Z><C|F|O|P|T>nn``); a name that does not parse gets ``-``. ``quality`` is the
session's channel flag (good/bad). A montage is an intersection of families, derived in the app — never stored.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .crud import Subject, jdump
from .db import uuid7
from .entities import flag_levels, label_set, labelling
from .matio import struct_rows

_UNIT_BY_TYPE = {
    "MEG": "T", "MEG REF": "T", "MEG MAG": "T", "MEG GRAD": "T",
    "EEG": "V", "SEEG": "V", "ECOG": "V", "EOG": "V", "ECG": "V", "EMG": "V",
}
_CTF = re.compile(r"^M([LRZ])([CFOPT])")


def channel_units_for_types(types: list[str]) -> list[str]:
    return [_UNIT_BY_TYPE.get(str(t).strip().upper(), "") for t in types]


def _family(values: list[str]) -> tuple[list[int], list[str]]:
    levels = list(dict.fromkeys(values))          # order-stable, unique
    return [levels.index(v) for v in values], levels


def export_sensors(sub: Subject, channel_mat: dict, channel_flag: np.ndarray, *, name: str, session: str,
                   source_sha1: str | None = None) -> dict:
    channels = struct_rows(channel_mat["Channel"], "Name")
    names = [str(c["Name"]) for c in channels]
    types = [str(c["Type"]) for c in channels]
    units = channel_units_for_types(types)
    loc = np.full((len(channels), 3), np.nan, dtype=np.float64)
    for i, c in enumerate(channels):
        l = np.asarray(c.get("Loc")) if c.get("Loc") is not None else None
        if l is not None and l.size:
            loc[i] = l.reshape(3, -1)[:, 0]        # first coil position
    modalities = list(dict.fromkeys(types))
    n = len(names)
    path = f"timeseries/{name}"

    # THE MANIFOLD — a 0-manifold: isolated elements, its node IS its positions
    mid = uuid7()
    pos = loc.astype(np.float32)
    sub.manifold(id=mid, name=name, path=path, type="points", n_vertices=n, session=session,
                 topology={"kind": "none", "rank": 0}, geometry={"form": "stored", "cs_key": "scs", "positions_path": path},
                 producer_json=jdump({"modalities": modalities, "channel_units_source": "derived-from-type",
                                      **({"source_sha1": source_sha1} if source_sha1 else {})}),
                 populate=lambda at: sub.write_array(path, pos, table="manifold", id=mid))

    # THE NAMES — one level per channel, the subject's own dictionary
    labelling(sub, f"{path}_names", np.arange(n), names, manifold_id=mid, dictionary=f"{name}_names", scope="subject",
              session=session, description="the channel names — one level per channel")

    # THE FAMILIES — one set, each family an app-wide dictionary
    families: list[tuple[str, list[int], list[str], np.ndarray | None]] = []
    codes, levels = _family(types)
    families.append(("type", codes, levels, None))
    codes, levels = _family(units)
    families.append(("unit", codes, levels, None))
    for fam, group in (("hemisphere", 1), ("lobe", 2)):
        parsed = [(_CTF.match(nm).group(group) if _CTF.match(nm) else "-") for nm in names]
        codes, levels = _family(parsed)
        if len(levels) > 1:
            families.append((fam, codes, levels, None))
    flags = np.asarray(channel_flag).ravel() if channel_flag is not None and np.size(channel_flag) else None
    if flags is not None and flags.size == n:
        codes, levels, colors = flag_levels(flags)
        families.append(("quality", codes, levels, np.array(colors, dtype=np.uint8)))
    label_set(sub, f"{path}_labels", [(f, np.asarray(c), l, col) for f, c, l, col in families], manifold_id=mid, session=session,
              description="the montage families — " + ", ".join(f[0] for f in families))
    return {"name": name, "id": mid, "n_channels": n, "modalities": modalities, "names": names}
