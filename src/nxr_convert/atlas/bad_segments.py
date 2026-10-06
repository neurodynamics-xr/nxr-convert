"""THE SAMPLE MASK the atlas excludes: a recording's BAD SEGMENTS are a Selection (D143) — ``<rec>_bad_segments``, spans on
its time Line, written by the converter from Brainstorm's extended events labelled ``bad*`` (``events.export_events``).
Every sample inside a span carries no weight, so the atlas's sums — and the ``samples`` count per frame — are over good
samples only, and stay mergeable."""
from __future__ import annotations

import numpy as np

from .nxr_store import Recording, SubjectStore


def sample_mask(store: SubjectStore, rec: Recording) -> tuple[np.ndarray, dict]:
    """[n] bool, True = a good sample: outside every span of the recording's bad segments."""
    start, stop = store.bad_spans(rec)
    good = np.ones(rec.n_samples, bool)
    i0 = np.clip(np.floor((start - rec.origin) * rec.sfreq).astype(np.int64), 0, rec.n_samples)
    i1 = np.clip(np.ceil((stop - rec.origin) * rec.sfreq).astype(np.int64), 0, rec.n_samples)
    for a, b in zip(i0, i1):
        good[a:b] = False
    return good, {"source": "selection" if len(start) else "none", "segments": int(len(start)),
                  "bad_fraction": float(1 - good.mean()) if rec.n_samples else 0.0}
