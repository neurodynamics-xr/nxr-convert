"""A minimal Brainstorm-shaped EEG protocol on disk.

EVERY existing gate is MEG, which is why all four EEG defects were latent: an
EEG link has no CTF compensation and an EEG head model has no MEG method, and
in Brainstorm both absences are EMPTY ARRAYS rather than missing fields — the
exact shape that slipped past `is not None` and `v not in (None, "")`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

EEG_NAMES = ["FZ", "CZ", "PZ", "OZ", "PO3", "PO7", "T7", "T8"]


def _surface(n_v: int = 6):
    """A closed octahedron — manifold, watertight, genus 0."""
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
                  [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]], np.int32)  # 1-based
    return v[:n_v] if n_v != 6 else v, f


def make_eeg_protocol(root: Path, *, subject: str = "S01", condition: str = "C",
                      use_default_anat: bool = False, n_chan: int = 8,
                      n_t: int = 400, sfreq: float = 100.0) -> dict:
    root = Path(root)
    names = EEG_NAMES[:n_chan]
    anat_name = "@default_subject" if use_default_anat else subject
    anat = root / "anat" / anat_name
    anat.mkdir(parents=True, exist_ok=True)
    (root / "anat" / subject).mkdir(parents=True, exist_ok=True)

    savemat(root / "anat" / subject / "brainstormsubject.mat",
            {"UseDefaultAnat": np.array([[1 if use_default_anat else 0]]),
             "Cortex": f"{anat_name}/tess_cortex_mid_low.mat"})

    v, f = _surface()
    # THREE cortex surfaces: the kernel names the one iCortex does NOT, so a
    # test can tell "resolved through the kernel" from "resolved through the
    # index" — and the literal last-resort (`tess_cortex_pial_low.mat`) is
    # ALSO present, so a half-fixed `_kernel_surface` that misses the kernel
    # falls all the way through to a surface that EXISTS rather than crashing
    # with FileNotFoundError. A wrong-but-present surface is what makes the
    # regression a silent one — a store that loads, renders, and is wrong
    # everywhere — which is the failure mode this fixture exists to catch.
    for nm, comment in [("tess_cortex_mid_low.mat", "mid_6V"),
                        ("tess_cortex_white_high.mat", "white_6V"),
                        ("tess_cortex_pial_low.mat", "pial_6V")]:
        savemat(anat / nm, {"Vertices": v, "Faces": f, "Comment": comment,
                            "Atlas": np.array([], dtype=object)})

    raw = root / "data" / subject / f"@raw{condition}"
    raw.mkdir(parents=True, exist_ok=True)

    savemat(raw / "channel.mat", {
        "Comment": "EEG channels",
        "Channel": np.array(
            [(n, "EEG", np.zeros((3, 1)), np.array([]), np.array([])) for n in names],
            dtype=[("Name", "O"), ("Type", "O"), ("Loc", "O"), ("Orient", "O"), ("Weight", "O")]),
    })

    bst = raw / f"{condition}.bst"
    epoch = 100
    data = np.zeros((n_chan, n_t), dtype="<f4")
    with open(bst, "wb") as fh:
        fh.write(b"\0" * 8)
        for e in range(n_t // epoch):
            fh.write(data[:, e * epoch:(e + 1) * epoch].tobytes())

    savemat(raw / f"data_0raw_{condition}.mat", {
        "F": {
            "format": "BST-BIN", "filename": str(bst), "device": "EDF",
            # THE EEG SHAPE: empty arrays, not absent fields.
            "prop": {"sfreq": sfreq, "times": np.array([0.0, (n_t - 1) / sfreq]),
                     "currCtfComp": np.array([]), "destCtfComp": np.array([])},
            "header": {"nchannels": n_chan, "nsamples": n_t,
                       "epochsize": epoch, "hdrsize": 8},
            # A single occurrence's times matrix ALWAYS squeezes to 1-D through
            # pymatreader regardless of orientation (2x1 and 1x2 both come back
            # shape (2,)) — this is exactly the shape of the real file's `L`
            # event (AnphySleep/EPCTL01: one 150 s span, 28560.0 -> 28710.0),
            # which a bare `np.atleast_2d` misreads as two zero-duration marks
            # instead of one extended span. `epochs` (Brainstorm's own
            # ``1 x nOccurrences``) is the disambiguator `parse_events` uses to
            # recover the (2, 1) split, so this is a genuine EXTENDED
            # one-occurrence event (2 rows, 1 column) with its `epochs` count
            # of 1 — the same shape as `L`, not a workaround around it.
            "events": {"label": ["N2"], "color": [np.array([0.1, 0.35, 0.8])],
                       "times": [np.array([[1.0], [2.0]])],
                       "epochs": [np.array([1.0])],
                       "channels": [[np.array([])]]},
        },
        "ChannelFlag": np.ones((n_chan, 1)),
    })

    n_src = v.shape[0]
    savemat(raw / "headmodel_surf_openmeeg.mat", {
        "Comment": "OpenMEEG BEM", "HeadModelType": "surface",
        "GridLoc": v, "GridOrient": v,
        "Gain": np.zeros((n_chan, 3 * n_src)),
        "EEGMethod": "openmeeg", "MEGMethod": np.array([]),   # the crash shape
        "SurfaceFile": f"{anat_name}/tess_cortex_mid_low.mat",
    })
    savemat(raw / "results_MN_EEG_KERNEL_000000_0000.mat", {
        "Comment": "MN: EEG(Unconstr) 2018",
        "ImagingKernel": np.zeros((3 * n_src, n_chan)),
        "GoodChannel": np.arange(1, n_chan + 1),
        "ChannelFlag": np.ones((n_chan, 1)),
        "nComponents": 3,
        "HeadModelFile": f"{subject}/@raw{condition}/headmodel_surf_openmeeg.mat",
        "SurfaceFile": f"{anat_name}/tess_cortex_mid_low.mat",
        "Options": {"InverseMethod": "minnorm", "InverseMeasure": "amplitude"},
    })
    return {"root": root, "subject": subject, "condition": condition, "raw_dir": raw,
            "anat_dir": anat, "channel_names": names,
            "surface": "tess_cortex_mid_low.mat"}
