"""The DYADIC ATLAS producer (D127–D137; ported from nsp's ``nsp/atlas``, D135).

Measurements on the cortex — static maps (PET SUVR), MEG source activity, fibres — reduced onto a bookkeeping
tree of the cortex (the app's surface ladder, bit-exact) and of time (frames: cycles of the day-anchored tower at level −18, ≈ 0.33 s, doubling), stored
as MERGEABLE sums at the leaves only and rolled up by arithmetic (``code >> k``, ``code >> m``).

    nxr-convert atlas default-subject <dataset> --templates <dir>    the dataset's @default_subject (D127), one composition
    nxr-convert atlas build <dataset> --subject S                    the subject-side rows + its atlas (D128/D129)
    nxr-convert atlas reduce <dataset>                               the group sums, on the default subject (D130/D131)

There is no atlas format: an atlas is measurement rows on tile partitions over plain arrays (``atlas_store``). Frames and
the vector heat method run on nxr-compute's Node binding (``compute.mjs``; the addon the app pins). The newer
DYNAMICS atlas of nsp (grid · placement · engine · spectrum …) is not ported (schema 49 §5).
"""
