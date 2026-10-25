# nxr-convert

Converts a [Brainstorm](https://neuroimage.usc.edu/brainstorm/) protocol (MEG/EEG recordings, cortical surfaces, MRI
volumes, head models, inverse kernels, source maps, fibres and connectomes) into an **nxr datastore**: a folder that the
[Cortical Flow](https://corticalflow.app) desktop app opens.

Version **0.2.9**, which writes **database schema 49**. Each dataset's `dataset.sqlite` carries the schema number as
`PRAGMA user_version`. The app opens only databases with the version it was built for, and the converter opens only its
own version and refuses any other. Use a converter release that matches your desktop app (0.2.x for Cortical Flow 0.2).

## What it writes

```
<datastore>/
  <dataset>/
    dataset.sqlite          the dataset's database: one row per subject, surface, recording, kernel, …
    <subject>.nxr.zarr/     one Zarr v3 store per subject; the bytes (positions, faces, timeseries, kernels …)
```

The rows are written first, and every `zarr.json` is then rewritten from the rows, so the store describes itself. A
Brainstorm **condition** becomes a *session* of the subject. The protocol's default anatomy (`@default_subject`) becomes
the dataset's **template** subject, and `Group_analysis` becomes its **group** subject.

## Install

Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install nxr-convert
```

or with [pipx](https://pipx.pypa.io/):

```bash
pipx install nxr-convert
```

Check the install:

```bash
nxr-convert --version        # nxr-convert 0.2.9 (database schema 49)
```

The `atlas` commands also need the `atlas` extra (`nibabel`), plus Node.js and the nxr-compute Node binding
(`NXR_COMPUTE`). Install the extra with `uv tool install "nxr-convert[atlas]"`.
The conversion commands below need neither.

## Use with the Cortical Flow desktop app

The desktop app's **Import** runs this converter. It looks for one in this order:

1. `NXR_CONVERT_CMD`, an explicit command (for example `uv run --project /path/to/nxr-convert nxr-convert`);
2. a development checkout of the app's monorepo, run through `uv`;
3. `nxr-convert` on `PATH`;
4. `~/.local/bin/nxr-convert` (`nxr-convert.exe` on Windows), where `uv tool install` and `pipx install` put it — found
   even when the app was started from the Dock or Finder and did not inherit your shell's `PATH`.

If none is found, the Import page shows a setup card. For an install elsewhere, set `NXR_CONVERT_CMD` to its full path.

## Commands

Every command prints its progress as JSON lines on stdout, one object per event keyed by `stage`. The app parses this
output. A failure ends with a `{"stage": "error", "message": …}` line and exit status 1.

### Datasets

```bash
nxr-convert dataset create <datastore> <name> [--source-tool brainstorm] [--source-protocol P] [--source-path S]
nxr-convert dataset delete <datastore> <name>
nxr-convert dataset recover <datastore>/<name>    # roll back unfinished writes, finish deletions, rewrite the store
```

`dataset create` makes `<datastore>/<name>/dataset.sqlite` from the schema. It also creates the datastore folder if
it does not exist yet.

### Reading a protocol

```bash
nxr-convert list <protocol>                       # its subjects and conditions, as JSON
nxr-convert scan <protocol> [--prefer protocol.mat|walk] [--out view.json] [--summary]
```

### Converting

```bash
# one whole subject as one composition: every session, MRI volumes, source maps, fibres, connectomes
nxr-convert subject <protocol> --subject <name> --dataset <datastore>/<dataset>

# one condition as a session of a subject (the subject is created if it is absent)
nxr-convert convert <protocol> --subject <name> --condition <condition> --dataset <datastore>/<dataset> \
    [--surface tess_*.mat] [--extra-surface tess_*.mat …] [--recording data_*.mat] [--channel-file F] \
    [--no-gain] [--all-surfaces] [--include-imported] [--no-raw]

# the protocol's default anatomy as the TEMPLATE subject, and its group results as the GROUP subject
nxr-convert template <protocol> --dataset <datastore>/<dataset> [--surface tess_*.mat …]
nxr-convert group    <protocol> --dataset <datastore>/<dataset>
```

`<protocol>` is the Brainstorm protocol folder, the one that holds `anat/` and `data/`.

`template` also converts the **static maps** on the default anatomy: every `results_*.mat` source map (one value per
vertex) and every `matrix_*.mat` region map (scout extraction: one value per region of its `Atlas`) in `data/@default_study`,
`data/@inter` or a `data/Group_analysis` condition whose `SurfaceFile` is a `@default_subject` surface. A map on a surface
the template does not export itself (fsLR32k, fsaverage) brings that surface as a secondary one. A region map is a Field
on the atlas's regions (element k is the region coded k; NaN where the file gives no value). Time-resolved maps are
reported and skipped.

### Adding to or removing from a subject

```bash
nxr-convert surface --dataset D --subject S --file tess_*.mat [tess_*.mat …] [--bst-root <protocol>]
nxr-convert mri     --dataset D --subject S --anat <protocol>/anat/<subject> [--only NAME …]
nxr-convert inverse --dataset D --subject S --file results_*KERNEL*.mat --session C --bst-root <protocol> \
    [--channels-id ID] [--surface-id ID] [--forward-id ID] [--name N] [--channel-file F]
nxr-convert map     --dataset D --subject S --file results_*.mat|matrix_*.mat … --bst-root <protocol> [--session C]
nxr-convert surface --dataset D --subject S --fs-dir <FreeSurfer subject dir> --name cortex_pial [--scs-from tess_*.mat]
nxr-convert hodge   --dataset D --subject S --from <dir> [--nsp-subject S] [--replace] [--surface NAME]
nxr-convert interp  --dataset D --subject S --from SRC --to DST [--replace]
nxr-convert remove  --dataset D --subject S [--node <store path>]   # the subject, or one node of it
```

`hodge` adds the Hodge–Dirac eigenmodes of the subject's primary cortex from a solve's outputs in `<dir>`
(split layout, 0.2.5: `<S>_hodge_exact_modes/_exact_lambda/_coexact_modes/_coexact_lambda.npy` in place of `_hodge_modes.npy`; found by which files exist)
(`<S>_hodge_mu/_modes/_mass.npy`, `<S>_glued_V/_keep.npy`, `<S>_idt_F.npy`, `<S>_hodge.json`; the solve is on the two
hemispheres glued into one closed surface). It is stored like every eigenbasis (no schema change): the axis
`eigen/hodgeDirac` (μ > 0 in 1/m), its `eigen/hodgeDirac_kind` (0 exact · 2 coexact), the operator
`hodgeDirac_eigenvectors` [V × K] holding one vertex potential per mode (A = ψ/√2 or S = ⟨χ/√2, ω_v⟩; the 1-form is
(∇A + n × ∇S)/μ), and the Selection `hodgeDirac_support` (the vertices the glued surface keeps). The solved vertices must
sit at the primary cortex's positions (1 µm). `--replace` re-adds it.

0.2.6: `--surface NAME` adds the package of another surface under the same names with `__NAME` after `hodgeDirac`
(`eigen/hodgeDirac__cortex_pial`, `hodgeDirac__cortex_pial_eigenvectors` …); the primary keeps the plain names. The operator is
chunked [4096 × 64 modes] and written 64 modes at a time (peak about 0.8 GB at V = 300 000, K = 6000). When the solve's
`source` and the surface both carry `sha256_lh`/`sha256_rh` they must be equal.

0.2.7: the operator is chunked ONE MODE a chunk ([V × 1]) inside one shard per 64 modes, so reading modes i..j decodes
only those modes (a basis written by 0.2.6 is re-added with `--replace` to get it).

`qc` adds sensor QC to every recording of a subject: the PSD Field `frequency/<rec>_psd_preprocessed` (Welch, 4 s Hann,
50 % overlap, one-sided density, every channel, on channels × a 0.25 Hz frequency Line; `label` = `preprocessed`), and
measurements of it on the recording's channel `_names` selection — per MEG channel and per recording (the median over good
MEG channels): `line_residual_db`, `muscle_power`, `muscle_ratio_db`, `alpha_peak_hz`, `alpha_peak_power`,
`aperiodic_slope`, `channel_status`, `excluded_fraction` and `qc_flag` (0 pass · 1 check · 2 fail), read through the
dataset's `measure`, `qc flag` (which states the rule) and `channel status` dictionaries. `--megqc <sub>.tar.gz` adds the
meg-qc pipeline's session metrics as `megqc_*` measurements of the rest recordings. The `clean` stage (SSP, excluded time)
is not written yet: `excluded_fraction` is 0 and the PSD's provenance says `"excluded": "none marked"`.

`surface --fs-dir` writes FreeSurfer's `surf/lh.pial` + `rh.pial` (else `.pial.T1`) as ONE secondary surface (lh then rh) in
`freesurfer-tkr` metres, with the hemisphere components, the `sphere.reg` pair (metres), the `aparc` / `aparc.a2009s` /
`aparc.DKTatlas` annotations as the atlas set, and the sha256 of every file read in its provenance. `--scs-from` fits and stores
the tkr → SCS affine (`<name>_scs`) against a Brainstorm surface of the same vertices (refused above 1e-5 m rms).

`interp` writes `<DST>_from_<SRC>`, a csr operator: each DST vertex in barycentric weights of the SRC triangle holding it on
the registration spheres, within the same hemisphere; it reports row-sum error, minimum weight, share of exact gathers and the
round trip of a smooth field. 0.2.8: a surface stored without hemisphere components (a schema-48 store) is split into its
mesh's two connected components, left first, checked against the coordinate system's left (`scs` +y, `freesurfer-tkr` −x).

### A shared dataset written by parallel jobs

SQLite takes one writer, so jobs that each add to one subject of a large dataset (on a cluster) write a FRAGMENT and one job
merges them:

```bash
nxr-convert fragment --dataset <datastore>/<dataset> --subject <sub> --to <scratch>   # → <scratch>/<dataset>
nxr-convert hodge|surface|interp|qc|atlas build --dataset <scratch>/<dataset> --subject <sub> …   # arrays land in the real store
nxr-convert merge --dataset <datastore>/<dataset> <scratch1>/<dataset> <scratch2>/<dataset> …     # serially, one job
```

A fragment holds its subject's rows, the dataset's shared rows and (0.2.9) its template subjects' rows, read-only, and links
the subject's and the templates' real stores, so `atlas build` finds the `@default_subject`; `merge` never writes template rows back. `merge` replaces each
fragment's subject in the dataset's database by the fragment's copy, adds shared rows it lacks (a dictionary the dataset
already has under the same name is the dataset's; the fragment's references move to it), rewrites that subject's nodes'
`zarr.json` from the merged rows, and removes no folder.

`migrate --from <nxr.sqlite> --dataset <datastore>/<dataset>` brings a schema-48 database (one shared by many subjects,
written before the dataset layout) into the dataset folder as `dataset.sqlite` at schema 49: the `status` column added
(`complete`), every view, index and trigger recreated from the schema, then every node's `zarr.json` rewritten from its row
and the catalog written. The subject stores must already be in the folder; a node missing on disk is reported, not made.
48 → 49 is the only step it knows.

### The atlas (optional)

```bash
nxr-convert atlas default-subject <datastore>/<dataset> --templates <FreeSurfer subjects dir> […]
nxr-convert atlas build  <datastore>/<dataset> --subject S [--replace] [--kernels …] […]
nxr-convert atlas reduce <datastore>/<dataset> [--from <datastore>/<other dataset> …] [--kernels …]   # members of other datasets too
nxr-convert atlas info   <datastore>/<dataset>
```

`--kernels` chooses the inverse kernels, in `build` and in `reduce`: comma tokens, tokens of one kind OR and kinds AND —
`all` · `constrained` | `free` · `latest` (per recording, method and orientation, the newest Brainstorm stamp) · a kernel's
method (`MN_MEG`), stamp (`261005_2149`), name, or a glob over its name. `dSPM-unscaled_MEG,constrained,latest` is one kernel
a recording. The group atlas sums one kernel per member recording, method and orientation: a member left with two (two
inverse solutions of one method) is refused, never pooled. Stamps are per subject (when Brainstorm made the kernel), so
across subjects choose by `latest` or a glob, not by one stamp.

Run `nxr-convert <command> --help` for every flag.

## One writer per dataset

Each command that writes a dataset first takes its **writer lock**, `<datastore>/<dataset>/.writer.lock`. The lock is a
small JSON file (`pid`, `host`, `program`, `started_utc`) created atomically. If another live process holds the lock
(the desktop app, or a second conversion), the command is refused, and the error names the holder. A lock left by a
process that has died on the same machine is taken over. A lock held by another host is never broken; remove it by hand
if that host is gone. When the desktop app runs the converter during its own import, it already holds the lock and
passes it on through `NXR_WRITER_LOCK_HELD`. The lock is advisory, so do not edit a dataset by other means while a
conversion runs.

## Development

```bash
uv sync
uv run pytest -q
```

Tests that need a real Brainstorm protocol skip unless `NXR_TEST_PROTOCOL` points at one (Brainstorm's
`TutorialAuditory`).

This repository is a published snapshot. The converter is developed inside the Cortical Flow monorepo, and its
database schema (`src/nxr_convert/model.sql`) is a copy of the app's schema.

## License

MIT. See [LICENSE](LICENSE).
