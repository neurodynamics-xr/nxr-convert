# nxr-convert

Converts a [Brainstorm](https://neuroimage.usc.edu/brainstorm/) protocol (MEG/EEG recordings, cortical surfaces, MRI
volumes, head models, inverse kernels, source maps, fibres and connectomes) into an **nxr datastore**: a folder that the
[Cortical Flow](https://corticalflow.app) desktop app opens.

Version **0.2.0**, which writes **database schema 49**. Each dataset's `dataset.sqlite` carries the schema number as
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
nxr-convert --version        # nxr-convert 0.2.0 (database schema 49)
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
nxr-convert scan <protocol> [--prefer protocol.mat|sqlite|walk] [--out view.json] [--summary]
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

### Adding to or removing from a subject

```bash
nxr-convert surface --dataset D --subject S --file tess_*.mat [tess_*.mat …] [--bst-root <protocol>]
nxr-convert mri     --dataset D --subject S --anat <protocol>/anat/<subject> [--only NAME …]
nxr-convert inverse --dataset D --subject S --file results_*KERNEL*.mat --session C --bst-root <protocol> \
    [--channels-id ID] [--surface-id ID] [--forward-id ID] [--name N] [--channel-file F]
nxr-convert remove  --dataset D --subject S [--node <store path>]   # the subject, or one node of it
```

### The atlas (optional)

```bash
nxr-convert atlas default-subject <datastore>/<dataset> --templates <FreeSurfer subjects dir> […]
nxr-convert atlas build  <datastore>/<dataset> --subject S [--replace] […]
nxr-convert atlas reduce <datastore>/<dataset>
nxr-convert atlas info   <datastore>/<dataset>
```

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
