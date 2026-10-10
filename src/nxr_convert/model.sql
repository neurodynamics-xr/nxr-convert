-- backend/schema/model.sql — THE DATA MODEL'S SCHEMA (DDL), the one source for TypeScript
-- (`backend/src/db/schema.ts`, through `schema.sql.ts` — generated from this file at build by
-- `backend/scripts/schema-module.mjs`, held equal by `schema.test.ts`) and Python (`nxr_convert/db.py`).
-- Register D45/O24. Run it once, inside a transaction, on an empty database; the caller sets
-- `PRAGMA foreign_keys = ON` on every connection (a pragma inside a transaction is a no-op).
--
-- THE DATA MODEL'S SCHEMA — one schema, one version, for the whole of nxr-cortical-flow.
-- 
-- THIS IS THE SCHEMA THE ATTRIBUTES SUBSCRIBE TO (register D17). The SQL tables and the
-- attributes written into every node's `zarr.json` are ONE AND THE SAME schema: a row's
-- columns are the node's attributes, and the key-value entries in `zarr.json` are what the
-- builder aggregates into rows. There is no separate per-kind on-disk contract with its own
-- version (`nxr.catalog@1.0`, `nxr.dataset@1.0`, `nxr.subject@5.0`, … are RETIRED): a
-- datastore is at ONE version, entirely — a subject at one version inside a dataset at
-- another is exactly what breaks applications — and a migration moves the whole datastore.
-- The version is `PRAGMA user_version`, set by the last statement of this file; a database
-- at another version is refused and migrated whole.
-- 
-- The database REPLACES the manifest (register D2): every node keeps its `zarr.json`, and
-- what used to be aggregated into `.zmetadata` is aggregated here instead. It is built with
-- the model, not with the app — nothing in `app/` reads it yet. (A phase-1 `.index.sqlite`
-- per subject store, the manifest's columns verbatim, was built and retired on 2026-09-22:
-- what it held was not the schema, and a file inside the store is not where the database
-- lives — register D12.)
-- 
-- THE COLUMNS ARE THE CONTRACT. A key that matters is a typed column; there is no catch-all
-- "attributes" column, and no key lives only in `zarr.json` — the writers write the columns.
-- 
-- PROVENANCE ON EVERY NODE TABLE (register D19): `created_utc` · `created_by` · `modified_utc`
-- · `modified_by`. Attributes of the NODE, written by whoever creates or changes it — never
-- assigned by this database, which is rebuilt whole and could only say when it was built
-- (that is `meta.built_utc`). ISO 8601 UTC text, which sorts as a string.
-- 
-- NAMING (register D20) — a column's name says its type and role:
--   tables        singular nouns, snake_case            subject, subject_variable
--   primary key   id                                    a UUID (D16)
--   foreign key   <table>_id                            dataset_id, subject_id
--   boolean       is_* / has_*, INTEGER 0|1 + CHECK     is_remote, has_events
--   timestamp     *_utc, ISO 8601 UTC text              created_utc
--   JSON          *_json                                levels_json
--   count         n_*                                   n_channels
--   quantity      *_<unit> where a unit applies         sfreq_hz, duration_s, size_bytes
--   enumeration   a plain noun + CHECK on the values    driver, type
--   location      url for an address, path for a relative location, name for a label
-- 
-- ONE DDL, EVERY ENGINE. `node:sqlite` (the builder, main), SQLite-wasm (the renderer) and
-- Python's `sqlite3` (the converter) all run this text verbatim; a parity test holds the
-- Python copy to it. Tables are STRICT so a value of the wrong type is refused, not coerced.
-- 
-- Levels, top down (Diellor, 2026-09-22):
-- 
--   datastore   the thing the app opens; holds datasets; one at a time. ONE DATASTORE IS
--               ONE LOCATION — a copy elsewhere is another datastore. What the row records
--               is the DRIVER the root lives on (file · http · s3 · …), TensorStore-style,
--               so the app knows what kind of storage it is working from (D13)
--   dataset     one converted protocol, holding subjects
--   subject     one `<dataset>/<subject>.nxr.zarr` store — FIRST CUT from the fixture's root
--               attributes, for Diellor's feedback (register D18)
--   variable_dictionary · subject_variable   the cohort's variables (D15): what is measured, and
--               each subject's values; empty until a writer emits them
--   space       an ANALYTIC MANIFOLD — a parametric geometry (a line, a volume, a circle, a
--               sphere, a torus) with its parameters and no discretisation — on which
--               manifolds are defined and placed and analytic fields (a Gaussian, a Mexican
--               hat, g(λ)) are written (register D26/D27). Its parametrisation gives it a
--               chart for free; it is the three.js scene's world space
--   coordinate_system   a coordinate map on a space (a CHART, in the mathematical sense —
--               the word is avoided because the app has charts on screen) — SCS · NCS · voxel
--               indices on head space; linear · log on the frequency line; eigenvalue ·
--               wavelength on a spectral line — with a unit and a TRANSITION MAP from the
--               space's reference system (register D25/D28). Positions are always numbers IN
--               A COORDINATE SYSTEM, so a manifold names one
--   topology    the NETWORK a manifold's connections form — its structural backbone: how the
--               vertices are connected (kind · rank · orientation · closed · orientable);
--               shared by reference (the sphere and the folded cortex are one topology) (D23/D28/D29)
--   topology_cell   a topology's STORED cells of one rank ≥ 1 — the cell → vertex array (a
--               mesh's faces, a graph's edges), which is also that rank's boundary map (D92)
--   geometry    the METRIC — everything that constructs the geometry of a manifold: in CLOSED
--               FORM (its `geometry_dimension` rows: count · spacing · origin per dimension, in a
--               coordinate system) or STORED (a positions array, in a coordinate system, with the
--               metric induced). The two forms are treated the same way (register D29)
--   manifold    THE CELL COMPLEX — a topology (how vertices connect) + a geometry (the metric),
--               with its f-vector (n_vertices · n_edges · n_faces · n_volumes) as PRIMARY shape
--               data, owned by a subject. The surface / operators the app opens by default are
--               `is_primary` rows (D21, D22, D28, D29)
--   geometry_dimension   a closed-form geometry, one row per DIMENSION: count · spacing · origin
--   field         THE ARRAY: values on a manifold — the recording, the T1, an operator's
--               eigenvalues (register D39)
--   operator      THE MAPPING: a stored matrix from one manifold to another — the kernel, the
--               leadfield, eigenvectors, mass, a transition map (D40)
--   dictionary · dictionary_entry   ONE table of dictionaries: every labelling's
--               codes translate through it — a code's name, colour, a measurement, a function.
--               An atlas's levels (app-wide defaults), a montage family's, a quality flag's, a
--               paging ladder's, the user's own labels (register D32/D34)
--   selection   THE INDEXING (root CLAUDE.md §1): which part of a manifold is in play — a
--               point · a span · a set of spans · a labelling (elements + an integer code the
--               dictionary translates) · picked indices · a kernel · a radial distance — or a SET
--               whose members are Selections (D32)
--   selection_element · selection_span   a labelling's per-element codes / a spans set's members
--   field · selection · operator   NEXT
-- 
-- APP STATE IS NOT HERE. Preferences, window and panel state, last-opened things belong to
-- the USER on a MACHINE, not to the data: this database is derived from a datastore that
-- may be read-only (R2) or shared (a network drive), so it can hold nothing of the user's.
-- App state lives in the app's own store under userData (decision register D11). That
-- includes which datastore was opened, and when.
-- 
-- ONE DATABASE PER DATASTORE, LOCATED BY THE APP (register D12). The database describes one
-- datastore. Where the database itself lives is the app's to know — a file anywhere, or
-- later a server answering the same SQL — and nothing here assumes a file inside the store.
-- The `datastore` row says where the ROOT is and on what driver; datasets are relative to
-- the root; everything below is addressed by the model's datastore-relative ids.
-- 
-- KEYS — EVERY TABLE'S PRIMARY KEY IS A UUID (register D16). Minted once at first write
-- (`uuid.ts`, version 7), stored in the node's `zarr.json` as `id`, carried by every copy;
-- opaque, robust, intended for databasing. NEVER derived from a path, a name or content: the
-- v5 rule `nxr:/<path>` is replaced, and `name` is the human-readable, user-facing label —
-- not an identifier. A `path` is where a node sits, unique within its parent, free to change.
-- (An extra column could one day identify a COPY of a dataset located elsewhere; that is
-- not the primary key's job.)

-- ── meta ────────────────────────────────────────────────────────────────────
-- The database about itself: schema_version (THE data-model version, D17), built_utc,
-- the builder's name.
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;

-- ── datastore ───────────────────────────────────────────────────────────────
-- The thing the app opens: a root group (`<root>/zarr.json`) holding datasets. ONE DATASTORE IS ONE LOCATION: `url` is where it is, unique. `driver`
-- is the kind of storage that location is on, in TensorStore's sense (a kvstore driver):
-- the app reads it to know how the root is reached and whether it can be written.
--   file    an absolute path on a local filesystem            (Electron; writable)
--   http    an http(s) base URL — R2 behind data.corticalflow.app (the web build; read-only)
--   s3, gcs… a bucket and prefix                                (later)
--   memory  a store that exists only for a test
CREATE TABLE datastore (
  id         TEXT PRIMARY KEY,           -- UUID, in the root's zarr.json
  url        TEXT NOT NULL UNIQUE,       -- the root as the app addresses it: a path, or a URL
  driver     TEXT NOT NULL               -- 'file' · 'http' · 's3' · 'gcs' · 'memory' · …
               CHECK (driver <> ''),
  name       TEXT NOT NULL,              -- the user-facing label; the root folder's name unless stored
  created_utc  TEXT,                     -- when the root was written; ISO 8601 UTC
  created_by   TEXT,                     -- by which tool
  modified_utc TEXT,                     -- last change to the root's attributes, by any writer
  modified_by  TEXT
) STRICT;

-- ── dataset ─────────────────────────────────────────────────────────────────
-- One converted protocol: `<root>/<dataset>/zarr.json`, holding subjects. Addressed RELATIVE to the root. A cohort's VARIABLES (age, sex, group…)
-- are not a column here: they are a `variable` dictionary per dataset and a
-- `subject_variable` value per subject — built with the subject table (register D15).
CREATE TABLE dataset (
  id              TEXT PRIMARY KEY,      -- UUID, in the dataset's zarr.json
  datastore_id    TEXT NOT NULL REFERENCES datastore (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- the user-facing label: the `dataset` attribute, else the folder name
  path            TEXT NOT NULL,         -- root-relative (today = the folder name)
  source_tool     TEXT,                  -- source.tool      e.g. 'brainstorm'
  source_protocol TEXT,                  -- source.protocol  the protocol's name in that tool
  source_path     TEXT,                  -- source.path      a HINT, never truth (drives move)
  created_utc     TEXT,
  created_by      TEXT,
  modified_utc    TEXT,
  modified_by     TEXT,
  UNIQUE (datastore_id, path)
) STRICT;
CREATE INDEX dataset_datastore ON dataset (datastore_id);

-- ── subject ─────────────────────────────────────────────────────────────────
-- One subject store: `<root>/<dataset>/<subject>.nxr.zarr/zarr.json`. The subject's anatomy,
-- surfaces, forwards and inverses are NOT here: they are `manifold` and `operator` rows
-- carrying `subject_id` (D21).
-- A DATASET IS A COLLECTION OF SUBJECTS, and what accumulates OVER subjects lives in
-- PSEUDO-SUBJECTS of the same dataset (register D55) — exactly Brainstorm's shape:
--   kind  subject    a person's data
--         template   the default anatomy (`anat/@default_subject`: the template T1, cortex,
--                    atlases) — the manifolds group results are on, and the anatomy a
--                    subject INHERITS (`template_id`) instead of copying
--         group      group averages and statistics (`Group_analysis`, `data/@inter`):
--                    Fields whose contributions (`field_contribution`) are the subjects'
-- No node table changes for this: a group average is an ordinary Field, on a manifold the
-- group or the template owns (a manifold_id may point into another subject — rows are
-- global; a registration Operator maps a subject's cortex to the template's).
CREATE TABLE subject (
  id              TEXT PRIMARY KEY,      -- UUID, in the store's zarr.json
  status          TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'deleting')),  -- D141: readers see only 'complete'
  dataset_id      TEXT NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- the user-facing label: the `subject` attribute ('sub-0002')
  path            TEXT NOT NULL,         -- dataset-relative: the store folder ('sub-0002.nxr.zarr')
  kind            TEXT NOT NULL DEFAULT 'subject' CHECK (kind IN ('subject', 'template', 'group')),
  template_id     TEXT REFERENCES subject (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,  -- the template whose anatomy this subject uses
  source_format   TEXT,                  -- sourceFormat   'brainstorm'
  source_path     TEXT,                  -- sourcePath     the subject's folder in that tool — a HINT
  created_utc     TEXT,                  -- when the store was written (the converter writes these two)
  created_by      TEXT,
  modified_utc    TEXT,                  -- last write to the store by any writer (extend, Prepare, the app)
  modified_by     TEXT,
  UNIQUE (dataset_id, path),
  CHECK (template_id IS NULL OR kind = 'subject')
) STRICT;
CREATE INDEX subject_dataset ON subject (dataset_id, kind);

-- ── topology ────────────────────────────────────────────────────────────────
-- The NETWORK a manifold's connections form — its structural backbone (register D23/D28/D29):
-- HOW the vertices are connected, and nothing else. The manifold is the cell complex (a
-- connected set of geometric elements); this row is the graph those elements' connections
-- make; the counts of what it connects are the manifold's; the metric is the geometry's.
-- Shared by reference where the arrays are bulk: the folded cortex and its registration
-- sphere are ONE topology.
--   kind     none (isolated vertices) · lattice (a regular grid — connectivity implied by the
--            geometry's dimension rows) · path · circle · simplicial (a mesh of triangles) ·
--            graph (explicit edges)
--   rank     the top cell's dimension: 0 points · 1 line · 2 surface · 3 volume
--   n_components  how many connected pieces the network has (the zeroth Betti number): a
--            manifold need not be connected — the cortex is ONE manifold with TWO components,
--            each a manifold in its own right (register D30)
--   closed   its boundary is empty (a sphere, the cortex, a circle) — 1; it has one (a line, a
--            patch, a lattice) — 0; null when not yet measured
--   orientable  whether its top cells can be wound consistently; null when not yet measured
-- Its CELLS of rank ≥ 1 that are STORED — a mesh's faces, a graph's edges, a tet mesh's tets —
-- are `topology_cell` rows; a lattice's, a path's and a circle's are implicit (indices) and have
-- none. The COUNTS of every rank are the manifold's f-vector: a product and a union have counts
-- but no cells of their own. Rank 0 is the manifold's own vertices (D92).
CREATE TABLE topology (
  id           TEXT PRIMARY KEY,         -- UUID
  subject_id   TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  kind         TEXT NOT NULL CHECK (kind IN ('none', 'lattice', 'path', 'circle', 'simplicial', 'graph', 'product')),  -- product: of its factors' topologies, when they do not reduce to one of the others (a surface × a line)
  rank         INTEGER NOT NULL CHECK (rank BETWEEN 0 AND 3),
  n_components INTEGER NOT NULL DEFAULT 1 CHECK (n_components >= 1),
  winding      TEXT,                     -- how the top cells are wound ('ccw-outward'), recorded by the producer
  closed       INTEGER CHECK (closed IN (0, 1)),
  orientable   INTEGER CHECK (orientable IN (0, 1)),
  CHECK ((kind = 'none') = (rank = 0))
) STRICT;
CREATE INDEX topology_subject ON topology (subject_id, kind);

-- ── topology_cell ───────────────────────────────────────────────────────────
-- A topology's STORED cells of one rank (register D92): the cell → vertex array — edges [E×2],
-- faces [F×3], tets [T×4] — at a store-relative path. It is also the BOUNDARY MAP of that rank
-- read with signs (a face's winding orients it), so the exterior derivative d = ∂ᵀ needs no
-- bytes of its own. A mesh's edges, when not stored, are derived from its faces at load. A
-- component's path may be its union's (sliced by `from_index`) until the writers store each
-- component's own. Its count is the manifold's f-vector entry for the rank.
CREATE TABLE topology_cell (
  topology_id TEXT NOT NULL REFERENCES topology (id) ON DELETE CASCADE,
  rank        INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 3),
  path        TEXT NOT NULL,             -- 'sources/cortex_pial_low/faces'
  PRIMARY KEY (topology_id, rank)
) STRICT;

-- ── space ───────────────────────────────────────────────────────────────────
-- An ANALYTIC MANIFOLD (register D26/D27): a parametric GEOMETRY — a line, a plane, a volume,
-- a circle, a sphere, a torus — with the geometry's own parameters (a radius) and NO
-- discretisation. It is what a lattice SAMPLES (the parameter domain, regularly), what a mesh
-- is PLACED in, and what an analytic field (a Gaussian, a Mexican hat, g(λ)) is written on.
-- Its parametrisation gives it a coordinate system for free — the parameter domain: the
-- identity on Rⁿ, (θ, φ) on a sphere — and every other system is a re-labelling of the same
-- points. Mathematically many rows are copies of one geometry ('head' and 'sphere' are two
-- copies of R³): a row anchors COMPARABILITY — two systems of one space are related by a
-- transition map, two spaces are not, which is why two copies are two rows. A space stores no
-- coordinates: positions live on the manifolds placed in it, as numbers in one of its systems.
-- Getting from one space into another (a sphere into R³, voxels into head space) is an
-- OPERATOR. Per subject today; some (a spectral line, a template space) will be shared once
-- templates are settled.
CREATE TABLE space (
  id              TEXT PRIMARY KEY,      -- UUID
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- which copy: 'head' · 'sphere' · 'time' · 'frequency' · 'spectrum:laplaceBeltrami'
  geometry        TEXT NOT NULL CHECK (geometry IN ('line', 'circle', 'plane', 'sphere', 'torus', 'volume')),
  rank            INTEGER NOT NULL,      -- the geometry's dimension, and the extrinsic dimension of everything placed in it
  parameters_json TEXT,                  -- the geometry's own parameters: {radius} for a circle or sphere, {major, minor} for a torus; NULL for Rⁿ
  description     TEXT,
  UNIQUE (subject_id, name),
  CHECK (rank = CASE geometry WHEN 'line' THEN 1 WHEN 'circle' THEN 1 WHEN 'plane' THEN 2 WHEN 'sphere' THEN 2 WHEN 'torus' THEN 2 WHEN 'volume' THEN 3 END)
) STRICT;

-- ── coordinate_system ───────────────────────────────────────────────────────
-- A COORDINATE MAP on a space (register D25/D28) — a CHART in the mathematical sense; the
-- table is not called that because the app has charts on screen. The rule that turns the
-- space's points into numbers, with a unit. A space has several — SCS, NCS and the T1's voxel
-- indices on head space; linear and log on the frequency line; eigenvalue and wavelength on a
-- spectral line; the renderer's on the scene — and every manifold's positions are numbers IN
-- ONE OF THEM. One per space is the REFERENCE; the others are defined by a TRANSITION MAP:
--   identity · affine (rotation · scale · translation: what T1_scs is) · log (a base) ·
--   power (a scale and an exponent: wavelength = 2π / √λ) · operator (anything else)
-- A transition map IS an operator (an affine matrix is a linear map). Until the operator table
-- exists its parameters sit here in `transition_json`; they move to `transition_operator_id`
-- then, so they have one home. A system that is not an isometry (log, power) presents the
-- space in other numbers; it does not change any manifold's metric.
CREATE TABLE coordinate_system (
  id              TEXT PRIMARY KEY,      -- UUID
  space_id        TEXT NOT NULL REFERENCES space (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- 'scs' · 'ncs' · 'voxel:T1_volume' · 'clock' · 'linear' · 'log10' · 'eigenvalue' · 'wavelength' · 'scene'
  unit            TEXT,                  -- of this system's numbers: 'm' · 'mm' · 'voxel' · 's' · 'Hz'
  is_reference    INTEGER NOT NULL DEFAULT 0 CHECK (is_reference IN (0, 1)),
  transition      TEXT NOT NULL DEFAULT 'identity'
                    CHECK (transition IN ('identity', 'affine', 'log', 'power', 'operator')),
  transition_json TEXT,                  -- the map's parameters FROM the reference system: {matrix, offset} · {base} · {scale, exponent}
  description     TEXT,
  UNIQUE (space_id, name),
  CHECK (is_reference = 0 OR transition = 'identity')
) STRICT;
CREATE UNIQUE INDEX coordinate_system_reference ON coordinate_system (space_id) WHERE is_reference = 1;

-- ── geometry ────────────────────────────────────────────────────────────────
-- THE METRIC — everything that CONSTRUCTS the geometry of a manifold (register D29). Two
-- forms, treated the same way: each is what the app reads to build the geometry, and the
-- database knows the manifold's shape counts without reading either.
--   closed_form  the sampling of the space's parameter domain, regular: one `geometry_dimension`
--                row per dimension (count · spacing · origin · unit); positions are GENERATED
--                (origin + index ⊙ spacing, then the space's parametrisation — the identity on
--                Rⁿ, the sphere's equation on a sphere); the metric is the spacing on a Euclidean
--                space, induced through the parametrisation on a curved one
--   stored       a positions array (`positions_path`), one position per vertex; the metric is
--                INDUCED from the positions
-- Either way the numbers are in a COORDINATE SYSTEM of a space — `coordinate_system_id` — whose
-- unit they carry and whose space's rank is the extrinsic dimension. A lattice is placed in its
-- OWN index system (the T1 in 'voxel:T1_volume'), whose transition to the reference system is
-- the affine (T1_scs).
CREATE TABLE geometry (
  id                   TEXT PRIMARY KEY,  -- UUID
  subject_id           TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  form                 TEXT NOT NULL CHECK (form IN ('closed_form', 'stored', 'product')),
  coordinate_system_id TEXT REFERENCES coordinate_system (id) ON DELETE CASCADE,  -- null for a PRODUCT: each factor carries its own (D37)
  positions_path       TEXT,              -- form = stored: 'sources/cortex_pial_low/vertices', 'timeseries/channel_ctf_acc1'
  direction_json       TEXT,              -- form = closed_form: the dimensions' orientation in parameter space, row-major, when not identity
  CHECK ((form = 'stored') = (positions_path IS NOT NULL)),
  CHECK (form = 'closed_form' OR direction_json IS NULL),
  CHECK ((form = 'product') = (coordinate_system_id IS NULL))
) STRICT;
CREATE INDEX geometry_subject ON geometry (subject_id, form);
CREATE INDEX geometry_coordinate_system ON geometry (coordinate_system_id);

-- ── geometry_dimension ──────────────────────────────────────────────────────
-- One row per dimension in the array's order (register D28/D29/D37). A CLOSED-FORM
-- dimension is written inline: how many vertices along it, the spacing between neighbours
-- (the smallest discretisation — 1 / spacing along time is the sampling rate), and where the
-- first one sits, in the geometry's coordinate system — a time axis is one row; a volume is
-- three; a sampled sphere's (θ, φ) lattice would be two, in parameter units. An inline row IS
-- a Line manifold in closed form: a volume is the product of three of them. A PRODUCT
-- dimension is a reference instead: `manifold_id` names the FACTOR that is this dimension
-- (the recording's manifold is Sensors × Time — dimension 0 → the sensors, 1 → the time
-- line), and the product's topology and f-vector follow from the factors (D37). A stored
-- geometry has no rows.
CREATE TABLE geometry_dimension (
  geometry_id TEXT NOT NULL REFERENCES geometry (id) ON DELETE CASCADE,
  ordinal     INTEGER NOT NULL,          -- 0-based position among the array's dimensions
  name        TEXT,                      -- 'time', 'frequency'; 'i' · 'j' · 'k' for voxel dimensions; 'theta' · 'phi'
  n_vertices  INTEGER NOT NULL,          -- vertices along this dimension (a factor's, for a product)
  spacing     REAL,                      -- closed form: between neighbouring vertices, in parameter units
  origin      REAL,                      -- closed form: the first vertex's coordinate; null when the producer wrote none
  unit        TEXT,                      -- closed form: 's', 'Hz', 'mm', 'rad'
  manifold_id TEXT REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- product: the FACTOR this dimension is
  PRIMARY KEY (geometry_id, ordinal),
  CHECK ((manifold_id IS NULL) = (spacing IS NOT NULL)),
  CHECK (manifold_id IS NULL OR (origin IS NULL AND unit IS NULL))
) STRICT;
CREATE INDEX geometry_dimension_factor ON geometry_dimension (manifold_id);
-- a product cannot outlive a factor: losing a factor's dimension row deletes the product's
-- geometry, and with it (UNIQUE geometry_id, ON DELETE CASCADE) the product manifold
CREATE TRIGGER geometry_dimension_factor_gone AFTER DELETE ON geometry_dimension WHEN OLD.manifold_id IS NOT NULL
BEGIN
  DELETE FROM geometry WHERE id = OLD.geometry_id;
END;

-- ── manifold ────────────────────────────────────────────────────────────────
-- THE CELL COMPLEX (register D28/D29): a connected set of geometric elements, owned by a
-- subject — its TOPOLOGY (how the vertices are connected) + its GEOMETRY (the metric, in a
-- coordinate system of a space) + its SHAPE COUNTS. The counts are PRIMARY data: the f-vector
-- (n_vertices · n_edges · n_faces · n_volumes, one per cell rank — a face is a 2-cell, a
-- volume a 3-cell; a field on a cell kind has that many values; the alternating sum is the
-- Euler characteristic) is known here regardless of what constructs the geometry, because
-- labelling, tiling and querying are computed from it. A manifold with no geometry (a graph
-- without a metric) has a null `geometry_id`. `type` is the model's vocabulary word for a
-- combination the columns spell out, kept because the app speaks it. Aliases: lattice ≡ grid;
-- simplicial ≡ mesh ≡ surface; path ≡ line topology.
-- COMPONENTS (register D30): a manifold need not be connected. The cortex is one manifold
-- with two components, and each hemisphere is a manifold in its own right — its own
-- topology, geometry and operators — that names the union it is a component of
-- (`component_of_id`), in which order (`component_ordinal`), and where its vertices start in
-- the union's ordering (`from_index`). A field on the union is the concatenation of fields on
-- its components in ordinal order. A subset of vertices that is NOT a component is a
-- Selection, never a manifold.
CREATE TABLE manifold (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
  status          TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'deleting')),  -- D141: readers see only 'complete'
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- the node's folder name (Brainstorm's file stem); a product's is its factors' joined by ' × '
  path            TEXT,                  -- store-relative: 'sources/cortex_pial_low', 'timeseries/<rec>_time'; NULL for a manifold no node stores — a product (D37)
  type            TEXT NOT NULL CHECK (type IN ('points', 'line', 'image', 'volume', 'curve', 'surface', 'tetrahedral', 'graph', 'product')),
  label           TEXT,                  -- what the manifold IS, for people: a component's 'left' / 'right' (from the Structures parcellation), INHERITED by an eigen axis from the hemisphere it derives from (D31)
  session         TEXT,                  -- the acquisition condition, on the nodes that share one (v5)
  is_primary      INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),  -- the one the app opens by default
  -- shape: the f-vector
  n_vertices      INTEGER NOT NULL,
  n_edges         INTEGER,               -- 1-cells; null when unknown without reading the arrays
  n_faces         INTEGER,               -- 2-cells: the triangles of a mesh, the rectangles of a lattice
  n_volumes       INTEGER,               -- 3-cells: the boxes of a rank-3 lattice
  -- structure
  topology_id     TEXT NOT NULL REFERENCES topology (id) ON DELETE CASCADE,
  geometry_id     TEXT REFERENCES geometry (id) ON DELETE CASCADE,   -- null: no metric, no place
  -- components
  component_of_id   TEXT REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- the disconnected union this is a connected component of
  component_ordinal INTEGER,             -- its order among the union's components
  from_index        INTEGER,             -- where its vertices start in the union's ordering
  -- derivation
  derived_from_id TEXT REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- EIGEN: the manifold whose operator this diagonalises; REGIONS: the surface partitioned
  partition_id    TEXT REFERENCES selection (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- REGIONS (D47/D55): a coarse points manifold whose vertices are the codes of this partition, in code order — averages over cortical areas are Fields on it
  operator        TEXT,                  -- EIGEN: which operator — 'laplaceBeltrami', 'relativeDirac'
  -- provenance
  comment         TEXT,                  -- the producer's own comment ('cortex_20484V')
  source          TEXT,                  -- the source file / object inside the producer's tool
  producer_json   TEXT,                  -- the rest of what the producer recorded, free-form
  created_utc     TEXT,
  created_by      TEXT,
  modified_utc    TEXT,
  modified_by     TEXT,
  UNIQUE (subject_id, path),
  UNIQUE (geometry_id),                  -- a geometry constructs ONE manifold
  UNIQUE (component_of_id, component_ordinal),
  CHECK ((component_of_id IS NULL) = (component_ordinal IS NULL)),
  CHECK (component_of_id IS NOT NULL OR from_index IS NULL)
) STRICT;
CREATE INDEX manifold_subject ON manifold (subject_id, type);
CREATE INDEX manifold_session ON manifold (session);
CREATE INDEX manifold_topology ON manifold (topology_id);
CREATE UNIQUE INDEX manifold_primary ON manifold (subject_id, type) WHERE is_primary = 1;

-- ── dictionary · dictionary_entry ───────────────────────────────
-- ONE table of DICTIONARIES (register D32/D34): the entries of every dictionary live in
-- `dictionary_entry`, keyed by which dictionary and which code; `dictionary` is
-- the header that gives a dictionary an id, a name and a SCOPE, so a labelling references it
-- by key. A labelling assigns an integer CODE to each element; the dictionary says what the
-- code means: a name, a colour, a MEASUREMENT (a threshold, a bucket width), a FUNCTION (an
-- aggregation, a kernel), and whatever else. The same structure serves an atlas's levels
-- (written as APP-wide defaults so they can be used throughout), a montage family's, a
-- quality flag's, a paging ladder's levels, and the labels a user makes to mark data.
--   scope   app (a default everywhere) · dataset · subject · user
CREATE TABLE dictionary (
  id          TEXT PRIMARY KEY,          -- UUID
  name        TEXT NOT NULL,             -- 'Desikan-Killiany' · 'channel type' · 'quality' · 'envelope ladder'
  scope       TEXT NOT NULL CHECK (scope IN ('app', 'dataset', 'subject', 'user')),
  dataset_id  TEXT REFERENCES dataset (id) ON DELETE CASCADE,     -- when scope = dataset
  subject_id  TEXT REFERENCES subject (id) ON DELETE CASCADE,     -- when scope = subject
  description TEXT,
  created_utc TEXT, created_by TEXT, modified_utc TEXT, modified_by TEXT,
  CHECK ((scope = 'dataset') = (dataset_id IS NOT NULL)),
  CHECK ((scope = 'subject') = (subject_id IS NOT NULL))
) STRICT;
CREATE UNIQUE INDEX dictionary_name ON dictionary (scope, coalesce(dataset_id, ''), coalesce(subject_id, ''), name);
CREATE TABLE dictionary_entry (
  dictionary_id        TEXT NOT NULL REFERENCES dictionary (id) ON DELETE CASCADE,
  code             INTEGER NOT NULL,     -- the integer a labelling assigns to an element
  name             TEXT NOT NULL,        -- what it means: 'bankssts L', 'MEG', 'good', 'bucket 32'
  color_json       TEXT,                 -- [r, g, b] 0–255, when the level has a colour
  measurement      REAL,                 -- a number the level stands for: a bucket width, a threshold, a band edge
  measurement_unit TEXT,
  function         TEXT,                 -- a function the level names: an aggregation ('max'), a kernel ('heat')
  params_json      TEXT,                 -- the function's parameters
  attributes_json  TEXT,                 -- anything else a code is given
  ordinal          INTEGER,              -- display order
  PRIMARY KEY (dictionary_id, code)
) STRICT;

-- ── selection ───────────────────────────────────────────────────────────────
-- THE INDEXING (register D32): which part of a manifold is in play. One table, `type` says
-- the kind, the type-specific columns are nullable:
--   point          a rule: `center` (an element) — or, ON A SPACE, `position` (below)
--   span           RECTANGULAR: its CORNERS — a start and a stop on each constrained axis of
--                  the coordinate system, rows in `selection_extent` (D35/D36). One row on a
--                  line is an interval; on a volume one row is a slab, three a box. Its RANK is
--                  its row count; the manifold's rank comes from the join
--   distance       CIRCULAR / SPHERICAL: everything within `radius` (in `unit`) of `center`
--                  under the manifold's METRIC — an interval on a line, a disc on a surface,
--                  a ball in a volume (D35). Its rank is the manifold's. On a surface it is
--                  the level set at `radius` of the heat distance solve from a delta at
--                  `center`; nothing of the evaluation is stored
--   spans          A TESSELLATION of a line: contiguous intervals covering it, in either of two
--                  forms, and they are the same kind (D87) --
--                    ROWS       its members in `selection_span` (events)
--                    CLOSED FORM `params_json.tiling = {origin, width, hop, count}` and no rows:
--                                a constant-Q LEVEL, where locating tile k is an O(1) multiply
--                                rather than a scan (D80); the chunk grid takes its form from
--                                the Field's stored chunk shape (D59)
--                  A member's ORDINAL IS ITS CODE -- `at(k)` gives id `${set}#${k}` (D57) -- so a
--                  tessellation needs no array to be coded, and `dictionary_id` names those codes
--                  when they mean something (a level's tile dictionary, D83).
--   indices        a subset: the picked elements (`selection_element` with a null code)
--   kernel         a rule: `weight_fn` + `params_json` at the values in use
--   set            a GROUP of Selections. Either its members are ROWS (`member_of_id` +
--                  `member_ordinal`, written by the member as `memberOf` + `ordinal`: the
--                  montage families, the parcellations of a surface, a set of ROIs), or it is a
--                  PARTITION stored as one array — an integer code per element (`array_path` +
--                  `dictionary_id`; the codes as `selection_element` rows when the manifold is
--                  points or a surface, the array alone for a lattice image or volume, D32):
--                  a parcellation, the channel names, a recording's flags. Its members are
--                  "the elements whose code is k", derivable, never rows (D48).
-- EVERY selection is a set of CELLS of one kind — vertices by default (`cell`, D92); two things
-- may be said OF one (D48):
--   `code`         an integer that groups it — an event's type, a chunk's index, a tile's label,
--                  an ROI's class — read with `dictionary_id` (its own, or its set's);
--   `selection_measurement`   values measured on it — a tile's band power, a chunk's max.
-- The `labels` type is gone: a labelling is a partition set, and a code is on any selection.
-- A Selection may belong to a Field (`of_field_id`: the recording's own channel flags, its chunk
-- grid). The app writes a Selection to the database and the store together (D33).
-- A Selection is ON a manifold OR ON a SPACE, never both (D95). On a manifold it is a chain of its
-- cells (D94); on a space — an analytic manifold, continuous, with no cells — it is a POSITION:
-- a `point` whose `position` is a number in one of the space's coordinate systems
-- (`coordinate_system_id`). The time cursor is one: t seconds in `clock` on the subject's `time`
-- space, which every recording's Line samples, each consumer resolving t on its own Line.
CREATE TABLE selection (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
  status          TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'deleting')),  -- D141: readers see only 'complete'
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,         -- the node's folder name, or a family's name inside a stacked node
  path            TEXT,                  -- store-relative; NULL for a Selection the app has not stored yet
  type            TEXT NOT NULL CHECK (type IN ('point', 'span', 'distance', 'spans', 'indices', 'kernel', 'set', 'ladder')),
  label           TEXT,                  -- for people
  session         TEXT,                  -- the acquisition condition, on the nodes that share one (v5)
  description     TEXT,
  manifold_id     TEXT REFERENCES manifold (id) ON DELETE CASCADE,  -- what it indexes — or NULL, on a space
  space_id        TEXT REFERENCES space (id) ON DELETE CASCADE,     -- the SPACE it is a position in (D95)
  cell            TEXT NOT NULL DEFAULT '0',  -- WHICH cells of it (D92): '0' vertices · '1' edges · '2' faces · '1,0' on a product
  of_field_id     TEXT REFERENCES field (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- the Field it belongs to, when it is one's (a recording's flags)
  member_of_id    TEXT REFERENCES selection (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- the set this is a member of
  member_ordinal  INTEGER,
  code            INTEGER,               -- what this selection IS, as an integer (D48) — read with dictionary_id
  dictionary_id   TEXT REFERENCES dictionary (id) ON DELETE RESTRICT,  -- the dictionary its code (or a partition's codes) is read with
  n_elements      INTEGER,               -- elements it covers: picked (indices) or labelled (a partition)
  n_members       INTEGER,               -- a set's members: rows, or a partition's distinct codes; a spans set's intervals
  array_path      TEXT,                  -- the stored per-element values (codes · indices); null for a rule
  center          INTEGER,               -- point · distance: the centre ELEMENT of the manifold
  position        REAL,                  -- a point on a SPACE: its coordinate, in `coordinate_system_id` (D95)
  coordinate_system_id TEXT REFERENCES coordinate_system (id) ON DELETE CASCADE,  -- the system `position` is written in
  radius          REAL,                  -- distance: in `unit`, under the manifold's metric
  unit            TEXT,                  -- distance: of the radius ('m')
  weight_fn       TEXT,                  -- kernel
  params_json     TEXT,                  -- kernel: the parameters at the values in use
  created_utc TEXT, created_by TEXT, modified_utc TEXT, modified_by TEXT,
  UNIQUE (subject_id, path),
  UNIQUE (member_of_id, member_ordinal),
  CHECK ((member_of_id IS NULL) = (member_ordinal IS NULL)),
  CHECK (type <> 'set' OR array_path IS NULL OR dictionary_id IS NOT NULL),   -- a partition's codes need a dictionary
  -- a LADDER is top-level and its members' codes are read with ITS dictionary (D65/D83)
  CHECK (type <> 'ladder' OR (member_of_id IS NULL AND dictionary_id IS NOT NULL)),
  CHECK (type <> 'distance' OR (center IS NOT NULL AND radius IS NOT NULL AND radius >= 0)),
  CHECK ((manifold_id IS NULL) <> (space_id IS NULL)),          -- on a manifold OR on a space (D95)
  -- on a space: a point, by its position in a coordinate system; on a manifold: no position
  CHECK (space_id IS NULL OR (type = 'point' AND position IS NOT NULL AND coordinate_system_id IS NOT NULL AND center IS NULL)),
  CHECK (space_id IS NOT NULL OR (position IS NULL AND coordinate_system_id IS NULL)),
  CHECK (type <> 'point' OR center IS NOT NULL OR position IS NOT NULL),
  CHECK (cell <> '' AND cell NOT GLOB '*[^0-3,*]*'),
  -- A SELECTION IS A CHAIN (D94): a span is its manifold's TOP-RANK cells in a box — edges `[start, stop)`
  -- on a Line — never its vertices (a set of vertices is `indices`)
  CHECK (type NOT IN ('span', 'spans') OR cell <> '0')
) STRICT;
CREATE INDEX selection_subject ON selection (subject_id, type);
CREATE INDEX selection_manifold ON selection (manifold_id);
CREATE INDEX selection_space ON selection (space_id);
CREATE INDEX selection_set ON selection (member_of_id);

-- ── selection_element ───────────────────────────────────────────────────────
-- A labelling's code per element, or a subset's picked elements (code null) — rows when the
-- manifold is points or a surface. "Vertices in the left precentral gyrus" is a join from
-- here to the dictionary entry.
CREATE TABLE selection_element (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  element      INTEGER NOT NULL,         -- the vertex / channel index on the manifold
  code         INTEGER,                  -- labels: → dictionary_entry.code; indices: null
  PRIMARY KEY (selection_id, element)
) STRICT;
CREATE INDEX selection_element_code ON selection_element (selection_id, code);

-- ── selection_extent ────────────────────────────────────────────────────────
-- A span's CORNERS, one row per CONSTRAINED axis (D35/D36) — the same shape as
-- `geometry_dimension`, which `axis` refers to, and the same thing as the slice Zarr reads
-- per dimension: half-open [start, stop) in the axis's coordinate. A span on a line has one
-- row; on a volume one row is a slab and three a box; the axes a span leaves out are
-- unconstrained. Centre + width is NOT how a span is written — that is a distance's rule.
CREATE TABLE selection_extent (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  axis         INTEGER NOT NULL,         -- → geometry_dimension.ordinal of the manifold's geometry
  start        REAL NOT NULL,
  stop         REAL NOT NULL CHECK (stop >= start),
  PRIMARY KEY (selection_id, axis)
) STRICT;

-- ── selection_measurement ───────────────────────────────────────────────────
-- What was MEASURED on a selection (D40/D48): one value per (selection, measure) — a tile's
-- band power or kurtosis, a chunk's max, an ROI's area, a detector's score. `measure` is a
-- name a dictionary entry defines (its unit, its function); the per-VERTEX values a measure
-- has are never here — those are a Field (an array) on the manifold or its tiling (D47).
-- A value measured ON a selection — or on ONE of its members (D48/D68). What makes it
-- queryable without loading anything: the FIELD it describes and the member CODE it is for, so
-- "the tiles of this level where this field's mean power exceeds T" is one indexed scan. Values
-- are stored at the FINEST level of a tiling only; a coarser view is the dyadic roll-up
-- `code >> shift` under the measure's reduction (the ladder numbers a tile's children 2t, 2t+1).
-- ARRAY-BACKED (D96, 2026-09-25): when a measure has a value per MEMBER per ROW of its field — the
-- envelope, a min and a max per window per channel, millions of values — the values are an ARRAY in
-- the store (`array_path`, `[rows × members × components]`, `component` naming its slice) and this is
-- ONE row per measure with `code` and `value` null: it says what is measured, on what, where the
-- numbers are. The roll-up is the same `code >> shift`, read off the array.
CREATE TABLE selection_measurement (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  code         INTEGER,                  -- the MEMBER this value is for (a tile, a window); null: the selection as a whole
  of_field_id  TEXT REFERENCES field (id) ON DELETE CASCADE,  -- the field measured; null: a geometric measurement of the selection itself (area, vertices)
  measure      TEXT NOT NULL,            -- 'mean' · 'max' · 'mean_square' · 'area' · … — registered in the app-wide `measure` dictionary
  value        REAL,                     -- null when the values are an array (below)
  weight       REAL,                     -- what this value counts for in a roll-up (a tile's area); null = 1
  -- WITHIN (2026-09-24): the value is over member `code` of `selection_id` RESTRICTED to member
  -- `within_code` of `within_selection_id` — a channel group, a parcel. So a measurement is
  -- rank 2: one member of the manifold it rolls up along, one of the manifold it is grouped by.
  -- Both null = the whole of that manifold, which is every row written before this existed.
  -- A CONJUNCTION (left AND frontal) is not a second pair; it is a labelling of its own (D30).
  within_selection_id TEXT REFERENCES selection (id) ON DELETE CASCADE,
  within_code         INTEGER,
  unit         TEXT,
  computed_utc TEXT,
  computed_by  TEXT,
  array_path   TEXT,                     -- ARRAY-BACKED (D96): the store-relative array holding one value per member per row of the field
  component    INTEGER,                  -- which slice of the array's last axis is this measure (the envelope: 0 min, 1 max)
  CHECK ((within_code IS NULL) >= (within_selection_id IS NULL)),   -- a code needs a selection to read it
  CHECK ((value IS NULL) = (array_path IS NOT NULL)),              -- a value here, or an array of them
  CHECK (array_path IS NULL OR (code IS NULL AND of_field_id IS NOT NULL))   -- an array is per member, of a field
) STRICT;
CREATE UNIQUE INDEX selection_measurement_key ON selection_measurement (selection_id, coalesce(code, -1), coalesce(of_field_id, ''), coalesce(within_selection_id, ''), coalesce(within_code, -1), measure);
CREATE INDEX selection_measurement_measure ON selection_measurement (measure, value);
CREATE INDEX selection_measurement_field ON selection_measurement (of_field_id, measure, value);
-- THE BOOKKEEPING'S OWN TWO (2026-09-24). `selection_measurement_key` above is an EXPRESSION
-- index (`coalesce`, which it needs: SQLite counts NULLs as distinct, so without it a selection's
-- own `code IS NULL` row could be written twice). An ordinary `code = ?` predicate cannot match
-- it, so every point seek fell back to `_field` and scanned that whole (field, measure) partition.
-- Measured on 180 006 rows of a constant-Q ladder's bookkeeping: reading one tile at all 16 levels
-- took 709 ms, and a cross-band join did not finish. With these it is 57 µs and 130 µs.
CREATE INDEX selection_measurement_tile ON selection_measurement (selection_id, code, measure);
CREATE INDEX selection_measurement_level ON selection_measurement (selection_id, measure, value);
-- "the windows where this band is strongest WITHIN the left hemisphere": seek the level, the
-- grouping and the group, then range over value.
CREATE INDEX selection_measurement_within ON selection_measurement (selection_id, within_selection_id, within_code, measure, value);

-- ── selection_span ──────────────────────────────────────────────────────────
-- The members of a spans set on a line: an event, a page — each by its corners, [start, stop)
-- in the line's coordinate system (D36); an instant has start = stop.
CREATE TABLE selection_span (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  ordinal      INTEGER NOT NULL,
  start        REAL NOT NULL,
  stop         REAL NOT NULL CHECK (stop >= start),
  code         INTEGER,                  -- the member's level (an event type), read with the set's dictionary
  PRIMARY KEY (selection_id, ordinal)
) STRICT;
CREATE INDEX selection_span_start ON selection_span (selection_id, start);

-- ── field ───────────────────────────────────────────────────────────────────
-- THE ARRAY (register D39): values on a manifold. Its SHAPE is its manifold's — a recording is
-- on Sensors × Time (a product manifold, D37), a T1 on its volume, an operator's eigenvalues
-- on the operator's mode axis. What the row adds over the array: identity, what the values
-- ARE (`value_type`, `n_components`, `unit`), how they are stored (`data_type`,
-- `chunk_shape_json` — what zarrita needs to read a chunk without the manifest, D2), and what
-- it is OF: a summary of a field (`of_field_id` + `function`), a by-product of an operator
-- (`of_operator_id`: eigenvalues), a derivation (`derived_from_id`), and the OPERATOR that
-- produced it from that (`by_operator_id`, D93): the divergence of a current names its divergence.
--   kind   what it is, open vocabulary: recording · image · eigenvalues · positions ·
--          orientations · index · … — a new kind of measurement is a row, not DDL
--   value_type   scalar · integer · complex · vector3 · quaternion (the model's) · vector2 ·
--          matrix3
--   cell   WHICH cells of the manifold the values sit on (register D92) — a CELL KEY: the rank
--          digit ('0' vertices · '1' edges · '2' faces · '3' volumes), '*' after it for a DUAL
--          cell, and on a PRODUCT one rank per factor joined by ',' ('0,0' the source currents on
--          cortex × time; '0,1' their time derivative; the default '0' is the vertices on a
--          product too). A lattice's vertices are its SAMPLE
--          POINTS, so a T1's values are '0'.
CREATE TABLE field (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
  status          TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'deleting')),  -- D141: readers see only 'complete'
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  path            TEXT,                  -- store-relative; NULL for a Field the app has not stored yet
  kind            TEXT NOT NULL,
  label           TEXT,
  session         TEXT,
  description     TEXT,
  manifold_id     TEXT NOT NULL REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  cell            TEXT NOT NULL DEFAULT '0',  -- which cells of the manifold (D92)
  value_type      TEXT NOT NULL CHECK (value_type IN ('scalar', 'integer', 'complex', 'vector2', 'vector3', 'matrix3', 'quaternion')),
  n_components    INTEGER NOT NULL CHECK (n_components >= 1),
  unit            TEXT,
  data_type       TEXT NOT NULL,         -- the stored dtype: 'float32', 'uint8'
  chunk_shape_json TEXT,                 -- the chunk grid, in the array's dimension order
  codecs_json     TEXT,                  -- the codec pipeline, verbatim from zarr.json (O21): [bytes] · [bytes, blosc]
  fill_value_json TEXT,                  -- what an absent chunk holds, verbatim (0, "NaN"); with the four above and the
                                         -- manifold's shape, everything zarrita needs to read a chunk without the manifest
  of_field_id     TEXT REFERENCES field (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,      -- a SUMMARY of another field
  of_operator_id  TEXT REFERENCES operator (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,   -- a by-product of: an operator's eigenvalues
  by_operator_id  TEXT REFERENCES operator (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,  -- PRODUCED BY: this field = that operator applied to `derived_from_id` (D93)
  function        TEXT,                  -- the roll-up or statistic this field IS: a summary's ('minmax' of its of_field), a group's ('mean' over its contributions, D55)
  derived_from_id TEXT REFERENCES field (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  comment         TEXT,
  source          TEXT,
  producer_json   TEXT,
  created_utc TEXT, created_by TEXT, modified_utc TEXT, modified_by TEXT,
  UNIQUE (subject_id, path),
  CHECK (cell <> '' AND cell NOT GLOB '*[^0-3,*]*')
) STRICT;
CREATE INDEX field_subject ON field (subject_id, kind);
CREATE INDEX field_manifold ON field (manifold_id);
CREATE INDEX field_session ON field (session);
CREATE INDEX field_of_field ON field (of_field_id);
CREATE INDEX field_of_operator ON field (of_operator_id);
CREATE INDEX field_by_operator ON field (by_operator_id);

-- ── field_contribution ──────────────────────────────────────────────────────
-- What a GROUP field accumulates (D55): the subjects — and which of their fields — a dataset-
-- level average or statistic is over, with a weight (1 for a mean; a subject's n for a
-- weighted one). `field.function` names the statistic (mean · std · t · count · …). A group
-- field's contributions come and go with the subjects that made them.
CREATE TABLE field_contribution (
  field_id        TEXT NOT NULL REFERENCES field (id) ON DELETE CASCADE,
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  source_field_id TEXT REFERENCES field (id) ON DELETE SET NULL,   -- the subject's own field it was taken from, when it is one
  weight          REAL NOT NULL DEFAULT 1,
  ordinal         INTEGER NOT NULL,
  PRIMARY KEY (field_id, ordinal),
  UNIQUE (field_id, subject_id, source_field_id)
) STRICT;
CREATE INDEX field_contribution_subject ON field_contribution (subject_id);

-- ── operator ────────────────────────────────────────────────────────────────
-- THE MAPPING (register D40): Field → Field, a stored matrix from one manifold to another —
-- the imaging kernel (sensors → cortex, vector3), the leadfield (the other way), a
-- hemisphere's eigenvectors (its eigen axis → the hemisphere), its mass matrix (sparse,
-- hemisphere → hemisphere), a projection onto modes, a TRANSITION MAP between coordinate
-- systems (voxel → scs, the 3×4 affine — `from/to_coordinate_system_id`; D25's "an affine is
-- an operator"). `from_selection_id` restricts the domain (the 270 channels a kernel maps).
-- `to_manifold_id` is null where the row space has no manifold row yet (the joint 800 modes
-- of sensorsToEigenmodes — O15).
--   kind     open vocabulary: inverse kernel · forward · eigenvectors · mass · projection ·
--            transition · covariance · whitener · …
--   layout   dense (one array) · csc / csr (a group of data · indices · indptr) · ANALYTIC —
--            no bytes at all: the matrix is evaluated by its node from the manifold's geometry
--            or a closed form (a gradient, a divergence, d₀, a Hodge star, the DFT), so it has
--            no path and no dtype; `n_rows` × `n_cols` are still its shape, in cells × components
--            of the two ends (register D93). What makes it a row is identity: a runtime operator
--            is an OBJECT the app can list, and a saved result names it (`field.by_operator_id`).
--   method   HOW an analytic operator is evaluated — 'fem-tangential' · 'fem-ambient' · 'dec' ·
--            'fft'; null for a stored matrix
--   params_json   an analytic operator's parameters at the values in use
--   from_cell / to_cell   the CELL KEYS of the two ends (D92, as on `field`). A differential
--            operator maps within ONE manifold and changes the cell: DEC d₀ is cortex '0' →
--            cortex '1', ★₁ is '1' → '1*', the FEM face gradient '0' → '2'.
CREATE TABLE operator (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
  status          TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('pending', 'complete', 'deleting')),  -- D141: readers see only 'complete'
  subject_id      TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  path            TEXT,
  kind            TEXT NOT NULL,
  label           TEXT,
  session         TEXT,
  description     TEXT,
  from_manifold_id TEXT NOT NULL REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  to_manifold_id   TEXT REFERENCES manifold (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  from_cell       TEXT NOT NULL DEFAULT '0',
  to_cell         TEXT NOT NULL DEFAULT '0',
  from_selection_id TEXT REFERENCES selection (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
  to_selection_id   TEXT REFERENCES selection (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
  from_value_type TEXT CHECK (from_value_type IN ('scalar', 'integer', 'complex', 'vector2', 'vector3', 'matrix3', 'quaternion')),
  to_value_type   TEXT CHECK (to_value_type IN ('scalar', 'integer', 'complex', 'vector2', 'vector3', 'matrix3', 'quaternion')),
  from_coordinate_system_id TEXT REFERENCES coordinate_system (id) ON DELETE CASCADE,  -- transition maps
  to_coordinate_system_id   TEXT REFERENCES coordinate_system (id) ON DELETE CASCADE,
  layout          TEXT NOT NULL CHECK (layout IN ('dense', 'csc', 'csr', 'analytic')),
  method          TEXT,                  -- analytic: how it is evaluated (D93)
  params_json     TEXT,                  -- analytic: its parameters
  n_rows          INTEGER NOT NULL,
  n_cols          INTEGER NOT NULL,
  nnz             INTEGER,               -- sparse layouts
  data_type       TEXT,
  chunk_shape_json TEXT,
  codecs_json     TEXT,                  -- dense: as on field (O21); a sparse group's arrays carry their own
  fill_value_json TEXT,
  derived_from_id TEXT REFERENCES operator (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,  -- a projection derived from the kernel
  -- THE PAIR (D106): the operator this one INVERTS, and how — stated on the inverse side (one forward may have several: MNE
  -- and dSPM of one leadfield). exact: B·A = I and A·B = I (the DFT) · left: B·A = I, A·B a projection (analysis Φᵀ M of a
  -- truncated Φ) · regularized: B·A = R, a resolution operator (W of G) · adjoint: B = Aᵀ under the metric (−div of grad).
  inverts_id      TEXT REFERENCES operator (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
  inverse_kind    TEXT CHECK (inverse_kind IN ('exact', 'left', 'regularized', 'adjoint')),
  comment         TEXT,
  source          TEXT,
  producer_json   TEXT,
  created_utc TEXT, created_by TEXT, modified_utc TEXT, modified_by TEXT,
  UNIQUE (subject_id, path),
  CHECK ((layout IN ('csc', 'csr')) = (nnz IS NOT NULL)),
  CHECK (layout <> 'analytic' OR (path IS NULL AND data_type IS NULL AND method IS NOT NULL)),   -- an analytic operator has no bytes, and says how it is evaluated
  CHECK ((from_coordinate_system_id IS NULL) = (to_coordinate_system_id IS NULL)),
  CHECK (from_cell <> '' AND from_cell NOT GLOB '*[^0-3,*]*'),
  CHECK (to_cell <> '' AND to_cell NOT GLOB '*[^0-3,*]*'),
  CHECK ((inverts_id IS NULL) = (inverse_kind IS NULL)),   -- a pair says how it inverts
  CHECK (inverts_id IS NULL OR inverts_id <> id)
) STRICT;
CREATE INDEX operator_subject ON operator (subject_id, kind);
CREATE INDEX operator_inverts ON operator (inverts_id);

-- AN INVERSE PAIR LOSES ITS PARTNER WHOLE (open defect 21, 2026-09-29). `inverts_id` is ON DELETE SET NULL, but
-- `inverse_kind` is not a key and was left behind, so the CHECK `(inverts_id IS NULL) = (inverse_kind IS NULL)` refused
-- the delete of any forward model with an inverse — and so the delete of any subject holding a kernel ↔ forward pair.
-- Cleared together, BEFORE the partner goes (the SET NULL then finds nothing to do). IF NOT EXISTS: `openDatabase`
-- also applies it to a database built before it (a trigger changes no table, so the version stands).
CREATE TRIGGER IF NOT EXISTS operator_inverse_partner_gone BEFORE DELETE ON operator BEGIN
  UPDATE operator SET inverts_id = NULL, inverse_kind = NULL WHERE inverts_id = OLD.id;
END;

-- A COMPOSED operator's FACTORS (D106), in APPLICATION order (ordinal 0 applied first): an operator, or a kernel Selection
-- (a gain g(λ) on a spectral manifold, D35). `derived_from_id` stays the primary parent; this is the exact chain —
-- `sensorsToEigenmodes` = [W, the Dirac analysis], G Φ_im = [the Dirac synthesis, G]. The parent's detail (D42 naming).
CREATE TABLE operator_factor (
  operator_id         TEXT NOT NULL REFERENCES operator (id) ON DELETE CASCADE,
  ordinal             INTEGER NOT NULL,
  factor_operator_id  TEXT REFERENCES operator (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  factor_selection_id TEXT REFERENCES selection (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
  transposed          INTEGER NOT NULL DEFAULT 0 CHECK (transposed IN (0, 1)),
  PRIMARY KEY (operator_id, ordinal),
  CHECK ((factor_operator_id IS NULL) <> (factor_selection_id IS NULL)),   -- one factor: an operator or a gain
  CHECK (factor_operator_id IS NULL OR factor_operator_id <> operator_id)
) STRICT;
CREATE INDEX operator_factor_operator ON operator_factor (factor_operator_id);
CREATE INDEX operator_from ON operator (from_manifold_id);
CREATE INDEX operator_to ON operator (to_manifold_id);
CREATE INDEX operator_session ON operator (session);

-- ── variable_dictionary · subject_variable ──────────────────────────────────
-- The cohort's variables (D15). `variable_dictionary` DEFINES what a dataset measures about
-- its subjects — name, type, unit, allowed levels — and holds no values (the
-- `participants.json` half). `subject_variable` holds one typed value per subject per
-- defined variable (the `participants.tsv` half). A missing value is a missing row. No
-- writer emits these yet.
CREATE TABLE variable_dictionary (
  id          TEXT PRIMARY KEY,          -- UUID
  dataset_id  TEXT NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
  name        TEXT NOT NULL,             -- 'age', 'sex', 'group', …
  type        TEXT NOT NULL CHECK (type IN ('number', 'string', 'boolean')),
  unit        TEXT,                      -- 'years', …
  description TEXT,
  levels_json TEXT,                      -- JSON list of allowed values, for a categorical variable
  UNIQUE (dataset_id, name)
) STRICT;
CREATE TABLE subject_variable (
  subject_id             TEXT NOT NULL REFERENCES subject (id) ON DELETE CASCADE,
  variable_dictionary_id TEXT NOT NULL REFERENCES variable_dictionary (id) ON DELETE CASCADE,
  value_num              REAL,           -- for type 'number' / 'boolean' (0/1)
  value_text             TEXT,           -- for type 'string'
  PRIMARY KEY (subject_id, variable_dictionary_id),
  CHECK ((value_num IS NULL) <> (value_text IS NULL))
) STRICT;

-- THE VERSION — of the tables AND of the attributes on disk. Bumped when either changes.
-- ── sync_job ────────────────────────────────────────────────────────────────
-- THE OUTBOX (register D44/O20). The rows are the truth; a node's `zarr.json` attributes are
-- a copy written FROM its row. A SQLite trigger cannot run code, so the triggers below only
-- RECORD that a node's row (or a child row that is part of its attributes) changed; the sync
-- writer (`backend/src/db/sync.ts`) drains the queue: `upsert` writes the attributes from the
-- row, `delete` removes the node folder named by `path` (captured at deletion, since the row
-- is gone). One pending job per (table, row, op) — a burst of child inserts is one job. The
-- queue survives a crash and is visible in DB Browser. `selection_element` is NOT a trigger
-- source: its content is the array, written by the class's write verb, never an attribute.
-- The bodies test for a pending twin with NOT EXISTS rather than INSERT OR IGNORE: a
-- statement's own conflict clause (our upserts' ON CONFLICT DO UPDATE) OVERRIDES the one in a
-- trigger body, and OR IGNORE then aborts.
CREATE TABLE sync_job (
  id          INTEGER PRIMARY KEY,
  table_name  TEXT NOT NULL CHECK (table_name IN ('dataset', 'subject', 'manifold', 'selection', 'field', 'operator')),
  row_id      TEXT NOT NULL,
  op          TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
  path        TEXT,                       -- delete: the node's LOCATION (datastore-relative: dataset/subject/node) when its row went
  created_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  done_utc    TEXT,
  error       TEXT
) STRICT;
CREATE UNIQUE INDEX sync_job_pending ON sync_job (table_name, row_id, op) WHERE done_utc IS NULL;
-- ── INDEXES ADDED WITHOUT A VERSION ── (2026-10-07) a parent's DELETE looks up its children by these columns; without an
-- index each cascade or SET NULL scans the child table (a selection delete scanned all of selection_measurement). Added
-- WITHOUT a version (an index changes no table): `openDatabase` runs this block on a database built before it.
CREATE INDEX IF NOT EXISTS selection_measurement_within_selection ON selection_measurement (within_selection_id);
CREATE INDEX IF NOT EXISTS operator_from_selection ON operator (from_selection_id);
CREATE INDEX IF NOT EXISTS operator_to_selection ON operator (to_selection_id);
CREATE INDEX IF NOT EXISTS operator_factor_selection ON operator_factor (factor_selection_id);
CREATE INDEX IF NOT EXISTS manifold_partition ON manifold (partition_id);
CREATE INDEX IF NOT EXISTS manifold_geometry ON manifold (geometry_id);
CREATE INDEX IF NOT EXISTS selection_of_field ON selection (of_field_id);
CREATE INDEX IF NOT EXISTS selection_dictionary ON selection (dictionary_id);
CREATE INDEX IF NOT EXISTS field_derived_from ON field (derived_from_id);
CREATE INDEX IF NOT EXISTS field_contribution_source ON field_contribution (source_field_id);
-- the sync queue's own: the drain seeks the oldest PENDING job, never scanning the finished history
CREATE INDEX IF NOT EXISTS sync_job_queue ON sync_job (id) WHERE done_utc IS NULL;
-- ── END INDEXES ADDED WITHOUT A VERSION ──

-- ─── JOBS (D59) — a user-triggered COMPUTATION, on either host. The standard backend shape: the
-- request is a row (what, over which inputs, with which parameters, asked by whom), a worker —
-- main's provisioners, or the renderer's GPU batch — runs it and reports status here, and the
-- results are ordinary node rows written through the repositories (`result_json` names them).
-- Not a node: no path, no sync trigger; it is the app's record of work, not the store's.
CREATE TABLE job (
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,                       -- the operation: 'prepare:eigen', 'batch:psd', 'filter:chunk', …
  host          TEXT NOT NULL CHECK (host IN ('main', 'renderer')),
  status        TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')) DEFAULT 'queued',
  dataset_id    TEXT REFERENCES dataset (id) ON DELETE CASCADE,
  subject_id    TEXT REFERENCES subject (id) ON DELETE CASCADE,
  inputs_json   TEXT,                                -- the ids of the rows it reads (fields, selections, operators)
  params_json   TEXT,                                -- its parameters, as given
  result_json   TEXT,                                -- the ids of the rows it wrote, once done
  progress      REAL CHECK (progress IS NULL OR (progress >= 0 AND progress <= 1)),
  error         TEXT,
  requested_by  TEXT,
  created_utc   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  started_utc   TEXT,
  finished_utc  TEXT
) STRICT;
CREATE INDEX job_status ON job (status, created_utc);
CREATE INDEX job_subject ON job (subject_id);

CREATE TRIGGER manifold_ai AFTER INSERT ON manifold BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER manifold_au AFTER UPDATE ON manifold BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER manifold_ad AFTER DELETE ON manifold WHEN OLD.path IS NOT NULL BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'manifold', OLD.id, 'delete', ds.path || '/' || s.path || '/' || OLD.path FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = OLD.subject_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_ai AFTER INSERT ON selection BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_au AFTER UPDATE ON selection BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_ad AFTER DELETE ON selection WHEN OLD.path IS NOT NULL BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'selection', OLD.id, 'delete', ds.path || '/' || s.path || '/' || OLD.path FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = OLD.subject_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
CREATE TRIGGER field_ai AFTER INSERT ON field BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'field', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'field' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER field_au AFTER UPDATE ON field BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'field', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'field' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER field_ad AFTER DELETE ON field WHEN OLD.path IS NOT NULL BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'field', OLD.id, 'delete', ds.path || '/' || s.path || '/' || OLD.path FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = OLD.subject_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'field' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
CREATE TRIGGER operator_ai AFTER INSERT ON operator BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'operator', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'operator' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER operator_au AFTER UPDATE ON operator BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'operator', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'operator' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER operator_ad AFTER DELETE ON operator WHEN OLD.path IS NOT NULL BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'operator', OLD.id, 'delete', ds.path || '/' || s.path || '/' || OLD.path FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = OLD.subject_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'operator' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
-- A PATH-LESS row (a product, a union, a chunk grid, a run's output Line) has no folder of its own: it rides in its
-- SUBJECT's `rows_without_path`, so deleting one must re-sync the subject — or the copy keeps a row the database no
-- longer has, and an import of the store trips over it (found 2026-09-29: a saved-then-deleted run's grids).
-- `IF NOT EXISTS`: `openDatabase` applies the same statements to a database built before them (no version change).
CREATE TRIGGER IF NOT EXISTS manifold_ad_pathless AFTER DELETE ON manifold WHEN OLD.path IS NULL BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', OLD.subject_id, 'upsert' WHERE EXISTS (SELECT 1 FROM subject WHERE id = OLD.subject_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.subject_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER IF NOT EXISTS selection_ad_pathless AFTER DELETE ON selection WHEN OLD.path IS NULL BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', OLD.subject_id, 'upsert' WHERE EXISTS (SELECT 1 FROM subject WHERE id = OLD.subject_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.subject_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER IF NOT EXISTS field_ad_pathless AFTER DELETE ON field WHEN OLD.path IS NULL BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', OLD.subject_id, 'upsert' WHERE EXISTS (SELECT 1 FROM subject WHERE id = OLD.subject_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.subject_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER IF NOT EXISTS operator_ad_pathless AFTER DELETE ON operator WHEN OLD.path IS NULL BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', OLD.subject_id, 'upsert' WHERE EXISTS (SELECT 1 FROM subject WHERE id = OLD.subject_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.subject_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
-- the dataset and subject rows are nodes too: their folders' zarr.json carry them (`class: dataset` / `subject`).
-- A DELETE of either enqueues ITS OWN folder: SQLite runs a cascade AFTER the parent row is gone, so a
-- cascaded node's trigger can no longer join its subject to compose a location — the folder goes whole.
CREATE TRIGGER dataset_ai AFTER INSERT ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'dataset', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER dataset_au AFTER UPDATE ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'dataset', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER dataset_ad AFTER DELETE ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'dataset', OLD.id, 'delete', OLD.path WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_ai AFTER INSERT ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_au AFTER UPDATE ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_ad AFTER DELETE ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'subject', OLD.id, 'delete', ds.path || '/' || OLD.path FROM dataset ds WHERE ds.id = OLD.dataset_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
-- a child row is part of its parent's attributes: the parent is re-synced
CREATE TRIGGER selection_extent_ai AFTER INSERT ON selection_extent BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', NEW.selection_id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = NEW.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_extent_ad AFTER DELETE ON selection_extent BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', OLD.selection_id, 'upsert' WHERE EXISTS (SELECT 1 FROM selection WHERE id = OLD.selection_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = OLD.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_span_ai AFTER INSERT ON selection_span BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', NEW.selection_id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = NEW.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_span_ad AFTER DELETE ON selection_span BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', OLD.selection_id, 'upsert' WHERE EXISTS (SELECT 1 FROM selection WHERE id = OLD.selection_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = OLD.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_measurement_ai AFTER INSERT ON selection_measurement BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', NEW.selection_id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = NEW.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER selection_measurement_ad AFTER DELETE ON selection_measurement BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'selection', OLD.selection_id, 'upsert' WHERE EXISTS (SELECT 1 FROM selection WHERE id = OLD.selection_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'selection' AND j.row_id = OLD.selection_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER field_contribution_ai AFTER INSERT ON field_contribution BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'field', NEW.field_id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'field' AND j.row_id = NEW.field_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER field_contribution_ad AFTER DELETE ON field_contribution BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'field', OLD.field_id, 'upsert' WHERE EXISTS (SELECT 1 FROM field WHERE id = OLD.field_id) AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'field' AND j.row_id = OLD.field_id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER geometry_dimension_ai AFTER INSERT ON geometry_dimension BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', manifold.id, 'upsert' FROM manifold WHERE geometry_id = NEW.geometry_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = manifold.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER topology_cell_ai AFTER INSERT ON topology_cell BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', manifold.id, 'upsert' FROM manifold WHERE topology_id = NEW.topology_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = manifold.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER topology_cell_ad AFTER DELETE ON topology_cell BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', manifold.id, 'upsert' FROM manifold WHERE topology_id = OLD.topology_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = manifold.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER topology_au AFTER UPDATE ON topology BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', manifold.id, 'upsert' FROM manifold WHERE topology_id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = manifold.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER geometry_au AFTER UPDATE ON geometry BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'manifold', manifold.id, 'upsert' FROM manifold WHERE geometry_id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'manifold' AND j.row_id = manifold.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;

-- ── VIEWS ───────────────────────────────────────────────────────────────────
-- The questions the app asks, each given a name (register D51): what `subjectAccess.ts`
-- filtered in JavaScript over every node of a subject. Read-only; versioned with the schema;
-- readable from both languages and DB Browser. A view is a query, not a cache — none of
-- these touches `selection_element`.

-- one row per manifold, with what it is made of and where it is placed, by NAME
CREATE VIEW manifold_view AS
SELECT m.*, t.kind AS topology_kind, t.rank AS rank, t.n_components AS n_components, t.closed AS closed,
       g.form AS geometry_form, cs.name AS coordinate_system, cs.unit AS unit, sp.name AS space,
       pm.name AS component_of, dm.name AS derived_from,
       (SELECT count(*) FROM manifold c WHERE c.component_of_id = m.id) AS n_component_rows
FROM manifold m
JOIN topology t ON t.id = m.topology_id
LEFT JOIN geometry g ON g.id = m.geometry_id
LEFT JOIN coordinate_system cs ON cs.id = g.coordinate_system_id
LEFT JOIN space sp ON sp.id = cs.space_id
LEFT JOIN manifold pm ON pm.id = m.component_of_id
LEFT JOIN manifold dm ON dm.id = m.derived_from_id;

-- one row per RECORDING: its two factors by name, its rate and length, its chunk grid
CREATE VIEW recording_view AS
SELECT f.id, f.subject_id, f.name, f.path, f.session, f.label, f.description, f.manifold_id, f.data_type, f.unit,
       sm.id AS sensors_id, sm.name AS sensors, sm.n_vertices AS n_channels,
       tm.id AS time_id, tm.name AS time_line, tm.n_vertices AS n_samples,
       td.spacing AS dt_s, 1.0 / td.spacing AS sfreq_hz, tm.n_vertices * td.spacing AS duration_s,
       json_extract(f.chunk_shape_json, '$[1]') AS chunk_samples,
       ch.id AS chunks_id, ch.n_members AS n_chunks,
       f.created_utc, f.created_by, f.modified_utc, f.modified_by
FROM field f
JOIN manifold pm ON pm.id = f.manifold_id AND pm.type = 'product'
JOIN geometry_dimension d0 ON d0.geometry_id = pm.geometry_id AND d0.ordinal = 0
JOIN geometry_dimension d1 ON d1.geometry_id = pm.geometry_id AND d1.ordinal = 1
JOIN manifold sm ON sm.id = d0.manifold_id
JOIN manifold tm ON tm.id = d1.manifold_id
JOIN geometry_dimension td ON td.geometry_id = tm.geometry_id AND td.ordinal = 0
LEFT JOIN selection ch ON ch.of_field_id = f.id AND ch.type = 'spans' AND ch.manifold_id = tm.id
WHERE f.kind = 'recording';

-- one row per operator, from and to by name, with what restricts it
CREATE VIEW operator_view AS
SELECT o.*, fm.name AS from_manifold, fm.type AS from_type, tm.name AS to_manifold, tm.type AS to_type,
       fs.name AS from_selection, fs.n_elements AS from_n_elements, ts.name AS to_selection,
       fcs.name AS from_coordinate_system, tcs.name AS to_coordinate_system, dop.name AS derived_from
FROM operator o
JOIN manifold fm ON fm.id = o.from_manifold_id
LEFT JOIN manifold tm ON tm.id = o.to_manifold_id
LEFT JOIN selection fs ON fs.id = o.from_selection_id
LEFT JOIN selection ts ON ts.id = o.to_selection_id
LEFT JOIN coordinate_system fcs ON fcs.id = o.from_coordinate_system_id
LEFT JOIN coordinate_system tcs ON tcs.id = o.to_coordinate_system_id
LEFT JOIN operator dop ON dop.id = o.derived_from_id;

-- one row per selection, with its manifold, its dictionary, its set and its field by name
CREATE VIEW selection_view AS
SELECT s.*, m.name AS manifold, m.type AS manifold_type, d.name AS dictionary, d.scope AS dictionary_scope,
       ps.name AS member_of, f.name AS of_field, f.kind AS of_field_kind,
       (s.type = 'set' AND s.array_path IS NOT NULL) AS is_partition,
       (SELECT count(*) FROM selection_measurement x WHERE x.selection_id = s.id) AS n_measurements
FROM selection s
JOIN manifold m ON m.id = s.manifold_id
LEFT JOIN dictionary d ON d.id = s.dictionary_id
LEFT JOIN selection ps ON ps.id = s.member_of_id
LEFT JOIN field f ON f.id = s.of_field_id;

-- one row per field: its manifold by name and type, what it is OF, and how many subjects it accumulates
CREATE VIEW field_view AS
SELECT f.*, m.name AS manifold, m.type AS manifold_type, m.n_vertices AS n_vertices, ms.name AS manifold_subject, ms.kind AS manifold_subject_kind,
       s.kind AS subject_kind, of_f.name AS of_field, of_o.name AS of_operator,
       (SELECT count(*) FROM field_contribution c WHERE c.field_id = f.id) AS n_contributions
FROM field f
JOIN subject s ON s.id = f.subject_id
JOIN manifold m ON m.id = f.manifold_id
JOIN subject ms ON ms.id = m.subject_id
LEFT JOIN field of_f ON of_f.id = f.of_field_id
LEFT JOIN operator of_o ON of_o.id = f.of_operator_id;

-- one row per (subject, session): what the session holds
CREATE VIEW session_view AS
SELECT x.subject_id, x.session,
       (SELECT count(*) FROM field f WHERE f.subject_id = x.subject_id AND f.session = x.session AND f.kind = 'recording') AS n_recordings,
       (SELECT count(*) FROM operator o WHERE o.subject_id = x.subject_id AND o.session = x.session AND o.kind = 'inverse kernel') AS n_kernels,
       (SELECT count(*) FROM operator o WHERE o.subject_id = x.subject_id AND o.session = x.session AND o.kind LIKE 'sensorsTo%') AS n_projections,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = x.subject_id AND m.session = x.session AND m.type = 'points') AS n_sensor_layouts,
       (SELECT sum(tm.n_vertices * td.spacing) FROM recording_view r JOIN manifold tm ON tm.id = r.time_id JOIN geometry_dimension td ON td.geometry_id = tm.geometry_id AND td.ordinal = 0 WHERE r.subject_id = x.subject_id AND r.session = x.session) AS duration_s
FROM (SELECT DISTINCT subject_id, session FROM field WHERE session IS NOT NULL
      UNION SELECT DISTINCT subject_id, session FROM operator WHERE session IS NOT NULL
      UNION SELECT DISTINCT subject_id, session FROM manifold WHERE session IS NOT NULL
      UNION SELECT DISTINCT subject_id, session FROM selection WHERE session IS NOT NULL) x;

-- THE ATLASES (D129): one row per (partition, field measured) — the atlas measurements grouped so they are
-- DISCOVERABLE. An atlas is measurement rows on a tile partition at its depth (spatial) or on a JOINT partition of tiles ×
-- frames (spatiotemporal), each partition's `params_json.atlas` naming its tree and its gauge (the canonical frames field);
-- a joint partition's `params_json.joint` names its space partition, its frame tiling and its bit layout; a recording's FRAME
-- tiling (`frames`) carries the good samples per frame. No version: a view changes no table, and `openDatabase` refreshes the
-- views of a database built before it (`refreshViews`); a served database published before it has none. The field is a map
-- (a PET surface), a byte-less source estimate (its recording and kernel), or a group field (its contributions).
CREATE VIEW atlas_view AS
SELECT x.subject_id, s.name AS subject, s.kind AS subject_kind,
       x.id AS partition_id, x.name AS partition,
       CASE WHEN x.type = 'spans' THEN 'frames' WHEN json_extract(x.params_json, '$.joint') IS NOT NULL THEN 'spatiotemporal' ELSE 'spatial' END AS kind,
       json_extract(x.params_json, '$.atlas.tree') AS tree,
       coalesce(json_extract(x.params_json, '$.atlas.level'), json_extract(sp.params_json, '$.atlas.level')) AS level,
       json_extract(x.params_json, '$.atlas.gauge') AS gauge_field_id,
       json_extract(x.params_json, '$.joint.space') AS space_partition_id,
       json_extract(x.params_json, '$.joint.time') AS time_tiling_id,
       json_extract(x.params_json, '$.joint.frames') AS n_frames,
       coalesce(json_extract(x.params_json, '$.joint.tiles'), x.n_members) AS n_tiles,
       m.of_field_id AS field_id, f.name AS field, f.kind AS field_kind, f.unit AS field_unit, f.function AS field_function,
       f.derived_from_id AS recording_id, f.by_operator_id AS kernel_id,
       (SELECT count(*) FROM field_contribution c WHERE c.field_id = f.id) AS n_contributions,
       group_concat(DISTINCT m.measure) AS measures,
       max(m.within_selection_id) AS bands_id, count(DISTINCT m.within_code) AS n_bands,
       count(*) AS n_rows
FROM selection x
JOIN subject s ON s.id = x.subject_id
JOIN selection_measurement m ON m.selection_id = x.id AND m.of_field_id IS NOT NULL
JOIN field f ON f.id = m.of_field_id
LEFT JOIN selection sp ON sp.id = json_extract(x.params_json, '$.joint.space')
WHERE json_extract(x.params_json, '$.atlas') IS NOT NULL
GROUP BY x.id, m.of_field_id;

-- one row per subject: what it holds, and whether it is WARM (its eigenbases, masses and
-- projections — the `sensorsTo*` operators — exist: what the app's prepare step computes)
CREATE VIEW subject_status AS
SELECT s.id, s.dataset_id, s.name, s.path, s.kind, s.template_id,
       (SELECT count(*) FROM session_view v WHERE v.subject_id = s.id) AS n_sessions,
       (SELECT count(*) FROM field f WHERE f.subject_id = s.id AND f.kind = 'recording') AS n_recordings,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.type = 'surface' AND m.component_of_id IS NULL) AS n_surfaces,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.type = 'volume') AS n_volumes,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id) AS n_manifolds,
       (SELECT count(*) FROM field f WHERE f.subject_id = s.id) AS n_fields,
       (SELECT count(*) FROM operator o WHERE o.subject_id = s.id) AS n_operators,
       (SELECT count(*) FROM selection x WHERE x.subject_id = s.id) AS n_selections,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.operator IS NOT NULL AND m.component_of_id IS NOT NULL) AS n_eigen_axes,
       (SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind = 'mass') AS n_masses,
       (SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind LIKE 'sensorsTo%') AS n_projections,
       (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.is_primary = 1 AND m.type = 'surface') AS has_primary_surface,
       ((SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.operator IS NOT NULL AND m.component_of_id IS NOT NULL) > 0
        AND (SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind = 'mass') > 0
        AND ((SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind LIKE 'sensorsTo%') > 0
             -- D107: a fusion is COMPOSED where a view opens — an inverse kernel and a Dirac basis to compose it from are enough
             OR ((SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind = 'inverse kernel') > 0
                 AND (SELECT count(*) FROM manifold m WHERE m.subject_id = s.id AND m.operator = 'relativeDirac') > 0))) AS is_warm,
       s.source_format, s.source_path, s.created_utc, s.modified_utc
FROM subject s;

-- the DASHBOARD's numbers in one read (app/src/pages/Dashboard.tsx): a total per class, and per kind or type the tallies its
-- breakdowns draw — the same rows the screens' views list (subject_status is one row a subject; session_view one a session)
CREATE VIEW dashboard_counts AS
SELECT 'subjects' AS what, NULL AS name, count(*) AS n FROM subject
UNION ALL SELECT 'sessions', NULL, count(*) FROM session_view
UNION ALL SELECT 'recordings', NULL, count(*) FROM recording_view
UNION ALL SELECT 'seconds', NULL, coalesce(sum(duration_s), 0) FROM recording_view
UNION ALL SELECT 'fields', kind, count(*) FROM field_view GROUP BY kind
UNION ALL SELECT 'manifolds', type, count(*) FROM manifold_view GROUP BY type
UNION ALL SELECT 'selections', type, count(*) FROM selection_view GROUP BY type
UNION ALL SELECT 'operators', kind, count(*) FROM operator_view GROUP BY kind;


-- ── BEGIN GENERATED: the attribute views (backend/scripts/gen-attribute-views.mjs) — do not edit ──
-- One view per node table: its zarr.json attributes as TEXT (D145). Every writer writes zarr.json from these, in any language.
CREATE VIEW manifold_attributes AS
SELECT m.id, json_object('class', 'manifold', 'id', m.id, 'status', m.status, 'subject_id', m.subject_id, 'name', m.name, 'path', m.path, 'type', m.type, 'label', m.label, 'session', m.session, 'is_primary', m.is_primary, 'n_vertices', m.n_vertices, 'n_edges', m.n_edges, 'n_faces', m.n_faces, 'n_volumes', m.n_volumes, 'topology_id', m.topology_id, 'geometry_id', m.geometry_id, 'component_of_id', m.component_of_id, 'component_ordinal', m.component_ordinal, 'from_index', m.from_index, 'derived_from_id', m.derived_from_id, 'partition_id', m.partition_id, 'operator', m.operator, 'comment', m.comment, 'source', m.source, 'producer_json', m.producer_json, 'created_utc', m.created_utc, 'created_by', m.created_by, 'modified_utc', m.modified_utc, 'modified_by', m.modified_by,
  'topology', (SELECT json_object('id', t.id, 'subject_id', t.subject_id, 'kind', t.kind, 'rank', t.rank, 'n_components', t.n_components, 'winding', t.winding, 'closed', t.closed, 'orientable', t.orientable, 'cells', (SELECT json_group_array(json_object('rank', c.rank, 'path', c.path) ORDER BY c.rank) FROM topology_cell c WHERE c.topology_id = t.id)) FROM topology t WHERE t.id = m.topology_id),
  'geometry', (SELECT json_object('id', g.id, 'subject_id', g.subject_id, 'form', g.form, 'coordinate_system_id', g.coordinate_system_id, 'positions_path', g.positions_path, 'direction_json', g.direction_json) FROM geometry g WHERE g.id = m.geometry_id),
  'dimensions', (SELECT json_group_array(json_object('geometry_id', d.geometry_id, 'ordinal', d.ordinal, 'name', d.name, 'n_vertices', d.n_vertices, 'spacing', d.spacing, 'origin', d.origin, 'unit', d.unit, 'manifold_id', d.manifold_id) ORDER BY d.ordinal) FROM geometry_dimension d WHERE d.geometry_id = m.geometry_id),
  'coordinate_system', (SELECT json_object('id', cs.id, 'space_id', cs.space_id, 'name', cs.name, 'unit', cs.unit, 'is_reference', cs.is_reference, 'transition', cs.transition, 'transition_json', cs.transition_json, 'description', cs.description) FROM geometry g JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE g.id = m.geometry_id),
  'space', (SELECT json_object('id', sp.id, 'subject_id', sp.subject_id, 'name', sp.name, 'geometry', sp.geometry, 'rank', sp.rank, 'parameters_json', sp.parameters_json, 'description', sp.description) FROM geometry g JOIN coordinate_system cs ON cs.id = g.coordinate_system_id JOIN space sp ON sp.id = cs.space_id WHERE g.id = m.geometry_id)
) AS attributes FROM manifold m;

CREATE VIEW selection_attributes AS
SELECT x.id, json_object('class', 'selection', 'id', x.id, 'status', x.status, 'subject_id', x.subject_id, 'name', x.name, 'path', x.path, 'type', x.type, 'label', x.label, 'session', x.session, 'description', x.description, 'manifold_id', x.manifold_id, 'space_id', x.space_id, 'cell', x.cell, 'of_field_id', x.of_field_id, 'member_of_id', x.member_of_id, 'member_ordinal', x.member_ordinal, 'code', x.code, 'dictionary_id', x.dictionary_id, 'n_elements', x.n_elements, 'n_members', x.n_members, 'array_path', x.array_path, 'center', x.center, 'position', x.position, 'coordinate_system_id', x.coordinate_system_id, 'radius', x.radius, 'unit', x.unit, 'weight_fn', x.weight_fn, 'params_json', x.params_json, 'created_utc', x.created_utc, 'created_by', x.created_by, 'modified_utc', x.modified_utc, 'modified_by', x.modified_by,
  'extents', (SELECT json_group_array(json_object('selection_id', e.selection_id, 'axis', e.axis, 'start', e.start, 'stop', e.stop) ORDER BY e.axis) FROM selection_extent e WHERE e.selection_id = x.id),
  'spans', (SELECT json_group_array(json_object('ordinal', p.ordinal, 'start', p.start, 'stop', p.stop, 'code', p.code) ORDER BY p.ordinal) FROM selection_span p WHERE p.selection_id = x.id),
  'measurements', (SELECT json_group_array(json_object('selection_id', m.selection_id, 'code', m.code, 'of_field_id', m.of_field_id, 'measure', m.measure, 'value', m.value, 'weight', m.weight, 'within_selection_id', m.within_selection_id, 'within_code', m.within_code, 'unit', m.unit, 'computed_utc', m.computed_utc, 'computed_by', m.computed_by, 'array_path', m.array_path, 'component', m.component) ORDER BY m.measure, coalesce(m.of_field_id, ''), coalesce(m.within_selection_id, ''), coalesce(m.within_code, -1)) FROM selection_measurement m WHERE m.selection_id = x.id AND m.code IS NULL),
  'dictionary', (SELECT json_object('id', dd.id, 'name', dd.name, 'scope', dd.scope, 'dataset_id', dd.dataset_id, 'subject_id', dd.subject_id, 'description', dd.description, 'created_utc', dd.created_utc, 'created_by', dd.created_by, 'modified_utc', dd.modified_utc, 'modified_by', dd.modified_by) FROM dictionary dd WHERE dd.id = x.dictionary_id),
  'dictionary_entries', (SELECT json_group_array(json_object('dictionary_id', de.dictionary_id, 'code', de.code, 'name', de.name, 'color_json', de.color_json, 'measurement', de.measurement, 'measurement_unit', de.measurement_unit, 'function', de.function, 'params_json', de.params_json, 'attributes_json', de.attributes_json, 'ordinal', de.ordinal) ORDER BY de.code) FROM dictionary_entry de WHERE de.dictionary_id = x.dictionary_id)
) AS attributes FROM selection x;

CREATE VIEW field_attributes AS
SELECT f.id, json_object('class', 'field', 'id', f.id, 'status', f.status, 'subject_id', f.subject_id, 'name', f.name, 'path', f.path, 'kind', f.kind, 'label', f.label, 'session', f.session, 'description', f.description, 'manifold_id', f.manifold_id, 'cell', f.cell, 'value_type', f.value_type, 'n_components', f.n_components, 'unit', f.unit, 'data_type', f.data_type, 'chunk_shape_json', f.chunk_shape_json, 'codecs_json', f.codecs_json, 'fill_value_json', f.fill_value_json, 'of_field_id', f.of_field_id, 'of_operator_id', f.of_operator_id, 'by_operator_id', f.by_operator_id, 'function', f.function, 'derived_from_id', f.derived_from_id, 'comment', f.comment, 'source', f.source, 'producer_json', f.producer_json, 'created_utc', f.created_utc, 'created_by', f.created_by, 'modified_utc', f.modified_utc, 'modified_by', f.modified_by,
  'contributions', (SELECT json_group_array(json_object('field_id', c.field_id, 'subject_id', c.subject_id, 'source_field_id', c.source_field_id, 'weight', c.weight, 'ordinal', c.ordinal) ORDER BY c.ordinal) FROM field_contribution c WHERE c.field_id = f.id)
) AS attributes FROM field f;

CREATE VIEW operator_attributes AS
SELECT o.id, json_object('class', 'operator', 'id', o.id, 'status', o.status, 'subject_id', o.subject_id, 'name', o.name, 'path', o.path, 'kind', o.kind, 'label', o.label, 'session', o.session, 'description', o.description, 'from_manifold_id', o.from_manifold_id, 'to_manifold_id', o.to_manifold_id, 'from_cell', o.from_cell, 'to_cell', o.to_cell, 'from_selection_id', o.from_selection_id, 'to_selection_id', o.to_selection_id, 'from_value_type', o.from_value_type, 'to_value_type', o.to_value_type, 'from_coordinate_system_id', o.from_coordinate_system_id, 'to_coordinate_system_id', o.to_coordinate_system_id, 'layout', o.layout, 'method', o.method, 'params_json', o.params_json, 'n_rows', o.n_rows, 'n_cols', o.n_cols, 'nnz', o.nnz, 'data_type', o.data_type, 'chunk_shape_json', o.chunk_shape_json, 'codecs_json', o.codecs_json, 'fill_value_json', o.fill_value_json, 'derived_from_id', o.derived_from_id, 'inverts_id', o.inverts_id, 'inverse_kind', o.inverse_kind, 'comment', o.comment, 'source', o.source, 'producer_json', o.producer_json, 'created_utc', o.created_utc, 'created_by', o.created_by, 'modified_utc', o.modified_utc, 'modified_by', o.modified_by,
  'factors', (SELECT json_group_array(json_object('factor_operator_id', k.factor_operator_id, 'factor_selection_id', k.factor_selection_id, 'transposed', k.transposed) ORDER BY k.ordinal) FROM operator_factor k WHERE k.operator_id = o.id),
  'from_coordinate_system', (SELECT json_object('id', fc.id, 'space_id', fc.space_id, 'name', fc.name, 'unit', fc.unit, 'is_reference', fc.is_reference, 'transition', fc.transition, 'transition_json', fc.transition_json, 'description', fc.description) FROM coordinate_system fc WHERE fc.id = o.from_coordinate_system_id),
  'from_space', (SELECT json_object('id', fs.id, 'subject_id', fs.subject_id, 'name', fs.name, 'geometry', fs.geometry, 'rank', fs.rank, 'parameters_json', fs.parameters_json, 'description', fs.description) FROM coordinate_system fc JOIN space fs ON fs.id = fc.space_id WHERE fc.id = o.from_coordinate_system_id),
  'to_coordinate_system', (SELECT json_object('id', tc.id, 'space_id', tc.space_id, 'name', tc.name, 'unit', tc.unit, 'is_reference', tc.is_reference, 'transition', tc.transition, 'transition_json', tc.transition_json, 'description', tc.description) FROM coordinate_system tc WHERE tc.id = o.to_coordinate_system_id),
  'to_space', (SELECT json_object('id', ts.id, 'subject_id', ts.subject_id, 'name', ts.name, 'geometry', ts.geometry, 'rank', ts.rank, 'parameters_json', ts.parameters_json, 'description', ts.description) FROM coordinate_system tc JOIN space ts ON ts.id = tc.space_id WHERE tc.id = o.to_coordinate_system_id)
) AS attributes FROM operator o;

CREATE VIEW subject_attributes AS
SELECT s.id, json_object('class', 'subject', 'schema_version', 49, 'id', s.id, 'status', s.status, 'dataset_id', s.dataset_id, 'name', s.name, 'path', s.path, 'kind', s.kind, 'template_id', s.template_id, 'source_format', s.source_format, 'source_path', s.source_path, 'created_utc', s.created_utc, 'created_by', s.created_by, 'modified_utc', s.modified_utc, 'modified_by', s.modified_by,
  'rows_without_path', json_object('manifold', (SELECT json_group_array(json(v.attributes) ORDER BY r.name) FROM manifold r JOIN manifold_attributes v ON v.id = r.id WHERE r.subject_id = s.id AND r.path IS NULL), 'selection', (SELECT json_group_array(json(v.attributes) ORDER BY r.name) FROM selection r JOIN selection_attributes v ON v.id = r.id WHERE r.subject_id = s.id AND r.path IS NULL), 'field', (SELECT json_group_array(json(v.attributes) ORDER BY r.name) FROM field r JOIN field_attributes v ON v.id = r.id WHERE r.subject_id = s.id AND r.path IS NULL), 'operator', (SELECT json_group_array(json(v.attributes) ORDER BY r.name) FROM operator r JOIN operator_attributes v ON v.id = r.id WHERE r.subject_id = s.id AND r.path IS NULL))
) AS attributes FROM subject s;

CREATE VIEW dataset_attributes AS
SELECT d.id, json_object('class', 'dataset', 'schema_version', 49, 'id', d.id, 'datastore_id', d.datastore_id, 'name', d.name, 'path', d.path, 'source_tool', d.source_tool, 'source_protocol', d.source_protocol, 'source_path', d.source_path, 'created_utc', d.created_utc, 'created_by', d.created_by, 'modified_utc', d.modified_utc, 'modified_by', d.modified_by) AS attributes FROM dataset d;
-- ── END GENERATED: the attribute views ──
PRAGMA user_version = 49;
