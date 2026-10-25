CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;
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
CREATE TABLE subject (
  id              TEXT PRIMARY KEY,      -- UUID, in the store's zarr.json
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
CREATE TABLE topology_cell (
  topology_id TEXT NOT NULL REFERENCES topology (id) ON DELETE CASCADE,
  rank        INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 3),
  path        TEXT NOT NULL,             -- 'sources/cortex_pial_low/faces'
  PRIMARY KEY (topology_id, rank)
) STRICT;
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
CREATE TRIGGER geometry_dimension_factor_gone AFTER DELETE ON geometry_dimension WHEN OLD.manifold_id IS NOT NULL
BEGIN
  DELETE FROM geometry WHERE id = OLD.geometry_id;
END;
CREATE TABLE manifold (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
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
CREATE TABLE selection (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
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
CREATE TABLE selection_element (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  element      INTEGER NOT NULL,         -- the vertex / channel index on the manifold
  code         INTEGER,                  -- labels: → dictionary_entry.code; indices: null
  PRIMARY KEY (selection_id, element)
) STRICT;
CREATE INDEX selection_element_code ON selection_element (selection_id, code);
CREATE TABLE selection_extent (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  axis         INTEGER NOT NULL,         -- → geometry_dimension.ordinal of the manifold's geometry
  start        REAL NOT NULL,
  stop         REAL NOT NULL CHECK (stop >= start),
  PRIMARY KEY (selection_id, axis)
) STRICT;
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
CREATE INDEX selection_measurement_tile ON selection_measurement (selection_id, code, measure);
CREATE INDEX selection_measurement_level ON selection_measurement (selection_id, measure, value);
CREATE INDEX selection_measurement_within ON selection_measurement (selection_id, within_selection_id, within_code, measure, value);
CREATE TABLE selection_span (
  selection_id TEXT NOT NULL REFERENCES selection (id) ON DELETE CASCADE,
  ordinal      INTEGER NOT NULL,
  start        REAL NOT NULL,
  stop         REAL NOT NULL CHECK (stop >= start),
  code         INTEGER,                  -- the member's level (an event type), read with the set's dictionary
  PRIMARY KEY (selection_id, ordinal)
) STRICT;
CREATE INDEX selection_span_start ON selection_span (selection_id, start);
CREATE TABLE field (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
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
CREATE TABLE operator (
  id              TEXT PRIMARY KEY,      -- UUID, in the node's zarr.json
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
CREATE TRIGGER dataset_ai AFTER INSERT ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'dataset', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER dataset_au AFTER UPDATE ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'dataset', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER dataset_ad AFTER DELETE ON dataset BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'dataset', OLD.id, 'delete', OLD.path WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'dataset' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_ai AFTER INSERT ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_au AFTER UPDATE ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', NEW.id, 'upsert' WHERE NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = NEW.id AND j.op = 'upsert' AND j.done_utc IS NULL); END;
CREATE TRIGGER subject_ad AFTER DELETE ON subject BEGIN INSERT INTO sync_job (table_name, row_id, op, path) SELECT 'subject', OLD.id, 'delete', ds.path || '/' || OLD.path FROM dataset ds WHERE ds.id = OLD.dataset_id AND NOT EXISTS (SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = OLD.id AND j.op = 'delete' AND j.done_utc IS NULL); END;
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
LEFT JOIN manifold dm ON dm.id = m.derived_from_id
/* manifold_view(id,subject_id,name,path,type,label,session,is_primary,n_vertices,n_edges,n_faces,n_volumes,topology_id,geometry_id,component_of_id,component_ordinal,from_index,derived_from_id,partition_id,operator,comment,source,producer_json,created_utc,created_by,modified_utc,modified_by,topology_kind,rank,n_components,closed,geometry_form,coordinate_system,unit,space,component_of,derived_from,n_component_rows) */;
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
WHERE f.kind = 'recording'
/* recording_view(id,subject_id,name,path,session,label,description,manifold_id,data_type,unit,sensors_id,sensors,n_channels,time_id,time_line,n_samples,dt_s,sfreq_hz,duration_s,chunk_samples,chunks_id,n_chunks,created_utc,created_by,modified_utc,modified_by) */;
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
LEFT JOIN operator dop ON dop.id = o.derived_from_id
/* operator_view(id,subject_id,name,path,kind,label,session,description,from_manifold_id,to_manifold_id,from_cell,to_cell,from_selection_id,to_selection_id,from_value_type,to_value_type,from_coordinate_system_id,to_coordinate_system_id,layout,method,params_json,n_rows,n_cols,nnz,data_type,chunk_shape_json,codecs_json,fill_value_json,derived_from_id,inverts_id,inverse_kind,comment,source,producer_json,created_utc,created_by,modified_utc,modified_by,from_manifold,from_type,to_manifold,to_type,from_selection,from_n_elements,to_selection,from_coordinate_system,to_coordinate_system,derived_from) */;
CREATE VIEW selection_view AS
SELECT s.*, m.name AS manifold, m.type AS manifold_type, d.name AS dictionary, d.scope AS dictionary_scope,
       ps.name AS member_of, f.name AS of_field, f.kind AS of_field_kind,
       (s.type = 'set' AND s.array_path IS NOT NULL) AS is_partition,
       (SELECT count(*) FROM selection_measurement x WHERE x.selection_id = s.id) AS n_measurements
FROM selection s
JOIN manifold m ON m.id = s.manifold_id
LEFT JOIN dictionary d ON d.id = s.dictionary_id
LEFT JOIN selection ps ON ps.id = s.member_of_id
LEFT JOIN field f ON f.id = s.of_field_id
/* selection_view(id,subject_id,name,path,type,label,session,description,manifold_id,space_id,cell,of_field_id,member_of_id,member_ordinal,code,dictionary_id,n_elements,n_members,array_path,center,position,coordinate_system_id,radius,unit,weight_fn,params_json,created_utc,created_by,modified_utc,modified_by,manifold,manifold_type,dictionary,dictionary_scope,member_of,of_field,of_field_kind,is_partition,n_measurements) */;
CREATE VIEW field_view AS
SELECT f.*, m.name AS manifold, m.type AS manifold_type, m.n_vertices AS n_vertices, ms.name AS manifold_subject, ms.kind AS manifold_subject_kind,
       s.kind AS subject_kind, of_f.name AS of_field, of_o.name AS of_operator,
       (SELECT count(*) FROM field_contribution c WHERE c.field_id = f.id) AS n_contributions
FROM field f
JOIN subject s ON s.id = f.subject_id
JOIN manifold m ON m.id = f.manifold_id
JOIN subject ms ON ms.id = m.subject_id
LEFT JOIN field of_f ON of_f.id = f.of_field_id
LEFT JOIN operator of_o ON of_o.id = f.of_operator_id
/* field_view(id,subject_id,name,path,kind,label,session,description,manifold_id,cell,value_type,n_components,unit,data_type,chunk_shape_json,codecs_json,fill_value_json,of_field_id,of_operator_id,by_operator_id,function,derived_from_id,comment,source,producer_json,created_utc,created_by,modified_utc,modified_by,manifold,manifold_type,n_vertices,manifold_subject,manifold_subject_kind,subject_kind,of_field,of_operator,n_contributions) */;
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
      UNION SELECT DISTINCT subject_id, session FROM selection WHERE session IS NOT NULL) x
/* session_view(subject_id,session,n_recordings,n_kernels,n_projections,n_sensor_layouts,duration_s) */;
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
        AND (SELECT count(*) FROM operator o WHERE o.subject_id = s.id AND o.kind LIKE 'sensorsTo%') > 0) AS is_warm,
       s.source_format, s.source_path, s.created_utc, s.modified_utc
FROM subject s
/* subject_status(id,dataset_id,name,path,kind,template_id,n_sessions,n_recordings,n_surfaces,n_volumes,n_manifolds,n_fields,n_operators,n_selections,n_eigen_axes,n_masses,n_projections,has_primary_surface,is_warm,source_format,source_path,created_utc,modified_utc) */;
