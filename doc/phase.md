# litehybrid Phase 1 Implementation Plan

> Phase 1 goal: build a loadable SQLite extension with a writable virtual table that persists vectors in a SQLite shadow table and performs brute-force (Flat) vector search.
>
> Core principle: **all data lives in SQLite. No in-memory persistent state.**
>
> Architecture:
> - `litehybrid-vec`: SQLite-aware vector index engine. Manages shadow tables, insert/delete/search.
> - `litehybrid-text`: full-text search engine (placeholder in Phase 1).
> - `litehybrid-core`: hybrid search orchestration (thin wrapper in Phase 1).
> - `litehybrid-ext`: SQLite virtual-table adapter. Forwards SQL operations to `litehybrid-vec`.
>
> Decisions: use **rusqlite**, start with **vector-only** search, split into **vec / text / core / ext** crates.

---

## Phase 1.0 — Workspace & Crate Bootstrap

- [x] Create Rust workspace `Cargo.toml` with shared `[workspace.package]` metadata.
- [x] Create `crates/litehybrid-vec/Cargo.toml`.
  - Vector search crate: metrics, vector types, Flat index.
  - Depends on `rusqlite`.
- [x] Create `crates/litehybrid-text/Cargo.toml`.
  - Placeholder crate for full-text search.
- [x] Create `crates/litehybrid-core/Cargo.toml`.
  - Hybrid orchestration crate.
  - Depends on `litehybrid-vec` and `litehybrid-text`.
- [x] Create `crates/litehybrid-ext/Cargo.toml`.
  - `crate-type = ["cdylib"]`.
  - Depends on `litehybrid-core`.
  - Depends on `rusqlite = { version = "0.40.1", features = ["vtab", "loadable_extension"] }`.
  - Note: `loadable_extension` is deferred; manual extension entry point will be used.
- [x] Move existing `types.rs` and `metrics.rs` into `litehybrid-vec`.
- [x] Update `litehybrid-core/src/lib.rs` to re-export vector types from `litehybrid-vec`.
- [x] Add cross-crate dependencies:
  - `litehybrid-core` → `litehybrid-vec`
  - `litehybrid-core` → `litehybrid-text`
  - `litehybrid-ext` → `litehybrid-core`
- [x] Add top-level `.gitignore` entries for Rust if missing (`/target`, `Cargo.lock`, `*.dylib`, `*.so`).
- [x] Run `cargo build` on the workspace and verify all crates compile.

---

## Phase 1.1 — Core Vector Types (`litehybrid-vec/src/types.rs`)

- [x] Define `RowId` as `pub type RowId = i64`.
- [x] Define `ScoredRowId` struct:
  ```rust
  pub struct ScoredRowId {
      pub rowid: RowId,
      pub score: f32,
  }
  ```
- [x] Define `VectorQuery` struct:
  ```rust
  pub struct VectorQuery {
      pub vector: Vec<f32>,
      pub topk: usize,
  }
  ```
- [x] Define `SearchResult` struct:
  ```rust
  pub struct SearchResult {
      pub hits: Vec<ScoredRowId>,
  }
  ```
- [x] Export all types from `litehybrid-vec/src/lib.rs`.
- [x] Re-export vector types from `litehybrid-core/src/lib.rs`.

---

## Phase 1.2 — Distance Metrics (`litehybrid-vec/src/metrics.rs`)

- [x] Define `Metric` enum in `metrics.rs`: `L2`, `Cosine`, `Dot`.
- [x] Implement `Metric::distance(self, a, b)` method dispatching to the concrete metric.
- [x] Implement `l2_distance(a, b)` returning squared Euclidean distance.
- [x] Implement `cosine_distance(a, b)` returning `1 - cosine_similarity`.
- [x] Implement `dot_distance(a, b)` returning negative dot product (so smaller is better, consistent with L2/cosine).
- [x] Add dimension mismatch guard panicking on mismatched lengths.
- [x] Add unit tests in `litehybrid-vec/src/metrics.rs` for the three metrics and `Metric::distance`.

---

## Phase 1.3 — VectorIndex Trait & SQLite-Backed FlatIndex

### Common abstractions (`litehybrid-vec/src/index/mod.rs`)

- [x] Define `IndexError` enum:
  ```rust
  pub enum IndexError {
      DimensionMismatch { expected: usize, got: usize },
      NotFound(RowId),
      Sqlite(rusqlite::Error),
  }
  ```
- [x] Implement `Display`, `Error`, and `From<rusqlite::Error>` for `IndexError`.
- [x] Define `VectorIndex` trait:
  ```rust
  pub trait VectorIndex {
      fn insert(&self, db: &Connection, rowid: RowId, vector: &[f32]) -> Result<(), IndexError>;
      fn delete(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError>;
      fn search(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError>;
  }
  ```

### FlatIndex (`litehybrid-vec/src/index/flat.rs`)

- [x] Define `FlatIndex` struct:
  ```rust
  pub struct FlatIndex {
      table_name: String,
      dim: usize,
      metric: Metric,
  }
  ```
- [x] Implement `VectorIndex` for `FlatIndex`:
  - `insert`: validate dimension, serialize vector to BLOB, `INSERT OR REPLACE` into shadow table.
  - `delete`: `DELETE FROM ... WHERE rowid = ?`, return `NotFound` if no row deleted.
  - `search`: read all vectors from shadow table, compute distances, return top-k with a binary max-heap.
- [x] Implement constructor `FlatIndex::create(db, table_name, dim, metric)`:
  - Creates shadow table `<table_name>_litehybrid_flat(rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)`.
- [x] Add helper function `serialize_vector(Vec<f32>) -> Vec<u8>` (little-endian `f32` bytes).
- [x] Add helper function `deserialize_blob(&[u8], expected_dim) -> Vec<f32>`.
- [x] Add unit tests using an in-memory SQLite connection (`Connection::open_in_memory`):
  - Insert vectors and search returns correct top-k ordering.
  - Dimension mismatch returns error.
  - Delete removes vector from subsequent searches.
  - Duplicate rowid insert overwrites old vector.

---

## Phase 1.4 — HybridIndex Facade (`litehybrid-core/src/index.rs`)

- [x] Create `litehybrid-core/src/index.rs`.
- [x] Define `HybridIndex` struct wrapping a `Box<dyn VectorIndex>`:
  ```rust
  pub struct HybridIndex {
      vector: Box<dyn VectorIndex>,
  }
  ```
- [x] Define `VectorIndexKind` enum (Phase 1: `Flat`).
- [x] Implement `HybridIndex::create(db, table_name, dim, metric, kind) -> Self`.
  - For Phase 1, instantiate `FlatIndex` when `kind == VectorIndexKind::Flat`.
- [x] Implement `insert_vector(&self, db, rowid, vector)` delegating to the trait.
- [x] Implement `delete_vector(&self, db, rowid)` delegating to the trait.
- [x] Implement `search_vector(&self, db, query) -> SearchResult` delegating to the trait.
- [x] Export `HybridIndex` from `litehybrid-core/src/lib.rs`.
- [x] Add unit test for insert/search/delete through `HybridIndex`.

---

## Phase 1.5 — Writable SQLite Virtual Table (`litehybrid-ext/src/lib.rs`)

- [x] Define `LitehybridVTab` struct holding a raw SQLite db pointer and an `Arc<HybridIndex>`.
- [x] Implement `VTab` trait:
  - `connect` parses `dim`, `metric`, and `index` from arguments.
  - Declares schema `CREATE TABLE x(embedding BLOB, distance REAL HIDDEN, k INT HIDDEN)` (rowid is implicit).
  - Calls `HybridIndex::create` to create the shadow table.
- [x] Implement `CreateVTab` trait with `KIND = VTabKind::Default`.
- [x] Implement `UpdateVTab` trait:
  - `insert(args)`: extract rowid and embedding BLOB, call `HybridIndex::insert_vector`.
  - `delete(value)`: extract rowid, call `HybridIndex::delete_vector`.
  - `update(args)`: treat as delete + insert for the same rowid.
- [x] Define `LitehybridCursor` struct holding search results and current position.
- [x] Implement `VTabCursor`:
  - `filter` handles the query constraint on `embedding` (using `=` in Phase 1; `MATCH` requires extra work).
  - Calls `HybridIndex::search_vector` and stores results.
  - `next`, `eof`, `column`, `rowid` iterate results.
- [x] Implement extension entry point `sqlite3_extension_init` registering module `litehybrid`.
- [x] Gate the loadable-extension entry point behind a Cargo feature (`extension`) so that
  the workspace can keep using `Connection::open_in_memory` in unit tests.
- [x] Add a fallback `sqlite3_extension_init` when `extension` is disabled that returns
  `SQLITE_ERROR` and prints the correct rebuild command (`cargo build -p litehybrid-ext --features extension`).
- [x] Add unit tests inside `litehybrid-ext` (gated by `not(feature = "extension")`) that
  register the module on an in-memory connection.

---

## Phase 1.6 — Argument Parsing in `connect`

- [x] Parse `dim=<usize>` from vtab arguments.
- [x] Parse `metric=<string>` supporting `l2`, `cosine`, `dot` (with optional quotes).
- [x] Parse `index=<string>` supporting `flat` (default `flat`).
- [x] Return `SQLITE_ERROR` with a clear message on invalid or missing arguments.
- [x] Store parsed `dim`, `metric`, and index kind inside `LitehybridVTab`.

---

## Phase 1.7 — `vec_f32` Scalar Helper

- [x] Register scalar function `vec_f32(text)` in `sqlite3_extension_init`:
  - Parse a string like `'[1.0, 2.0, 3.0]'` into `Vec<f32>`.
  - Return as a BLOB of little-endian `f32` values.
- [x] Add unit test for the parser.
- [x] Update manual tests to use `vec_f32(...)`.

---

## Phase 1.8 — Build, Load, and Manual Test

- [x] Run `cargo build -p litehybrid-ext --features extension`.
- [x] Load in `sqlite3` CLI and run:
  ```sql
  .load target/debug/liblitehybrid_ext
  CREATE VIRTUAL TABLE idx USING litehybrid(dim=3, metric='l2');
  INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'));
  SELECT rowid, distance FROM idx WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') LIMIT 2;
  ```
- [x] Verify the extension loads and returns correct top-k results.
- [x] Verify persistence: close `sqlite3`, reopen, run `SELECT` without re-inserting, and confirm results are identical.

---

## Phase 1.9 — Cleanup and Documentation

- [x] Run `cargo fmt --all -- --check`.
- [x] Run `cargo clippy --all-features -- -D warnings`.
- [x] Run `cargo test` for `litehybrid-vec`, `litehybrid-core`, and `litehybrid-ext`.
- [x] Update `README.md` with:
  - Project one-liner.
  - Build instructions.
  - Phase 1 usage example.
- [x] Update this `doc/phase.md` to mark completed steps.

---

## Out of Scope for Phase 1

The following are intentionally deferred to Phase 2:

- `litehybrid-text` full implementation (FTS5 integration).
- Scalar metadata / `WHERE` filters in `litehybrid-core`.
- Hybrid fusion (RRF / weighted sum) in `litehybrid-core`.
- Advanced index types (IVF, HNSW).
- Pro / free feature split.
- Multi-language bindings.

---

## Definition of Done for Phase 1

> A user can build `litehybrid-ext`, load it in `sqlite3`, create a virtual table with `USING litehybrid(dim=..., metric='...')`, insert vectors, close and reopen the database, and run a `MATCH` query that returns the nearest neighbors in the correct order without re-inserting data.


---

## Phase 2 — Dynamic Schema & Metadata Filtering (sqlite-vec-style)

> Phase 2 goal: evolve the virtual table from a hard-coded vector-only schema to a dynamically-declared, self-contained table aligned with the `sqlite-vec` API style. Support one vector column plus scalar metadata columns that can be filtered during KNN search.
>
> Core principle: **the user declares columns inside `USING litehybrid(...)`, and the extension dynamically generates the virtual table schema and backing shadow tables.**

Example target SQL:

```sql
CREATE VIRTUAL TABLE items USING litehybrid(
  embedding float[384],
  category text,
  year int
);

INSERT INTO items(rowid, embedding, category, year)
VALUES (1, vec_f32('[0.1, ...]'), 'tech', 2024);

SELECT rowid, distance
FROM items
WHERE embedding MATCH vec_f32('[0.1, ...]')
  AND category = 'tech'
  AND year > 2020
ORDER BY distance
LIMIT 10;
```

---

### Phase 2.0 — Column Declaration Parser

Location: `litehybrid-ext/src/vtab.rs`

- [ ] Replace the current key-value `parse_arguments` with a scanner/tokenizer that understands sqlite-vec-style pseudo-column declarations.
- [ ] Supported syntax (Phase 2 subset):
  - `name float[N]` — vector column (`N` is the dimension).
  - `name int8[N]` — int8 vector column (Phase 2 recognizes the syntax; full serialization is Phase 3).
  - `name bit[N]` — bit vector column (Phase 2 recognizes the syntax; full serialization is Phase 3).
  - `name text` — scalar metadata column.
  - `name integer` — scalar metadata column.
  - `name real` — scalar metadata column.
- [ ] Define internal `ColumnDecl` enum/struct capturing:
  - column name
  - SQLite storage type (`BLOB`, `TEXT`, `INTEGER`, `REAL`)
  - role: `Vector { dim, element_type }` or `Metadata { sql_type }`
- [ ] Validate that exactly one vector column is declared in Phase 2.
- [ ] Return clear `SQLITE_ERROR` messages for unknown types or malformed declarations.

---

### Phase 2.1 — Dynamic Schema Generation

Location: `litehybrid-ext/src/vtab.rs` (`connect`)

- [ ] Build the virtual table `CREATE TABLE x(...)` string dynamically from `ColumnDecl`s.
- [ ] Vector columns are declared as `BLOB` in the virtual table schema (the `float[N]` syntax is parsed but mapped to SQLite `BLOB`).
- [ ] Metadata columns keep their SQLite type (`TEXT`, `INTEGER`, `REAL`).
- [ ] Append HIDDEN columns:
  - `distance REAL HIDDEN`
  - `k INT HIDDEN`
- [ ] Persist the parsed column definitions in `LitehybridVTab` so that `xBestIndex`, `xUpdate`, and `xColumn` can reference them by index.

Example generated schema for:
```sql
CREATE VIRTUAL TABLE items USING litehybrid(embedding float[384], category text, year int);
```

is:
```sql
CREATE TABLE items(
  embedding BLOB,
  category TEXT,
  year INT,
  distance REAL HIDDEN,
  k INT HIDDEN
);
```

---

### Phase 2.2 — Shadow Table Schema for Metadata

Location: `litehybrid-vec/src/index/flat.rs` and related storage code

- [ ] Introduce an `{table}_litehybrid_info(key TEXT PRIMARY KEY, value ANY)` shadow table to store:
  - schema version
  - vector dimension
  - metric
  - serialized column definitions (so schema can be reconstructed on reconnect)
- [ ] Keep the existing `{table}_litehybrid_flat(rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)` for vector storage.
- [ ] Add `{table}_litehybrid_metadata(rowid INTEGER PRIMARY KEY, col_0, col_1, ...)` to store scalar metadata columns.
  - One column per declared metadata column.
  - Use SQLite native types.
- [ ] On `FlatIndex::create`, create all three shadow tables.
- [ ] On `FlatIndex::open` / reconnect, read `{table}_litehybrid_info` and validate that the declared columns match the stored schema.

---

### Phase 2.3 — Insert / Update / Delete with Metadata

Location: `litehybrid-ext/src/vtab.rs` (`UpdateVTab`)

- [ ] `insert`: parse vector BLOB from the vector column and metadata values from remaining columns; insert into all shadow tables atomically inside the same SQLite transaction.
- [ ] `delete`: delete the row from `{table}_litehybrid_flat` and `{table}_litehybrid_metadata`.
- [ ] `update`: implement as delete + insert for the same rowid, preserving metadata columns when not changed.
- [ ] Validate vector dimension and BLOB length on insert/update.

---

### Phase 2.4 — `xBestIndex` for Vector + Metadata Constraints

Location: `litehybrid-ext/src/vtab.rs` (`best_index`)

- [ ] Identify the vector column constraint (`embedding MATCH ?` or `embedding = ?`) and mark it as the KNN driver.
- [ ] Identify optional `k = ?` constraint.
- [ ] Identify metadata column constraints (`=`, `!=`, `<`, `<=`, `>`, `>=`).
- [ ] Encode which constraints are used into `idxNum` / `idxStr` so that `xFilter` receives the right arguments in order.
- [ ] Reject queries that do not have a vector column constraint.
- [ ] Set `estimated_cost` appropriately.

---

### Phase 2.5 — `xFilter` with Metadata Filtering

Location: `litehybrid-ext/src/vtab.rs` (`LitehybridCursor::filter`)

- [ ] Parse the query vector BLOB and optional `k`.
- [ ] Pass metadata constraints down to the index layer.
- [ ] Extend `VectorQuery` (or add a new query type) to carry:
  - query vector
  - top-k
  - metadata filters
- [ ] In `FlatIndex::search`:
  - Read vectors from `{table}_litehybrid_flat`.
  - For each candidate rowid, look up metadata in `{table}_litehybrid_metadata` and apply filters.
  - Only score rows that pass metadata filters.
  - Return top-k scored rowids.
- [ ] Optional optimization: keep metadata in the same shadow table row or use a single `SELECT ... WHERE rowid IN (...)` batch lookup instead of N point queries.

---

### Phase 2.6 — `xColumn` and Result Reading

- [ ] `xColumn` for vector column returns `NULL` (same as Phase 1).
- [ ] `xColumn` for metadata columns reads from cached search result + metadata lookup.
- [ ] `xColumn` for `distance` returns the score.
- [ ] `xColumn` for `k` returns the requested top-k.

---

### Phase 2.7 — `vec_f32` Scalar Helper

- [ ] Move Phase 1.7 (`vec_f32(text)`) into Phase 2 if not already done.
- [ ] Register `vec_f32(text)` as a scalar SQL function in `sqlite3_extension_init`.
- [ ] Parse `'[1.0, 2.0, 3.0]'` into little-endian `f32` BLOB.
- [ ] Add unit tests.

---

### Phase 2.8 — Tests & Documentation

- [ ] Unit tests for the column declaration parser.
- [ ] Unit tests for dynamic schema generation.
- [ ] Integration test: create table with `embedding float[3], category text`, insert rows, query with metadata filter, verify results.
- [ ] Integration test: update metadata without changing vector.
- [ ] Integration test: delete row removes it from metadata-filtered queries.
- [ ] Update `README.md` with Phase 2 usage example.
- [ ] Update this `doc/phase.md` to mark completed steps.

---

## Out of Scope for Phase 2

The following are intentionally deferred to later phases:

- Multiple vector columns in one virtual table.
- Partition keys.
- Auxiliary columns (`+contents text` style).
- Full-text search (`litehybrid-text` / FTS5 integration).
- Hybrid fusion across vector + text (RRF / weighted sum).
- Advanced index types (IVF, HNSW).
- Pro / free feature split.
- Multi-language bindings.

---

## Definition of Done for Phase 2

> A user can build `litehybrid-ext`, load it in `sqlite3`, create a virtual table with `USING litehybrid(embedding float[N], ...)` declaring one vector column and multiple scalar metadata columns, insert rows, and run a `WHERE embedding MATCH vec_f32('[...]') AND metadata_col = 'value'` query that returns correctly filtered nearest neighbors ordered by distance.


---

## Phase 3 — Multi-Element Vector Types (int8 / bit)

> Phase 3 goal: extend `litehybrid` to support vector columns whose elements are not only `float32`, but also `int8` and `bit`, matching `sqlite-vec`'s `vec0(embedding int8[768])` and `vec0(embedding bit[256])` capabilities.
>
> Core principle: **the virtual table API stays the same; only the internal serialization, distance kernels, and scalar constructor functions change per element type.**

Example target SQL:

```sql
-- int8 vectors
CREATE VIRTUAL TABLE items_i8 USING litehybrid(embedding int8[384], category text);

INSERT INTO items_i8(rowid, embedding, category)
VALUES (1, vec_int8('[10, -20, 30, ...]'), 'tech');

-- bit vectors
CREATE VIRTUAL TABLE items_bit USING litehybrid(embedding bit[256], category text);

INSERT INTO items_bit(rowid, embedding, category)
VALUES (1, vec_bit('[1, 0, 1, 1, 0, ...]'), 'tech');
```

---

### Phase 3.0 — Vector Element Type Abstraction

Location: `litehybrid-vec/src/types.rs`

- [ ] Introduce `VectorElementType` enum: `F32`, `Int8`, `Bit`.
- [ ] Update `VectorQuery` to carry both the element type and the raw query data.
- [ ] Introduce a `Vector` enum or generic container that can hold:
  - `Vec<f32>` for `F32`
  - `Vec<i8>` for `Int8`
  - `BitVec` or `Vec<u8>` packed bits for `Bit`
- [ ] Update `ColumnDecl::Vector` to include `element_type: VectorElementType`.

---

### Phase 3.1 — Serialization

Location: `litehybrid-vec/src/index/flat.rs` and serialization helpers

- [ ] `F32`: little-endian 4 bytes per element (already implemented).
- [ ] `Int8`: 1 signed byte per element.
- [ ] `Bit`: packed bits, 8 elements per byte, least-significant bit first (align with `sqlite-vec` if possible).
- [ ] Add validation: BLOB length must match `dim * element_size`.

---

### Phase 3.2 — Distance Metrics per Element Type

Location: `litehybrid-vec/src/metrics.rs`

- [ ] `F32`: L2, Cosine, Dot (already implemented).
- [ ] `Int8`: L2, Cosine, Dot over `&[i8]`.
- [ ] `Bit`: Hamming distance (popcount of XOR) and optionally Jaccard distance.
- [ ] Update `Metric::distance` to dispatch based on vector element type, returning a clear error on type/metric mismatch.

---

### Phase 3.3 — Scalar Constructor Functions

Location: `litehybrid-ext/src/lib.rs` (`sqlite3_extension_init`)

- [ ] `vec_int8(text)` — parse JSON-array string of integers into `Vec<i8>` BLOB.
- [ ] `vec_bit(text)` — parse JSON-array string of `0`/`1` into packed-bit BLOB.
- [ ] Ensure each function produces a BLOB with a distinguishable format/subtype so that downstream functions can validate element type without re-parsing.
- [ ] Add unit tests for both constructors.

---

### Phase 3.4 — Virtual Table Schema and Parsing

Location: `litehybrid-ext/src/vtab.rs`

- [ ] Column parser already recognizes `int8[N]` and `bit[N]` from Phase 2; now actually use the parsed `element_type`.
- [ ] Virtual table schema still declares vector columns as `BLOB` regardless of element type.
- [ ] Store `element_type` in `{table}_litehybrid_info` so reconnect can validate query vector type matches index type.

---

### Phase 3.5 — FlatIndex Support for int8 / bit

Location: `litehybrid-vec/src/index/flat.rs`

- [ ] `FlatIndex` stores `element_type` alongside `dim` and `metric`.
- [ ] On insert, validate incoming BLOB matches the index's element type and dimension.
- [ ] On search, deserialize stored vectors according to `element_type` and run the matching distance kernel.
- [ ] Top-k logic remains the same.

---

### Phase 3.6 — Mixed-Type Constraints

- [ ] Reject `INSERT` of an `int8` vector into an `f32` index with a clear error.
- [ ] Reject `WHERE embedding MATCH vec_f32('[...]')` on an `int8` or `bit` index.
- [ ] Ensure `vec_int8` / `vec_bit` can still be used as plain scalar functions on non-indexed data if useful.

---

### Phase 3.7 — Tests & Documentation

- [ ] Unit tests for `int8` serialization/deserialization.
- [ ] Unit tests for `bit` packing/unpacking.
- [ ] Unit tests for `Int8` L2 / Cosine / Dot distances.
- [ ] Unit tests for `Bit` Hamming distance.
- [ ] Integration test: create `int8` index, insert, search, verify ordering.
- [ ] Integration test: create `bit` index, insert, search, verify Hamming ordering.
- [ ] Update `README.md` with `vec_int8` and `vec_bit` examples.
- [ ] Update this `doc/phase.md` to mark completed steps.

---

## Out of Scope for Phase 3

The following are intentionally deferred to later phases:

- Quantization-aware indexes (binary-quantized Flat, Product Quantization).
- Advanced approximate indexes (IVF, HNSW) for int8/bit vectors.
- Full-text search (`litehybrid-text` / FTS5 integration).
- Hybrid fusion across vector + text (RRF / weighted sum).
- Pro / free feature split.
- Multi-language bindings.

---

## Definition of Done for Phase 3

> A user can build `litehybrid-ext`, load it in `sqlite3`, create virtual tables with `embedding int8[N]` and `embedding bit[N]`, insert rows using `vec_int8('[...]')` and `vec_bit('[...]')`, and run KNN queries that return correctly ordered nearest neighbors using the appropriate distance metric for each element type.
