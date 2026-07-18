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
- [ ] Add `rusqlite` dependency to `litehybrid-vec`.
  - `rusqlite = { version = "0.40.1", features = ["vtab"] }` (loadable_extension feature not required for core engine).
- [x] Create `crates/litehybrid-text/Cargo.toml`.
  - Placeholder crate for full-text search.
- [x] Create `crates/litehybrid-core/Cargo.toml`.
  - Depends on `litehybrid-vec` and `litehybrid-text`.
- [x] Create `crates/litehybrid-ext/Cargo.toml`.
  - `crate-type = ["cdylib"]`.
  - Depends on `litehybrid-core`.
  - Depends on `rusqlite = { version = "0.40.1", features = ["vtab", "loadable_extension"] }`.
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
- [x] Define `Metric` enum: `L2`, `Cosine`, `Dot`.
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

- [x] Define function signature:
  ```rust
  pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32;
  ```
- [x] Implement `l2_distance(a, b)` returning squared Euclidean distance.
- [x] Implement `cosine_distance(a, b)` returning `1 - cosine_similarity`.
- [x] Implement `dot_distance(a, b)` returning negative dot product (so smaller is better, consistent with L2/cosine).
- [x] Add dimension mismatch guard panicking on mismatched lengths.
- [x] Add unit tests in `litehybrid-vec/src/metrics.rs` for the three metrics.

---

## Phase 1.3 — SQLite-Backed Flat Vector Index (`litehybrid-vec/src/index/flat.rs`)

- [ ] Create module path `litehybrid-vec/src/index/flat.rs`.
- [ ] Define `IndexError` enum:
  ```rust
  pub enum IndexError {
      DimensionMismatch { expected: usize, got: usize },
      DuplicateRowId(RowId),
      NotFound(RowId),
      Sqlite(rusqlite::Error),
  }
  ```
- [ ] Define `FlatIndex` struct:
  ```rust
  pub struct FlatIndex {
      table_name: String,
      dim: usize,
      metric: Metric,
  }
  ```
- [ ] Implement constructor:
  ```rust
  pub fn create(
      db: &Connection,
      table_name: &str,
      dim: usize,
      metric: Metric,
  ) -> Result<Self, IndexError>
  ```
  - Creates shadow table `<table_name>_litehybrid_vectors(rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)`.
  - Returns configured `FlatIndex` instance.
- [ ] Implement `insert(&self, db: &Connection, rowid: RowId, vector: &[f32]) -> Result<(), IndexError>`:
  - Validate `vector.len() == self.dim`.
  - Serialize vector to BLOB (little-endian `f32` bytes).
  - Use `INSERT OR REPLACE INTO ... VALUES (?, ?)` so duplicate rowids overwrite.
- [ ] Implement `delete(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError>`:
  - Execute `DELETE FROM ... WHERE rowid = ?`.
  - Return `NotFound` if no row was deleted.
- [ ] Implement `search(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError>`:
  - Validate `query.vector.len() == self.dim`.
  - Query `SELECT rowid, embedding FROM <shadow_table>`.
  - Deserialize each BLOB to `Vec<f32>`.
  - Compute distance using `self.metric`.
  - Keep top-k smallest distances with a binary max-heap.
  - Return `SearchResult` ordered best-first.
- [ ] Add helper function `serialize_vector(Vec<f32>) -> Vec<u8>`.
- [ ] Add helper function `deserialize_blob(&[u8]) -> Vec<f32>`.
- [ ] Add unit tests using an in-memory SQLite connection (`Connection::open_in_memory`):
  - Insert vectors and search returns correct top-k ordering.
  - Dimension mismatch returns error.
  - Delete removes vector from subsequent searches.
  - Duplicate rowid insert overwrites old vector.

---

## Phase 1.4 — `VectorIndex` Facade (`litehybrid-vec/src/index.rs`)

- [ ] Create `litehybrid-vec/src/index.rs`.
- [ ] Define `VectorIndex` struct wrapping `FlatIndex`:
  ```rust
  pub struct VectorIndex {
      inner: FlatIndex,
  }
  ```
- [ ] Implement `VectorIndex::create(db, table_name, dim, metric)` delegating to `FlatIndex::create`.
- [ ] Implement `insert`, `delete`, `search` delegating to `FlatIndex`.
- [ ] Export `VectorIndex` and `FlatIndex` from `litehybrid-vec/src/lib.rs`.

---

## Phase 1.5 — HybridIndex Facade (`litehybrid-core/src/index.rs`)

- [ ] Create `litehybrid-core/src/index.rs`.
- [ ] Define `HybridIndex` struct wrapping `VectorIndex` from `litehybrid-vec`:
  ```rust
  pub struct HybridIndex {
      vector: VectorIndex,
  }
  ```
- [ ] Implement `HybridIndex::create(db, table_name, metric, dim) -> Self`.
- [ ] Implement `insert_vector(&self, db, rowid, vector)`.
- [ ] Implement `delete_vector(&self, db, rowid)`.
- [ ] Implement `search_vector(&self, db, query) -> SearchResult`.
- [ ] Export `HybridIndex` from `litehybrid-core/src/lib.rs`.

---

## Phase 1.6 — Writable SQLite Virtual Table (`litehybrid-ext/src/lib.rs`)

- [ ] Define `LitehybridVTab` struct:
  ```rust
  #[repr(C)]
  pub struct LitehybridVTab {
      base: ffi::sqlite3_vtab,
      index: HybridIndex,
  }
  ```
- [ ] Implement `VTab` trait:
  - `connect` parses `dim` and `metric` from arguments.
  - Declares schema `CREATE TABLE x(rowid INTEGER, embedding BLOB, distance HIDDEN, k HIDDEN)`.
  - Calls `HybridIndex::create` to create the shadow table.
- [ ] Implement `CreateVTab` trait with `KIND = VTabKind::Default`.
- [ ] Implement `UpdateVTab` trait:
  - `insert(args)`: extract rowid and embedding BLOB, call `HybridIndex::insert_vector`.
  - `delete(value)`: extract rowid, call `HybridIndex::delete_vector`.
  - `update(args)`: treat as delete + insert for the same rowid.
- [ ] Define `LitehybridCursor` struct holding search results and current position.
- [ ] Implement `VTabCursor`:
  - `filter` handles the `MATCH` constraint on `embedding`.
  - Calls `HybridIndex::search_vector` and stores results.
  - `next`, `eof`, `column`, `rowid` iterate results.
- [ ] Implement extension entry point `sqlite3_extension_init` registering module `litehybrid0`.

---

## Phase 1.7 — `vec_f32` Scalar Helper

- [ ] Register scalar function `vec_f32(text)` in `sqlite3_extension_init`:
  - Parse a string like `'[1.0, 2.0, 3.0]'` into `Vec<f32>`.
  - Return as a BLOB of little-endian `f32` bytes.
- [ ] Add unit test for the parser.
- [ ] Update manual tests to use `vec_f32(...)`.

---

## Phase 1.8 — Build, Load, and Manual Test

- [ ] Run `cargo build -p litehybrid-ext`.
- [ ] Load in `sqlite3` CLI and run:
  ```sql
  .load target/debug/liblitehybrid_ext
  CREATE VIRTUAL TABLE idx USING litehybrid0(dim=3, metric='l2');
  INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'));
  SELECT rowid, distance FROM idx WHERE embedding MATCH vec_f32('[1.0, 0.1, 0.1]') LIMIT 2;
  ```
- [ ] Verify the extension loads and returns correct top-k results.
- [ ] Verify persistence: close `sqlite3`, reopen, run `SELECT` without re-inserting, and confirm results are identical.

---

## Phase 1.9 — Cleanup and Documentation

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo clippy --all-features -- -D warnings`.
- [ ] Run `cargo test` for `litehybrid-vec`, `litehybrid-core`, and `litehybrid-ext`.
- [ ] Update `README.md` with:
  - Project one-liner.
  - Build instructions.
  - Phase 1 usage example.
- [ ] Update this `doc/phase.md` to mark completed steps.

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

> A user can build `litehybrid-ext`, load it in `sqlite3`, create a virtual table with `USING litehybrid0(dim=..., metric='...')`, insert vectors, close and reopen the database, and run a `MATCH` query that returns the nearest neighbors in the correct order without re-inserting data.
