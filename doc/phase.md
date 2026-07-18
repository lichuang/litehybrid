# litehybrid Phase 1 Implementation Plan

> Phase 1 goal: build a loadable SQLite extension that exposes a read-only virtual table for brute-force (Flat) vector search.
>
> Architecture:
> - `litehybrid-vec`: vector search engine (metrics, Flat index)
> - `litehybrid-text`: full-text search engine (placeholder in Phase 1)
> - `litehybrid-core`: hybrid search orchestration (query, planner, fusion, scalar filters)
> - `litehybrid-ext`: SQLite virtual-table adapter
>
> Decisions: use **rusqlite**, start with **vector-only** search, split into **vec / text / core / ext** crates.

---

## Phase 1.0 — Workspace & Crate Bootstrap

- [x] Create Rust workspace `Cargo.toml` with shared `[workspace.package]` metadata.
- [x] Create `crates/litehybrid-vec/Cargo.toml`.
  - Vector search crate: metrics, vector types, Flat index.
  - No SQLite-specific dependencies.
- [x] Create `crates/litehybrid-text/Cargo.toml`.
  - Placeholder crate for full-text search; may be empty in Phase 1.
- [x] Create `crates/litehybrid-core/Cargo.toml`.
  - Hybrid orchestration crate.
  - Depends on `litehybrid-vec` and `litehybrid-text`.
- [x] Create `crates/litehybrid-ext/Cargo.toml`.
  - `crate-type = ["cdylib"]`.
  - Depends on `litehybrid-core`.
  - Depends on `rusqlite = { version = "0.40.1", features = ["vtab", "loadable_extension"] }`.
- [x] Move existing `types.rs` and `metrics.rs` from `litehybrid-core` into `litehybrid-vec`.
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

## Phase 1.3 — Storage Abstraction (`litehybrid-vec/src/storage.rs`)

- [ ] Define `Document` struct holding a single row:
  ```rust
  pub struct Document {
      pub rowid: RowId,
      pub vector: Vec<f32>,
  }
  ```
- [ ] Define `Storage` trait:
  ```rust
  pub trait Storage {
      fn insert(&mut self, doc: Document) -> Result<(), StorageError>;
      fn delete(&mut self, rowid: RowId) -> Result<(), StorageError>;
      fn get(&self, rowid: RowId) -> Option<&Document>;
      fn iter(&self) -> impl Iterator<Item = &Document>;
  }
  ```
- [ ] Implement `InMemoryStorage` using `HashMap<RowId, Document>`.
- [ ] Define `StorageError` enum.
- [ ] Add basic unit tests for insert/get/delete/iter.

---

## Phase 1.4 — Flat Vector Index (`litehybrid-vec/src/index/flat.rs`)

- [ ] Create module path `litehybrid-vec/src/index/flat.rs`.
- [ ] Define `FlatIndex` struct:
  ```rust
  pub struct FlatIndex<S: Storage> {
      storage: S,
      metric: Metric,
      dim: usize,
  }
  ```
- [ ] Implement constructor `FlatIndex::new(storage: S, metric: Metric, dim: usize) -> Self`.
- [ ] Implement `add(&mut self, rowid: RowId, vector: Vec<f32>)` validating dimension.
- [ ] Implement `remove(&mut self, rowid: RowId)`.
- [ ] Implement `search(&self, query: &VectorQuery) -> SearchResult`:
  - Iterate all stored vectors.
  - Compute distance with configured metric.
  - Keep top-k smallest distances using a binary heap or `Vec::sort_by`.
  - Return ordered results (best first).
- [ ] Add unit tests for search with known vectors and metrics.

---

## Phase 1.5 — `VectorIndex` Facade (`litehybrid-vec/src/index.rs`)

- [ ] Define `VectorIndex` struct wrapping `FlatIndex<InMemoryStorage>`:
  ```rust
  pub struct VectorIndex {
      flat: FlatIndex<InMemoryStorage>,
  }
  ```
- [ ] Implement `VectorIndex::new(metric: Metric, dim: usize) -> Self`.
- [ ] Implement `insert(&mut self, rowid: RowId, vector: Vec<f32>)`.
- [ ] Implement `delete(&mut self, rowid: RowId)`.
- [ ] Implement `search(&self, query: &VectorQuery) -> SearchResult`.
- [ ] Export `VectorIndex` from `litehybrid-vec/src/lib.rs`.
- [ ] Add integration test exercising insert + search end-to-end.

---

## Phase 1.6 — HybridIndex Facade (`litehybrid-core/src/index.rs`)

- [ ] Define `HybridIndex` struct wrapping `VectorIndex` from `litehybrid-vec`:
  ```rust
  pub struct HybridIndex {
      vector: VectorIndex,
  }
  ```
- [ ] Implement `HybridIndex::new(metric: Metric, dim: usize) -> Self`.
- [ ] Implement `insert_vector(&mut self, rowid: RowId, vector: Vec<f32>)`.
- [ ] Implement `delete_vector(&mut self, rowid: RowId)`.
- [ ] Implement `search_vector(&self, query: &VectorQuery) -> SearchResult`.
- [ ] Export `HybridIndex` from `litehybrid-core/src/lib.rs`.
- [ ] Add integration test exercising insert + search end-to-end.

---

## Phase 1.7 — SQLite Extension Adapter (`litehybrid-ext/src/lib.rs`)

- [ ] Define `LitehybridVTab` struct:
  ```rust
  #[repr(C)]
  pub struct LitehybridVTab {
      base: ffi::sqlite3_vtab,
      index: RefCell<HybridIndex>,
  }
  ```
- [ ] Implement `VTab` trait for `LitehybridVTab`:
  - `connect` parses arguments and returns schema `CREATE TABLE x(rowid INTEGER, embedding BLOB)`.
  - For Phase 1, accept arguments `dim=...` and `metric=...` (e.g. `metric='cosine'`).
  - `best_index` performs full scan.
  - `open` returns a cursor.
- [ ] Implement `CreateVTab` trait with `KIND = VTabKind::Default`.
- [ ] Define `LitehybridCursor` struct holding current rowid / results.
- [ ] Implement `VTabCursor`:
  - `filter` ignores arguments and triggers a full scan.
  - `next`, `eof`, `column`, `rowid` return rows from a static or seeded result set.
- [ ] **Important:** wire `MATCH` constraint so that `WHERE embedding MATCH vec_f32('[...]')` invokes `HybridIndex::search_vector`.
  - In `best_index`, detect `ConstraintOperator::MATCH` on column 1 (`embedding`).
  - Pass the vector bytes through `filter` args.
  - In `filter`, parse the BLOB into `Vec<f32>` and call `index.search_vector`.
- [ ] Implement extension entry point `sqlite3_extension_init` registering module `litehybrid0`.

---

## Phase 1.8 — Argument Parsing in `connect`

- [ ] Parse `dim=<usize>` from `VTabArguments::arguments`.
- [ ] Parse `metric=<string>` supporting `l2`, `cosine`, `dot`.
- [ ] Return `SQLITE_ERROR` with a clear message on invalid arguments.
- [ ] Store parsed `metric` and `dim` inside `LitehybridVTab`.

---

## Phase 1.9 — Build, Load, and Manual Test

- [ ] Run `cargo build -p litehybrid-ext`.
- [ ] Load in `sqlite3` CLI:
  ```sql
  .load target/debug/liblitehybrid_ext
  CREATE VIRTUAL TABLE idx USING litehybrid0(dim=3, metric='l2');
  INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'));
  INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'));
  SELECT rowid, distance FROM idx WHERE embedding MATCH vec_f32('[1.0, 0.1, 0.1]') LIMIT 2;
  ```
  > Note: `vec_f32` is not provided by this crate in Phase 1. Decide whether to also register a scalar helper or use raw BLOB literals for testing.
- [ ] Verify the extension loads and returns correct top-k results.

---

## Phase 1.10 — Scalar Helper `vec_f32` (Optional but Recommended)

- [ ] Register a scalar function `vec_f32(text)` in `sqlite3_extension_init`:
  - Parse a string like `'[1.0, 2.0, 3.0]'` into `Vec<f32>`.
  - Return as a BLOB of little-endian `f32` values.
- [ ] Add unit test for the parser.
- [ ] Update manual test to use `vec_f32(...)`.

---

## Phase 1.11 — Cleanup and Documentation

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
- Persistent storage to SQLite BLOB (keep `InMemoryStorage`).
- Writable virtual table (`INSERT` through vtab).
- Pro / free feature split.
- Multi-language bindings.

---

## Definition of Done for Phase 1

> A user can build `litehybrid-ext`, load it in `sqlite3`, create a virtual table with `USING litehybrid0(dim=..., metric='...')`, insert vectors, and run a `MATCH` query that returns the nearest neighbors in the correct order.
