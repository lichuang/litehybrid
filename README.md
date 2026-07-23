# litehybrid

The hybrid search engine for SQLite-powered AI agents.

**Vector + full-text + scalar search, all in a single SQLite file.**

> **Status:** Phase 1 complete — loadable SQLite extension with brute-force (Flat)
> vector search via a writable virtual table.

## Features (Phase 1)

- Loadable SQLite extension (`litehybrid-ext`)
- Writable virtual table: `CREATE VIRTUAL TABLE ... USING litehybrid(...)`
- Flat (brute-force) vector index
- Distance metrics: L2, Cosine, Dot
- `vec_f32(text)` scalar helper for human-readable vector literals
- All data stored in SQLite shadow tables — persistence and ACID by default

## Build

```bash
# Build the loadable extension (.dylib on macOS, .so on Linux)
cargo build -p litehybrid-ext --features extension
```

The extension artifact is written to `target/debug/liblitehybrid_ext.dylib`
(macOS) or `target/debug/liblitehybrid_ext.so` (Linux).

## Usage

```bash
sqlite3
```

```sql
.load target/debug/liblitehybrid_ext

CREATE VIRTUAL TABLE idx USING litehybrid(dim=3, metric='l2');

INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'));
INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'));
INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'));

SELECT rowid, distance
FROM idx
WHERE embedding = vec_f32('[1.0, 0.1, 0.1]')
LIMIT 2;
```

Close and reopen the database — the vectors and index remain available without
re-inserting data.

## Development

```bash
# Run all tests
cargo test --all

# Format and lint
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
```

## Project Structure

```
crates/
  litehybrid-vec/    # Vector types, metrics, Flat index
  litehybrid-text/   # Full-text search placeholder (Phase 2)
  litehybrid-core/   # Hybrid orchestration facade
  litehybrid-ext/    # SQLite loadable extension
```

## Roadmap

- **Phase 1** ✅ SQLite loadable extension with Flat vector search
- **Phase 2** Dynamic sqlite-vec-style schema, metadata filtering
- **Phase 3** int8 / bit vector support

See [`doc/phase.md`](doc/phase.md) for the full implementation plan.

## License

Apache-2.0
