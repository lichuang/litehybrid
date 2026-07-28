# litehybrid

The hybrid search engine for SQLite-powered AI agents.

**Vector + full-text + scalar search, all in a single SQLite file.**

> **Status:** Phase 2 complete — sqlite-vec-style column declarations, dynamic
> schema generation, metadata filtering, and metadata value reading.

## Features

- Loadable SQLite extension (`litehybrid-ext`)
- Writable virtual table: `CREATE VIRTUAL TABLE ... USING litehybrid(...)`
- Flat (brute-force) vector index
- Distance metrics: L2, Cosine, Dot, Hamming
- Vector element types: `float[N]`, `int8[N]`, `bit[N]`
- Scalar metadata columns: `text`, `integer`, `real`
- Metadata filtering in `WHERE` clauses
- `vec_f32(text)`, `vec_int8(text)`, `vec_bit(text)` scalar helpers for human-readable vector literals
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

CREATE VIRTUAL TABLE idx USING litehybrid(embedding float[3], metric='l2');

INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'));
INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'));
INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'));

SELECT rowid, distance
FROM idx
WHERE embedding = vec_f32('[1.0, 0.1, 0.1]')
LIMIT 2;
```

### Metadata columns and filtering

```sql
CREATE VIRTUAL TABLE items USING litehybrid(
  embedding float[384],
  category text,
  year int
);

INSERT INTO items(rowid, embedding, category, year)
VALUES (1, vec_f32('[0.1, ...]'), 'tech', 2024);

INSERT INTO items(rowid, embedding, category, year)
VALUES (2, vec_f32('[0.2, ...]'), 'science', 2023);

SELECT rowid, category, year, distance
FROM items
WHERE embedding = vec_f32('[0.1, ...]')
  AND category = 'tech'
  AND year > 2020
ORDER BY distance
LIMIT 10;
```

Update metadata without re-specifying the vector:

```sql
UPDATE items SET category = 'updated' WHERE rowid = 1;
```

Delete a row:

```sql
DELETE FROM items WHERE rowid = 2;
```

### int8 and bit vectors

```sql
CREATE VIRTUAL TABLE items_i8 USING litehybrid(embedding int8[3]);
INSERT INTO items_i8(rowid, embedding) VALUES (1, vec_int8('[10, 0, 0]'));
SELECT rowid, distance FROM items_i8 WHERE embedding = vec_int8('[10, 1, 1]') LIMIT 2;

CREATE VIRTUAL TABLE items_bit USING litehybrid(embedding bit[4]);
INSERT INTO items_bit(rowid, embedding) VALUES (1, vec_bit('[1, 0, 0, 0]'));
SELECT rowid, distance FROM items_bit WHERE embedding = vec_bit('[1, 0, 1, 0]') LIMIT 2;
```

Close and reopen the database — the vectors, metadata, and index remain
available without re-inserting data.

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
- **Phase 2** ✅ Dynamic sqlite-vec-style schema, metadata filtering, and metadata reading
- **Phase 3** ✅ int8 / bit vector support

See [`doc/phase.md`](doc/phase.md) for the full implementation plan.

## License

Apache-2.0
