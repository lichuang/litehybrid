# Agent Instructions

## Project Overview

`litehybrid` is a hybrid search engine for SQLite-powered AI agents. It is
implemented as a Rust workspace that produces a loadable SQLite extension.

Long-term goal: **vector + full-text + scalar search, all in a single SQLite
file.**

Current status (Phase 1): loadable SQLite extension with a writable virtual
table and brute-force (Flat) vector search.

The implementation roadmap lives in [`doc/phase.md`](doc/phase.md). Read it
before starting non-trivial work.

## Workspace Structure

```
crates/
  litehybrid-vec/    # Vector types, distance metrics, index traits, Flat index
  litehybrid-text/   # Full-text search placeholder (Phase 2)
  litehybrid-core/   # Hybrid orchestration facade over vec/text
  litehybrid-ext/    # SQLite loadable extension entry point and vtab
```

Guidelines for where to add code:

- **New distance metrics / vector serialization** → `litehybrid-vec`
- **New index types** (IVF, HNSW, etc.) → `litehybrid-vec/src/index/`
- **Hybrid query orchestration / fusion** → `litehybrid-core`
- **SQLite virtual table logic / scalar SQL functions** → `litehybrid-ext`
- **FTS5 integration** → `litehybrid-text`

## Rust Toolchain

Pinned to stable **Rust 1.95.0** via `rust-toolchain.toml`. Do not use nightly
features.

## Code Style

Governed by `rustfmt.toml`:

- `max_width = 120`
- `chain_width = 100`
- `tab_spaces = 2`
- `reorder_imports = true`
- `merge_derives = false`

Always run `cargo fmt --all` before committing.

## Code Quality Requirements

All code changes must pass the following checks before they are considered
complete:

```bash
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
cargo test --all
```

If you modified the loadable extension entry point or scalar functions, also
verify:

```bash
cargo build -p litehybrid-ext --features extension
```

Run these commands locally and fix any reported issues before finishing a task.

## Build Commands

```bash
# Build all workspace crates
cargo build

# Build the actual SQLite loadable extension (.dylib / .so)
cargo build -p litehybrid-ext --features extension
```

The extension artifact is written to `target/debug/liblitehybrid_ext.dylib`
(macOS) or `target/debug/liblitehybrid_ext.so` (Linux).

## Testing

```bash
# Run all unit tests
cargo test --all

# Test only the extension crate (without the loadable_extension feature)
cargo test -p litehybrid-ext
```

The `extension` Cargo feature enables `rusqlite/loadable_extension`. It is
disabled by default because it conflicts with `Connection::open_in_memory` used
in unit tests.

## SQLite Virtual Table Conventions

- The virtual table module name is **`litehybrid`** (registered in
  `litehybrid-ext/src/lib.rs`).
- Virtual table schema generation currently lives in
  `litehybrid-ext/src/vtab.rs` (`connect`).
- Scalar SQL functions (e.g., `vec_f32`) are registered alongside the module in
  `register_module`.
- Keep all persistent state in SQLite shadow tables; do not rely on in-memory
  state across reconnections.

## Adding Dependencies

The workspace tries to stay dependency-light. Avoid adding new crates unless
necessary. If you do add one:

1. Add it to the workspace root `Cargo.toml` `[workspace.dependencies]`.
2. Reference it from the relevant crate `Cargo.toml`.
3. Prefer MIT / Apache-2.0 / BSD licenses to avoid GPL/LGPL contamination.

## Documentation and Planning

- Keep [`doc/phase.md`](doc/phase.md) up to date when completing planned steps.
- Do not fabricate project plans or decisions that are not documented in
  `doc/phase.md`, `README.md`, or the codebase. If you are inferring a design
  choice, clearly label it as inference, not as an established plan.
- Update `README.md` if user-facing usage or build instructions change.
- Update `AGENTS.md` if you change workflows, build steps, or conventions
  covered here.

## Commit Messages

Use conventional commit prefixes (`feat:`, `fix:`, `doc:`, `refactor:`, etc.).
When a task is complete, write a one-line English summary to the
`commit-message` file for the user to use.
