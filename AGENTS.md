# Agent Instructions

## Code Quality Requirements

All code changes must pass the following checks before they are considered complete:

```bash
cargo fmt --all -- --check
cargo clippy --all-features -- -D warnings
```

Run these commands locally and fix any reported issues before finishing a task.
