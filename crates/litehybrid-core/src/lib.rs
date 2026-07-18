//! litehybrid-core
//!
//! Hybrid search orchestration layer. Combines vector search from
//! `litehybrid-vec` and full-text search from `litehybrid-text` into a
//! unified search interface.

#![deny(missing_docs)]

pub use litehybrid_vec::{Metric, RowId, ScoredRowId, SearchResult, VectorQuery};
