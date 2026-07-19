//! litehybrid-vec
//!
//! Vector search engine for litehybrid. Provides distance metrics and
//! SQLite-backed vector indexing primitives used by the hybrid search layer.

#![deny(missing_docs)]

pub mod index;
pub mod metrics;
pub mod types;

pub use index::{FlatIndex, IndexError, VectorIndex};
pub use metrics::{cosine_distance, distance, dot_distance, l2_distance};
pub use rusqlite::Connection;
pub use types::{Metric, RowId, ScoredRowId, SearchResult, VectorQuery};
