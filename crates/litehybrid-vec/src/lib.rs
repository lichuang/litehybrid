//! litehybrid-vec
//!
//! Vector search engine for litehybrid. Provides distance metrics and
//! SQLite-backed vector indexing primitives used by the hybrid search layer.

#![deny(missing_docs)]

pub mod index;
pub mod metrics;
pub mod serialize;
pub mod types;

pub use index::{FlatIndex, IndexError, VectorIndex};
pub use metrics::{Metric, cosine_distance, dot_distance, l2_distance};
pub use rusqlite::Connection;
pub use serialize::{SerializationError, deserialize_vector};
pub use types::{RowId, ScoredRowId, SearchResult, Vector, VectorElementType, VectorQuery};
