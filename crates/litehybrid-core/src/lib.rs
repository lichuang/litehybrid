//! litehybrid-core
//!
//! Hybrid search orchestration layer. Combines vector search from
//! `litehybrid-vec` and full-text search from `litehybrid-text` into a
//! unified search interface.

#![deny(missing_docs)]

pub mod index;

pub use index::{HybridIndex, VectorIndexKind};
pub use litehybrid_vec::{
  Metric, RowId, ScoredRowId, SearchResult, SerializationError, Vector, VectorElementType, VectorQuery,
  deserialize_vector,
};
