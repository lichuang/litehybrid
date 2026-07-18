//! litehybrid-core
//!
//! The core hybrid search engine. This crate is database-agnostic and can be
//! tested independently of SQLite.

#![deny(missing_docs)]

pub mod types;

pub use types::{Metric, RowId, ScoredRowId, SearchResult, VectorQuery};
