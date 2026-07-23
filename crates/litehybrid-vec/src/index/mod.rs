//! Vector index implementations.

pub mod flat;
pub(crate) mod topk;

pub use flat::FlatIndex;

use rusqlite::Connection;

use crate::{Metric, RowId, SearchResult, SerializationError, Vector, VectorElementType, VectorQuery};

/// Errors that can occur when operating on a vector index.
#[derive(Debug)]
pub enum IndexError {
  /// The provided vector dimension does not match the index dimension.
  DimensionMismatch {
    /// Expected dimension.
    expected: usize,
    /// Actual dimension received.
    got: usize,
  },
  /// The requested rowid was not found.
  NotFound(RowId),
  /// The requested vector element type is not supported by the index yet.
  UnsupportedElementType(VectorElementType),
  /// Two vectors have different element types.
  MismatchedElementTypes {
    /// Element type of the left-hand vector.
    left: VectorElementType,
    /// Element type of the right-hand vector.
    right: VectorElementType,
  },
  /// The requested metric is not valid for the vector element type.
  UnsupportedMetricForType {
    /// Metric that was requested.
    metric: Metric,
    /// Element type for which the metric is invalid.
    element_type: VectorElementType,
  },
  /// A vector BLOB could not be serialized or deserialized.
  Serialization(SerializationError),
  /// An underlying SQLite error.
  Sqlite(rusqlite::Error),
}

impl std::fmt::Display for IndexError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      IndexError::DimensionMismatch { expected, got } => {
        write!(f, "dimension mismatch: expected {}, got {}", expected, got)
      }
      IndexError::NotFound(rowid) => write!(f, "rowid {} not found", rowid),
      IndexError::UnsupportedElementType(ty) => write!(f, "unsupported vector element type: {:?}", ty),
      IndexError::MismatchedElementTypes { left, right } => {
        write!(f, "mismatched vector element types: {:?} vs {:?}", left, right)
      }
      IndexError::UnsupportedMetricForType { metric, element_type } => {
        write!(f, "metric {:?} is not supported for {:?} vectors", metric, element_type)
      }
      IndexError::Serialization(err) => write!(f, "serialization error: {}", err),
      IndexError::Sqlite(err) => write!(f, "sqlite error: {}", err),
    }
  }
}

impl std::error::Error for IndexError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      IndexError::Sqlite(err) => Some(err),
      _ => None,
    }
  }
}

impl From<rusqlite::Error> for IndexError {
  fn from(err: rusqlite::Error) -> Self {
    IndexError::Sqlite(err)
  }
}

impl From<SerializationError> for IndexError {
  fn from(err: SerializationError) -> Self {
    IndexError::Serialization(err)
  }
}

/// Common interface for all vector indexes.
///
/// Implementations include brute-force Flat indexes, IVF, HNSW, etc.
pub trait VectorIndex: Send + Sync {
  /// Insert or replace a vector for the given rowid.
  fn insert(&self, db: &Connection, rowid: RowId, vector: &Vector) -> Result<(), IndexError>;

  /// Delete the vector for the given rowid.
  fn delete(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError>;

  /// Search for the top-k nearest vectors.
  fn search(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError>;
}
