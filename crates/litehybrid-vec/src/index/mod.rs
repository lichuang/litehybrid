//! Vector index implementations.

pub mod flat;
pub(crate) mod topk;

pub use flat::FlatIndex;

use rusqlite::Connection;

use crate::{MetadataValue, Metric, RowId, SearchResult, SerializationError, Vector, VectorElementType, VectorQuery};

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
  /// The stored schema does not match the requested schema.
  SchemaMismatch {
    /// Expected schema value.
    expected: String,
    /// Actual schema value found in the info table.
    got: String,
  },
  /// The number of metadata values does not match the number of metadata columns.
  MetadataCountMismatch {
    /// Expected metadata value count.
    expected: usize,
    /// Actual metadata value count received.
    got: usize,
  },
  /// A metadata value does not match the declared scalar type of its column.
  MetadataTypeMismatch {
    /// Declared scalar type for the column.
    expected: crate::ScalarType,
    /// Actual metadata value that was supplied.
    got: crate::MetadataValue,
  },
  /// A stored metadata value could not be converted back to a typed metadata value.
  MetadataDeserialization {
    /// Column that caused the error.
    column: String,
    /// Error message from the conversion.
    message: String,
  },
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
      IndexError::SchemaMismatch { expected, got } => {
        write!(f, "schema mismatch: expected {}, got {}", expected, got)
      }
      IndexError::MetadataCountMismatch { expected, got } => {
        write!(f, "metadata count mismatch: expected {}, got {}", expected, got)
      }
      IndexError::MetadataTypeMismatch { expected, got } => {
        write!(
          f,
          "metadata type mismatch: expected {}, got {:?}",
          expected.as_str(),
          got
        )
      }
      IndexError::MetadataDeserialization { column, message } => {
        write!(f, "metadata deserialization failed for column {}: {}", column, message)
      }
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
  /// Insert or replace a vector and its metadata for the given rowid.
  fn insert(
    &self,
    db: &Connection,
    rowid: RowId,
    vector: &Vector,
    metadata: &[Option<MetadataValue>],
  ) -> Result<(), IndexError>;

  /// Delete the vector and metadata for the given rowid.
  fn delete(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError>;

  /// Search for the top-k nearest vectors.
  fn search(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError>;

  /// Read the stored metadata values for the given rowid.
  ///
  /// Returns one optional value per metadata column, in the same order as the
  /// index's metadata column declaration.
  fn read_metadata(&self, db: &Connection, rowid: RowId) -> Result<Vec<Option<MetadataValue>>, IndexError>;

  /// Return the rowid of every vector stored in the index.
  ///
  /// This is used for scans that do not have a vector query constraint, for
  /// example when SQLite needs to locate a row by rowid for UPDATE/DELETE.
  fn scan(&self, db: &Connection) -> Result<Vec<RowId>, IndexError>;

  /// Read the stored vector for the given rowid.
  fn read_vector(&self, db: &Connection, rowid: RowId) -> Result<Vector, IndexError>;
}
