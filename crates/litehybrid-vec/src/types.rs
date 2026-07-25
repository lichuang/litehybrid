//! Core types used throughout the litehybrid search engine.

/// SQLite rowid type. All indexed documents are identified by this value.
pub type RowId = i64;

/// Element type stored in a vector column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorElementType {
  /// 32-bit IEEE 754 floating point values.
  F32,
  /// 8-bit signed integer values.
  Int8,
  /// Binary values packed into bytes.
  Bit,
}

impl VectorElementType {
  /// Return the canonical string representation of this element type.
  pub fn as_str(&self) -> &'static str {
    match self {
      VectorElementType::F32 => "float",
      VectorElementType::Int8 => "int8",
      VectorElementType::Bit => "bit",
    }
  }
}

impl std::str::FromStr for VectorElementType {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "float" | "f32" => Ok(VectorElementType::F32),
      "int8" => Ok(VectorElementType::Int8),
      "bit" => Ok(VectorElementType::Bit),
      _ => Err(format!("unknown vector element type: {}", s)),
    }
  }
}

/// Scalar SQL type for a metadata column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
  /// TEXT column.
  Text,
  /// INTEGER column.
  Integer,
  /// REAL column.
  Real,
}

impl ScalarType {
  /// Return the canonical string representation of this scalar type.
  pub fn as_str(&self) -> &'static str {
    match self {
      ScalarType::Text => "text",
      ScalarType::Integer => "integer",
      ScalarType::Real => "real",
    }
  }
}

impl std::str::FromStr for ScalarType {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "text" => Ok(ScalarType::Text),
      "integer" | "int" => Ok(ScalarType::Integer),
      "real" => Ok(ScalarType::Real),
      _ => Err(format!("unknown scalar type: {}", s)),
    }
  }
}

/// Metadata column declaration used to build the metadata shadow table.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataColumn {
  /// Column name as declared in the virtual table.
  pub name: String,
  /// SQLite storage type for this scalar column.
  pub scalar_type: ScalarType,
}

/// A dense vector whose element type is known at runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
  /// 32-bit float vector.
  F32(Vec<f32>),
  /// 8-bit signed integer vector.
  Int8(Vec<i8>),
  /// Packed binary vector.
  Bit {
    /// Packed bytes. Each byte holds up to 8 bits, least-significant bit first.
    data: Vec<u8>,
    /// Number of valid bits in `data`.
    dim: usize,
  },
}

impl Vector {
  /// Return the element type of this vector.
  pub fn element_type(&self) -> VectorElementType {
    match self {
      Vector::F32(_) => VectorElementType::F32,
      Vector::Int8(_) => VectorElementType::Int8,
      Vector::Bit { .. } => VectorElementType::Bit,
    }
  }

  /// Return the vector dimension (number of elements).
  ///
  /// For `Bit` vectors this is the number of valid bits, not the byte length.
  pub fn dim(&self) -> usize {
    match self {
      Vector::F32(v) => v.len(),
      Vector::Int8(v) => v.len(),
      Vector::Bit { dim, .. } => *dim,
    }
  }
}

/// A single search hit, pairing a rowid with its relevance score.
///
/// For distance-based metrics (L2, cosine distance), lower scores are better.
/// For similarity-style metrics, the score semantics are defined by the
/// underlying index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredRowId {
  /// Identifies the matched document.
  pub rowid: RowId,
  /// Relevance or distance score. Lower is better for distance metrics.
  pub score: f32,
}

/// A vector-based search query.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
  /// Query embedding. Its dimension must match the dimension configured on
  /// the index.
  pub vector: Vector,
  /// Maximum number of results to return.
  pub topk: usize,
}

/// Result of a search operation.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
  /// Matching documents ordered by score (best first).
  pub hits: Vec<ScoredRowId>,
}

impl SearchResult {
  /// Create an empty result.
  pub fn empty() -> Self {
    Self { hits: Vec::new() }
  }

  /// Create a result from a list of hits.
  pub fn new(hits: Vec<ScoredRowId>) -> Self {
    Self { hits }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn scored_row_id_orders_by_score() {
    let a = ScoredRowId { rowid: 1, score: 0.1 };
    let b = ScoredRowId { rowid: 2, score: 0.2 };
    assert!(a.score < b.score);
    assert_ne!(a, b);
  }

  #[test]
  fn search_result_empty() {
    let r = SearchResult::empty();
    assert!(r.hits.is_empty());
  }

  #[test]
  fn vector_reports_element_type_and_dim() {
    assert_eq!(Vector::F32(vec![1.0, 2.0, 3.0]).element_type(), VectorElementType::F32);
    assert_eq!(Vector::F32(vec![1.0, 2.0, 3.0]).dim(), 3);

    assert_eq!(Vector::Int8(vec![1, 2]).element_type(), VectorElementType::Int8);
    assert_eq!(Vector::Int8(vec![1, 2]).dim(), 2);

    assert_eq!(
      Vector::Bit {
        data: vec![0b0000_0011],
        dim: 7
      }
      .element_type(),
      VectorElementType::Bit
    );
    assert_eq!(
      Vector::Bit {
        data: vec![0b0000_0011],
        dim: 7
      }
      .dim(),
      7
    );
  }
}
