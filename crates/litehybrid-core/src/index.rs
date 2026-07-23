//! Hybrid search orchestration layer.

use litehybrid_vec::{
  Connection, FlatIndex, IndexError, Metric, RowId, SearchResult, Vector, VectorElementType, VectorIndex, VectorQuery,
};

/// Vector index kind used when creating a `HybridIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorIndexKind {
  /// Brute-force Flat index.
  Flat,
}

/// A hybrid search index that combines vector and text indexes.
///
/// In Phase 1, only vector search is implemented. The text index is a
/// placeholder for Phase 2.
pub struct HybridIndex {
  vector: Box<dyn VectorIndex>,
}

impl HybridIndex {
  /// Create a new hybrid index backed by the requested vector index kind.
  ///
  /// The `table_name` is used to derive the underlying shadow table name.
  pub fn create(
    db: &Connection,
    table_name: &str,
    dim: usize,
    metric: Metric,
    kind: VectorIndexKind,
  ) -> Result<Self, IndexError> {
    let vector: Box<dyn VectorIndex> = match kind {
      VectorIndexKind::Flat => Box::new(FlatIndex::create(db, table_name, dim, metric, VectorElementType::F32)?),
    };
    Ok(Self { vector })
  }

  /// Insert or replace a vector for the given rowid.
  pub fn insert_vector(&self, db: &Connection, rowid: RowId, vector: &Vector) -> Result<(), IndexError> {
    self.vector.insert(db, rowid, vector)
  }

  /// Delete the vector for the given rowid.
  pub fn delete_vector(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError> {
    self.vector.delete(db, rowid)
  }

  /// Search for the top-k nearest vectors.
  pub fn search_vector(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError> {
    self.vector.search(db, query)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use litehybrid_vec::Vector;

  #[test]
  fn insert_search_and_delete() {
    let db = Connection::open_in_memory().unwrap();
    let index = HybridIndex::create(&db, "test_hybrid", 3, Metric::L2, VectorIndexKind::Flat).unwrap();

    index.insert_vector(&db, 1, &Vector::F32(vec![1.0, 0.0, 0.0])).unwrap();
    index.insert_vector(&db, 2, &Vector::F32(vec![0.0, 1.0, 0.0])).unwrap();
    index.insert_vector(&db, 3, &Vector::F32(vec![0.0, 0.0, 1.0])).unwrap();

    let query = VectorQuery {
      vector: Vector::F32(vec![1.0, 0.1, 0.1]),
      topk: 2,
    };
    let result = index.search_vector(&db, &query).unwrap();
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].rowid, 1);

    index.delete_vector(&db, 1).unwrap();
    let result = index.search_vector(&db, &query).unwrap();
    assert_eq!(result.hits.len(), 2);
    assert_ne!(result.hits[0].rowid, 1);
  }
}
