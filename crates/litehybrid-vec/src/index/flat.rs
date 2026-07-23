//! Flat (brute-force) vector index backed by a SQLite shadow table.

use std::collections::BinaryHeap;

use rusqlite::{Connection, params};

use crate::index::IndexError;
use crate::index::topk::Candidate;
use crate::serialize::deserialize_vector;
use crate::{Metric, RowId, ScoredRowId, SearchResult, Vector, VectorElementType, VectorQuery};

/// A brute-force vector index that stores all vectors in a SQLite shadow table.
///
/// The index itself does not keep vectors in memory. Vectors are read from the
/// shadow table on every search.
#[derive(Debug, Clone)]
pub struct FlatIndex {
  table_name: String,
  dim: usize,
  element_type: VectorElementType,
  metric: Metric,
}

impl crate::index::VectorIndex for FlatIndex {
  fn insert(&self, db: &Connection, rowid: RowId, vector: &Vector) -> Result<(), IndexError> {
    self.check_dimension(vector.dim())?;
    if vector.element_type() != self.element_type {
      return Err(IndexError::UnsupportedElementType(vector.element_type()));
    }
    let blob = vector.serialize();
    let sql = format!(
      "INSERT OR REPLACE INTO \"{}\" (rowid, embedding) VALUES (?1, ?2)",
      self.shadow_table_name()
    );
    db.execute(&sql, params![rowid, blob])?;
    Ok(())
  }

  fn delete(&self, db: &Connection, rowid: RowId) -> Result<(), IndexError> {
    let sql = format!("DELETE FROM \"{}\" WHERE rowid = ?1", self.shadow_table_name());
    let deleted = db.execute(&sql, params![rowid])?;
    if deleted == 0 {
      return Err(IndexError::NotFound(rowid));
    }
    Ok(())
  }

  fn search(&self, db: &Connection, query: &VectorQuery) -> Result<SearchResult, IndexError> {
    self.check_dimension(query.vector.dim())?;

    let sql = format!("SELECT rowid, embedding FROM \"{}\"", self.shadow_table_name());
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
      let rowid: RowId = row.get(0)?;
      let blob: Vec<u8> = row.get(1)?;
      Ok((rowid, blob))
    })?;

    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(query.topk);
    for row in rows {
      let (rowid, blob) = row?;
      let vector = deserialize_vector(self.element_type, self.dim, &blob)?;
      let score = self.metric.distance_vector(&query.vector, &vector)?;
      let candidate = Candidate { rowid, score };

      if heap.len() < query.topk {
        heap.push(candidate);
      } else if heap.peek().is_some_and(|worst| candidate.score < worst.score) {
        heap.pop();
        heap.push(candidate);
      }
    }

    let mut hits: Vec<ScoredRowId> = heap.into_iter().map(|c| c.into()).collect();
    hits.sort_by(|a, b| a.score.total_cmp(&b.score));
    Ok(SearchResult::new(hits))
  }
}

impl FlatIndex {
  /// Create a new `FlatIndex` and its shadow table.
  ///
  /// The shadow table is named `<table_name>_litehybrid_flat`.
  pub fn create(
    db: &Connection,
    table_name: &str,
    dim: usize,
    metric: Metric,
    element_type: VectorElementType,
  ) -> Result<Self, IndexError> {
    Self::validate_metric_for_element_type(metric, element_type)?;

    let shadow_table = Self::shadow_table_name_for(table_name);
    let sql = format!(
      "CREATE TABLE IF NOT EXISTS \"{}\" (rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)",
      shadow_table
    );
    db.execute(&sql, [])?;
    Ok(Self {
      table_name: table_name.to_string(),
      dim,
      element_type,
      metric,
    })
  }

  fn shadow_table_name(&self) -> String {
    Self::shadow_table_name_for(&self.table_name)
  }

  fn shadow_table_name_for(table_name: &str) -> String {
    format!("{}_litehybrid_flat", table_name)
  }

  fn validate_metric_for_element_type(metric: Metric, element_type: VectorElementType) -> Result<(), IndexError> {
    let valid = match element_type {
      VectorElementType::F32 | VectorElementType::Int8 => {
        matches!(metric, Metric::L2 | Metric::Cosine | Metric::Dot)
      }
      VectorElementType::Bit => metric == Metric::Hamming,
    };
    if valid {
      Ok(())
    } else {
      Err(IndexError::UnsupportedMetricForType { metric, element_type })
    }
  }

  fn check_dimension(&self, got: usize) -> Result<(), IndexError> {
    if got != self.dim {
      Err(IndexError::DimensionMismatch {
        expected: self.dim,
        got,
      })
    } else {
      Ok(())
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::VectorIndex;

  fn in_memory_index(dim: usize, metric: Metric) -> (Connection, FlatIndex) {
    in_memory_index_with_type(dim, metric, VectorElementType::F32)
  }

  fn in_memory_index_with_type(dim: usize, metric: Metric, element_type: VectorElementType) -> (Connection, FlatIndex) {
    let db = Connection::open_in_memory().unwrap();
    let index = FlatIndex::create(&db, "test_idx", dim, metric, element_type).unwrap();
    (db, index)
  }

  #[test]
  fn insert_and_search() {
    let (db, index) = in_memory_index(3, Metric::L2);
    index.insert(&db, 1, &Vector::F32(vec![1.0, 0.0, 0.0])).unwrap();
    index.insert(&db, 2, &Vector::F32(vec![0.0, 1.0, 0.0])).unwrap();
    index.insert(&db, 3, &Vector::F32(vec![0.0, 0.0, 1.0])).unwrap();

    let query = VectorQuery {
      vector: Vector::F32(vec![1.0, 0.1, 0.1]),
      topk: 2,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].rowid, 1);
  }

  #[test]
  fn search_orders_by_score() {
    let (db, index) = in_memory_index(2, Metric::L2);
    index.insert(&db, 1, &Vector::F32(vec![0.0, 0.0])).unwrap();
    index.insert(&db, 2, &Vector::F32(vec![1.0, 0.0])).unwrap();
    index.insert(&db, 3, &Vector::F32(vec![2.0, 0.0])).unwrap();

    let query = VectorQuery {
      vector: Vector::F32(vec![0.0, 0.0]),
      topk: 3,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits[0].rowid, 1);
    assert_eq!(result.hits[1].rowid, 2);
    assert_eq!(result.hits[2].rowid, 3);
  }

  #[test]
  fn insert_overwrites_duplicate_rowid() {
    let (db, index) = in_memory_index(2, Metric::L2);
    index.insert(&db, 1, &Vector::F32(vec![0.0, 0.0])).unwrap();
    index.insert(&db, 1, &Vector::F32(vec![10.0, 10.0])).unwrap();

    let query = VectorQuery {
      vector: Vector::F32(vec![0.0, 0.0]),
      topk: 1,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits[0].rowid, 1);
    assert!((result.hits[0].score - 200.0).abs() < 1e-3);
  }

  #[test]
  fn delete_removes_vector() {
    let (db, index) = in_memory_index(2, Metric::L2);
    index.insert(&db, 1, &Vector::F32(vec![0.0, 0.0])).unwrap();
    index.insert(&db, 2, &Vector::F32(vec![1.0, 0.0])).unwrap();
    index.delete(&db, 1).unwrap();

    let query = VectorQuery {
      vector: Vector::F32(vec![0.0, 0.0]),
      topk: 10,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].rowid, 2);
  }

  #[test]
  fn delete_missing_returns_error() {
    let (db, index) = in_memory_index(2, Metric::L2);
    let err = index.delete(&db, 1).unwrap_err();
    assert!(matches!(err, IndexError::NotFound(1)));
  }

  #[test]
  fn dimension_mismatch_on_insert() {
    let (db, index) = in_memory_index(2, Metric::L2);
    let err = index.insert(&db, 1, &Vector::F32(vec![1.0, 2.0, 3.0])).unwrap_err();
    assert!(matches!(err, IndexError::DimensionMismatch { expected: 2, got: 3 }));
  }

  #[test]
  fn dimension_mismatch_on_search() {
    let (db, index) = in_memory_index(2, Metric::L2);
    let query = VectorQuery {
      vector: Vector::F32(vec![1.0, 2.0, 3.0]),
      topk: 1,
    };
    let err = index.search(&db, &query).unwrap_err();
    assert!(matches!(err, IndexError::DimensionMismatch { expected: 2, got: 3 }));
  }

  #[test]
  fn insert_and_retrieve_int8_vector() {
    let (db, index) = in_memory_index_with_type(4, Metric::L2, VectorElementType::Int8);
    index.insert(&db, 1, &Vector::Int8(vec![10, -20, 30, -40])).unwrap();

    let stmt = "SELECT embedding FROM test_idx_litehybrid_flat WHERE rowid = 1";
    let blob: Vec<u8> = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert_eq!(blob, vec![10u8, 236, 30, 216]);
  }

  #[test]
  fn insert_and_retrieve_bit_vector() {
    let (db, index) = in_memory_index_with_type(10, Metric::Hamming, VectorElementType::Bit);
    let data = vec![0b0000_0011u8, 0b1000_0000u8];
    index
      .insert(
        &db,
        1,
        &Vector::Bit {
          data: data.clone(),
          dim: 10,
        },
      )
      .unwrap();

    let stmt = "SELECT embedding FROM test_idx_litehybrid_flat WHERE rowid = 1";
    let blob: Vec<u8> = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert_eq!(blob, data);
  }

  #[test]
  fn insert_mismatched_element_type_fails() {
    let (db, index) = in_memory_index_with_type(2, Metric::L2, VectorElementType::F32);
    let err = index.insert(&db, 1, &Vector::Int8(vec![1, 2])).unwrap_err();
    assert!(matches!(
      err,
      IndexError::UnsupportedElementType(VectorElementType::Int8)
    ));
  }
}
