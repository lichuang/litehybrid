//! Flat (brute-force) vector index backed by a SQLite shadow table.

use std::collections::BinaryHeap;

use rusqlite::{Connection, Result as SqliteResult, params};

use crate::index::IndexError;
use crate::index::topk::Candidate;
use crate::{Metric, RowId, ScoredRowId, SearchResult, VectorQuery, distance};

/// A brute-force vector index that stores all vectors in a SQLite shadow table.
///
/// The index itself does not keep vectors in memory. Vectors are read from the
/// shadow table on every search.
#[derive(Debug, Clone)]
pub struct FlatIndex {
  table_name: String,
  dim: usize,
  metric: Metric,
}

impl crate::index::VectorIndex for FlatIndex {
  fn insert(&self, db: &Connection, rowid: RowId, vector: &[f32]) -> Result<(), IndexError> {
    self.check_dimension(vector.len())?;
    let blob = serialize_vector(vector);
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
    self.check_dimension(query.vector.len())?;

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
      let vector = deserialize_blob(&blob, self.dim)?;
      let score = distance(self.metric, &query.vector, &vector);
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
  pub fn create(db: &Connection, table_name: &str, dim: usize, metric: Metric) -> Result<Self, IndexError> {
    let shadow_table = Self::shadow_table_name_for(table_name);
    let sql = format!(
      "CREATE TABLE IF NOT EXISTS \"{}\" (rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)",
      shadow_table
    );
    db.execute(&sql, [])?;
    Ok(Self {
      table_name: table_name.to_string(),
      dim,
      metric,
    })
  }

  fn shadow_table_name(&self) -> String {
    Self::shadow_table_name_for(&self.table_name)
  }

  fn shadow_table_name_for(table_name: &str) -> String {
    format!("{}_litehybrid_flat", table_name)
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

/// Serialize a vector into little-endian `f32` bytes.
fn serialize_vector(vector: &[f32]) -> Vec<u8> {
  vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deserialize little-endian `f32` bytes into a vector.
fn deserialize_blob(blob: &[u8], expected_dim: usize) -> SqliteResult<Vec<f32>> {
  if blob.len() != expected_dim * 4 {
    return Err(rusqlite::Error::InvalidColumnType(
      1,
      "embedding".to_string(),
      rusqlite::types::Type::Blob,
    ));
  }
  let mut vector = Vec::with_capacity(expected_dim);
  for chunk in blob.chunks_exact(4) {
    let bytes: [u8; 4] = chunk.try_into().expect("chunk size is 4");
    vector.push(f32::from_le_bytes(bytes));
  }
  Ok(vector)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::VectorIndex;

  fn in_memory_index(dim: usize, metric: Metric) -> (Connection, FlatIndex) {
    let db = Connection::open_in_memory().unwrap();
    let index = FlatIndex::create(&db, "test_idx", dim, metric).unwrap();
    (db, index)
  }

  #[test]
  fn insert_and_search() {
    let (db, index) = in_memory_index(3, Metric::L2);
    index.insert(&db, 1, &[1.0, 0.0, 0.0]).unwrap();
    index.insert(&db, 2, &[0.0, 1.0, 0.0]).unwrap();
    index.insert(&db, 3, &[0.0, 0.0, 1.0]).unwrap();

    let query = VectorQuery {
      vector: vec![1.0, 0.1, 0.1],
      topk: 2,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].rowid, 1);
  }

  #[test]
  fn search_orders_by_score() {
    let (db, index) = in_memory_index(2, Metric::L2);
    index.insert(&db, 1, &[0.0, 0.0]).unwrap();
    index.insert(&db, 2, &[1.0, 0.0]).unwrap();
    index.insert(&db, 3, &[2.0, 0.0]).unwrap();

    let query = VectorQuery {
      vector: vec![0.0, 0.0],
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
    index.insert(&db, 1, &[0.0, 0.0]).unwrap();
    index.insert(&db, 1, &[10.0, 10.0]).unwrap();

    let query = VectorQuery {
      vector: vec![0.0, 0.0],
      topk: 1,
    };
    let result = index.search(&db, &query).unwrap();
    assert_eq!(result.hits[0].rowid, 1);
    assert!((result.hits[0].score - 200.0).abs() < 1e-3);
  }

  #[test]
  fn delete_removes_vector() {
    let (db, index) = in_memory_index(2, Metric::L2);
    index.insert(&db, 1, &[0.0, 0.0]).unwrap();
    index.insert(&db, 2, &[1.0, 0.0]).unwrap();
    index.delete(&db, 1).unwrap();

    let query = VectorQuery {
      vector: vec![0.0, 0.0],
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
    let err = index.insert(&db, 1, &[1.0, 2.0, 3.0]).unwrap_err();
    assert!(matches!(err, IndexError::DimensionMismatch { expected: 2, got: 3 }));
  }

  #[test]
  fn dimension_mismatch_on_search() {
    let (db, index) = in_memory_index(2, Metric::L2);
    let query = VectorQuery {
      vector: vec![1.0, 2.0, 3.0],
      topk: 1,
    };
    let err = index.search(&db, &query).unwrap_err();
    assert!(matches!(err, IndexError::DimensionMismatch { expected: 2, got: 3 }));
  }
}
