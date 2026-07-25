//! Flat (brute-force) vector index backed by SQLite shadow tables.

use std::collections::BinaryHeap;

use rusqlite::{Connection, params};

use crate::index::IndexError;
use crate::index::topk::Candidate;
use crate::serialize::deserialize_vector;
use crate::{MetadataColumn, Metric, RowId, ScoredRowId, SearchResult, Vector, VectorElementType, VectorQuery};

const SCHEMA_VERSION: &str = "1";

/// A brute-force vector index that stores all vectors in a SQLite shadow table.
///
/// The index itself does not keep vectors in memory. Vectors are read from the
/// shadow table on every search. Scalar metadata columns are stored in a
/// separate shadow table managed by the same `FlatIndex`.
#[derive(Debug, Clone)]
pub struct FlatIndex {
  table_name: String,
  dim: usize,
  element_type: VectorElementType,
  metric: Metric,
  metadata_columns: Vec<MetadataColumn>,
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
  /// Create or reconnect to a `FlatIndex` and its shadow tables.
  ///
  /// Shadow tables are created with `IF NOT EXISTS`, so this function works for
  /// both initial creation and reconnecting to an existing index. When
  /// reconnecting, the stored schema in the info table is validated against the
  /// requested schema.
  pub fn create(
    db: &Connection,
    table_name: &str,
    dim: usize,
    metric: Metric,
    element_type: VectorElementType,
    metadata_columns: &[MetadataColumn],
  ) -> Result<Self, IndexError> {
    Self::validate_metric_for_element_type(metric, element_type)?;

    let index = Self {
      table_name: table_name.to_string(),
      dim,
      element_type,
      metric,
      metadata_columns: metadata_columns.to_vec(),
    };

    index.create_shadow_tables(db)?;
    index.validate_or_write_schema(db)?;

    Ok(index)
  }

  fn create_shadow_tables(&self, db: &Connection) -> Result<(), IndexError> {
    // Main vector storage.
    let sql = format!(
      "CREATE TABLE IF NOT EXISTS \"{}\" (rowid INTEGER PRIMARY KEY, embedding BLOB NOT NULL)",
      self.shadow_table_name()
    );
    db.execute(&sql, [])?;

    // Metadata storage for scalar columns.
    let mut sql = format!(
      "CREATE TABLE IF NOT EXISTS \"{}\" (rowid INTEGER PRIMARY KEY",
      self.metadata_table_name()
    );
    for col in &self.metadata_columns {
      let storage_type = match col.scalar_type {
        crate::ScalarType::Text => "TEXT",
        crate::ScalarType::Integer => "INTEGER",
        crate::ScalarType::Real => "REAL",
      };
      sql.push_str(&format!(
        ", \"{}\" {}",
        Self::escape_identifier(&col.name),
        storage_type
      ));
    }
    sql.push(')');
    db.execute(&sql, [])?;

    // Info table for schema validation on reconnect.
    let sql = format!(
      "CREATE TABLE IF NOT EXISTS \"{}\" (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
      self.info_table_name()
    );
    db.execute(&sql, [])?;

    Ok(())
  }

  /// Validate that the stored schema matches this index configuration, or write
  /// it if the info table is empty (fresh index).
  fn validate_or_write_schema(&self, db: &Connection) -> Result<(), IndexError> {
    let stored = self.read_info(db)?;

    if stored.is_empty() {
      self.write_schema(db)?;
      return Ok(());
    }

    let expected = self.schema_info();
    for (key, expected_value) in &expected {
      match stored.get(*key) {
        Some(actual) if actual == expected_value => {}
        Some(actual) => {
          return Err(IndexError::SchemaMismatch {
            expected: format!("{}={}", key, expected_value),
            got: format!("{}={}", key, actual),
          });
        }
        None => {
          return Err(IndexError::SchemaMismatch {
            expected: format!("{}={}", key, expected_value),
            got: format!("{} missing", key),
          });
        }
      }
    }

    Ok(())
  }

  fn write_schema(&self, db: &Connection) -> Result<(), IndexError> {
    let info = self.schema_info();
    let sql = format!(
      "INSERT OR REPLACE INTO \"{}\" (key, value) VALUES (?1, ?2)",
      self.info_table_name()
    );
    for (key, value) in info {
      db.execute(&sql, params![key, value])?;
    }
    Ok(())
  }

  fn read_info(&self, db: &Connection) -> Result<std::collections::HashMap<String, String>, IndexError> {
    let sql = format!("SELECT key, value FROM \"{}\"", self.info_table_name());
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
      let key: String = row.get(0)?;
      let value: String = row.get(1)?;
      Ok((key, value))
    })?;
    let mut map = std::collections::HashMap::new();
    for row in rows {
      let (k, v) = row?;
      map.insert(k, v);
    }
    Ok(map)
  }

  fn schema_info(&self) -> Vec<(&'static str, String)> {
    vec![
      ("version", SCHEMA_VERSION.to_string()),
      ("dim", self.dim.to_string()),
      ("metric", self.metric.as_str().to_string()),
      ("element_type", self.element_type.as_str().to_string()),
      ("columns", self.columns_descriptor()),
    ]
  }

  fn columns_descriptor(&self) -> String {
    let mut parts = Vec::new();
    parts.push(format!(
      "{}:{}:{}",
      Self::escape_descriptor_field("embedding"),
      self.element_type.as_str(),
      self.dim
    ));
    for col in &self.metadata_columns {
      parts.push(format!(
        "{}:{}",
        Self::escape_descriptor_field(&col.name),
        col.scalar_type.as_str()
      ));
    }
    parts.join("|")
  }

  /// Escape a column name for use in the columns descriptor.
  /// Pipe and colon characters are escaped so they cannot collide with separators.
  fn escape_descriptor_field(s: &str) -> String {
    s.replace('\\', "\\\\").replace('|', "\\|").replace(':', "\\:")
  }

  /// Escape an identifier for safe use in a SQLite DDL string.
  fn escape_identifier(s: &str) -> String {
    s.replace('"', "\"\"")
  }

  fn shadow_table_name(&self) -> String {
    Self::shadow_table_name_for(&self.table_name)
  }

  fn shadow_table_name_for(table_name: &str) -> String {
    format!("{}_litehybrid_flat", table_name)
  }

  fn metadata_table_name(&self) -> String {
    format!("{}_litehybrid_metadata", self.table_name)
  }

  fn info_table_name(&self) -> String {
    format!("{}_litehybrid_info", self.table_name)
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
    let index = FlatIndex::create(&db, "test_idx", dim, metric, element_type, &[]).unwrap();
    (db, index)
  }

  fn in_memory_index_with_metadata(
    dim: usize,
    metric: Metric,
    element_type: VectorElementType,
    metadata_columns: &[MetadataColumn],
  ) -> (Connection, FlatIndex) {
    let db = Connection::open_in_memory().unwrap();
    let index = FlatIndex::create(&db, "test_idx", dim, metric, element_type, metadata_columns).unwrap();
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

  #[test]
  fn creates_metadata_shadow_table() {
    let metadata = vec![
      MetadataColumn {
        name: "category".to_string(),
        scalar_type: crate::ScalarType::Text,
      },
      MetadataColumn {
        name: "year".to_string(),
        scalar_type: crate::ScalarType::Integer,
      },
    ];
    let (db, _index) = in_memory_index_with_metadata(3, Metric::L2, VectorElementType::F32, &metadata);

    let stmt = "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'test_idx_litehybrid_metadata'";
    let exists: bool = db.query_row(stmt, [], |_| Ok(true)).unwrap();
    assert!(exists);

    let stmt = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'test_idx_litehybrid_metadata'";
    let sql: String = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert!(sql.contains("category"));
    assert!(sql.contains("year"));
  }

  #[test]
  fn writes_schema_to_info_table() {
    let metadata = vec![MetadataColumn {
      name: "category".to_string(),
      scalar_type: crate::ScalarType::Text,
    }];
    let (db, _index) = in_memory_index_with_metadata(3, Metric::L2, VectorElementType::F32, &metadata);

    let stmt = "SELECT value FROM test_idx_litehybrid_info WHERE key = 'version'";
    let version: String = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert_eq!(version, SCHEMA_VERSION);

    let stmt = "SELECT value FROM test_idx_litehybrid_info WHERE key = 'dim'";
    let dim: String = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert_eq!(dim, "3");

    let stmt = "SELECT value FROM test_idx_litehybrid_info WHERE key = 'columns'";
    let columns: String = db.query_row(stmt, [], |row| row.get(0)).unwrap();
    assert!(columns.contains("embedding:float:3"));
    assert!(columns.contains("category:text"));
  }

  #[test]
  fn reconnect_validates_matching_schema() {
    let (db, _index) = in_memory_index_with_metadata(
      3,
      Metric::L2,
      VectorElementType::F32,
      &[MetadataColumn {
        name: "category".to_string(),
        scalar_type: crate::ScalarType::Text,
      }],
    );

    // Reconnecting with the same schema should succeed.
    let metadata = vec![MetadataColumn {
      name: "category".to_string(),
      scalar_type: crate::ScalarType::Text,
    }];
    FlatIndex::create(&db, "test_idx", 3, Metric::L2, VectorElementType::F32, &metadata).unwrap();
  }

  #[test]
  fn reconnect_rejects_mismatched_schema() {
    let (db, _index) = in_memory_index(3, Metric::L2);

    // Reconnecting with a different dimension should fail.
    let err = FlatIndex::create(&db, "test_idx", 4, Metric::L2, VectorElementType::F32, &[]).unwrap_err();
    assert!(matches!(err, IndexError::SchemaMismatch { .. }));
  }
}
