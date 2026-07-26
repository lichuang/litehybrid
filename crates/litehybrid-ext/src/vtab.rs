//! SQLite virtual table implementation for litehybrid vector search.

use std::borrow::Cow;
use std::ffi::{CStr, CString, c_int};
use std::sync::Arc;

use litehybrid_core::{
  HybridIndex, MetadataColumn, MetadataValue, Metric, RowId, ScalarType, ScoredRowId, VectorElementType,
  VectorIndexKind, VectorQuery, deserialize_vector,
};
use rusqlite::ffi;
use rusqlite::types::{Value, ValueRef};
use rusqlite::vtab::{
  Context, CreateVTab, Filters, IndexConstraintOp, IndexInfo, Inserts, UpdateVTab, Updates, VTab, VTabConnection,
  VTabCursor, VTabKind,
};
use rusqlite::{Connection, Error, Result};

const DEFAULT_TOPK: usize = 10;

/// Declared SQL type for a litehybrid virtual table column.
#[derive(Debug, Clone, PartialEq)]
enum SqlType {
  /// Vector column with a known element type and dimension.
  Vector {
    element_type: VectorElementType,
    dim: usize,
  },
  /// Scalar text metadata column.
  Text,
  /// Scalar integer metadata column.
  Integer,
  /// Scalar real metadata column.
  Real,
}

/// A parsed column declaration from the `CREATE VIRTUAL TABLE` argument list.
#[derive(Debug, Clone, PartialEq)]
struct ColumnDecl {
  name: String,
  sql_type: SqlType,
  type_name: String,
}

impl ColumnDecl {
  /// Return the SQLite column type name to use in the declared schema.
  ///
  /// Vector columns are declared as `BLOB` even though the original
  /// declaration uses `float[N]`, `int8[N]`, or `bit[N]`.
  fn type_name(&self) -> &str {
    match &self.sql_type {
      SqlType::Vector { .. } => "BLOB",
      _ => &self.type_name,
    }
  }
}

/// SQLite virtual table state for `litehybrid`.
#[repr(C)]
pub struct LitehybridVTab {
  base: ffi::sqlite3_vtab,
  db: *mut ffi::sqlite3,
  index: Arc<HybridIndex>,
  columns: Vec<ColumnDecl>,
  vector_column_index: i32,
  distance_column_index: i32,
  k_column_index: i32,
  metric: Metric,
}

// Safety: the raw `db` pointer is owned by SQLite and remains valid for the
// lifetime of the virtual table. SQLite serializes access in serialized mode.
unsafe impl Send for LitehybridVTab {}
unsafe impl Sync for LitehybridVTab {}

/// Cursor over a vector search result set.
#[repr(C)]
pub struct LitehybridCursor {
  base: ffi::sqlite3_vtab_cursor,
  db: *mut ffi::sqlite3,
  index: Arc<HybridIndex>,
  dim: usize,
  element_type: VectorElementType,
  topk: usize,
  results: Vec<ScoredRowId>,
  position: usize,
  num_columns: usize,
  vector_column_index: i32,
  distance_column_index: i32,
  k_column_index: i32,
}

// Safety: same reasoning as `LitehybridVTab`.
unsafe impl Send for LitehybridCursor {}
unsafe impl Sync for LitehybridCursor {}

unsafe impl VTab<'_> for LitehybridVTab {
  type Aux = ();
  type Cursor = LitehybridCursor;

  fn connect(
    db: &mut VTabConnection,
    _aux: Option<&Self::Aux>,
    _module_name: &[u8],
    _database_name: &[u8],
    table_name: &[u8],
    args: &[&[u8]],
  ) -> Result<(Cow<'static, CStr>, Self)> {
    let table_name_str =
      std::str::from_utf8(table_name).map_err(|e| Error::ModuleError(format!("invalid table name: {}", e)))?;
    let (columns, metric, kind) = parse_arguments(args)?;

    let vector_column_index = columns
      .iter()
      .position(|c| matches!(c.sql_type, SqlType::Vector { .. }))
      .ok_or_else(|| Error::ModuleError("litehybrid requires exactly one vector column".to_string()))?;
    if columns.iter().filter(|c| matches!(c.sql_type, SqlType::Vector { .. })).count() != 1 {
      return Err(Error::ModuleError(
        "litehybrid requires exactly one vector column".to_string(),
      ));
    }
    let (dim, element_type) = match &columns[vector_column_index].sql_type {
      SqlType::Vector { element_type, dim } => (*dim, *element_type),
      _ => unreachable!(),
    };

    let metadata_columns: Vec<MetadataColumn> = columns
      .iter()
      .filter(|c| !matches!(c.sql_type, SqlType::Vector { .. }))
      .map(|c| MetadataColumn {
        name: c.name.clone(),
        scalar_type: match c.sql_type {
          SqlType::Text => ScalarType::Text,
          SqlType::Integer => ScalarType::Integer,
          SqlType::Real => ScalarType::Real,
          SqlType::Vector { .. } => unreachable!(),
        },
      })
      .collect();

    let db_ptr = unsafe { db.handle() };
    let conn = unsafe { Connection::from_handle(db_ptr)? };
    let index = HybridIndex::create(
      &conn,
      table_name_str,
      dim,
      metric,
      element_type,
      kind,
      &metadata_columns,
    )
    .map_err(|e| Error::ModuleError(e.to_string()))?;

    let mut schema = format!("CREATE TABLE \"{}\" (", table_name_str);
    for (i, col) in columns.iter().enumerate() {
      if i > 0 {
        schema.push_str(", ");
      }
      schema.push_str(&col.name);
      schema.push(' ');
      schema.push_str(col.type_name());
    }
    schema.push_str(", distance REAL HIDDEN, k INT HIDDEN)");
    let schema_cstr = CString::new(schema)?;

    let num_columns = columns.len();
    Ok((
      Cow::Owned(schema_cstr),
      Self {
        base: ffi::sqlite3_vtab::default(),
        db: db_ptr,
        index: Arc::new(index),
        columns,
        vector_column_index: vector_column_index as i32,
        distance_column_index: num_columns as i32,
        k_column_index: (num_columns + 1) as i32,
        metric,
      },
    ))
  }

  fn best_index(&self, info: &mut IndexInfo) -> Result<bool> {
    // Tracks the 1-based argv position for the next constraint we consume.
    // SQLite passes constraint values to xFilter in the order we assign here.
    let mut argv_index = 1;
    // Whether we found a MATCH/EQ constraint on the vector column.
    // A vector search query is unusable without this.
    let mut has_match = false;
    // Whether we found an EQ constraint on the hidden k column.
    let mut has_k = false;

    // Examine every WHERE-clause constraint offered by SQLite.
    for (constraint, mut usage) in info.constraints_and_usages() {
      // Constraints may be unusable due to join ordering; skip those.
      if !constraint.is_usable() {
        continue;
      }
      let col = constraint.column();
      let op = constraint.operator();

      // Vector column constraint (= or MATCH) drives the KNN search.
      if col == self.vector_column_index
        && (op == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_MATCH
          || op == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ)
      {
        // Tell SQLite to pass the constraint value to xFilter at this argv position.
        usage.set_argv_index(argv_index);
        // The virtual table guarantees this constraint will be satisfied, so SQLite
        // does not need to double-check it on each returned row.
        usage.set_omit(true);
        argv_index += 1;
        has_match = true;
      }
      // Hidden k column constraint overrides the default top-k value.
      else if col == self.k_column_index && op == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ {
        usage.set_argv_index(argv_index);
        argv_index += 1;
        has_k = true;
      }
    }

    // Reject any plan that does not constrain the vector column, because we
    // cannot perform a KNN search without a query vector.
    if !has_match {
      return Ok(false);
    }

    // Encode which constraints were consumed into idx_num; xFilter uses this to
    // know which argv values are present.
    // bit 0: query vector present
    // bit 1: k value present
    let mut idx_num = 0;
    if has_match {
      idx_num |= 1;
    }
    if has_k {
      idx_num |= 2;
    }
    info.set_idx_num(idx_num);
    // A low estimated cost encourages SQLite to choose the vector-index plan.
    info.set_estimated_cost(1000.0);
    Ok(true)
  }

  fn open(&mut self) -> Result<Self::Cursor> {
    let (dim, element_type) = match &self.columns[self.vector_column_index as usize].sql_type {
      SqlType::Vector { element_type, dim } => (*dim, *element_type),
      _ => unreachable!(),
    };
    Ok(LitehybridCursor {
      base: ffi::sqlite3_vtab_cursor::default(),
      db: self.db,
      index: Arc::clone(&self.index),
      dim,
      element_type,
      topk: DEFAULT_TOPK,
      results: Vec::new(),
      position: 0,
      num_columns: self.columns.len(),
      vector_column_index: self.vector_column_index,
      distance_column_index: self.distance_column_index,
      k_column_index: self.k_column_index,
    })
  }
}

impl CreateVTab<'_> for LitehybridVTab {
  const KIND: VTabKind = VTabKind::Default;

  fn destroy(&self) -> Result<()> {
    Ok(())
  }
}

impl UpdateVTab<'_> for LitehybridVTab {
  fn insert(&mut self, args: &Inserts<'_>) -> Result<i64> {
    let rowid: Option<RowId> = args.get(1)?;
    let rowid = rowid.ok_or_else(|| Error::ModuleError("rowid is required".to_string()))?;
    let embedding: Option<Vec<u8>> = args.get(self.vector_column_index as usize + 2)?;
    let embedding = embedding.ok_or_else(|| Error::ModuleError("embedding is required".to_string()))?;
    let vector = deserialize_vector(self.element_type()?, self.dim()?, &embedding)
      .map_err(|e| Error::ModuleError(e.to_string()))?;
    let metadata = self.extract_metadata(args)?;

    let conn = unsafe { Connection::from_handle(self.db)? };
    self
      .index
      .insert_vector(&conn, rowid, &vector, &metadata)
      .map_err(|e| Error::ModuleError(e.to_string()))?;
    Ok(rowid)
  }

  fn delete(&mut self, arg: ValueRef<'_>) -> Result<()> {
    let rowid = value_as_rowid(arg)?;
    let conn = unsafe { Connection::from_handle(self.db)? };
    self.index.delete_vector(&conn, rowid).map_err(|e| Error::ModuleError(e.to_string()))?;
    Ok(())
  }

  fn update(&mut self, args: &Updates<'_>) -> Result<()> {
    let old_rowid: Option<RowId> = args.get(0)?;
    let old_rowid = old_rowid.ok_or_else(|| Error::ModuleError("old rowid is required for update".to_string()))?;
    let new_rowid: Option<RowId> = args.get(1)?;
    let new_rowid = new_rowid.ok_or_else(|| Error::ModuleError("new rowid is required for update".to_string()))?;
    let embedding: Option<Vec<u8>> = args.get(self.vector_column_index as usize + 2)?;
    let embedding = embedding.ok_or_else(|| Error::ModuleError("embedding is required".to_string()))?;
    let vector = deserialize_vector(self.element_type()?, self.dim()?, &embedding)
      .map_err(|e| Error::ModuleError(e.to_string()))?;
    let metadata = self.extract_metadata(args)?;

    let conn = unsafe { Connection::from_handle(self.db)? };
    self.index.delete_vector(&conn, old_rowid).map_err(|e| Error::ModuleError(e.to_string()))?;
    self
      .index
      .insert_vector(&conn, new_rowid, &vector, &metadata)
      .map_err(|e| Error::ModuleError(e.to_string()))?;
    Ok(())
  }
}

impl LitehybridVTab {
  fn element_type(&self) -> Result<VectorElementType> {
    match &self.columns[self.vector_column_index as usize].sql_type {
      SqlType::Vector { element_type, .. } => Ok(*element_type),
      _ => Err(Error::ModuleError(
        "vector_column_index points to a non-vector column".to_string(),
      )),
    }
  }

  fn dim(&self) -> Result<usize> {
    match &self.columns[self.vector_column_index as usize].sql_type {
      SqlType::Vector { dim, .. } => Ok(*dim),
      _ => Err(Error::ModuleError(
        "vector_column_index points to a non-vector column".to_string(),
      )),
    }
  }

  /// Extract metadata values from an insert or update argument list.
  ///
  /// Column values in `args` start at index 2 (`args[0]` is the insert marker,
  /// `args[1]` is the rowid). The vector column is skipped.
  fn extract_metadata(&self, args: &rusqlite::vtab::Values<'_>) -> Result<Vec<Option<MetadataValue>>> {
    let mut metadata = Vec::new();
    for (i, _col) in self.columns.iter().enumerate() {
      if i == self.vector_column_index as usize {
        continue;
      }
      let raw: Option<Value> = args.get(i + 2)?;
      let value = match raw {
        None | Some(Value::Null) => None,
        Some(v) => Some(MetadataValue::try_from(v).map_err(Error::ModuleError)?),
      };
      metadata.push(value);
    }
    Ok(metadata)
  }
}

unsafe impl VTabCursor for LitehybridCursor {
  fn filter(&mut self, idx_num: c_int, _idx_str: Option<&str>, args: &Filters<'_>) -> Result<()> {
    // idx_num is a bitmask produced by best_index telling us which constraint
    // values were passed in through `args`.
    let has_match = (idx_num & 1) != 0;
    let has_k = (idx_num & 2) != 0;

    // A vector search is impossible without a query vector. best_index should
    // already have rejected such plans; this is a defensive check.
    if !has_match {
      return Err(Error::ModuleError(
        "MATCH constraint on vector column is required".to_string(),
      ));
    }

    // The first consumed constraint (argv_index 1 in best_index) is the query
    // vector BLOB, which becomes args index 0 here.
    let query_blob: Vec<u8> = args.get(0)?;

    // The second consumed constraint, if present, is the hidden k column.
    // Otherwise fall back to the default top-k value.
    self.topk = if has_k {
      args.get::<i64>(1)? as usize
    } else {
      DEFAULT_TOPK
    };

    // Convert the raw BLOB into a typed Vector, validating element type and dim.
    let query_vector =
      deserialize_vector(self.element_type, self.dim, &query_blob).map_err(|e| Error::ModuleError(e.to_string()))?;

    // Build a temporary Connection from the raw db handle stored in the cursor.
    let conn = unsafe { Connection::from_handle(self.db)? };

    // Run the KNN search through the underlying HybridIndex.
    let result = self
      .index
      .search_vector(
        &conn,
        &VectorQuery {
          vector: query_vector,
          topk: self.topk,
        },
      )
      .map_err(|e| Error::ModuleError(e.to_string()))?;

    // Cache the scored results and reset the cursor position so iteration
    // starts at the first hit.
    self.results = result.hits;
    self.position = 0;
    Ok(())
  }

  fn next(&mut self) -> Result<()> {
    self.position += 1;
    Ok(())
  }

  fn eof(&self) -> bool {
    self.position >= self.results.len()
  }

  fn column(&self, ctx: &mut Context, i: c_int) -> Result<()> {
    if i == self.k_column_index {
      ctx.set_result(&(self.topk as i64))?;
      return Ok(());
    }
    if i == self.distance_column_index {
      // Hidden distance column returns the score of the current hit.
      let hit = &self.results[self.position];
      ctx.set_result(&hit.score)?;
      return Ok(());
    }
    if i == self.vector_column_index {
      // The stored vector is not returned by the search cursor.
      ctx.set_result(&Value::Null)?;
      return Ok(());
    }
    if (0..self.num_columns as i32).contains(&i) {
      // Metadata columns are not yet populated from stored rows.
      ctx.set_result(&Value::Null)?;
      return Ok(());
    }
    Err(Error::ModuleError(format!("unknown column index: {}", i)))
  }

  fn rowid(&self) -> Result<i64> {
    Ok(self.results[self.position].rowid)
  }
}

fn parse_arguments(args: &[&[u8]]) -> Result<(Vec<ColumnDecl>, Metric, VectorIndexKind)> {
  let mut columns = Vec::new();
  let mut metric = None;
  let mut kind = None;

  for arg in args {
    let s = std::str::from_utf8(arg).map_err(|e| Error::ModuleError(format!("invalid argument: {}", e)))?;
    let s = s.trim();
    if s.is_empty() {
      continue;
    }
    if let Some(value) = s.strip_prefix("metric=") {
      metric = Some(parse_metric(value)?);
    } else if let Some(value) = s.strip_prefix("index=") {
      kind = Some(parse_index_kind(value)?);
    } else {
      columns.push(parse_column_decl(s)?);
    }
  }

  if columns.is_empty() {
    return Err(Error::ModuleError(
      "at least one column declaration is required".to_string(),
    ));
  }

  let vector_columns: Vec<_> = columns.iter().filter(|c| matches!(c.sql_type, SqlType::Vector { .. })).collect();
  if vector_columns.len() != 1 {
    return Err(Error::ModuleError(format!(
      "litehybrid requires exactly one vector column, found {}",
      vector_columns.len()
    )));
  }
  let element_type = match &vector_columns[0].sql_type {
    SqlType::Vector { element_type, .. } => *element_type,
    _ => unreachable!(),
  };

  let metric = metric.unwrap_or_else(|| default_metric_for(element_type));
  let kind = kind.unwrap_or(VectorIndexKind::Flat);

  Ok((columns, metric, kind))
}

fn default_metric_for(element_type: VectorElementType) -> Metric {
  match element_type {
    VectorElementType::F32 | VectorElementType::Int8 => Metric::L2,
    VectorElementType::Bit => Metric::Hamming,
  }
}

fn parse_column_decl(s: &str) -> Result<ColumnDecl> {
  let s = s.trim();
  let (name, rest) = split_name_and_type(s)?;
  let name = parse_identifier(name)?;
  let (sql_type, type_name) = parse_sql_type(rest)?;
  Ok(ColumnDecl {
    name,
    sql_type,
    type_name,
  })
}

fn split_name_and_type(s: &str) -> Result<(&str, &str)> {
  let s = s.trim();
  if let Some(quoted) = s.strip_prefix('"') {
    let end = quoted
      .find('"')
      .ok_or_else(|| Error::ModuleError(format!("unterminated quoted column name in '{}'", s)))?;
    let name = &s[..=end + 1];
    let rest = s[end + 2..].trim();
    Ok((name, rest))
  } else {
    let mut parts = s.splitn(2, char::is_whitespace);
    let name = parts.next().ok_or_else(|| Error::ModuleError(format!("expected 'name type', got '{}'", s)))?;
    let rest = parts.next().ok_or_else(|| Error::ModuleError(format!("expected 'name type', got '{}'", s)))?;
    Ok((name, rest.trim()))
  }
}

fn parse_identifier(s: &str) -> Result<String> {
  let s = s.trim();
  if s.starts_with('"') {
    if !s.ends_with('"') || s.len() < 2 {
      return Err(Error::ModuleError(format!("unterminated quoted identifier: '{}'", s)));
    }
    Ok(s[1..s.len() - 1].replace("\"\"", "\""))
  } else if s.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false)
    && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
  {
    Ok(s.to_string())
  } else {
    Err(Error::ModuleError(format!("invalid column name: '{}'", s)))
  }
}

fn parse_sql_type(s: &str) -> Result<(SqlType, String)> {
  let s = s.trim();
  if let Some(inner) = s.strip_prefix("float[").and_then(|s| s.strip_suffix("]")) {
    let dim = inner
      .trim()
      .parse::<usize>()
      .map_err(|e| Error::ModuleError(format!("invalid vector dimension '{}': {}", inner, e)))?;
    if dim == 0 {
      return Err(Error::ModuleError(
        "vector dimension must be greater than zero".to_string(),
      ));
    }
    return Ok((
      SqlType::Vector {
        element_type: VectorElementType::F32,
        dim,
      },
      s.to_lowercase(),
    ));
  }
  if let Some(inner) = s.strip_prefix("int8[").and_then(|s| s.strip_suffix("]")) {
    let dim = inner
      .trim()
      .parse::<usize>()
      .map_err(|e| Error::ModuleError(format!("invalid vector dimension '{}': {}", inner, e)))?;
    if dim == 0 {
      return Err(Error::ModuleError(
        "vector dimension must be greater than zero".to_string(),
      ));
    }
    return Ok((
      SqlType::Vector {
        element_type: VectorElementType::Int8,
        dim,
      },
      s.to_lowercase(),
    ));
  }
  if let Some(inner) = s.strip_prefix("bit[").and_then(|s| s.strip_suffix("]")) {
    let dim = inner
      .trim()
      .parse::<usize>()
      .map_err(|e| Error::ModuleError(format!("invalid vector dimension '{}': {}", inner, e)))?;
    if dim == 0 {
      return Err(Error::ModuleError(
        "vector dimension must be greater than zero".to_string(),
      ));
    }
    return Ok((
      SqlType::Vector {
        element_type: VectorElementType::Bit,
        dim,
      },
      s.to_lowercase(),
    ));
  }
  let lower = s.to_lowercase();
  match lower.as_str() {
    "text" => Ok((SqlType::Text, lower)),
    "integer" | "int" => Ok((SqlType::Integer, lower)),
    "real" => Ok((SqlType::Real, lower)),
    _ => Err(Error::ModuleError(format!("unsupported column type: '{}'", s))),
  }
}

fn parse_metric(value: &str) -> Result<Metric> {
  match unquote(value).trim().to_lowercase().as_str() {
    "l2" => Ok(Metric::L2),
    "cosine" => Ok(Metric::Cosine),
    "dot" => Ok(Metric::Dot),
    "hamming" => Ok(Metric::Hamming),
    _ => Err(Error::ModuleError(format!("unknown metric: {}", value))),
  }
}

fn parse_index_kind(value: &str) -> Result<VectorIndexKind> {
  match unquote(value).trim().to_lowercase().as_str() {
    "flat" => Ok(VectorIndexKind::Flat),
    _ => Err(Error::ModuleError(format!("unknown index kind: {}", value))),
  }
}

fn unquote(value: &str) -> &str {
  let value = value.trim();
  if value.len() >= 2 {
    let bytes = value.as_bytes();
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];
    if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
      return &value[1..value.len() - 1];
    }
  }
  value
}

fn value_as_rowid(value: ValueRef<'_>) -> Result<RowId> {
  match value {
    ValueRef::Integer(i) => Ok(i),
    _ => Err(Error::ModuleError(format!(
      "expected integer rowid, got {:?}",
      value.data_type()
    ))),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_single_float_vector_column() {
    let (columns, metric, kind) = parse_arguments(&[b"embedding float[384]"]).unwrap();
    assert_eq!(columns.len(), 1);
    assert_eq!(columns[0].name, "embedding");
    assert_eq!(
      columns[0].sql_type,
      SqlType::Vector {
        element_type: VectorElementType::F32,
        dim: 384,
      }
    );
    assert_eq!(metric, Metric::L2);
    assert_eq!(kind, VectorIndexKind::Flat);
  }

  #[test]
  fn parse_int8_vector_with_metadata() {
    let (columns, metric, kind) = parse_arguments(&[
      b"embedding int8[64]",
      b"category text",
      b"year int",
      b"metric='cosine'",
      b"index='flat'",
    ])
    .unwrap();
    assert_eq!(columns.len(), 3);
    assert_eq!(columns[0].name, "embedding");
    assert_eq!(
      columns[0].sql_type,
      SqlType::Vector {
        element_type: VectorElementType::Int8,
        dim: 64,
      }
    );
    assert_eq!(columns[1].name, "category");
    assert_eq!(columns[1].sql_type, SqlType::Text);
    assert_eq!(columns[2].name, "year");
    assert_eq!(columns[2].sql_type, SqlType::Integer);
    assert_eq!(metric, Metric::Cosine);
    assert_eq!(kind, VectorIndexKind::Flat);
  }

  #[test]
  fn parse_bit_vector_defaults_to_hamming() {
    let (columns, metric, _kind) = parse_arguments(&[b"embedding bit[128]"]).unwrap();
    assert_eq!(
      columns[0].sql_type,
      SqlType::Vector {
        element_type: VectorElementType::Bit,
        dim: 128,
      }
    );
    assert_eq!(metric, Metric::Hamming);
  }

  #[test]
  fn parse_rejects_zero_vector_dimension() {
    let err = parse_arguments(&[b"embedding float[0]"]).unwrap_err();
    assert!(err.to_string().contains("greater than zero"));
  }

  #[test]
  fn parse_rejects_missing_vector_column() {
    let err = parse_arguments(&[b"category text"]).unwrap_err();
    assert!(err.to_string().contains("exactly one vector column"));
  }

  #[test]
  fn parse_rejects_multiple_vector_columns() {
    let err = parse_arguments(&[b"a float[3]", b"b float[3]"]).unwrap_err();
    assert!(err.to_string().contains("exactly one vector column"));
  }

  #[test]
  fn parse_quoted_column_name() {
    let (columns, _, _) = parse_arguments(&[b"\"weird name\" float[3]"]).unwrap();
    assert_eq!(columns[0].name, "weird name");
  }

  #[test]
  fn vector_column_declared_as_blob() {
    let (columns, _, _) = parse_arguments(&[b"embedding float[384]", b"category text"]).unwrap();
    assert_eq!(columns[0].type_name(), "BLOB");
    assert_eq!(columns[1].type_name(), "text");
  }
}
