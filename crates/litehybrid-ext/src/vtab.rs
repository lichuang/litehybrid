//! SQLite virtual table implementation for litehybrid vector search.

use std::borrow::Cow;
use std::ffi::{CStr, CString, c_int};
use std::sync::Arc;

use litehybrid_core::{HybridIndex, Metric, RowId, ScoredRowId, VectorIndexKind, VectorQuery};
use rusqlite::ffi;
use rusqlite::types::{Value, ValueRef};
use rusqlite::vtab::{
  Context, CreateVTab, Filters, IndexConstraintOp, IndexInfo, Inserts, UpdateVTab, Updates, VTab, VTabConnection,
  VTabCursor, VTabKind,
};
use rusqlite::{Connection, Error, Result};

const DEFAULT_TOPK: usize = 10;

/// Column indices in the declared virtual table schema.
///
/// `rowid` is implicit in SQLite and is not declared as a regular column.
mod column {
  pub const EMBEDDING: i32 = 0;
  pub const DISTANCE: i32 = 1;
  pub const K: i32 = 2;
}

/// SQLite virtual table state for `litehybrid`.
#[repr(C)]
pub struct LitehybridVTab {
  base: ffi::sqlite3_vtab,
  db: *mut ffi::sqlite3,
  index: Arc<HybridIndex>,
  dim: usize,
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
  topk: usize,
  results: Vec<ScoredRowId>,
  position: usize,
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
    let (dim, metric, kind) = parse_arguments(args)?;

    let db_ptr = unsafe { db.handle() };
    let conn = unsafe { Connection::from_handle(db_ptr)? };
    let index =
      HybridIndex::create(&conn, table_name_str, dim, metric, kind).map_err(|e| Error::ModuleError(e.to_string()))?;

    let schema = format!(
      "CREATE TABLE \"{}\" (embedding BLOB, distance REAL HIDDEN, k INT HIDDEN)",
      table_name_str
    );
    let schema_cstr = CString::new(schema)?;
    Ok((
      Cow::Owned(schema_cstr),
      Self {
        base: ffi::sqlite3_vtab::default(),
        db: db_ptr,
        index: Arc::new(index),
        dim,
      },
    ))
  }

  fn best_index(&self, info: &mut IndexInfo) -> Result<bool> {
    let mut argv_index = 1;
    let mut has_match = false;
    let mut has_k = false;

    for (constraint, mut usage) in info.constraints_and_usages() {
      if !constraint.is_usable() {
        continue;
      }
      match (constraint.column(), constraint.operator()) {
        (
          column::EMBEDDING,
          IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_MATCH | IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ,
        ) => {
          usage.set_argv_index(argv_index);
          usage.set_omit(true);
          argv_index += 1;
          has_match = true;
        }
        (column::K, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ) => {
          usage.set_argv_index(argv_index);
          argv_index += 1;
          has_k = true;
        }
        _ => {}
      }
    }

    if !has_match {
      return Ok(false);
    }

    let mut idx_num = 0;
    if has_match {
      idx_num |= 1;
    }
    if has_k {
      idx_num |= 2;
    }
    info.set_idx_num(idx_num);
    info.set_estimated_cost(1000.0);
    Ok(true)
  }

  fn open(&mut self) -> Result<Self::Cursor> {
    Ok(LitehybridCursor {
      base: ffi::sqlite3_vtab_cursor::default(),
      db: self.db,
      index: Arc::clone(&self.index),
      dim: self.dim,
      topk: DEFAULT_TOPK,
      results: Vec::new(),
      position: 0,
    })
  }
}

impl CreateVTab<'_> for LitehybridVTab {
  const KIND: VTabKind = VTabKind::Default;
}

impl UpdateVTab<'_> for LitehybridVTab {
  fn insert(&mut self, args: &Inserts<'_>) -> Result<i64> {
    let rowid: Option<RowId> = args.get(1)?;
    let rowid = rowid.ok_or_else(|| Error::ModuleError("rowid is required".to_string()))?;
    let embedding: Option<Vec<u8>> = args.get(2)?;
    let embedding = embedding.ok_or_else(|| Error::ModuleError("embedding is required".to_string()))?;
    let vector = deserialize_embedding(&embedding, self.dim)?;

    let conn = unsafe { Connection::from_handle(self.db)? };
    self.index.insert_vector(&conn, rowid, &vector).map_err(|e| Error::ModuleError(e.to_string()))?;
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
    let embedding: Option<Vec<u8>> = args.get(2)?;
    let embedding = embedding.ok_or_else(|| Error::ModuleError("embedding is required".to_string()))?;
    let vector = deserialize_embedding(&embedding, self.dim)?;

    let conn = unsafe { Connection::from_handle(self.db)? };
    self.index.delete_vector(&conn, old_rowid).map_err(|e| Error::ModuleError(e.to_string()))?;
    self.index.insert_vector(&conn, new_rowid, &vector).map_err(|e| Error::ModuleError(e.to_string()))?;
    Ok(())
  }
}

unsafe impl VTabCursor for LitehybridCursor {
  fn filter(&mut self, idx_num: c_int, _idx_str: Option<&str>, args: &Filters<'_>) -> Result<()> {
    let has_match = (idx_num & 1) != 0;
    let has_k = (idx_num & 2) != 0;
    if !has_match {
      return Err(Error::ModuleError(
        "MATCH constraint on embedding is required".to_string(),
      ));
    }

    let query_blob: Vec<u8> = args.get(0)?;
    self.topk = if has_k {
      args.get::<i64>(1)? as usize
    } else {
      DEFAULT_TOPK
    };
    let query_vector = deserialize_embedding(&query_blob, self.dim)?;

    let conn = unsafe { Connection::from_handle(self.db)? };
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
    let hit = &self.results[self.position];
    match i {
      column::EMBEDDING => ctx.set_result(&Value::Null),
      column::DISTANCE => ctx.set_result(&hit.score),
      column::K => ctx.set_result(&(self.topk as i64)),
      _ => Err(Error::ModuleError(format!("unknown column index: {}", i))),
    }
  }

  fn rowid(&self) -> Result<i64> {
    Ok(self.results[self.position].rowid)
  }
}

fn parse_arguments(args: &[&[u8]]) -> Result<(usize, Metric, VectorIndexKind)> {
  let mut dim = None;
  let mut metric = None;
  let mut kind = None;

  for arg in args {
    let s = std::str::from_utf8(arg).map_err(|e| Error::ModuleError(format!("invalid argument: {}", e)))?;
    if let Some(value) = s.strip_prefix("dim=") {
      dim = Some(
        value
          .parse::<usize>()
          .map_err(|e| Error::ModuleError(format!("invalid dim value '{}': {}", value, e)))?,
      );
    } else if let Some(value) = s.strip_prefix("metric=") {
      metric = Some(parse_metric(value)?);
    } else if let Some(value) = s.strip_prefix("index=") {
      kind = Some(parse_index_kind(value)?);
    }
  }

  let dim = dim.ok_or_else(|| Error::ModuleError("missing dim= argument".to_string()))?;
  let metric = metric.unwrap_or(Metric::L2);
  let kind = kind.unwrap_or(VectorIndexKind::Flat);
  Ok((dim, metric, kind))
}

fn parse_metric(value: &str) -> Result<Metric> {
  match unquote(value).trim().to_lowercase().as_str() {
    "l2" => Ok(Metric::L2),
    "cosine" => Ok(Metric::Cosine),
    "dot" => Ok(Metric::Dot),
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

fn deserialize_embedding(blob: &[u8], dim: usize) -> Result<Vec<f32>> {
  let expected = dim * 4;
  if blob.len() != expected {
    return Err(Error::ModuleError(format!(
      "embedding size mismatch: expected {} bytes for dim {}, got {}",
      expected,
      dim,
      blob.len()
    )));
  }
  let mut vector = Vec::with_capacity(dim);
  for chunk in blob.chunks_exact(4) {
    let bytes: [u8; 4] = chunk.try_into().expect("chunk size is 4");
    vector.push(f32::from_le_bytes(bytes));
  }
  Ok(vector)
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
