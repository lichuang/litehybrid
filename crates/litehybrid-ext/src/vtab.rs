//! SQLite virtual table implementation for litehybrid vector search.

use std::borrow::Cow;
use std::ffi::{CStr, CString, c_int};
use std::sync::Arc;

use litehybrid_core::{
  HybridIndex, MetadataColumn, MetadataConstraint, MetadataConstraintOp, MetadataValue, Metric, RowId, ScalarType,
  ScoredRowId, VectorElementType, VectorIndexKind, VectorQuery, deserialize_vector,
};
use rusqlite::ffi;
use rusqlite::types::{Value, ValueRef};
use rusqlite::vtab::{
  Context, CreateVTab, Filters, IndexConstraintOp, IndexInfo, Inserts, UpdateVTab, Updates, VTab, VTabConnection,
  VTabCursor, VTabKind,
};
use rusqlite::{Connection, Error, Result};

const DEFAULT_TOPK: usize = 10;

fn metadata_constraint_op_from_index_op(op: &IndexConstraintOp) -> Option<MetadataConstraintOp> {
  match op {
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ => Some(MetadataConstraintOp::Eq),
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_NE => Some(MetadataConstraintOp::Ne),
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_LT => Some(MetadataConstraintOp::Lt),
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_LE => Some(MetadataConstraintOp::Le),
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_GT => Some(MetadataConstraintOp::Gt),
    IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_GE => Some(MetadataConstraintOp::Ge),
    _ => None,
  }
}

/// A metadata constraint that `best_index` decided to consume.
#[derive(Debug, Clone)]
struct MetadataConstraintSpec {
  column_index: i32,
  op: MetadataConstraintOp,
}

/// Map a virtual table column index to the corresponding metadata column index.
///
/// Returns `None` if the column is the vector column, a hidden column, or out of
/// range.  Virtual table columns and metadata columns have different index spaces
/// because the vector column is excluded from the metadata column list.
fn metadata_column_index(columns: &[ColumnDecl], vector_column_index: i32, virtual_index: i32) -> Option<usize> {
  if virtual_index < 0 || virtual_index as usize >= columns.len() {
    return None;
  }
  if virtual_index == vector_column_index {
    return None;
  }
  if matches!(columns[virtual_index as usize].sql_type, SqlType::Vector { .. }) {
    return None;
  }
  let metadata_idx = columns
    .iter()
    .take(virtual_index as usize)
    .filter(|c| !matches!(c.sql_type, SqlType::Vector { .. }))
    .count();
  Some(metadata_idx)
}

/// Encode consumed metadata constraints into a compact string for `xFilter`.
///
/// Format: `column_index:op,column_index:op,...`
/// Example: `2:=,3:>` means "column 2 equals ? and column 3 greater than ?".
fn encode_metadata_constraints(constraints: &[MetadataConstraintSpec]) -> String {
  constraints
    .iter()
    .map(|c| format!("{}:{}", c.column_index, c.op.as_str()))
    .collect::<Vec<_>>()
    .join(",")
}

/// Decode the metadata constraint plan produced by `best_index`.
fn decode_metadata_constraints(idx_str: Option<&str>) -> Result<Vec<MetadataConstraintSpec>> {
  let s = idx_str.unwrap_or("");
  if s.is_empty() {
    return Ok(Vec::new());
  }
  s.split(',')
    .map(|part| {
      let (col_str, op_str) =
        part.split_once(':').ok_or_else(|| Error::ModuleError(format!("malformed idx_str: {}", s)))?;
      let column_index = col_str
        .parse::<i32>()
        .map_err(|e| Error::ModuleError(format!("invalid column index in idx_str: {}", e)))?;
      let op: MetadataConstraintOp =
        op_str.parse().map_err(|_| Error::ModuleError(format!("unknown operator in idx_str: {}", op_str)))?;
      Ok(MetadataConstraintSpec { column_index, op })
    })
    .collect::<Result<Vec<_>>>()
}

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
  /// Metadata column declarations, in metadata-column index order.
  metadata_columns: Vec<MetadataColumn>,
  /// Cached metadata values for each row in `results`.
  metadata_cache: Vec<Vec<Option<MetadataValue>>>,
  /// Map from virtual table column index to metadata column index.
  /// `None` means the column is not a metadata column.
  metadata_column_map: Vec<Option<usize>>,
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
    // Whether we found a MATCH/EQ constraint on the vector column.
    // A vector search query is unusable without this.
    let mut has_match = false;
    // Whether we found an EQ constraint on the hidden k column.
    let mut has_k = false;
    // Metadata column constraints that we will consume in `xFilter`.
    let mut metadata_constraints = Vec::new();

    // Collect usable constraints first so we can assign argv indices in a fixed
    // order regardless of the order SQLite happens to present them.  xFilter
    // relies on: argv 1 = vector, argv 2 = k (if present), argv 3+ = metadata.
    let mut usable_constraints = Vec::new();
    for (constraint, usage) in info.constraints_and_usages() {
      if !constraint.is_usable() {
        continue;
      }
      usable_constraints.push((constraint.column(), constraint.operator(), usage));
    }

    // Tracks the 1-based argv position for the next constraint we consume.
    // SQLite passes constraint values to xFilter in the order we assign here.
    let mut argv_index = 1;

    // First pass: vector column constraint (= or MATCH) drives the KNN search.
    for (col, op, usage) in &mut usable_constraints {
      if *col == self.vector_column_index
        && matches!(
          *op,
          IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_MATCH | IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ
        )
      {
        usage.set_argv_index(argv_index);
        // The virtual table guarantees this constraint will be satisfied, so SQLite
        // does not need to double-check it on each returned row.
        usage.set_omit(true);
        argv_index += 1;
        has_match = true;
      }
    }

    // Second pass: hidden k column constraint overrides the default top-k value.
    for (col, op, usage) in &mut usable_constraints {
      if *col == self.k_column_index && matches!(*op, IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ) {
        usage.set_argv_index(argv_index);
        usage.set_omit(true);
        argv_index += 1;
        has_k = true;
      }
    }

    // Third pass: scalar metadata column constraints are consumed so their values
    // reach `xFilter`.  The index layer now applies these predicates, so SQLite
    // does not need to double-check them on returned rows.
    for (col, op, usage) in &mut usable_constraints {
      let Some(metadata_idx) = metadata_column_index(&self.columns, self.vector_column_index, *col) else {
        continue;
      };
      let Some(meta_op) = metadata_constraint_op_from_index_op(op) else {
        continue;
      };
      usage.set_argv_index(argv_index);
      usage.set_omit(true);
      argv_index += 1;
      metadata_constraints.push(MetadataConstraintSpec {
        column_index: metadata_idx as i32,
        op: meta_op,
      });
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
    // Encode metadata constraints into idx_str so xFilter knows which argv
    // values correspond to which columns and operators.
    info.set_idx_str(&encode_metadata_constraints(&metadata_constraints));

    if !has_match {
      // Allow plans without a vector constraint so that UPDATE/DELETE by rowid
      // can proceed.  Such plans are very expensive and return every row in
      // xFilter so SQLite can locate the target row; real vector queries should
      // always include a MATCH/EQ on the vector column.
      info.set_estimated_cost(1_000_000.0);
      return Ok(true);
    }

    // A low estimated cost encourages SQLite to choose the vector-index plan.
    info.set_estimated_cost(1000.0);
    Ok(true)
  }

  fn open(&mut self) -> Result<Self::Cursor> {
    let (dim, element_type) = match &self.columns[self.vector_column_index as usize].sql_type {
      SqlType::Vector { element_type, dim } => (*dim, *element_type),
      _ => unreachable!(),
    };

    let metadata_columns: Vec<MetadataColumn> = self
      .columns
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

    let mut metadata_column_map = vec![None; self.columns.len()];
    for (metadata_idx, virtual_idx) in self
      .columns
      .iter()
      .enumerate()
      .filter(|(_, c)| !matches!(c.sql_type, SqlType::Vector { .. }))
      .zip(0usize..)
    {
      metadata_column_map[metadata_idx.0] = Some(virtual_idx);
    }

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
      metadata_columns,
      metadata_cache: Vec::new(),
      metadata_column_map,
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
    let element_type = self.element_type()?;
    let dim = self.dim()?;
    let vector = deserialize_vector(element_type, dim, &embedding).map_err(|e| {
      Error::ModuleError(format!(
        "invalid embedding for {}[{}] index: {}",
        element_type.as_str(),
        dim,
        e
      ))
    })?;
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

    let conn = unsafe { Connection::from_handle(self.db)? };
    let element_type = self.element_type()?;
    let dim = self.dim()?;
    let vector = match embedding {
      Some(blob) => deserialize_vector(element_type, dim, &blob).map_err(|e| {
        Error::ModuleError(format!(
          "invalid embedding for {}[{}] index: {}",
          element_type.as_str(),
          dim,
          e
        ))
      })?,
      None => self.index.read_vector(&conn, old_rowid).map_err(|e| Error::ModuleError(e.to_string()))?,
    };
    let metadata = self.extract_metadata(args)?;

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
  fn filter(&mut self, idx_num: c_int, idx_str: Option<&str>, args: &Filters<'_>) -> Result<()> {
    // idx_num is a bitmask produced by best_index telling us which constraint
    // values were passed in through `args`.
    let has_match = (idx_num & 1) != 0;
    let has_k = (idx_num & 2) != 0;

    // Build a temporary Connection from the raw db handle stored in the cursor.
    let conn = unsafe { Connection::from_handle(self.db)? };

    if !has_match {
      // A plan without a vector constraint is used by SQLite for UPDATE/DELETE
      // by rowid.  Return every row so SQLite can locate the target row.
      self.topk = DEFAULT_TOPK;
      let rowids = self.index.scan(&conn).map_err(|e| Error::ModuleError(e.to_string()))?;
      self.results = rowids.into_iter().map(|rowid| ScoredRowId { rowid, score: 0.0 }).collect();
      self.position = 0;
      self.cache_metadata(&conn)?;
      return Ok(());
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

    // Read metadata constraint values in the same order that best_index assigned
    // argv indices and convert them into predicates for the index layer.
    let metadata_specs = decode_metadata_constraints(idx_str)?;
    let mut metadata_constraints = Vec::with_capacity(metadata_specs.len());
    let start_index = if has_k { 2 } else { 1 };
    for (offset, spec) in metadata_specs.iter().enumerate() {
      let arg_index = start_index + offset;
      let value: Value = args.get(arg_index)?;
      let value = MetadataValue::try_from(value)
        .map_err(|e| Error::ModuleError(format!("invalid metadata constraint value: {}", e)))?;
      metadata_constraints.push(MetadataConstraint {
        column_index: spec.column_index as usize,
        op: spec.op,
        value,
      });
    }

    // Convert the raw BLOB into a typed Vector, validating element type and dim.
    let query_vector = deserialize_vector(self.element_type, self.dim, &query_blob).map_err(|e| {
      Error::ModuleError(format!(
        "invalid query vector for {}[{}] index: {}",
        self.element_type.as_str(),
        self.dim,
        e
      ))
    })?;

    // Run the KNN search through the underlying HybridIndex.
    let result = self
      .index
      .search_vector(
        &conn,
        &VectorQuery {
          vector: query_vector,
          topk: self.topk,
          constraints: metadata_constraints,
        },
      )
      .map_err(|e| Error::ModuleError(e.to_string()))?;

    // Cache the scored results and reset the cursor position so iteration
    // starts at the first hit.
    self.results = result.hits;
    self.position = 0;
    self.cache_metadata(&conn)?;
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
      // Metadata columns return the cached value read from the shadow table.
      if let Some(metadata_idx) = self.metadata_column_map[i as usize] {
        let value = &self.metadata_cache[self.position][metadata_idx];
        match value {
          Some(v) => ctx.set_result(v)?,
          None => ctx.set_result(&Value::Null)?,
        }
        return Ok(());
      }
      // Any other non-metadata, non-vector column returns NULL.
      ctx.set_result(&Value::Null)?;
      return Ok(());
    }
    Err(Error::ModuleError(format!("unknown column index: {}", i)))
  }

  fn rowid(&self) -> Result<i64> {
    Ok(self.results[self.position].rowid)
  }
}

impl LitehybridCursor {
  /// Read metadata values for each row in `self.results` and cache them so that
  /// `xColumn` can return metadata without querying the shadow table repeatedly.
  fn cache_metadata(&mut self, conn: &Connection) -> Result<()> {
    self.metadata_cache.clear();
    self.metadata_cache.reserve(self.results.len());
    for hit in &self.results {
      let values = self.index.read_metadata(conn, hit.rowid).map_err(|e| Error::ModuleError(e.to_string()))?;
      self.metadata_cache.push(values);
    }
    Ok(())
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

  #[test]
  fn encode_and_decode_metadata_constraints() {
    let constraints = vec![
      MetadataConstraintSpec {
        column_index: 2,
        op: MetadataConstraintOp::Eq,
      },
      MetadataConstraintSpec {
        column_index: 3,
        op: MetadataConstraintOp::Gt,
      },
    ];
    let encoded = encode_metadata_constraints(&constraints);
    assert_eq!(encoded, "2:=,3:>");

    let decoded = decode_metadata_constraints(Some(&encoded)).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].column_index, 2);
    assert_eq!(decoded[0].op, MetadataConstraintOp::Eq);
    assert_eq!(decoded[1].column_index, 3);
    assert_eq!(decoded[1].op, MetadataConstraintOp::Gt);
  }

  #[test]
  fn decode_empty_metadata_constraints() {
    assert!(decode_metadata_constraints(None).unwrap().is_empty());
    assert!(decode_metadata_constraints(Some("")).unwrap().is_empty());
  }
}
