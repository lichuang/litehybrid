//! SQLite loadable extension for litehybrid.

use rusqlite::vtab::Module;
use rusqlite::{Connection, Result};

mod scalar;
mod vtab;

use scalar::register_scalar_functions;
use vtab::LitehybridVTab;

/// Register the `litehybrid` module on the given connection.
///
/// This is public so that unit tests below can register the module on an
/// in-memory connection when the `extension` feature is disabled. It is not
/// part of the loadable-extension public API.
#[doc(hidden)]
pub fn register_module(conn: &Connection) -> Result<()> {
  static MODULE: Module<'static, LitehybridVTab> = Module::update_module();
  conn.create_module("litehybrid", &MODULE, None)?;
  register_scalar_functions(conn)
}

#[cfg(feature = "extension")]
mod entry_point {
  use std::ffi::{c_char, c_int};

  use rusqlite::ffi;
  use rusqlite::{Connection, Result};

  use super::register_module;

  /// SQLite extension entry point.
  ///
  /// # Safety
  ///
  /// Must be called by SQLite with a valid `db` handle, optional error message
  /// pointer, and the extension API routines pointer, as per the
  /// `sqlite3_extension_init` contract.
  #[unsafe(no_mangle)]
  pub unsafe extern "C" fn sqlite3_extension_init(
    db: *mut ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
  ) -> c_int {
    unsafe { Connection::extension_init2(db, pz_err_msg, p_api, init_extension) }
  }

  fn init_extension(conn: Connection) -> Result<bool> {
    register_module(&conn)?;
    Ok(false)
  }
}

#[cfg(not(feature = "extension"))]
mod fallback_entry_point {
  use std::ffi::{c_char, c_int, c_void};
  use std::ptr;

  use rusqlite::ffi;

  /// Fallback entry point used when the crate is built without the
  /// `extension` feature. It reports a clear error so users know how to
  /// rebuild the loadable extension.
  ///
  /// # Safety
  ///
  /// Must be called by SQLite with a valid optional error message pointer,
  /// as per the `sqlite3_extension_init` contract. This function assumes the
  /// host process exports `sqlite3_malloc` so the error message can be
  /// allocated.
  #[unsafe(no_mangle)]
  pub unsafe extern "C" fn sqlite3_extension_init(
    _db: *mut ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    _p_api: *mut ffi::sqlite3_api_routines,
  ) -> c_int {
    unsafe extern "C" {
      fn sqlite3_malloc(n: usize) -> *mut c_void;
    }

    const MSG: &str = "litehybrid-ext was built without the 'extension' feature; rebuild with: cargo build -p litehybrid-ext --features extension";

    unsafe {
      if !pz_err_msg.is_null() {
        let bytes = MSG.as_bytes();
        let ptr = sqlite3_malloc(bytes.len() + 1) as *mut c_char;
        if !ptr.is_null() {
          ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), ptr, bytes.len());
          ptr.add(bytes.len()).write(0);
          *pz_err_msg = ptr;
        }
      }
    }

    ffi::SQLITE_ERROR
  }
}

#[cfg(all(test, not(feature = "extension")))]
mod tests {
  use super::*;

  fn in_memory_db() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    register_module(&db).unwrap();
    db
  }

  #[test]
  fn vec_f32_scalar_function() {
    let db = in_memory_db();
    let blob: Vec<u8> = db.query_row("SELECT vec_f32('[1.0, 2.0, 3.0]')", [], |row| row.get(0)).unwrap();

    let expected = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    assert_eq!(blob, expected);
  }

  #[test]
  fn create_virtual_table_and_search_with_vec_f32() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx USING litehybrid(embedding float[3], metric='l2', index='flat')",
      [],
    )
    .unwrap();

    db.execute(
      "INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))",
      [],
    )
    .unwrap();

    let mut stmt = db
      .prepare("SELECT rowid, distance FROM idx WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') LIMIT 2")
      .unwrap();
    let rows: Vec<(i64, f32)> =
      stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().collect::<Result<_>>().unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
  }

  #[test]
  fn create_virtual_table_and_search_with_vec_int8() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx_i8 USING litehybrid(embedding int8[3], metric='l2', index='flat')",
      [],
    )
    .unwrap();

    db.execute(
      "INSERT INTO idx_i8(rowid, embedding) VALUES (1, vec_int8('[10, 0, 0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_i8(rowid, embedding) VALUES (2, vec_int8('[0, 10, 0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_i8(rowid, embedding) VALUES (3, vec_int8('[0, 0, 10]'))",
      [],
    )
    .unwrap();

    let mut stmt = db
      .prepare("SELECT rowid, distance FROM idx_i8 WHERE embedding = vec_int8('[10, 1, 1]') LIMIT 2")
      .unwrap();
    let rows: Vec<(i64, f32)> =
      stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().collect::<Result<_>>().unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
  }

  #[test]
  fn create_virtual_table_and_search_with_vec_bit() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx_bit USING litehybrid(embedding bit[4], index='flat')",
      [],
    )
    .unwrap();

    db.execute(
      "INSERT INTO idx_bit(rowid, embedding) VALUES (1, vec_bit('[1, 0, 0, 0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_bit(rowid, embedding) VALUES (2, vec_bit('[0, 1, 0, 0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_bit(rowid, embedding) VALUES (3, vec_bit('[0, 0, 1, 0]'))",
      [],
    )
    .unwrap();

    let mut stmt = db
      .prepare("SELECT rowid, distance FROM idx_bit WHERE embedding = vec_bit('[1, 0, 1, 0]') LIMIT 2")
      .unwrap();
    let rows: Vec<(i64, f32)> =
      stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().collect::<Result<_>>().unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
  }

  #[test]
  fn dynamic_schema_includes_metadata_columns() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx_schema USING litehybrid(embedding float[3], category text, year int, metric='l2')",
      [],
    )
    .unwrap();

    let mut stmt = db.prepare("PRAGMA table_xinfo(idx_schema)").unwrap();
    let columns: Vec<(String, String, i32)> = stmt
      .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(6)?)))
      .unwrap()
      .collect::<Result<_>>()
      .unwrap();

    assert_eq!(
      columns,
      vec![
        ("embedding".to_string(), "BLOB".to_string(), 0),
        ("category".to_string(), "TEXT".to_string(), 0),
        ("year".to_string(), "INT".to_string(), 0),
        ("distance".to_string(), "REAL".to_string(), 1),
        ("k".to_string(), "INT".to_string(), 1),
      ]
    );
  }

  #[test]
  fn metadata_constraint_filters_search_results() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx_meta USING litehybrid(embedding float[3], category text, metric='l2')",
      [],
    )
    .unwrap();

    db.execute(
      "INSERT INTO idx_meta(rowid, embedding, category) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'), 'tech')",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_meta(rowid, embedding, category) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'), 'science')",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_meta(rowid, embedding, category) VALUES (3, vec_f32('[0.9, 0.0, 0.0]'), 'tech')",
      [],
    )
    .unwrap();

    let mut stmt = db
      .prepare("SELECT rowid, category FROM idx_meta WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') AND category = 'tech' ORDER BY distance")
      .unwrap();
    let rows: Vec<(i64, String)> =
      stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?))).unwrap().collect::<Result<_>>().unwrap();

    assert_eq!(rows, vec![(1, "tech".to_string()), (3, "tech".to_string())]);
  }

  #[test]
  fn metadata_constraint_with_vector_column_not_first() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx_meta2 USING litehybrid(category text, embedding float[3], year int, metric='l2')",
      [],
    )
    .unwrap();

    db.execute(
      "INSERT INTO idx_meta2(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_meta2(rowid, category, year, embedding) VALUES (2, 'science', 2023, vec_f32('[0.0, 1.0, 0.0]'))",
      [],
    )
    .unwrap();
    db.execute(
      "INSERT INTO idx_meta2(rowid, category, year, embedding) VALUES (3, 'tech', 2022, vec_f32('[0.9, 0.0, 0.0]'))",
      [],
    )
    .unwrap();

    let mut stmt = db
      .prepare(
        "SELECT rowid, category, year FROM idx_meta2 WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') \
         AND category = 'tech' AND year > 2020 ORDER BY distance",
      )
      .unwrap();
    let rows: Vec<(i64, String, i64)> = stmt
      .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
      .unwrap()
      .collect::<Result<_>>()
      .unwrap();

    assert_eq!(rows, vec![(1, "tech".to_string(), 2024), (3, "tech".to_string(), 2022)]);
  }
}
