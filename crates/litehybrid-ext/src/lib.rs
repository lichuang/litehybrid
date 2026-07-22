//! SQLite loadable extension for litehybrid.

use rusqlite::vtab::Module;
use rusqlite::{Connection, Result};

mod vtab;

use vtab::LitehybridVTab;

/// Register the `litehybrid` module on the given connection.
///
/// This is public so that unit tests below can register the module on an
/// in-memory connection when the `extension` feature is disabled. It is not
/// part of the loadable-extension public API.
#[doc(hidden)]
pub fn register_module(conn: &Connection) -> Result<()> {
  static MODULE: Module<'static, LitehybridVTab> = Module::update_module();
  conn.create_module("litehybrid", &MODULE, None)
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
  use rusqlite::params;

  fn serialize_f32(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
  }

  fn in_memory_db() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    register_module(&db).unwrap();
    db
  }

  #[test]
  fn create_virtual_table_and_search() {
    let db = in_memory_db();

    db.execute(
      "CREATE VIRTUAL TABLE idx USING litehybrid(dim=3, metric='l2', index='flat')",
      [],
    )
    .unwrap();

    let blob1 = serialize_f32(&[1.0, 0.0, 0.0]);
    let blob2 = serialize_f32(&[0.0, 1.0, 0.0]);
    let blob3 = serialize_f32(&[0.0, 0.0, 1.0]);

    db.execute("INSERT INTO idx(rowid, embedding) VALUES (1, ?1)", params![blob1]).unwrap();
    db.execute("INSERT INTO idx(rowid, embedding) VALUES (2, ?1)", params![blob2]).unwrap();
    db.execute("INSERT INTO idx(rowid, embedding) VALUES (3, ?1)", params![blob3]).unwrap();

    let query = serialize_f32(&[1.0, 0.1, 0.1]);
    let mut stmt = db.prepare("SELECT rowid, distance FROM idx WHERE embedding = ?1 LIMIT 2").unwrap();
    let rows: Vec<(i64, f32)> = stmt
      .query_map(params![query], |row| Ok((row.get(0)?, row.get(1)?)))
      .unwrap()
      .collect::<Result<_>>()
      .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
  }
}
