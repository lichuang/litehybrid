//! SQLite scalar functions for litehybrid.

use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, Error, Result};
use std::fmt;

/// Error returned when parsing a vector literal fails.
#[derive(Debug)]
pub struct VecF32ParseError(String);

impl fmt::Display for VecF32ParseError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl std::error::Error for VecF32ParseError {}

/// Parse a string vector literal such as `'[1.0, 2.0, 3.0]'` into `Vec<f32>`.
///
/// The literal must be surrounded by square brackets and elements are separated
/// by commas. Arbitrary whitespace around brackets and elements is allowed.
pub fn parse_vec_f32(s: &str) -> Result<Vec<f32>, VecF32ParseError> {
  let trimmed = s.trim();
  if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
    return Err(VecF32ParseError(format!(
      "expected a vector literal like '[1.0, 2.0, 3.0]', got '{}'",
      s
    )));
  }

  let inner = trimmed[1..trimmed.len() - 1].trim();
  if inner.is_empty() {
    return Ok(Vec::new());
  }

  inner
    .split(',')
    .map(|part| {
      let part = part.trim();
      part
        .parse::<f32>()
        .map_err(|e| VecF32ParseError(format!("invalid float value '{}' in vector literal: {}", part, e)))
    })
    .collect()
}

/// Serialize a slice of `f32` values into a little-endian byte vector.
pub fn serialize_f32_vec(vector: &[f32]) -> Vec<u8> {
  vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Register all litehybrid scalar SQL functions on the given connection.
pub fn register_scalar_functions(conn: &Connection) -> Result<()> {
  conn.create_scalar_function(
    "vec_f32",
    1,
    FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
    |ctx| {
      let text: String = ctx.get(0)?;
      let vector = parse_vec_f32(&text).map_err(|e| Error::UserFunctionError(Box::new(e)))?;
      Ok(serialize_f32_vec(&vector))
    },
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_simple_vector() {
    assert_eq!(parse_vec_f32("[1.0, 2.0, 3.0]").unwrap(), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn parse_integer_elements() {
    assert_eq!(parse_vec_f32("[1, 2, 3]").unwrap(), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn parse_with_whitespace() {
    assert_eq!(parse_vec_f32(" [ 1.0 , 2.0 ] ").unwrap(), vec![1.0, 2.0]);
  }

  #[test]
  fn parse_empty_vector() {
    assert_eq!(parse_vec_f32("[]").unwrap(), Vec::<f32>::new());
    assert_eq!(parse_vec_f32("[  ]").unwrap(), Vec::<f32>::new());
  }

  #[test]
  fn parse_missing_brackets_fails() {
    assert!(parse_vec_f32("1.0, 2.0").is_err());
  }

  #[test]
  fn parse_invalid_number_fails() {
    assert!(parse_vec_f32("[1.0, foo]").is_err());
  }

  #[test]
  fn serialize_round_trip() {
    let vector = [1.0f32, 2.0, 3.0];
    let bytes = serialize_f32_vec(&vector);
    assert_eq!(bytes.len(), 12);
    let mut deserialized = Vec::with_capacity(vector.len());
    for chunk in bytes.chunks_exact(4) {
      deserialized.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(deserialized, vec![1.0, 2.0, 3.0]);
  }
}
