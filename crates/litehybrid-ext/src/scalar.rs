//! SQLite scalar functions for litehybrid.

use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, Error, Result};
use std::fmt;

use litehybrid_core::Vector;

/// Error returned when parsing a vector literal fails.
#[derive(Debug)]
pub struct VecParseError(String);

impl fmt::Display for VecParseError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl std::error::Error for VecParseError {}

/// Extract the inner content of a bracketed vector literal such as `'[...]'`.
fn parse_bracketed(s: &str) -> Result<&str, VecParseError> {
  let trimmed = s.trim();
  if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
    return Err(VecParseError(format!(
      "expected a vector literal like '[...]', got '{}'",
      s
    )));
  }
  Ok(trimmed[1..trimmed.len() - 1].trim())
}

/// Parse a string vector literal such as `'[1.0, 2.0, 3.0]'` into `Vec<f32>`.
///
/// The literal must be surrounded by square brackets and elements are separated
/// by commas. Arbitrary whitespace around brackets and elements is allowed.
pub fn parse_vec_f32(s: &str) -> Result<Vec<f32>, VecParseError> {
  let inner = parse_bracketed(s)?;
  if inner.is_empty() {
    return Ok(Vec::new());
  }

  inner
    .split(',')
    .map(|part| {
      let part = part.trim();
      part
        .parse::<f32>()
        .map_err(|e| VecParseError(format!("invalid float value '{}' in vector literal: {}", part, e)))
    })
    .collect()
}

/// Parse a string vector literal such as `'[10, -20, 30]'` into `Vec<i8>`.
///
/// The literal must be surrounded by square brackets and elements are separated
/// by commas. Arbitrary whitespace around brackets and elements is allowed.
pub fn parse_vec_int8(s: &str) -> Result<Vec<i8>, VecParseError> {
  let inner = parse_bracketed(s)?;
  if inner.is_empty() {
    return Ok(Vec::new());
  }

  inner
    .split(',')
    .map(|part| {
      let part = part.trim();
      part
        .parse::<i8>()
        .map_err(|e| VecParseError(format!("invalid int8 value '{}' in vector literal: {}", part, e)))
    })
    .collect()
}

/// Parse a string vector literal such as `'[1, 0, 1, 1]'` into packed bit bytes.
///
/// Each element must be `0` or `1`. Bits are packed least-significant bit first.
/// Returns the packed bytes and the original dimension (number of bits).
pub fn parse_vec_bit(s: &str) -> Result<(Vec<u8>, usize), VecParseError> {
  let inner = parse_bracketed(s)?;
  if inner.is_empty() {
    return Ok((Vec::new(), 0));
  }

  let values: Vec<u8> = inner
    .split(',')
    .map(|part| {
      let part = part.trim();
      part
        .parse::<u8>()
        .map_err(|e| VecParseError(format!("invalid bit value '{}' in vector literal: {}", part, e)))
        .and_then(|v| {
          if v <= 1 {
            Ok(v)
          } else {
            Err(VecParseError(format!(
              "bit vector elements must be 0 or 1, got '{}'",
              part
            )))
          }
        })
    })
    .collect::<Result<_, _>>()?;

  Ok(pack_bits(&values))
}

/// Pack a slice of `0`/`1` values into bytes, least-significant bit first.
fn pack_bits(values: &[u8]) -> (Vec<u8>, usize) {
  let dim = values.len();
  let byte_len = dim.div_ceil(8);
  let mut data = vec![0u8; byte_len];
  for (i, &bit) in values.iter().enumerate() {
    if bit == 1 {
      data[i / 8] |= 1 << (i % 8);
    }
  }
  (data, dim)
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
      Ok(Vector::F32(vector).serialize())
    },
  )?;

  conn.create_scalar_function(
    "vec_int8",
    1,
    FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
    |ctx| {
      let text: String = ctx.get(0)?;
      let vector = parse_vec_int8(&text).map_err(|e| Error::UserFunctionError(Box::new(e)))?;
      Ok(Vector::Int8(vector).serialize())
    },
  )?;

  conn.create_scalar_function(
    "vec_bit",
    1,
    FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
    |ctx| {
      let text: String = ctx.get(0)?;
      let (data, dim) = parse_vec_bit(&text).map_err(|e| Error::UserFunctionError(Box::new(e)))?;
      Ok(Vector::Bit { data, dim }.serialize())
    },
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_simple_f32_vector() {
    assert_eq!(parse_vec_f32("[1.0, 2.0, 3.0]").unwrap(), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn parse_integer_elements_as_f32() {
    assert_eq!(parse_vec_f32("[1, 2, 3]").unwrap(), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn parse_f32_with_whitespace() {
    assert_eq!(parse_vec_f32(" [ 1.0 , 2.0 ] ").unwrap(), vec![1.0, 2.0]);
  }

  #[test]
  fn parse_empty_f32_vector() {
    assert_eq!(parse_vec_f32("[]").unwrap(), Vec::<f32>::new());
    assert_eq!(parse_vec_f32("[  ]").unwrap(), Vec::<f32>::new());
  }

  #[test]
  fn parse_f32_missing_brackets_fails() {
    assert!(parse_vec_f32("1.0, 2.0").is_err());
  }

  #[test]
  fn parse_f32_invalid_number_fails() {
    assert!(parse_vec_f32("[1.0, foo]").is_err());
  }

  #[test]
  fn parse_simple_int8_vector() {
    assert_eq!(parse_vec_int8("[10, -20, 30]").unwrap(), vec![10, -20, 30]);
  }

  #[test]
  fn parse_int8_out_of_range_fails() {
    assert!(parse_vec_int8("[10, 200]").is_err());
    assert!(parse_vec_int8("[10, -200]").is_err());
  }

  #[test]
  fn parse_empty_int8_vector() {
    assert_eq!(parse_vec_int8("[]").unwrap(), Vec::<i8>::new());
  }

  #[test]
  fn parse_simple_bit_vector() {
    let (data, dim) = parse_vec_bit("[1, 0, 1, 1]").unwrap();
    assert_eq!(dim, 4);
    // LSB-first: positions 0=1, 1=0, 2=1, 3=1 -> 0b0000_1101
    assert_eq!(data, vec![0b0000_1101]);
  }

  #[test]
  fn parse_bit_vector_packs_lsb_first() {
    // 10 bits -> 2 bytes
    let (data, dim) = parse_vec_bit("[1, 1, 0, 1, 0, 0, 1, 0, 1, 0]").unwrap();
    assert_eq!(dim, 10);
    assert_eq!(data[0], 0b0100_1011); // bits 0-7
    assert_eq!(data[1] & 0b0000_0011, 0b0000_0001); // bits 8-9
  }

  #[test]
  fn parse_bit_invalid_value_fails() {
    assert!(parse_vec_bit("[1, 2, 0]").is_err());
    assert!(parse_vec_bit("[1, -1, 0]").is_err());
  }

  #[test]
  fn parse_empty_bit_vector() {
    let (data, dim) = parse_vec_bit("[]").unwrap();
    assert!(data.is_empty());
    assert_eq!(dim, 0);
  }

  #[test]
  fn vec_f32_serializes_to_little_endian_blob() {
    let blob = Vector::F32(vec![1.0f32, 2.0, 3.0]).serialize();
    assert_eq!(blob.len(), 12);
    let mut deserialized = Vec::with_capacity(3);
    for chunk in blob.chunks_exact(4) {
      deserialized.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(deserialized, vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn vec_int8_serializes_to_raw_bytes() {
    let blob = Vector::Int8(vec![10i8, -20, 30]).serialize();
    assert_eq!(blob, vec![10u8, 236, 30]);
  }

  #[test]
  fn vec_bit_serializes_to_packed_bytes() {
    let blob = Vector::Bit {
      data: vec![0b0000_1011],
      dim: 4,
    }
    .serialize();
    assert_eq!(blob, vec![0b0000_1011]);
  }
}
