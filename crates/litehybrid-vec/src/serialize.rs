//! Serialization and deserialization of typed vectors to/from SQLite BLOBs.

use crate::{Vector, VectorElementType};

/// Number of bytes required to store `dim` bits.
fn bit_byte_len(dim: usize) -> usize {
  dim.div_ceil(8)
}

/// Serialize a slice of `f32` values into a little-endian byte vector.
fn serialize_f32(vector: &[f32]) -> Vec<u8> {
  vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deserialize little-endian `f32` bytes into a vector.
///
/// Returns an error if the BLOB length does not match `dim * 4`.
fn deserialize_f32(blob: &[u8], dim: usize) -> Result<Vec<f32>, SerializationError> {
  let expected = dim * 4;
  if blob.len() != expected {
    return Err(SerializationError::LengthMismatch {
      expected,
      got: blob.len(),
    });
  }
  let mut vector = Vec::with_capacity(dim);
  for chunk in blob.chunks_exact(4) {
    let bytes: [u8; 4] = chunk.try_into().expect("chunk size is 4");
    vector.push(f32::from_le_bytes(bytes));
  }
  Ok(vector)
}

/// Serialize a slice of `i8` values into raw bytes.
fn serialize_int8(vector: &[i8]) -> Vec<u8> {
  vector.iter().map(|v| *v as u8).collect()
}

/// Deserialize raw bytes into a vector of `i8` values.
///
/// Returns an error if the BLOB length does not match `dim`.
fn deserialize_int8(blob: &[u8], dim: usize) -> Result<Vec<i8>, SerializationError> {
  if blob.len() != dim {
    return Err(SerializationError::LengthMismatch {
      expected: dim,
      got: blob.len(),
    });
  }
  Ok(blob.iter().map(|v| *v as i8).collect())
}

/// Deserialize packed bit bytes.
///
/// Returns an error if the BLOB length does not match `(dim + 7) / 8`.
fn deserialize_bit(blob: &[u8], dim: usize) -> Result<Vec<u8>, SerializationError> {
  let expected = bit_byte_len(dim);
  if blob.len() != expected {
    return Err(SerializationError::LengthMismatch {
      expected,
      got: blob.len(),
    });
  }
  Ok(blob.to_vec())
}

/// Deserialize a BLOB into a `Vector` according to its element type and dimension.
pub fn deserialize_vector(
  element_type: VectorElementType,
  dim: usize,
  blob: &[u8],
) -> Result<Vector, SerializationError> {
  match element_type {
    VectorElementType::F32 => deserialize_f32(blob, dim).map(Vector::F32),
    VectorElementType::Int8 => deserialize_int8(blob, dim).map(Vector::Int8),
    VectorElementType::Bit => deserialize_bit(blob, dim).map(|data| Vector::Bit { data, dim }),
  }
}

/// Errors that can occur during vector serialization/deserialization.
#[derive(Debug, PartialEq)]
pub enum SerializationError {
  /// The BLOB length does not match the expected size for the element type and
  /// dimension.
  LengthMismatch {
    /// Expected byte length.
    expected: usize,
    /// Actual byte length.
    got: usize,
  },
}

impl std::fmt::Display for SerializationError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      SerializationError::LengthMismatch { expected, got } => {
        write!(f, "blob length mismatch: expected {} bytes, got {}", expected, got)
      }
    }
  }
}

impl std::error::Error for SerializationError {}

impl Vector {
  /// Serialize this vector into a BLOB.
  pub fn serialize(&self) -> Vec<u8> {
    match self {
      Vector::F32(v) => serialize_f32(v),
      Vector::Int8(v) => serialize_int8(v),
      Vector::Bit { data, .. } => data.clone(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn f32_round_trip() {
    let vector = [1.0f32, 2.0, 3.0];
    let blob = serialize_f32(&vector);
    assert_eq!(blob.len(), 12);
    assert_eq!(deserialize_f32(&blob, 3).unwrap(), vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn f32_length_mismatch() {
    let blob = serialize_f32(&[1.0f32, 2.0]);
    assert_eq!(
      deserialize_f32(&blob, 3),
      Err(SerializationError::LengthMismatch { expected: 12, got: 8 })
    );
  }

  #[test]
  fn int8_round_trip() {
    let vector = [10i8, -20, 127, -128];
    let blob = serialize_int8(&vector);
    assert_eq!(blob, vec![10u8, 236, 127, 128]);
    assert_eq!(deserialize_int8(&blob, 4).unwrap(), vec![10, -20, 127, -128]);
  }

  #[test]
  fn int8_length_mismatch() {
    let blob = serialize_int8(&[1i8, 2i8]);
    assert_eq!(
      deserialize_int8(&blob, 3),
      Err(SerializationError::LengthMismatch { expected: 3, got: 2 })
    );
  }

  #[test]
  fn bit_round_trip() {
    let dim = 10;
    let data = vec![0b0000_0011u8, 0b0000_0001u8];
    let vector = Vector::Bit {
      data: data.clone(),
      dim,
    };
    let blob = vector.serialize();
    assert_eq!(deserialize_bit(&blob, dim).unwrap(), data);
  }

  #[test]
  fn bit_length_mismatch() {
    let blob = vec![0b0000_0011u8];
    assert_eq!(
      deserialize_bit(&blob, 10),
      Err(SerializationError::LengthMismatch { expected: 2, got: 1 })
    );
  }

  #[test]
  fn vector_serialize_dispatch() {
    assert_eq!(Vector::F32(vec![1.0, 2.0]).serialize(), serialize_f32(&[1.0, 2.0]));
    assert_eq!(Vector::Int8(vec![1, 2]).serialize(), serialize_int8(&[1, 2]));
    assert_eq!(
      Vector::Bit {
        data: vec![0b0000_0011],
        dim: 7
      }
      .serialize(),
      vec![0b0000_0011]
    );
  }

  #[test]
  fn deserialize_vector_dispatch() {
    let f32_blob = serialize_f32(&[1.0, 2.0]);
    assert_eq!(
      deserialize_vector(VectorElementType::F32, 2, &f32_blob).unwrap(),
      Vector::F32(vec![1.0, 2.0])
    );

    let int8_blob = serialize_int8(&[1, 2]);
    assert_eq!(
      deserialize_vector(VectorElementType::Int8, 2, &int8_blob).unwrap(),
      Vector::Int8(vec![1, 2])
    );

    let bit_blob = vec![0b0000_0011u8];
    assert_eq!(
      deserialize_vector(VectorElementType::Bit, 7, &bit_blob).unwrap(),
      Vector::Bit { data: bit_blob, dim: 7 }
    );
  }
}
