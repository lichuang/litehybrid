//! Distance and similarity metrics for dense vectors.

use crate::{IndexError, Vector, VectorElementType};

/// Distance or similarity metric used to compare dense vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
  /// Squared Euclidean distance. Lower is better.
  L2,
  /// Cosine distance, defined as `1 - cosine_similarity`. Lower is better.
  Cosine,
  /// Negative dot product. Lower is better.
  Dot,
  /// Hamming distance for binary vectors. Lower is better.
  Hamming,
}

impl Metric {
  /// Compute the distance between two f32 vectors according to this metric.
  ///
  /// # Panics
  ///
  /// Panics if `a` and `b` have different lengths, or if this metric is not
  /// defined for f32 vectors.
  pub fn distance_f32(self, a: &[f32], b: &[f32]) -> f32 {
    match self {
      Metric::L2 => l2_distance_f32(a, b),
      Metric::Cosine => cosine_distance_f32(a, b),
      Metric::Dot => dot_distance_f32(a, b),
      Metric::Hamming => panic!("Hamming distance is only defined for bit vectors"),
    }
  }

  /// Compute the distance between two int8 vectors according to this metric.
  ///
  /// Returns an error if this metric is not defined for int8 vectors.
  pub fn distance_i8(self, a: &[i8], b: &[i8]) -> Result<f32, IndexError> {
    match self {
      Metric::L2 => Ok(l2_distance_i8(a, b)),
      Metric::Cosine => Ok(cosine_distance_i8(a, b)),
      Metric::Dot => Ok(dot_distance_i8(a, b)),
      Metric::Hamming => Err(IndexError::UnsupportedMetricForType {
        metric: self,
        element_type: VectorElementType::Int8,
      }),
    }
  }

  /// Compute the distance between two typed vectors according to this metric.
  ///
  /// Returns an error if the element type and metric combination is not
  /// supported.
  pub fn distance_vector(self, a: &Vector, b: &Vector) -> Result<f32, IndexError> {
    match (a, b) {
      (Vector::F32(a), Vector::F32(b)) => Ok(self.distance_f32(a, b)),
      (Vector::Int8(a), Vector::Int8(b)) => self.distance_i8(a, b),
      (
        Vector::Bit { data: a, dim },
        Vector::Bit {
          data: b,
          dim: other_dim,
        },
      ) => {
        debug_assert_eq!(dim, other_dim);
        match self {
          Metric::Hamming => Ok(hamming_distance(a, b, *dim)),
          _ => Err(IndexError::UnsupportedMetricForType {
            metric: self,
            element_type: VectorElementType::Bit,
          }),
        }
      }
      _ => Err(IndexError::MismatchedElementTypes {
        left: a.element_type(),
        right: b.element_type(),
      }),
    }
  }
}

/// Squared Euclidean distance for f32 vectors. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
  assert_equal_length(a, b);
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| {
      let d = x - y;
      d * d
    })
    .sum()
}

/// Cosine distance for f32 vectors, defined as `1 - cosine_similarity`. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
  assert_equal_length(a, b);
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (x, y) in a.iter().zip(b.iter()) {
    dot += x * y;
    norm_a += x * x;
    norm_b += y * y;
  }
  let norm = norm_a.sqrt() * norm_b.sqrt();
  if norm == 0.0 { 1.0 } else { 1.0 - (dot / norm) }
}

/// Negative dot product for f32 vectors. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn dot_distance_f32(a: &[f32], b: &[f32]) -> f32 {
  assert_equal_length(a, b);
  -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Squared Euclidean distance for int8 vectors. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn l2_distance_i8(a: &[i8], b: &[i8]) -> f32 {
  assert_equal_length(a, b);
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| {
      let d = *x as f32 - *y as f32;
      d * d
    })
    .sum()
}

/// Cosine distance for int8 vectors. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn cosine_distance_i8(a: &[i8], b: &[i8]) -> f32 {
  assert_equal_length(a, b);
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (x, y) in a.iter().zip(b.iter()) {
    let xf = *x as f32;
    let yf = *y as f32;
    dot += xf * yf;
    norm_a += xf * xf;
    norm_b += yf * yf;
  }
  let norm = norm_a.sqrt() * norm_b.sqrt();
  if norm == 0.0 { 1.0 } else { 1.0 - (dot / norm) }
}

/// Negative dot product for int8 vectors. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn dot_distance_i8(a: &[i8], b: &[i8]) -> f32 {
  assert_equal_length(a, b);
  -a.iter().zip(b.iter()).map(|(x, y)| (*x as f32) * (*y as f32)).sum::<f32>()
}

/// Hamming distance for packed bit vectors. Lower is better.
///
/// Only the first `dim` bits are compared; remaining bits in the last byte are
/// masked off.
///
/// # Panics
///
/// Panics if `a` and `b` have different byte lengths.
pub fn hamming_distance(a: &[u8], b: &[u8], dim: usize) -> f32 {
  assert_equal_length(a, b);
  let full_bytes = dim / 8;
  let remaining_bits = dim % 8;
  let mut count = 0usize;

  for i in 0..full_bytes {
    count += (a[i] ^ b[i]).count_ones() as usize;
  }

  if remaining_bits > 0 {
    let mask = (1u8 << remaining_bits) - 1;
    count += ((a[full_bytes] ^ b[full_bytes]) & mask).count_ones() as usize;
  }

  count as f32
}

fn assert_equal_length<T>(a: &[T], b: &[T]) {
  assert_eq!(a.len(), b.len(), "vector length mismatch: {} vs {}", a.len(), b.len());
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn l2_between_orthogonal_unit_vectors() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((l2_distance_f32(&a, &b) - 2.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_same_vector_is_zero() {
    let a = [1.0f32, 2.0, 3.0];
    assert!(cosine_distance_f32(&a, &a).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_orthogonal_vectors_is_one() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((cosine_distance_f32(&a, &b) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_opposite_vectors_is_two() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [-1.0f32, 0.0, 0.0];
    assert!((cosine_distance_f32(&a, &b) - 2.0).abs() < 1e-6);
  }

  #[test]
  fn dot_distance_orders_correctly() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    // dot(a, b) == 0, so distance is -0 == 0
    assert!(dot_distance_f32(&a, &b).abs() < 1e-6);
  }

  #[test]
  fn distance_method_matches_l2() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((Metric::L2.distance_f32(&a, &b) - l2_distance_f32(&a, &b)).abs() < 1e-6);
  }

  #[test]
  fn distance_method_dispatches_all_metrics() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!(Metric::L2.distance_f32(&a, &b) > 0.0);
    assert!(Metric::Cosine.distance_f32(&a, &b) > 0.0);
    assert!(Metric::Dot.distance_f32(&a, &b).abs() < 1e-6);
  }

  #[test]
  #[should_panic(expected = "vector length mismatch")]
  fn mismatched_lengths_panic() {
    let a = [1.0f32, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    l2_distance_f32(&a, &b);
  }

  #[test]
  fn l2_i8_matches_f32() {
    let a_i8 = [1i8, 2, 3];
    let b_i8 = [4i8, 5, 6];
    let a_f32 = [1.0f32, 2.0, 3.0];
    let b_f32 = [4.0f32, 5.0, 6.0];
    assert!((l2_distance_i8(&a_i8, &b_i8) - l2_distance_f32(&a_f32, &b_f32)).abs() < 1e-6);
  }

  #[test]
  fn cosine_i8_matches_f32() {
    let a_i8 = [1i8, 0, 0];
    let b_i8 = [0i8, 1, 0];
    assert!((cosine_distance_i8(&a_i8, &b_i8) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn dot_i8_orders_correctly() {
    let a = [1i8, 0, 0];
    let b = [0i8, 1, 0];
    assert!(dot_distance_i8(&a, &b).abs() < 1e-6);
  }

  #[test]
  fn hamming_distance_counts_differing_bits() {
    // bits: 0b00000011 vs 0b00000001 -> differ in 1 bit
    let a = vec![0b0000_0011u8];
    let b = vec![0b0000_0001u8];
    assert_eq!(hamming_distance(&a, &b, 8), 1.0);

    // only compare first 2 bits: 0b11 vs 0b01 -> differ in 1 bit
    assert_eq!(hamming_distance(&a, &b, 2), 1.0);

    // only compare first 1 bit: 0b1 vs 0b1 -> differ in 0 bits
    assert_eq!(hamming_distance(&a, &b, 1), 0.0);
  }

  #[test]
  fn hamming_distance_with_partial_byte() {
    let a = vec![0b1111_1111u8, 0b0000_0001u8];
    let b = vec![0b0000_0000u8, 0b0000_0010u8];
    // dim = 10: first byte 8 bits differ + next 2 bits differ
    assert_eq!(hamming_distance(&a, &b, 10), 10.0);
  }

  #[test]
  fn distance_vector_int8() {
    let a = Vector::Int8(vec![1, 0, 0]);
    let b = Vector::Int8(vec![0, 1, 0]);
    assert!((Metric::L2.distance_vector(&a, &b).unwrap() - 2.0).abs() < 1e-6);
  }

  #[test]
  fn distance_vector_bit_hamming() {
    let a = Vector::Bit {
      data: vec![0b0000_0011u8],
      dim: 8,
    };
    let b = Vector::Bit {
      data: vec![0b0000_0001u8],
      dim: 8,
    };
    assert_eq!(Metric::Hamming.distance_vector(&a, &b).unwrap(), 1.0);
  }

  #[test]
  fn distance_vector_rejects_mismatched_element_types() {
    let a = Vector::F32(vec![1.0, 0.0, 0.0]);
    let b = Vector::Int8(vec![1, 0, 0]);
    assert!(matches!(
      Metric::L2.distance_vector(&a, &b),
      Err(IndexError::MismatchedElementTypes {
        left: VectorElementType::F32,
        right: VectorElementType::Int8
      })
    ));
  }

  #[test]
  fn distance_vector_rejects_hamming_on_int8() {
    let a = Vector::Int8(vec![1, 0, 0]);
    let b = Vector::Int8(vec![0, 1, 0]);
    assert!(matches!(
      Metric::Hamming.distance_vector(&a, &b),
      Err(IndexError::UnsupportedMetricForType {
        metric: Metric::Hamming,
        element_type: VectorElementType::Int8
      })
    ));
  }

  #[test]
  fn distance_vector_rejects_l2_on_bit() {
    let a = Vector::Bit {
      data: vec![0b0000_0011u8],
      dim: 8,
    };
    let b = Vector::Bit {
      data: vec![0b0000_0001u8],
      dim: 8,
    };
    assert!(matches!(
      Metric::L2.distance_vector(&a, &b),
      Err(IndexError::UnsupportedMetricForType {
        metric: Metric::L2,
        element_type: VectorElementType::Bit
      })
    ));
  }
}
