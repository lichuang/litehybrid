//! Distance and similarity metrics for dense vectors.

use crate::Metric;

/// Compute the distance between two vectors according to the given metric.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
  match metric {
    Metric::L2 => l2_distance(a, b),
    Metric::Cosine => cosine_distance(a, b),
    Metric::Dot => dot_distance(a, b),
  }
}

/// Squared Euclidean distance. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
  assert_equal_length(a, b);
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| {
      let d = x - y;
      d * d
    })
    .sum()
}

/// Cosine distance, defined as `1 - cosine_similarity`. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
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

/// Negative dot product. Lower is better.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
  assert_equal_length(a, b);
  -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

fn assert_equal_length(a: &[f32], b: &[f32]) {
  assert_eq!(a.len(), b.len(), "vector length mismatch: {} vs {}", a.len(), b.len());
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn l2_between_orthogonal_unit_vectors() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((l2_distance(&a, &b) - 2.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_same_vector_is_zero() {
    let a = [1.0f32, 2.0, 3.0];
    assert!(cosine_distance(&a, &a).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_orthogonal_vectors_is_one() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_between_opposite_vectors_is_two() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [-1.0f32, 0.0, 0.0];
    assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
  }

  #[test]
  fn dot_distance_orders_correctly() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    // dot(a, b) == 0, so distance is -0 == 0
    assert!(dot_distance(&a, &b).abs() < 1e-6);
  }

  #[test]
  fn distance_dispatcher_matches_l2() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert!((distance(Metric::L2, &a, &b) - l2_distance(&a, &b)).abs() < 1e-6);
  }

  #[test]
  #[should_panic(expected = "vector length mismatch")]
  fn mismatched_lengths_panic() {
    let a = [1.0f32, 0.0];
    let b = [1.0f32, 0.0, 0.0];
    l2_distance(&a, &b);
  }
}
