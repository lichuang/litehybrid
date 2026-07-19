//! Top-k candidate utilities for vector index searches.

use std::cmp::Ordering;

use crate::{RowId, ScoredRowId};

/// A single search result candidate used to build a top-k heap.
///
/// `Candidate` is crate-internal and not exposed to users of `litehybrid-vec`.
/// The ordering is deliberately by `score` so that a `BinaryHeap` behaves as a
/// max-heap: the worst-scoring candidate sits at the top and can be evicted
/// first when a better candidate arrives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) rowid: RowId,
    pub(crate) score: f32,
}

impl From<Candidate> for ScoredRowId {
    fn from(candidate: Candidate) -> Self {
        ScoredRowId {
            rowid: candidate.rowid,
            score: candidate.score,
        }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score).is_eq()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural ordering makes BinaryHeap a max-heap by score.
        // The largest score sits at the top so it can be evicted first.
        self.score.total_cmp(&other.score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_orders_by_score() {
        let a = Candidate {
            rowid: 1,
            score: 1.0,
        };
        let b = Candidate {
            rowid: 2,
            score: 2.0,
        };
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn converts_to_scored_row_id() {
        let candidate = Candidate {
            rowid: 42,
            score: 0.5,
        };
        let scored: ScoredRowId = candidate.into();
        assert_eq!(scored.rowid, 42);
        assert_eq!(scored.score, 0.5);
    }
}
