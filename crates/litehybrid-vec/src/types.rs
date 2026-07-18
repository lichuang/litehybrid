//! Core types used throughout the litehybrid search engine.

/// SQLite rowid type. All indexed documents are identified by this value.
pub type RowId = i64;

/// A single search hit, pairing a rowid with its relevance score.
///
/// For distance-based metrics (L2, cosine distance), lower scores are better.
/// For similarity-style metrics, the score semantics are defined by the
/// underlying index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredRowId {
    /// Identifies the matched document.
    pub rowid: RowId,
    /// Relevance or distance score. Lower is better for distance metrics.
    pub score: f32,
}

/// A vector-based search query.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
    /// Query embedding. Its length must match the dimension configured on
    /// the index.
    pub vector: Vec<f32>,
    /// Maximum number of results to return.
    pub topk: usize,
}

/// Distance or similarity metric used to compare dense vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared Euclidean distance. Lower is better.
    L2,
    /// Cosine distance, defined as `1 - cosine_similarity`. Lower is better.
    Cosine,
    /// Negative dot product. Lower is better.
    Dot,
}

/// Result of a search operation.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Matching documents ordered by score (best first).
    pub hits: Vec<ScoredRowId>,
}

impl SearchResult {
    /// Create an empty result.
    pub fn empty() -> Self {
        Self { hits: Vec::new() }
    }

    /// Create a result from a list of hits.
    pub fn new(hits: Vec<ScoredRowId>) -> Self {
        Self { hits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_row_id_orders_by_score() {
        let a = ScoredRowId {
            rowid: 1,
            score: 0.1,
        };
        let b = ScoredRowId {
            rowid: 2,
            score: 0.2,
        };
        assert!(a.score < b.score);
        assert_ne!(a, b);
    }

    #[test]
    fn search_result_empty() {
        let r = SearchResult::empty();
        assert!(r.hits.is_empty());
    }
}
