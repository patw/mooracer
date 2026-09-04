//! Vector index — brute-force cosine similarity over embedded documents.
//!
//! Design notes (perf posture, see spec "Indexes" / "Search" /
//! "Performance posture"):
//!
//! - A [`VectorIndex`] is created per **top-level field** with a fixed
//!   configured `dim`. Every document whose field holds a numeric array of
//!   exactly that length carries one entry; a document whose field is **missing**
//!   is simply not in the index (nothing to search), and a present field with
//!   the wrong length (or a non-numeric element) is a write error
//!   ([`StoreError::VectorDimMismatch`]).
//! - **Storage normalizes once at write time.** Each stored vector is unit
//!   normalized when it enters the index, so a search never re-normalizes the
//!   corpus: it normalizes the query *once*, then every score is a single
//!   dot product of two unit vectors — the cosine. This is the whole perf win
//!   of brute force: the per-query per-document work is one `f32` dot product
//!   over a contiguous `dim`-strided slice (no norm, no division, no per-doc
//!   allocation).
//! - **SIMD**: the dot product is a straight `zip`/`sum` over the slice; with
//!   `-C target-cpu=native` (see `.cargo/config.toml`) and the release profile
//!   it autovectorizes to FMA-wide SIMD. No hand-rolled intrinsic was measured
//!   to beat it, so we keep the portable loop (no `unsafe`).
//! - Layout: two parallel `Vec`s — `ids` (document `_id`s, insertion order) and
//!   a single flat `vecs: Vec<f32>` (`doc i` owns `vecs[i*dim..(i+1)*dim]`). The
//!   flat buffer is cache-friendly for the dot product and costs one
//!   `remove`/`swap_remove` (two memmoves) per delete instead of a boxed
//!   per-doc vector.
//! - No auto-embedding / inference: the engine only stores and compares the
//!   caller-supplied vectors (a `Value::Array` of numbers).

use crate::value::Value;

/// Cosine top-k result: the full document clone and its cosine score in
/// `[-1, 1]` (best / most similar first).
pub type VectorHit = (Value, f32);

/// A single-field vector index with a configured dimension.
pub struct VectorIndex {
    field: String,
    dim: usize,
    /// Document ids, in insertion order (index `i` owns vector `i`).
    ids: Vec<String>,
    /// Flat unit-normalized vectors: `ids[i]` → `vecs[i*dim..(i+1)*dim]`.
    vecs: Vec<f32>,
}

impl VectorIndex {
    /// Create an empty index over `field` with the configured `dim`.
    pub fn new(field: &str, dim: usize) -> Self {
        VectorIndex {
            field: field.to_string(),
            dim,
            ids: Vec::new(),
            vecs: Vec::new(),
        }
    }

    /// The indexed field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// The configured vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed documents (one entry per doc that carries a valid
    /// vector for the field).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` when no document is indexed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The document ids currently indexed (insertion order).
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Insert (or replace) `id`'s vector, unit-normalizing it on the way in.
    /// `values.len()` must equal the configured `dim`.
    pub fn insert(&mut self, id: String, values: &[f32]) {
        let n = self.ids.len();
        self.ids.push(id);
        let mut norm_sq = 0.0f32;
        for x in values {
            norm_sq += x * x;
        }
        let scale = if norm_sq > 0.0 { 1.0 / norm_sq.sqrt() } else { 1.0 };
        self.vecs.reserve(values.len());
        for x in values {
            self.vecs.push(x * scale);
        }
        debug_assert_eq!(self.vecs.len(), (n + 1) * self.dim);
    }

    /// Remove the entry for `id` (a no-op when absent). The flat buffer stays
    /// packed: a `swap_remove` collapses the tail over the gap.
    pub fn remove(&mut self, id: &str) -> bool {
        match self.ids.iter().position(|s| s == id) {
            None => false,
            Some(i) => {
                self.ids.swap_remove(i);
                // Drop the deleted doc's vector; the flat buffer stays packed.
                self.vecs.drain(i * self.dim..(i + 1) * self.dim);
                debug_assert_eq!(self.vecs.len(), self.ids.len() * self.dim);
                true
            }
        }
    }

    /// Brute-force cosine search over the indexed vectors.
    ///
    /// `query` must have length `dim`. `limit` is the top-k: the top `limit`
    /// documents by descending cosine score (ties by `_id`, ascending) are
    /// returned; `limit == 0` means "no limit" (return every entry, in
    /// best-first order). An empty index returns an empty vec.
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<(usize, f32)> {
        let n = self.ids.len();
        if n == 0 || query.len() != self.dim {
            return Vec::new();
        }
        // Normalize the query once; a zero query makes every cosine 0.
        let mut qnorm_sq = 0.0f32;
        for x in query {
            qnorm_sq += x * x;
        }
        let scale = if qnorm_sq > 0.0 { 1.0 / qnorm_sq.sqrt() } else { 0.0 };
        let q: Vec<f32> = query.iter().map(|x| x * scale).collect();

        // One dot product per document (autovectorized to SIMD).
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let v = &self.vecs[i * self.dim..(i + 1) * self.dim];
            let dot = dot(&q, v);
            scored.push((i, dot));
        }
        // Best first (descending score), ties by id (index order is stable, so
        // equal scores keep insertion order — deterministic for a given store).
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if limit > 0 {
            scored.truncate(limit);
        }
        scored
    }
}

/// Dot product of two equal-length `f32` slices. Written as a plain `zip`/`sum`
/// so `-C target-cpu=native` + release autovectorizes it to SIMD; no `unsafe`.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Coerce a document field value into a numeric vector of `dim` length, if the
/// field is present and valid. Returns `None` when the field is missing or is
/// not an array of exactly `dim` numbers (non-numeric elements included).
pub fn as_vector(v: Option<&Value>, dim: usize) -> Option<Vec<f32>> {
    let arr = v?.as_array()?;
    if arr.len() != dim {
        return None;
    }
    let mut out = Vec::with_capacity(dim);
    for e in arr {
        // I64 and F64 both convert to f32; any other type disqualifies.
        let f = match e {
            Value::I64(n) => *n as f32,
            Value::F64(x) => *x as f32,
            _ => return None,
        };
        out.push(f);
    }
    Some(out)
}

/// Present-and-valid check (does not allocate): the field holds a numeric array
/// of exactly `dim` elements. A missing field is `false`.
pub fn is_vector(v: Option<&Value>, dim: usize) -> bool {
    let arr = match v.and_then(Value::as_array) {
        Some(a) if a.len() == dim => a,
        _ => return false,
    };
    arr.iter()
        .all(|e| matches!(e, Value::I64(_) | Value::F64(_)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn vi(dim: usize) -> VectorIndex {
        VectorIndex::new("embedding", dim)
    }

    #[test]
    fn empty_index_searches_to_empty() {
        let ix = vi(3);
        assert!(ix.is_empty());
        assert_eq!(ix.search(&[1.0, 2.0, 3.0], 5), Vec::<(usize, f32)>::new());
    }

    #[test]
    fn as_vector_coerces_numbers_and_rejects_bad_shapes() {
        let good = Value::array_from(vec![
            Value::i64(1),
            Value::f64(2.0),
            Value::i64(3),
        ]);
        assert_eq!(as_vector(Some(&good), 3), Some(vec![1.0, 2.0, 3.0]));
        // wrong dim
        assert!(as_vector(Some(&good), 2).is_none());
        // non-numeric element
        let bad = Value::array_from(vec![Value::i64(1), Value::str("x"), Value::i64(3)]);
        assert!(as_vector(Some(&bad), 3).is_none());
        // not an array
        assert!(as_vector(Some(&Value::i64(5)), 3).is_none());
        // missing
        assert!(!is_vector(None, 3));
        assert!(is_vector(Some(&good), 3));
        assert!(!is_vector(Some(&good), 2));
    }

    #[test]
    fn insert_stores_unit_vectors_and_search_ranks_by_cosine() {
        let mut ix = vi(2);
        // a=(1,0) unit already; b=(0,1); c=(1,1)/norm
        ix.insert("a".into(), &[1.0, 0.0]);
        ix.insert("b".into(), &[0.0, 1.0]);
        ix.insert("c".into(), &[1.0, 1.0]);
        assert_eq!(ix.len(), 3);
        assert_eq!(ix.ids(), vec!["a", "b", "c"]);

        // query (1, 0): most similar is a (cos 1), then c (cos ~0.707), then b (cos 0).
        let res = ix.search(&[1.0, 0.0], 0);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].0, 0, "a is the top hit");
        assert_eq!(res[1].0, 2, "c (45°) is the middle hit");
        assert_eq!(res[2].0, 1, "b (orthogonal) is the worst");
        // scores strictly descending
        assert!(res[0].1 > res[1].1 && res[1].1 > res[2].1);
        // a's cosine is exactly 1 (aligned)
        assert!((res[0].1 - 1.0).abs() < 1e-5);
        // b is orthogonal -> cosine ~0
        assert!((res[2].1 - 0.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_is_scale_invariant_for_query_and_doc() {
        // "Normalize-unit-vector correctness": a non-unit doc/query must give
        // the same cosine as its unit form (cosine is scale-invariant).
        let mut ix = vi(3);
        ix.insert("short".into(), &[1.0, 0.0, 0.0]);
        ix.insert("long".into(), &[10.0, 0.0, 0.0]); // parallel to short, 10x longer

        // Query in the same direction but non-unit: (5,0,0).
        let res = ix.search(&[5.0, 0.0, 0.0], 0);
        assert_eq!(res.len(), 2);
        // Both are perfectly aligned with the query -> cosine 1 for each.
        for (_, s) in &res {
            assert!((s - 1.0).abs() < 1e-5, "aligned vectors score 1.0, got {s}");
        }
        // Unit query gives the identical scores (scale-invariance).
        let res_unit = ix.search(&[1.0, 0.0, 0.0], 0);
        assert_eq!(
            res.iter().map(|(_, s)| *s).collect::<Vec<f32>>(),
            res_unit.iter().map(|(_, s)| *s).collect::<Vec<f32>>()
        );
    }

    #[test]
    fn opposite_vectors_score_negative_and_zero_vector_scores_zero() {
        let mut ix = vi(2);
        ix.insert("pos".into(), &[1.0, 0.0]);
        ix.insert("neg".into(), &[-1.0, 0.0]);
        ix.insert("zero".into(), &[0.0, 0.0]);
        let res = ix.search(&[1.0, 0.0], 0);
        let by_id: Vec<(usize, f32)> = res;
        assert_eq!(by_id.len(), 3);
        // top is pos (cos +1), then zero (cos 0), then neg (cos -1)
        assert!((by_id[0].1 - 1.0).abs() < 1e-5);
        assert!((by_id[2].1 + 1.0).abs() < 1e-5);
        // zero vector scores exactly 0 against everything
        let zero_score = by_id.iter().find(|(i, _)| *i == 2).unwrap().1;
        assert_eq!(zero_score, 0.0);
    }

    #[test]
    fn limit_is_top_k_and_zero_means_all() {
        let mut ix = vi(2);
        for i in 0..10 {
            // spread vectors around; all unit, distinct
            let a = (i as f32) * 0.3;
            ix.insert(format!("d{i}"), &[a.cos(), a.sin()]);
        }
        assert_eq!(ix.search(&[1.0, 0.0], 3).len(), 3, "limit 3 -> top 3");
        assert_eq!(ix.search(&[1.0, 0.0], 0).len(), 10, "limit 0 -> all");
        assert_eq!(ix.search(&[1.0, 0.0], 100).len(), 10, "limit > n -> all");
    }

    #[test]
    fn remove_shrinks_and_repacks() {
        let mut ix = vi(2);
        ix.insert("a".into(), &[1.0, 0.0]);
        ix.insert("b".into(), &[0.0, 1.0]);
        ix.insert("c".into(), &[1.0, 1.0]);
        assert!(ix.remove("b"));
        assert!(!ix.remove("b"), "second remove is a no-op");
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.ids(), vec!["a", "c"]);
        // search still sees the two remaining docs
        let res = ix.search(&[1.0, 1.0], 0);
        assert_eq!(res.len(), 2);
        assert!((res[0].1 - 1.0).abs() < 1e-5, "c (aligned) stays top");
    }
}
