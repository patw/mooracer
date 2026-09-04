//! `Collection` — a named, in-memory document store.
//!
//! Design notes (perf posture, see spec "Data model" / "Performance posture"):
//!
//! - Storage is a single flat `HashMap<String, Value>`: `_id` (string) →
//!   document (always a `Value::Object`). No per-doc wrapper structs; the id
//!   is the map key *and* the first entry of the stored doc, so a lookup is
//!   one hash and docs stay contiguous value trees. If profiling later shows
//!   SipHash cost on the hot path, the map can swap hashers (e.g. `ahash`)
//!   behind this same API.
//! - Generated `_id`s come from one process-wide `AtomicU64` counter rendered
//!   as 24 zero-padded lowercase hex digits (MongoDB's shape): one atomic add
//!   + one 24-byte stack buffer per id. No RNG syscalls, no per-collection
//!   state, uniqueness across every collection in the process. The engine's
//!   write model serializes writes, so cross-collection atomic contention is
//!   a non-issue.
//! - `insert_many` is **staged (atomic)**: every doc is validated and its id
//!   checked for uniqueness against the store *and* the rest of the batch
//!   before anything commits. One bad doc rejects the whole batch and leaves
//!   the collection untouched (MongoDB `insertMany` semantics).
//! - Indexes live in a parallel [`IndexSet`] (`engine/src/index.rs`): the
//!   primary `_id` index always exists, and every `insert` / `set_doc` /
//!   `remove_doc` keeps all field indexes in lockstep (missing fields are
//!   indexed as `Null`). Field indexes are created with `create_index`
//!   (backfilled from the current docs) and dropped with `drop_index`.

// The prose above deliberately uses lazy-continuation (non-indented) list
// items; clippy's `doc_lazy_continuation` is a noise-level style nit here.
#![allow(clippy::doc_lazy_continuation)]

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
/// RRF fusion constant: the classic Reciprocal Rank Fusion value (the original
/// Cormack et al. paper uses `60`). A document at 1-based rank `r` in a ranking
/// contributes `1 / (RRF_K + r)` to its fused score; a larger `k` dampens the
/// influence of any single top rank.
pub const RRF_K: usize = 60;

/// Hybrid top-k result: the full document clone and its fused RRF score
/// (higher = more relevant; the RRF sum over the BM25 and vector rankings).
pub type HybridHit = (Value, f64);

use crate::index::{FieldIndex, IndexSet};
use crate::query::Query;
use crate::text::{TextHit, TextIndex, text_tokens};
use crate::value::Value;
use crate::vector::{VectorHit, VectorIndex, as_vector, is_vector};

/// Length of a generated `_id`: 24 hex characters.
pub const ID_LEN: usize = 24;

/// The key every document carries and is stored under.
pub const ID_KEY: &str = "_id";

/// Process-wide generator counter (see module docs).
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Next generated `_id`: 24 zero-padded lowercase hex digits.
fn next_id() -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    let mut buf = [b'0'; ID_LEN];
    for i in 0..16usize {
        buf[ID_LEN - 1 - i] = HEX[(n >> (i * 4)) as usize & 0xF];
    }
    // `collect::<String>()` from chars: one exact-capacity allocation, no
    // UTF-8 validation pass, no intermediate `Vec<u8>`.
    buf.iter().map(|&b| b as char).collect()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from document-store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The document is not an object (a document must be a `Value::Object`).
    NotAnObject,
    /// The document carries an `_id` that is not a string.
    IdMustBeString,
    /// A document with this `_id` is already in the collection (or in the
    /// same `insert_many` batch).
    DuplicateId(String),
    /// A replacement document carries an `_id` different from the document
    /// it is meant to replace (`_id` cannot be changed).
    IdMismatch { expected: String, found: String },
    /// `drop_index` was called for an index that was never created.
    NoIndex(String),
    /// The primary `_id` index cannot be dropped.
    PrimaryIndex,
    /// `update_one`'s filter matched no document (MongoDB's `updateOne`
    /// reports matched=0; the spec's "errors on no-match" surfaces that here).
    NoMatch,
    /// The update specification is malformed: a non-object spec, an unknown
    /// operator, a non-object `$set`/`$inc` operand, a `$set`/`$unset` of
    /// `_id`, a non-numeric `$inc` operand, or a `$inc` onto a non-numeric
    /// field.
    InvalidUpdate(String),
    /// A write to a document whose field `field` (vector-indexed) is present
    /// but is not a numeric array of the configured dimension `expected`
    /// (it held `found` elements, or none because it was not an array).
    VectorDimMismatch {
        field: String,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::NotAnObject => f.write_str("document must be an object"),
            StoreError::IdMustBeString => write!(f, "`{ID_KEY}` must be a string"),
            StoreError::DuplicateId(id) => write!(f, "duplicate `_id`: {id}"),
            StoreError::IdMismatch { expected, found } => {
                write!(f, "`{ID_KEY}` cannot change from {expected:?} to {found:?}")
            }
            StoreError::NoIndex(field) => write!(f, "no such index: {field}"),
            StoreError::PrimaryIndex => f.write_str("the primary `_id` index cannot be dropped"),
            StoreError::NoMatch => f.write_str("update matched no document"),
            StoreError::InvalidUpdate(msg) => write!(f, "invalid update: {msg}"),
            StoreError::VectorDimMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "field `{field}` is vector-indexed at dimension {expected} but a {found}-element value was written"
            ),
        }
    }
}

impl std::error::Error for StoreError {}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// A named in-memory store of documents, keyed by string `_id`.
///
/// Documents are stored as `Value::Object` (always carrying a string
/// `_id`); reads are `&self`, writes take `&mut self` (the server layer
/// serializes writes and allows concurrent reads with an RwLock later).
pub struct Collection {
    name: String,
    /// `_id` → document. Invariant: every value is a `Value::Object` whose
    /// `_id` entry equals the key.
    docs: HashMap<String, Value>,
    /// Index layer: the primary `_id` index plus every created field index.
    /// Invariant: each index holds exactly one entry per stored document
    /// (missing field → `Null`), maintained by insert / set_doc / remove_doc.
    indexes: IndexSet,
    /// Vector index layer: one [`VectorIndex`] per vector-indexed top-level
    /// field (created via `create_vector_index`). Invariant: each index holds
    /// one unit-normalized entry per document whose field is a valid
    /// `dim`-length numeric array (a missing field simply has no entry),
    /// maintained by insert / set_doc / remove_doc.
    vector_indexes: HashMap<String, VectorIndex>,
    /// Text index layer: one [`TextIndex`] per text-indexed top-level field
    /// (created via `create_text_index`). Invariant: each index holds one
    /// entry per document whose field is a string / array of strings (any
    /// other value, including a missing field, simply has no entry),
    /// maintained by insert / set_doc / remove_doc.
    text_indexes: HashMap<String, TextIndex>,
}

impl Collection {
    /// Create an empty collection with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Collection {
            name: name.into(),
            docs: HashMap::new(),
            indexes: IndexSet::new(),
            vector_indexes: HashMap::new(),
            text_indexes: HashMap::new(),
        }
    }

    /// The collection's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of stored documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// `true` when no documents are stored.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Borrow the document with this `_id`, if present.
    pub fn get(&self, id: &str) -> Option<&Value> {
        self.docs.get(id)
    }

    /// `true` when a document with this `_id` exists.
    pub fn contains(&self, id: &str) -> bool {
        self.docs.contains_key(id)
    }

    /// Iterate over all documents (order is unspecified — it is the hash
    /// map's; deterministic ordering arrives with the query builder).
    pub fn iter(&self) -> impl Iterator<Item = &Value> + '_ {
        self.docs.values()
    }

    /// Insert one document.
    ///
    /// - The document must be an object ([`StoreError::NotAnObject`]).
    /// - If it lacks `_id`, one is generated: 24 lowercase hex chars from
    ///   the process counter, inserted as the **first** key of the doc.
    /// - A present `_id` must be a string ([`StoreError::IdMustBeString`]);
    ///   its position in the doc is preserved (docs are not reordered).
    /// - An existing `_id` is a [`StoreError::DuplicateId`].
    ///
    /// Returns the document's `_id` (the generated one when auto-assigned).
    pub fn insert(&mut self, doc: Value) -> Result<String, StoreError> {
        let (id, doc) = normalize_doc(doc)?;
        // Check-then-insert (not insert-then-detect): a duplicate must NOT
        // clobber the existing document before the error is returned.
        if self.docs.contains_key(&id) {
            return Err(StoreError::DuplicateId(id));
        }
        self.check_vectors(&doc)?; // a bad vector is a no-op error
        self.indexes.index_doc(&id, &doc);
        self.index_vector_doc(&id, &doc);
        self.index_text_doc(&id, &doc);
        self.docs.insert(id.clone(), doc);
        Ok(id)
    }

    /// Insert many documents atomically (staged): all docs are validated
    /// and uniqueness-checked (against the store and each other) before
    /// any doc is committed. On the first error the collection is left
    /// exactly as it was. Returns the number of documents inserted.
    pub fn insert_many(
        &mut self,
        docs: impl IntoIterator<Item = Value>,
    ) -> Result<usize, StoreError> {
        let mut staged: Vec<(String, Value)> = Vec::new();
        for doc in docs {
            let (id, doc) = normalize_doc(doc)?;
            if self.docs.contains_key(&id) || staged.iter().any(|(k, _)| *k == id) {
                return Err(StoreError::DuplicateId(id));
            }
            self.check_vectors(&doc)?;
            staged.push((id, doc));
        }
        let n = staged.len();
        for (id, doc) in staged {
            self.indexes.index_doc(&id, &doc);
            self.index_vector_doc(&id, &doc);
            self.index_text_doc(&id, &doc);
            self.docs.insert(id, doc);
        }
        Ok(n)
    }

    // -- indexes ----------------------------------------------------------------

    /// Create (or rebuild) the index on top-level field `field`, backfilled
    /// from the documents currently stored. The primary `_id` index always
    /// exists; calling this for `_id` is a no-op.
    pub fn create_index(&mut self, field: &str) -> Result<(), StoreError> {
        if field == ID_KEY {
            return Ok(());
        }
        let mut idx = FieldIndex::new(field);
        for (id, doc) in &self.docs {
            idx.insert(index_value(doc, field), id.clone());
        }
        self.indexes.insert_index(field.to_string(), idx);
        Ok(())
    }

    /// Drop the index on field `field`. The primary `_id` index cannot be
    /// dropped (`StoreError::PrimaryIndex`); dropping an index that was
    /// never created is `StoreError::NoIndex`.
    pub fn drop_index(&mut self, field: &str) -> Result<(), StoreError> {
        self.indexes.drop(field)
    }

    /// All indexed field names, sorted (deterministic; always includes `_id`).
    pub fn index_names(&self) -> Vec<String> {
        self.indexes.names()
    }

    /// Borrow the index on field `field` if one exists — the query layer
    /// uses this to decide "indexed" vs full scan. The `_id` index is
    /// always present.
    pub fn index(&self, field: &str) -> Option<&FieldIndex> {
        self.indexes.get(field)
    }

    // -- vector indexes -------------------------------------------------------

    /// Create (or rebuild) the vector index on top-level field `field`,
    /// configured for vectors of exactly `dim` dimensions, backfilled from the
    /// documents currently stored (a doc whose `field` is a valid `dim`-length
    /// numeric array is indexed; a doc whose field is missing is not).
    ///
    /// This is the entry point for [`Collection::vector_search`]. Calling it
    /// again with a different `dim` rebuilds the index from the current docs
    /// (docs that no longer match the new dim are simply dropped from it).
    pub fn create_vector_index(&mut self, field: &str, dim: usize) {
        let mut idx = VectorIndex::new(field, dim);
        for (id, doc) in &self.docs {
            if let Some(v) = as_vector(doc.get(field), dim) {
                idx.insert(id.clone(), &v);
            }
        }
        self.vector_indexes.insert(field.to_string(), idx);
    }

    /// Drop the vector index on field `field` (a no-op when absent).
    pub fn drop_vector_index(&mut self, field: &str) {
        self.vector_indexes.remove(field);
    }

    /// `true` when `field` has a vector index.
    pub fn has_vector_index(&self, field: &str) -> bool {
        self.vector_indexes.contains_key(field)
    }

    /// Borrow the vector index on field `field`, if any.
    pub fn vector_index(&self, field: &str) -> Option<&VectorIndex> {
        self.vector_indexes.get(field)
    }

    /// All vector-indexed field names, sorted (deterministic).
    pub fn vector_index_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.vector_indexes.keys().cloned().collect();
        v.sort();
        v
    }

    /// Brute-force **cosine** similarity search over the vector index on
    /// `field`.
    ///
    /// `query` must have exactly the index's configured dimension (a query of
    /// the wrong length matches nothing and returns an empty vec). Returns the
    /// top `limit` documents (full clones) with their cosine score in `[-1, 1]`,
    /// best (most similar) first; ties break by `_id` ascending. `limit == 0`
    /// means no limit (return every indexed document in best-first order).
    ///
    /// Errors: [`StoreError::NoIndex`] when `field` has no vector index. An
    /// empty index (or a wrong-length query) returns an empty vec, not an
    /// error.
    pub fn vector_search(
        &self,
        field: &str,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorHit>, StoreError> {
        let ix = self
            .vector_indexes
            .get(field)
            .ok_or_else(|| StoreError::NoIndex(field.to_string()))?;
        let mut hits: Vec<VectorHit> = Vec::new();
        for (i, score) in ix.search(query, limit) {
            let id = ix.ids()[i].clone();
            if let Some(doc) = self.docs.get(&id).cloned() {
                hits.push((doc, score));
            }
        }
        Ok(hits)
    }

    /// Validate `doc` against every vector index: a **present** field that is
    /// not a `dim`-length numeric array is [`StoreError::VectorDimMismatch`]; a
    /// **missing** field is fine (the doc is simply not indexed). Called by the
    /// write paths *before* they mutate, so a bad write changes nothing.
    fn check_vectors(&self, doc: &Value) -> Result<(), StoreError> {
        // Names cloned (not borrowed) so the `&self` lookups are clean; there
        // are always few vector indexes.
        let fields: Vec<String> = self.vector_indexes.keys().cloned().collect();
        for field in &fields {
            let Some(ix) = self.vector_indexes.get(field) else {
                continue;
            };
            if let Some(v) = doc.get(field)
                && !is_vector(Some(v), ix.dim())
            {
                let found = v.as_array().map(|a| a.len()).unwrap_or(0);
                return Err(StoreError::VectorDimMismatch {
                    field: field.clone(),
                    expected: ix.dim(),
                    found,
                });
            }
        }
        Ok(())
    }

    /// Register one entry of `doc` in every vector index (a valid `dim`-length
    /// numeric field is indexed; a missing field is skipped). Invariant: a doc
    /// already passed [`Collection::check_vectors`].
    fn index_vector_doc(&mut self, id: &str, doc: &Value) {
        let fields: Vec<String> = self.vector_indexes.keys().cloned().collect();
        for field in &fields {
            let Some(ix) = self.vector_indexes.get_mut(field) else {
                continue;
            };
            if let Some(v) = as_vector(doc.get(field), ix.dim()) {
                ix.insert(id.to_string(), &v);
            }
        }
    }

    /// Remove one entry of `doc` from every vector index (inverse of
    /// [`Collection::index_vector_doc`]; a no-op for a field that was never
    /// indexed).
    fn deindex_vector_doc(&mut self, id: &str) {
        let fields: Vec<String> = self.vector_indexes.keys().cloned().collect();
        for field in &fields {
            if let Some(ix) = self.vector_indexes.get_mut(field) {
                ix.remove(id);
            }
        }
    }

    // -- text indexes -------------------------------------------------------

    /// Create (or rebuild) the BM25 text index on top-level field `field`,
    /// backfilled from the documents currently stored (a doc whose `field`
    /// is a string or an array of strings is indexed; anything else —
    /// including a missing field — is not). Calling it again deterministically
    /// rebuilds the index from the current docs.
    ///
    /// This is the entry point for [`Collection::text_search`].
    pub fn create_text_index(&mut self, field: &str) {
        let mut idx = TextIndex::new(field);
        let pairs: Vec<(String, Vec<String>)> = self
            .docs
            .iter()
            .filter_map(|(id, doc)| text_tokens(doc.get(field)).map(|t| (id.clone(), t)))
            .collect();
        idx.load(pairs);
        self.text_indexes.insert(field.to_string(), idx);
    }

    /// Drop the text index on field `field` (a no-op when absent).
    pub fn drop_text_index(&mut self, field: &str) {
        self.text_indexes.remove(field);
    }

    /// `true` when `field` has a text index.
    pub fn has_text_index(&self, field: &str) -> bool {
        self.text_indexes.contains_key(field)
    }

    /// Borrow the text index on field `field`, if any.
    pub fn text_index(&self, field: &str) -> Option<&TextIndex> {
        self.text_indexes.get(field)
    }

    /// All text-indexed field names, sorted (deterministic).
    pub fn text_index_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.text_indexes.keys().cloned().collect();
        v.sort();
        v
    }

    /// **BM25** full-text search over the text index on `field`.
    ///
    /// `query` is tokenized the same way as the indexed documents (lowercased
    /// `[a-z0-9]` runs, Porter-stemmed); each distinct query stem counts
    /// once. Returns the top `limit` documents (full clones) with their BM25
    /// score (higher = more relevant), best first; ties break by index order
    /// (deterministic). Only documents with a strictly positive score are
    /// returned; `limit == 0` means no limit.
    ///
    /// Errors: [`StoreError::NoIndex`] when `field` has no text index. An
    /// empty index (or a query with no tokens) returns an empty vec, not an
    /// error.
    pub fn text_search(
        &self,
        field: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<TextHit>, StoreError> {
        let ix = self
            .text_indexes
            .get(field)
            .ok_or_else(|| StoreError::NoIndex(field.to_string()))?;
        let mut hits: Vec<TextHit> = Vec::new();
        for (i, score) in ix.search(query, limit) {
            let id = ix.ids()[i].clone();
            if let Some(doc) = self.docs.get(&id).cloned() {
                hits.push((doc, score));
            }
        }
        Ok(hits)
    }

    // -- hybrid search ------------------------------------------------------

    /// **Hybrid** search: reciprocal-rank fusion (RRF) of the BM25 ranking
    /// over `text_field` and the brute-force cosine ranking over `vec_field`.
    ///
    /// RRF fuses *ranks*, not raw scores: a document at 1-based rank `r` in a
    /// ranking earns `1 / (RRF_K + r)`, and its fused score is the **sum**
    /// over every ranking it appears in (the classic RRF constant
    /// [`RRF_K`] = 60). Both full (unlimited) rankings are computed and fused
    /// over their **union**, so a document ranked by only one signal still
    /// surfaces — the two signals need not agree. Returns the top `limit`
    /// documents (full clones) with their fused RRF score, best first; ties
    /// break by `_id` ascending (deterministic). `limit == 0` means no limit
    /// (the same `0` convention as the query pipeline and the single searches).
    ///
    /// Errors: [`StoreError::NoIndex`] when `text_field` has no text index or
    /// `vec_field` has no vector index. An empty fusion (no ranked docs) returns
    /// an empty vec, not an error.
    pub fn hybrid_search(
        &self,
        text_field: &str,
        vec_field: &str,
        query_text: &str,
        query_vec: &[f32],
        limit: usize,
    ) -> Result<Vec<HybridHit>, StoreError> {
        let text_ix = self
            .text_indexes
            .get(text_field)
            .ok_or_else(|| StoreError::NoIndex(text_field.to_string()))?;
        let vec_ix = self
            .vector_indexes
            .get(vec_field)
            .ok_or_else(|| StoreError::NoIndex(vec_field.to_string()))?;

        // Full (unlimited) rankings: RRF depends on rank position, so the
        // final `limit` is applied only *after* fusion.
        let text_rank = text_ix.search(query_text, 0);
        let vec_rank = vec_ix.search(query_vec, 0);

        // id -> fused RRF score (a `f64` accumulation per document).
        let mut fused: HashMap<String, f64> = HashMap::new();
        for (r, &(i, _)) in text_rank.iter().enumerate() {
            let id = &text_ix.ids()[i];
            *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K as f64 + (r + 1) as f64);
        }
        for (r, &(i, _)) in vec_rank.iter().enumerate() {
            let id = &vec_ix.ids()[i];
            *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K as f64 + (r + 1) as f64);
        }

        // Best fused score first; ties break by `_id` ascending (total order,
        // so `sort_unstable_by` is deterministic). The final limit truncates.
        let mut ranked: Vec<(String, f64)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if limit > 0 {
            ranked.truncate(limit);
        }

        let mut hits: Vec<HybridHit> = Vec::with_capacity(ranked.len());
        for (id, score) in ranked {
            if let Some(doc) = self.docs.get(&id).cloned() {
                hits.push((doc, score));
            }
        }
        Ok(hits)
    }

    /// Register one entry of `doc` in every text index (a string /
    /// array-of-strings field is indexed; anything else is skipped — a text
    /// index never rejects a write).
    fn index_text_doc(&mut self, id: &str, doc: &Value) {
        let fields: Vec<String> = self.text_indexes.keys().cloned().collect();
        for field in &fields {
            let Some(ix) = self.text_indexes.get_mut(field) else {
                continue;
            };
            if let Some(tokens) = text_tokens(doc.get(field)) {
                ix.insert(id.to_string(), tokens);
            }
        }
    }

    /// Remove one entry of `doc` from every text index (inverse of
    /// [`Collection::index_text_doc`]; a no-op for a field that was never
    /// indexed).
    fn deindex_text_doc(&mut self, id: &str) {
        let fields: Vec<String> = self.text_indexes.keys().cloned().collect();
        for field in &fields {
            if let Some(ix) = self.text_indexes.get_mut(field) {
                ix.remove(id);
            }
        }
    }

    /// Snapshot of store + index state: document count, index count,
    /// per-index entry/distinct counts and capacity-based memory estimates.
    /// `per_index` is sorted by field name (deterministic; includes `_id`).
    /// See spec "Stats & reindex decisions".
    pub fn stats(&self) -> CollectionStats {
        let mut docs_memory: usize = std::mem::size_of::<Collection>()
            + self.docs.len() * (std::mem::size_of::<(String, Value)>() + ID_LEN);
        for (id, doc) in &self.docs {
            docs_memory += id.capacity() + crate::index::value_heap(doc);
        }
        let mut per_index: Vec<IndexStats> = Vec::new();
        for field in self.indexes.names() {
            if let Some(ix) = self.indexes.get(&field) {
                per_index.push(IndexStats {
                    field,
                    entries: ix.len(),
                    distinct: ix.distinct(),
                    memory: ix.memory_size(),
                });
            }
        }
        let total_memory = docs_memory + per_index.iter().map(|s| s.memory).sum::<usize>();
        CollectionStats {
            docs: self.docs.len(),
            docs_memory,
            indexes: per_index.len(),
            per_index,
            total_memory,
        }
    }

    /// Deterministically rebuild **every** index (including the primary `_id`
    /// index and every text index) from the documents currently stored,
    /// replacing incremental maintenance with a fresh pass (`FieldIndex::load`
    /// / `TextIndex::load`). Returns the number of *value* indexes rebuilt
    /// (text indexes are rebuilt too but not counted).
    pub fn reindex(&mut self) -> usize {
        let fields = self.indexes.names();
        for field in &fields {
            let pairs: Vec<(String, Value)> = self
                .docs
                .iter()
                .map(|(id, doc)| (id.clone(), index_value(doc, field)))
                .collect();
            if let Some(idx) = self.indexes.get_mut(field) {
                idx.load(pairs);
            }
        }
        let text_fields: Vec<String> = self.text_indexes.keys().cloned().collect();
        for field in &text_fields {
            let pairs: Vec<(String, Vec<String>)> = self
                .docs
                .iter()
                .filter_map(|(id, doc)| text_tokens(doc.get(field)).map(|t| (id.clone(), t)))
                .collect();
            if let Some(idx) = self.text_indexes.get_mut(field) {
                idx.load(pairs);
            }
        }
        fields.len()
    }

    // -- query entry points -------------------------------------------------------

    /// Start a **lazy** query over this collection with `filter` (a Mongo
    /// object; `{}` matches every document). The scan is deferred to a
    /// terminal on the returned [`Query`]: `.to_list()`, `.first()`,
    /// `.count()` (and the later `.sort`/`.skip`/`.limit`).
    pub fn find(&self, filter: Value) -> Query<'_> {
        Query::new(self, filter)
    }

    /// Eager convenience: the first document matching `filter`
    /// (storage order), or `None` when nothing matches.
    pub fn find_one(&self, filter: Value) -> Option<Value> {
        self.find(filter).first()
    }

    /// Eager convenience: the number of documents matching `filter`.
    pub fn count(&self, filter: Value) -> usize {
        self.find(filter).count()
    }

    /// Eager convenience: `true` when at least one document matches `filter`.
    pub fn exists(&self, filter: Value) -> bool {
        self.find(filter).first().is_some()
    }

    // -- update operators ($set / $inc / $unset) ----------------------------------

    /// Update **one** document matching `filter` (storage order — the first
    /// match) with the `update` operator spec, refreshing every index entry.
    ///
    /// `update` is a Mongo operator object: `{"$set": {…}, "$inc": {…},
    /// "$unset": {…}}` — the operators apply in the order they appear in the
    /// spec. Returns the number of matched documents (always `1` on success);
    /// `Err(StoreError::NoMatch)` when nothing matches (the spec's "errors on
    /// no-match"), and `Err(StoreError::InvalidUpdate)` when the spec or the
    /// applied change is malformed (nothing is changed on error).
    pub fn update_one(&mut self, filter: Value, update: Value) -> Result<usize, StoreError> {
        let Some(doc) = self.find(filter).first() else {
            return Err(StoreError::NoMatch);
        };
        let id = doc
            .get(ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let new = apply_update(&doc, &update)?;
        match self.set_doc(&id, new)? {
            Some(_) => Ok(1),
            None => Err(StoreError::NoMatch), // vanished: cannot happen (single-thread write)
        }
    }

    /// Update **every** document matching `filter` with the `update` operator
    /// spec (operators applied in spec order per document), refreshing every
    /// index entry. Returns the number of documents matched (0 is a valid
    /// result — no error, unlike [`Collection::update_one`]). A malformed spec
    /// is `Err(StoreError::InvalidUpdate)`, applied to **no** document (the
    /// spec is shape-validated before any document is touched).
    pub fn update_many(&mut self, filter: Value, update: Value) -> Result<usize, StoreError> {
        // Shape-validate the spec up front so a malformed update never
        // partially applies (MongoDB `updateMany` validates before running).
        validate_update(&update)?;
        let matched = self
            .find(filter)
            .to_list()
            .iter()
            .filter_map(|d| d.get(ID_KEY).and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<String>>();
        let mut n = 0usize;
        for id in matched {
            let Some(doc) = self.get(&id).cloned() else {
                continue;
            };
            let new = apply_update(&doc, &update)?;
            if self.set_doc(&id, new)?.is_some() {
                n += 1;
            }
        }
        Ok(n)
    }
    // -- replace -----------------------------------------------------------------

    /// Replace **one** document matching `filter` (storage order — the first
    /// match) with `new_doc`, **preserving the matched document's `_id`**,
    /// and refreshing every index entry.
    ///
    /// The matched document is replaced wholesale: fields not present in
    /// `new_doc` are gone (this is a replacement, not an update). `new_doc`
    /// is normalized via [`Collection::set_doc`]: it must be an object
    /// ([`StoreError::NotAnObject`]); a present `_id` must be a string **equal
    /// to the matched document's** (a different one is [`StoreError::IdMismatch`],
    /// since `_id` cannot change); a missing `_id` is restored as the matched
    /// document's (first key). Returns the number of matched documents (always
    /// `1` on success); `Err(StoreError::NoMatch)` when nothing matches.
    pub fn replace_one(&mut self, filter: Value, new_doc: Value) -> Result<usize, StoreError> {
        let Some(doc) = self.find(filter).first() else {
            return Err(StoreError::NoMatch);
        };
        let id = doc
            .get(ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        match self.set_doc(&id, new_doc)? {
            Some(_) => Ok(1),
            None => Err(StoreError::NoMatch), // vanished: cannot happen (single-thread write)
        }
    }

    // -- removal / refresh primitives --------------------------------------------
    // -- delete -----------------------------------------------------------------

    /// Delete **one** document matching `filter` (storage order — the first
    /// match), removing it from the store and all field indexes.
    ///
    /// Returns `true` when a document was deleted, `false` when nothing
    /// matched. The filter is the same Mongo object as in [`Collection::find`].
    pub fn delete_one(&mut self, filter: Value) -> bool {
        let Some(doc) = self.find(filter).first() else {
            return false;
        };
        let id = doc
            .get(ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        self.remove_doc(&id).is_some()
    }

    /// Delete **every** document matching `filter`, removing each from the
    /// store and all field indexes. Returns the number of documents deleted
    /// (0 is a valid result — no error).
    pub fn delete_many(&mut self, filter: Value) -> usize {
        let matched = self
            .find(filter)
            .to_list()
            .iter()
            .filter_map(|d| d.get(ID_KEY).and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<String>>();
        let mut n = 0usize;
        for id in matched {
            if self.remove_doc(&id).is_some() {
                n += 1;
            }
        }
        n
    }

    /// Remove a document by `_id` (the delete primitive — `delete_one` /
    /// `delete_many` build on this), removing all of its index entries.
    /// Returns the removed document, or `None` when the id is not stored.
    pub fn remove_doc(&mut self, id: &str) -> Option<Value> {
        let old = self.docs.get(id)?.clone();
        self.indexes.deindex_doc(id, &old);
        self.deindex_vector_doc(id);
        self.deindex_text_doc(id);
        self.docs.remove(id)
    }

    /// Replace a document's content, preserving its `_id` (the update /
    /// replace primitive — filter-based `update_*` / `replace_one` build on
    /// this), refreshing every index entry along the way.
    ///
    /// - `Ok(None)`: no document with this `_id` — nothing changed, and the
    ///   incoming document is deliberately *not* validated (the filter-
    ///   matches-first semantics of the later filter APIs).
    /// - `Ok(Some(old))`: replaced; `old` is the previous document.
    /// - `Err(…)`: the incoming document is not an object, its `_id` is not
    ///   a string, or it differs from `id` (`StoreError::IdMismatch` —
    ///   `_id` cannot change). On error the store and all indexes are
    ///   untouched.
    pub fn set_doc(&mut self, id: &str, doc: Value) -> Result<Option<Value>, StoreError> {
        let old = match self.docs.get(id) {
            Some(old) => old.clone(),
            None => return Ok(None),
        };
        let new = normalize_replace(id, doc)?; // validated BEFORE mutation
        self.check_vectors(&new)?; // a bad vector is a no-op error
        self.indexes.deindex_doc(id, &old);
        self.indexes.index_doc(id, &new);
        self.deindex_vector_doc(id);
        self.index_vector_doc(id, &new);
        self.deindex_text_doc(id);
        self.index_text_doc(id, &new);
        self.docs.insert(id.to_string(), new);
        Ok(Some(old))
    }

    /// Begin an **atomic batch** (transaction) over this collection.
    ///
    /// The returned [`Transaction`] stages writes in order; reads issued
    /// through it see the **pre-batch** state (staged writes are not visible
    /// until commit). `commit()` applies every staged write to the store and
    /// **all** field indexes at once; `rollback()` (or dropping the
    /// transaction without committing) discards them, leaving the collection
    /// exactly as it was. Any write that errors marks the transaction failed
    /// and makes `commit()` a no-op.
    pub fn begin(&mut self) -> Transaction<'_> {
        Transaction::new(self)
    }
}

// ---------------------------------------------------------------------------
// Atomic batch (transaction)
// ---------------------------------------------------------------------------

/// A single staged write, resolved against the pre-batch snapshot.
#[derive(Debug)]
enum BatchOp {
    /// Put a full document at `id` (insert, or overwrite an existing/prior
    /// op). The document's `_id` equals `id`.
    Put(String, Value),
    /// Delete the document at `id` (a no-op if it is absent at commit time).
    Delete(String),
}

/// An atomic batch of writes over a [`Collection`].
///
/// All writes are evaluated against the **pre-batch** snapshot and buffered;
/// the live store is untouched until `commit()`. `commit()` applies the whole
/// batch (store + every field index) as one unit; `rollback()` or dropping
/// the transaction without committing discards all buffered writes. If any
/// staged write errors, the transaction is marked failed and `commit()`
/// becomes a no-op (rollback on error).
///
/// Writes are **id-scoped concrete mutations** (`BatchOp::Put` /
/// `BatchOp::Delete`): a batch's writes compose per id (a later op on an id
/// already written by an earlier op overwrites it), so commit can apply each
/// id at most once. Deletes and inserts are both `BatchOp::Put`/`Delete`
/// (an insert is a `Put` of a new id).
pub struct Transaction<'a> {
    col: &'a mut Collection,
    /// Staged ops, in write order.
    ops: Vec<BatchOp>,
    /// Ids already written by an earlier op in this batch (in order) — lets a
    /// later op overwrite an earlier one and lets `insert` reject intra-batch
    /// duplicates.
    seen: Vec<String>,
    /// Whether a write has already errored (transaction is now failed).
    failed: bool,
    /// The first error that failed the transaction (if any).
    err: Option<StoreError>,
}

impl<'a> Transaction<'a> {
    fn new(col: &'a mut Collection) -> Self {
        Transaction {
            col,
            ops: Vec::new(),
            seen: Vec::new(),
            failed: false,
            err: None,
        }
    }

    /// The error that failed this transaction, if it is failed.
    pub fn error(&self) -> Option<&StoreError> {
        self.err.as_ref()
    }

    /// `true` once a write has errored (subsequent writes and `commit()` are
    /// no-ops).
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Stage an error: mark the transaction failed (first error wins).
    fn fail(&mut self, e: StoreError) {
        if !self.failed {
            self.failed = true;
            self.err = Some(e);
        }
    }

    // -- reads (always the pre-batch snapshot) --------------------------------

    /// Start a **lazy** query over the **pre-batch** state (staged writes are
    /// not visible). Same filter semantics as [`Collection::find`].
    pub fn find(&self, filter: Value) -> Query<'_> {
        self.col.find(filter)
    }

    /// The first document matching `filter` in the pre-batch state.
    pub fn find_one(&self, filter: Value) -> Option<Value> {
        self.col.find(filter).first()
    }

    /// The number of pre-batch documents matching `filter`.
    pub fn count(&self, filter: Value) -> usize {
        self.col.find(filter).count()
    }

    /// The pre-batch document with this `_id`, if present.
    pub fn get(&self, id: &str) -> Option<&Value> {
        self.col.get(id)
    }

    /// `true` when the pre-batch store holds a document with this `_id`.
    pub fn contains(&self, id: &str) -> bool {
        self.col.contains(id)
    }

    /// The pre-batch document count.
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// `true` when the pre-batch store is empty.
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// Borrow the pre-batch index on `field`, if any.
    pub fn index(&self, field: &str) -> Option<&FieldIndex> {
        self.col.index(field)
    }

    // -- writes (buffered; applied on commit) ---------------------------------

    /// Stage an insert. The doc is normalized (a missing `_id` is generated),
    /// and a `_id` already in the pre-batch store **or** staged earlier in
    /// this batch is a [`StoreError::DuplicateId`] that fails the
    /// transaction. Returns the doc's `_id`.
    pub fn insert(&mut self, doc: Value) -> Result<String, StoreError> {
        if self.failed {
            return Err(self.err.clone().unwrap());
        }
        let (id, doc) = match normalize_doc(doc) {
            Ok(v) => v,
            Err(e) => {
                self.fail(e.clone());
                return Err(e);
            }
        };
        if self.col.contains(&id) || self.seen.iter().any(|s| s == &id) {
            let e = StoreError::DuplicateId(id.clone());
            self.fail(e.clone());
            return Err(e);
        }
        if let Err(e) = self.col.check_vectors(&doc) {
            self.fail(e.clone());
            return Err(e);
        }
        self.seen.push(id.clone());
        self.ops.push(BatchOp::Put(id.clone(), doc));
        Ok(id)
    }

    /// Stage many inserts. The first error fails the whole transaction
    /// (which `commit()` turns into a full rollback). Returns the number of
    /// docs staged.
    pub fn insert_many(
        &mut self,
        docs: impl IntoIterator<Item = Value>,
    ) -> Result<usize, StoreError> {
        let mut n = 0;
        for d in docs {
            self.insert(d)?;
            n += 1;
        }
        Ok(n)
    }

    /// Stage an update of **every** pre-batch document matching `filter`,
    /// using the `update` operator spec (same semantics as
    /// [`Collection::update_many`]). Returns the number of matched documents.
    /// A malformed spec fails the transaction.
    pub fn update(&mut self, filter: Value, update: Value) -> Result<usize, StoreError> {
        if self.failed {
            return Err(self.err.clone().unwrap());
        }
        validate_update(&update).inspect_err(|e| {
            self.fail(e.clone());
        })?;
        let matched = self
            .col
            .find(filter)
            .to_list()
            .iter()
            .filter_map(|d| d.get(ID_KEY).and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<String>>();
        let n = matched.len();
        for id in matched {
            let Some(doc) = self.col.get(&id).cloned() else {
                continue;
            };
            let new = apply_update(&doc, &update).inspect_err(|e| {
                self.fail(e.clone());
            })?;
            self.put(id, new);
        }
        Ok(n)
    }

    /// Stage a replacement of the **first** pre-batch document matching
    /// `filter` with `new_doc`, preserving its `_id` (same semantics as
    /// [`Collection::replace_one`]). `new_doc` is validated up front; a
    /// `_id` mismatch or non-object doc fails the transaction. A filter that
    /// matches nothing is [`StoreError::NoMatch`]. Returns the match count.
    pub fn replace(&mut self, filter: Value, new_doc: Value) -> Result<usize, StoreError> {
        if self.failed {
            return Err(self.err.clone().unwrap());
        }
        let Some(doc) = self.col.find(filter).first() else {
            let e = StoreError::NoMatch;
            self.fail(e.clone());
            return Err(e);
        };
        let id = doc
            .get(ID_KEY)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let new = match normalize_replace(&id, new_doc) {
            Ok(v) => v,
            Err(e) => {
                self.fail(e.clone());
                return Err(e);
            }
        };
        self.put(id, new);
        Ok(1)
    }

    /// Stage a delete of **every** pre-batch document matching `filter`
    /// (same semantics as [`Collection::delete_many`]). Returns the number of
    /// matched documents (deletes are applied on commit; a delete of an id
    /// already staged for a put is dropped from the batch).
    pub fn delete(&mut self, filter: Value) -> usize {
        if self.failed {
            return 0;
        }
        let matched = self
            .col
            .find(filter)
            .to_list()
            .iter()
            .filter_map(|d| d.get(ID_KEY).and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<String>>();
        let n = matched.len();
        for id in matched {
            // Compose per id: an earlier staged put on this id is overwritten
            // (a pre-batch doc that was then deleted becomes a Delete; a doc
            // that was only inserted in this batch is removed entirely, so its
            // net effect is a no-op).
            if let Some(i) = self.seen.iter().position(|s| s == &id) {
                if self.col.contains(&id) {
                    self.ops[i] = BatchOp::Delete(id);
                } else {
                    self.ops.remove(i);
                    self.seen.remove(i);
                }
            } else {
                self.seen.push(id.clone());
                self.ops.push(BatchOp::Delete(id));
            }
        }
        n
    }

    /// Overwrite-or-stage the doc at `id` (a later write replaces an earlier
    /// op on the same id so each id commits at most once).
    fn put(&mut self, id: String, doc: Value) {
        if let Some(i) = self.seen.iter().position(|s| s == &id) {
            self.ops[i] = BatchOp::Put(id, doc);
        } else {
            self.seen.push(id.clone());
            self.ops.push(BatchOp::Put(id, doc));
        }
    }

    // -- commit / rollback ----------------------------------------------------

    /// Apply the whole batch to the store and **all** indexes as one unit.
    /// A failed transaction (a write that already errored) applies nothing
    /// and returns the stored error. Returns `Ok(())` on success (including
    /// an empty batch).
    pub fn commit(self) -> Result<(), StoreError> {
        if self.failed {
            return Err(self.err.clone().unwrap());
        }
        for op in self.ops {
            match op {
                BatchOp::Put(id, doc) => {
                    if self.col.docs.contains_key(&id) {
                        self.col.set_doc(&id, doc)?;
                    } else {
                        self.col.check_vectors(&doc)?;
                        self.col.indexes.index_doc(&id, &doc);
                        self.col.index_vector_doc(&id, &doc);
                        self.col.index_text_doc(&id, &doc);
                        self.col.docs.insert(id, doc);
                    }
                }
                BatchOp::Delete(id) => {
                    self.col.remove_doc(&id);
                }
            }
        }
        Ok(())
    }

    /// Discard every staged write; the collection is left exactly as it was
    /// before `begin()`. Idempotent and never fails. (Dropping the
    /// transaction without `commit()` has the same effect: no write is ever
    /// applied to the store until `commit()`, so the live collection is
    /// already untouched.)
    pub fn rollback(mut self) {
        self.ops.clear();
        self.seen.clear();
    }
}

/// Value registered for `field` in a field index: the field's value, or
/// `Null` when the document lacks the field (MongoDB convention — see
/// `index.rs`).
fn index_value(doc: &Value, field: &str) -> Value {
    doc.get(field).cloned().unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Per-index statistics, as reported by [`Collection::stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    /// The indexed field (`_id` for the primary index).
    pub field: String,
    /// Number of entries — one per stored document (missing field → a
    /// `Null` entry), so `entries == Collection::len()` for every index.
    pub entries: usize,
    /// Number of distinct values (engine total order: `I64(1)` and
    /// `F64(1.0)` count as one).
    pub distinct: usize,
    /// Estimated memory footprint in bytes (capacity-based estimate).
    pub memory: usize,
}

/// A point-in-time snapshot of a collection, as reported by
/// [`Collection::stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionStats {
    /// Number of stored documents.
    pub docs: usize,
    /// Estimated memory of the document store in bytes: the map + every id
    /// string + the recursive value trees (capacity-based estimate).
    pub docs_memory: usize,
    /// Number of indexes (always `>= 1` — the primary `_id` index).
    pub indexes: usize,
    /// Per-index statistics, sorted by field name (deterministic; always
    /// includes `_id`).
    pub per_index: Vec<IndexStats>,
    /// `docs_memory` + the sum of every index's memory estimate.
    pub total_memory: usize,
}

/// Validate a replacement document for a stored `_id` and normalize it to
/// storage form: must be an object; a present `_id` must be a string equal
/// to `id` (position preserved); a missing `_id` is prepended as the first
/// key (same rule as `insert`).
fn normalize_replace(id: &str, doc: Value) -> Result<Value, StoreError> {
    let entries = match doc {
        Value::Object(entries) => entries,
        _ => return Err(StoreError::NotAnObject),
    };
    match entries.iter().find(|(k, _)| k.as_str() == ID_KEY) {
        Some((_, Value::Str(s))) if s == id => Ok(Value::Object(entries)),
        Some((_, Value::Str(s))) => Err(StoreError::IdMismatch {
            expected: id.to_string(),
            found: s.clone(),
        }),
        Some((_, _)) => Err(StoreError::IdMustBeString),
        None => {
            let mut with_id = Vec::with_capacity(entries.len() + 1);
            with_id.push((ID_KEY.to_string(), Value::Str(id.to_string())));
            with_id.extend(entries);
            Ok(Value::Object(with_id))
        }
    }
}

/// Validate a document and make it storage-ready. Returns `(id, doc)` where
/// the doc is a `Value::Object` whose `_id` string equals `id`.
///
/// On success the doc is the input doc, either untouched (user `_id`) or
/// with a generated `_id` prepended (allocation: one re-buffered entry list
/// only in the auto-id case).
fn normalize_doc(doc: Value) -> Result<(String, Value), StoreError> {
    let entries = match doc {
        Value::Object(entries) => entries,
        _ => return Err(StoreError::NotAnObject),
    };
    match entries.iter().find(|(k, _)| k.as_str() == ID_KEY) {
        // Invariant (see `Value`): object keys are unique, so at most one hit.
        Some((_, Value::Str(id))) => Ok((id.clone(), Value::Object(entries))),
        Some((_, _)) => Err(StoreError::IdMustBeString),
        None => {
            let id = next_id();
            let mut with_id = Vec::with_capacity(entries.len() + 1);
            with_id.push((ID_KEY.to_string(), Value::Str(id.clone())));
            with_id.extend(entries);
            Ok((id, Value::Object(with_id)))
        }
    }
}

// ---------------------------------------------------------------------------
// Update operators ($set / $inc / $unset)
// ---------------------------------------------------------------------------

/// Shape-validate an update spec before it touches any document: it must be
/// an object, every operator must be `$set`/`$inc`/`$unset`, `$set` and `$inc`
/// operands must be objects, and no operator may name `_id` (the `_id` of a
/// document cannot be changed — MongoDB rejects it). Returns `Ok(())` or
/// `Err(InvalidUpdate)`. Per-field *value* errors (a non-numeric `$inc` onto a
/// non-numeric field) are doc-dependent and are caught by [`apply_update`].
fn validate_update(update: &Value) -> Result<(), StoreError> {
    let Value::Object(entries) = update else {
        return Err(StoreError::InvalidUpdate(
            "update spec must be an object".into(),
        ));
    };
    for (op, operand) in entries {
        let keys = match op.as_str() {
            "$set" | "$inc" => match operand {
                Value::Object(ks) => ks,
                _ => {
                    return Err(StoreError::InvalidUpdate(format!(
                        "{op} operand must be an object"
                    )));
                }
            },
            // `$unset` operand is either an object (keys) or an array of
            // string field names — both are checked per field below.
            "$unset" => continue,
            _ => {
                return Err(StoreError::InvalidUpdate(format!(
                    "unknown operator {op:?}"
                )));
            }
        };
        for (k, _) in keys {
            if k.as_str() == ID_KEY {
                return Err(StoreError::InvalidUpdate(
                    "`_id` cannot be set, incremented, or unset".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Apply an update operator spec to `doc`, returning the new document (the
/// original `_id` entry is left untouched, so [`Collection::set_doc`] sees a
/// matching `_id` and no reordering). The spec must have passed
/// [`validate_update`]; per-value errors (`$inc` onto a non-numeric field) are
/// `Err(InvalidUpdate)` with the original document untouched.
fn apply_update(doc: &Value, update: &Value) -> Result<Value, StoreError> {
    let entries = match update {
        Value::Object(entries) => entries,
        _ => {
            return Err(StoreError::InvalidUpdate(
                "update spec must be an object".into(),
            ));
        }
    };
    let mut out = doc.clone();
    for (op, operand) in entries {
        match op.as_str() {
            "$set" => apply_set(&mut out, operand)?,
            "$inc" => apply_inc(&mut out, operand)?,
            "$unset" => apply_unset(&mut out, operand)?,
            other => {
                return Err(StoreError::InvalidUpdate(format!(
                    "unknown operator {other:?}"
                )));
            }
        }
    }
    Ok(out)
}

/// `$set: {path: value, …}` — set (or create) each path to its value,
/// creating missing intermediate objects / sparse arrays per
/// [`Value::set_path`]. `_id` is rejected (validated up front, re-checked
/// here so `apply_update` is safe standalone).
fn apply_set(doc: &mut Value, operand: &Value) -> Result<(), StoreError> {
    let Value::Object(pairs) = operand else {
        return Err(StoreError::InvalidUpdate(
            "$set operand must be an object".into(),
        ));
    };
    for (k, v) in pairs {
        if k.as_str() == ID_KEY {
            return Err(StoreError::InvalidUpdate("`_id` cannot be set".into()));
        }
        doc.set_path(k, v.clone())
            .map_err(|e| StoreError::InvalidUpdate(format!("$set {k}: {e}")))?;
    }
    Ok(())
}

/// `$inc: {path: number, …}` — add each numeric operand to the field's
/// current value. A **missing** field is created with the operand as its
/// value (MongoDB: increment from zero). An existing **non-numeric** field is
/// an error (`$inc` applies only to numbers). Integer + integer stays an
/// integer unless it overflows `i64` (then it widens to `F64`); any float
/// operand or operand value makes the result a float.
fn apply_inc(doc: &mut Value, operand: &Value) -> Result<(), StoreError> {
    let Value::Object(pairs) = operand else {
        return Err(StoreError::InvalidUpdate(
            "$inc operand must be an object".into(),
        ));
    };
    for (k, delta) in pairs {
        if k.as_str() == ID_KEY {
            return Err(StoreError::InvalidUpdate(
                "`_id` cannot be incremented".into(),
            ));
        }
        // The numeric value of the operand (both as i64 when it is one, and
        // as f64 for the float math).
        let d_f = match delta {
            Value::I64(n) => *n as f64,
            Value::F64(x) => *x,
            _ => {
                return Err(StoreError::InvalidUpdate(format!(
                    "$inc {k}: operand must be numeric"
                )));
            }
        };
        let d_i = delta.as_i64(); // Some(…) for an I64 operand, None for F64
        let new = match doc.get_path(k) {
            None => delta.clone(), // missing field: create with the operand
            Some(Value::I64(cur)) => match d_i {
                // i64 + i64 stays an i64 unless it overflows (then widen).
                Some(di) => ((*cur) as i128 + di as i128)
                    .try_into()
                    .map(Value::I64)
                    .unwrap_or_else(|_| Value::F64(*cur as f64 + d_f)),
                None => Value::F64(*cur as f64 + d_f),
            },
            Some(Value::F64(cur)) => Value::F64(cur + d_f),
            Some(other) => {
                return Err(StoreError::InvalidUpdate(format!(
                    "$inc {k}: cannot increment a non-numeric value ({})",
                    other.type_name()
                )));
            }
        };
        doc.set_path(k, new)
            .map_err(|e| StoreError::InvalidUpdate(format!("$inc {k}: {e}")))?;
    }
    Ok(())
}

/// `$unset: {path: <any>}` or `$unset: [path, …]` — remove each path. The
/// operand's values are ignored (MongoDB convention); only the keys (or array
/// string elements) matter. Removing a missing path is a no-op. `_id` is
/// rejected (the collection invariant: a document always keeps its `_id`).
fn apply_unset(doc: &mut Value, operand: &Value) -> Result<(), StoreError> {
    let paths: Vec<String> = match operand {
        Value::Object(pairs) => pairs.iter().map(|(k, _)| k.clone()).collect(),
        Value::Array(items) => {
            let mut paths = Vec::with_capacity(items.len());
            for e in items {
                match e {
                    Value::Str(s) => paths.push(s.clone()),
                    other => {
                        return Err(StoreError::InvalidUpdate(format!(
                            "$unset array element must be a string (got {})",
                            other.type_name()
                        )));
                    }
                }
            }
            paths
        }
        _ => {
            return Err(StoreError::InvalidUpdate(
                "$unset operand must be an object or an array of field names".into(),
            ));
        }
    };
    for k in &paths {
        if k.as_str() == ID_KEY {
            return Err(StoreError::InvalidUpdate("`_id` cannot be unset".into()));
        }
        doc.remove_path(k)
            .map_err(|e| StoreError::InvalidUpdate(format!("$unset {k}: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn doc(pairs: &[(&str, Value)]) -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    fn is_hex(s: &str) -> bool {
        s.bytes().all(|b| b.is_ascii_hexdigit())
    }

    fn assert_auto_id(id: &str) {
        assert_eq!(id.len(), ID_LEN, "auto id has {ID_LEN} chars: {id}");
        assert!(is_hex(id), "auto id is lowercase hex: {id}");
        assert!(
            id.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        );
    }

    // -- basics ---------------------------------------------------------------

    #[test]
    fn new_collection_is_empty() {
        let c = Collection::new("cows");
        assert_eq!(c.name(), "cows");
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert!(c.iter().count() == 0);
    }

    #[test]
    fn insert_with_explicit_id() {
        let mut c = Collection::new("t");
        let id = c
            .insert(doc(&[("_id", Value::str("abc")), ("x", Value::i64(1))]))
            .unwrap();
        assert_eq!(id, "abc");
        assert_eq!(c.len(), 1);
        assert!(c.contains("abc"));
        assert_eq!(c.get("abc").unwrap().get("x"), Some(&Value::i64(1)));
        assert!(c.get("nope").is_none());
        assert!(!c.contains("nope"));
    }

    #[test]
    fn insert_generates_id_when_missing() {
        let mut c = Collection::new("t");
        let id = c.insert(doc(&[("x", Value::i64(1))])).unwrap();
        assert_auto_id(&id);
        let stored = c.get(&id).unwrap();
        assert_eq!(stored.get("_id"), Some(&Value::Str(id.clone())));
        // generated `_id` becomes the first key
        assert_eq!(stored.keys().next(), Some("_id"));
    }

    #[test]
    fn insert_empty_object_gets_id_only() {
        let mut c = Collection::new("t");
        let id = c.insert(Value::object()).unwrap();
        assert_auto_id(&id);
        let stored = c.get(&id).unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored.get("_id"), Some(&Value::Str(id)));
    }

    #[test]
    fn user_supplied_id_position_is_preserved() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("a", Value::i64(1)), ("_id", Value::str("k"))]))
            .unwrap();
        let keys: Vec<_> = c.get("k").unwrap().keys().collect();
        assert_eq!(keys, vec!["a", "_id"]); // not reordered
    }

    #[test]
    fn duplicate_insert_errors() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("dup"))])).unwrap();
        let err = c
            .insert(doc(&[("_id", Value::str("dup")), ("v", Value::i64(2))]))
            .unwrap_err();
        assert_eq!(err, StoreError::DuplicateId("dup".into()));
        // store unchanged: still one doc, original content intact
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("dup").unwrap().get("v"), None);
    }

    #[test]
    fn non_object_docs_rejected() {
        let mut c = Collection::new("t");
        assert_eq!(c.insert(Value::array()), Err(StoreError::NotAnObject));
        assert_eq!(c.insert(Value::i64(5)), Err(StoreError::NotAnObject));
        assert_eq!(c.insert(Value::Null), Err(StoreError::NotAnObject));
        assert_eq!(c.insert(Value::str("x")), Err(StoreError::NotAnObject));
        assert!(c.is_empty());
    }

    #[test]
    fn id_must_be_a_string() {
        let mut c = Collection::new("t");
        for bad in [
            Value::i64(7),
            Value::f64(7.5),
            Value::bool(true),
            Value::Null,
            Value::array(),
            Value::object(),
        ] {
            let err = c.insert(doc(&[("_id", bad.clone())])).unwrap_err();
            assert_eq!(err, StoreError::IdMustBeString, "bad id {bad}");
        }
        assert!(c.is_empty());
    }

    #[test]
    fn generated_ids_are_unique_and_well_formed() {
        let mut c = Collection::new("t");
        for i in 0..1000 {
            let id = c.insert(Value::object()).unwrap();
            assert_auto_id(&id);
            assert!(c.contains(&id));
            let _ = i;
        }
        let ids: std::collections::HashSet<&str> = c
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(ids.len(), 1000, "all generated ids unique");
        assert_eq!(c.len(), 1000);
    }

    // -- insert_many ----------------------------------------------------------

    #[test]
    fn insert_many_mixed_ids() {
        let mut c = Collection::new("t");
        let n = c
            .insert_many([
                doc(&[("_id", Value::str("a"))]),
                doc(&[("v", Value::i64(1))]), // auto
                doc(&[("v", Value::i64(2))]), // auto
            ])
            .unwrap();
        assert_eq!(n, 3);
        assert_eq!(c.len(), 3);
        assert!(c.contains("a"));
        for d in c.iter() {
            match d.get("_id") {
                Some(Value::Str(s)) => assert_auto_id_or(s, "a"),
                other => panic!("missing/invalid _id: {other:?}"),
            }
        }
    }

    fn assert_auto_id_or(s: &str, explicit: &str) {
        if s == explicit {
            return;
        }
        assert_auto_id(s);
    }

    #[test]
    fn insert_many_empty_is_noop() {
        let mut c = Collection::new("t");
        assert_eq!(c.insert_many(Vec::<Value>::new()).unwrap(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn insert_many_rejects_batch_when_store_duplicate() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("keep"))])).unwrap();
        let err = c
            .insert_many([
                doc(&[("_id", Value::str("fresh1"))]),
                doc(&[("_id", Value::str("keep"))]), // collides with store
                doc(&[("_id", Value::str("fresh2"))]),
            ])
            .unwrap_err();
        assert_eq!(err, StoreError::DuplicateId("keep".into()));
        // atomic: none of the batch landed
        assert_eq!(c.len(), 1);
        assert!(!c.contains("fresh1") && !c.contains("fresh2"));
    }

    #[test]
    fn insert_many_rejects_intra_batch_duplicate() {
        let mut c = Collection::new("t");
        let err = c
            .insert_many([
                doc(&[("_id", Value::str("twin"))]),
                doc(&[("_id", Value::str("twin"))]),
                doc(&[("v", Value::i64(1))]),
            ])
            .unwrap_err();
        assert_eq!(err, StoreError::DuplicateId("twin".into()));
        assert!(c.is_empty(), "nothing from the batch was inserted");
    }

    #[test]
    fn insert_many_rejects_invalid_doc_late_in_batch() {
        let mut c = Collection::new("t");
        let err = c
            .insert_many([
                doc(&[("_id", Value::str("ok1"))]),
                doc(&[("v", Value::i64(9))]),    // auto id, fine
                doc(&[("_id", Value::i64(42))]), // invalid: not a string
            ])
            .unwrap_err();
        assert_eq!(err, StoreError::IdMustBeString);
        assert!(
            c.is_empty(),
            "staging must not commit the earlier valid docs"
        );
    }

    #[test]
    fn insert_many_rejects_non_object_in_batch() {
        let mut c = Collection::new("t");
        let err = c
            .insert_many([doc(&[("_id", Value::str("ok"))]), Value::i64(1)])
            .unwrap_err();
        assert_eq!(err, StoreError::NotAnObject);
        assert!(c.is_empty());
    }

    // -- accessors ------------------------------------------------------------

    #[test]
    fn accessors_track_docs() {
        let mut c = Collection::new("moo");
        assert_eq!(c.name(), "moo");
        let id1 = c.insert(doc(&[("a", Value::i64(1))])).unwrap();
        let id2 = c.insert(doc(&[("b", Value::i64(2))])).unwrap();
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
        assert!(c.contains(&id1) && c.contains(&id2));
        assert_eq!(c.iter().count(), 2);
    }

    #[test]
    fn stored_doc_display_is_json_like() {
        let mut c = Collection::new("t");
        let id = c
            .insert(doc(&[("_id", Value::str("d1")), ("n", Value::i64(3))]))
            .unwrap();
        let s = format!("{}", c.get(&id).unwrap());
        assert!(s.starts_with('{') && s.contains("\"_id\": \"d1\"") && s.contains("\"n\": 3"));
    }

    // -- update operators ($set / $inc / $unset) --------------------------------

    /// An update operator spec (a Mongo operator object), e.g.
    /// `doc(&[("$set", doc(&[("a", Value::i64(1))]))])`.
    fn uop(pairs: &[(&str, Value)]) -> Value {
        doc(pairs)
    }

    fn by_id(id: &str) -> Value {
        doc(&[("_id", Value::str(id))])
    }

    #[test]
    fn update_one_set_changes_field_and_refreshes_index() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.create_index("age").unwrap();
        let upd = uop(&[(
            "$set",
            doc(&[("age", Value::i64(99)), ("tag", Value::str("moo"))]),
        )]);
        let n = c.update_one(by_id("a"), upd).unwrap();
        assert_eq!(n, 1);
        assert_eq!(c.get("a").unwrap().get("age"), Some(&Value::i64(99)));
        assert_eq!(c.get("a").unwrap().get("tag"), Some(&Value::str("moo")));
        // b untouched
        assert_eq!(c.get("b").unwrap().get("age"), Some(&Value::i64(40)));
        // index followed the change: 25 gone, 99 added, 40 still there
        let age = c.index("age").unwrap();
        assert!(age.ids_equal(&Value::i64(25)).is_empty());
        assert_eq!(age.ids_equal(&Value::i64(99)), vec!["a"]);
        assert_eq!(age.ids_equal(&Value::i64(40)), vec!["b"]);
        // a fresh query sees the update
        assert_eq!(c.count(doc(&[("age", Value::i64(99))])), 1);
    }

    #[test]
    fn update_one_no_match_is_an_error() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(1))]))
            .unwrap();
        let r = c.update_one(
            by_id("ghost"),
            uop(&[("$set", doc(&[("x", Value::i64(1))]))]),
        );
        assert_eq!(r, Err(StoreError::NoMatch));
        // store untouched
        assert_eq!(c.get("a").unwrap().get("x"), None);
    }

    #[test]
    fn update_one_set_creates_missing_and_nested_paths() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("top", Value::i64(1))]))
            .unwrap();
        let upd = uop(&[(
            "$set",
            doc(&[("a.b.c", Value::i64(7)), ("tag", Value::str("x"))]),
        )]);
        c.update_one(by_id("a"), upd).unwrap();
        let d = c.get("a").unwrap();
        assert_eq!(d.get_path("a.b.c"), Some(&Value::i64(7)));
        assert_eq!(d.get("tag"), Some(&Value::str("x")));
        assert_eq!(d.get("top"), Some(&Value::i64(1)));
        // sparse array create for a missing intermediate array field
        let mut c2 = Collection::new("t");
        c2.insert(doc(&[("_id", Value::str("a"))])).unwrap();
        let upd = uop(&[("$set", doc(&[("arr.2", Value::i64(5))]))]);
        c2.update_one(by_id("a"), upd).unwrap();
        assert_eq!(
            c2.get("a").unwrap().get("arr"),
            Some(&Value::array_from(vec![
                Value::Null,
                Value::Null,
                Value::i64(5)
            ]))
        );
    }

    #[test]
    fn update_one_inc_integer_float_and_create() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("n", Value::i64(5)),
            ("f", Value::f64(1.5)),
        ]))
        .unwrap();
        let upd = uop(&[(
            "$inc",
            doc(&[
                ("n", Value::i64(10)), // 5 + 10 = 15 (i64)
                ("f", Value::i64(1)),  // 1.5 + 1 = 2.5 (f64, cross-type)
                ("m", Value::i64(3)),  // missing -> created as 3
            ]),
        )]);
        c.update_one(by_id("a"), upd).unwrap();
        let d = c.get("a").unwrap();
        assert_eq!(d.get("n"), Some(&Value::i64(15)));
        assert_eq!(d.get("f"), Some(&Value::f64(2.5)));
        assert_eq!(d.get("m"), Some(&Value::i64(3)));
    }

    #[test]
    fn update_one_inc_overflow_widens_to_float() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("n", Value::i64(i64::MAX)),
        ]))
        .unwrap();
        let upd = uop(&[("$inc", doc(&[("n", Value::i64(1))]))]);
        c.update_one(by_id("a"), upd).unwrap();
        // i64::MAX + 1 overflows i64 -> widened to f64
        assert_eq!(
            c.get("a").unwrap().get("n"),
            Some(&Value::f64(i64::MAX as f64 + 1.0))
        );
    }

    #[test]
    fn update_one_inc_on_non_numeric_is_an_error() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("s", Value::str("hi"))]))
            .unwrap();
        let r = c.update_one(by_id("a"), uop(&[("$inc", doc(&[("s", Value::i64(1))]))]));
        assert!(matches!(r, Err(StoreError::InvalidUpdate(_))));
        // store untouched on error
        assert_eq!(c.get("a").unwrap().get("s"), Some(&Value::str("hi")));
    }

    #[test]
    fn update_one_inc_non_numeric_operand_is_an_error() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let r = c.update_one(by_id("a"), uop(&[("$inc", doc(&[("n", Value::str("x"))]))]));
        assert!(matches!(r, Err(StoreError::InvalidUpdate(_))));
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(1)));
    }

    #[test]
    fn update_one_unset_object_and_array_forms() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("x", Value::i64(1)),
            ("y", Value::i64(2)),
        ]))
        .unwrap();
        // object form: value ignored
        let upd = uop(&[("$unset", doc(&[("x", Value::str(""))]))]);
        c.update_one(by_id("a"), upd).unwrap();
        assert_eq!(c.get("a").unwrap().get("x"), None);
        assert_eq!(c.get("a").unwrap().get("y"), Some(&Value::i64(2)));
        // array form
        let upd = uop(&[("$unset", Value::array_from(vec![Value::str("y")]))]);
        c.update_one(by_id("a"), upd).unwrap();
        assert_eq!(c.get("a").unwrap().get("y"), None);
        // unset a missing path is a no-op (not an error)
        let upd = uop(&[("$unset", doc(&[("nope", Value::Null)]))]);
        assert_eq!(c.update_one(by_id("a"), upd), Ok(1));
    }

    #[test]
    fn update_one_applies_operators_in_spec_order() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(5))]))
            .unwrap();
        // $set then $inc on the same field: 5 -> set 10 -> +3 = 13
        let upd = uop(&[
            ("$set", doc(&[("n", Value::i64(10))])),
            ("$inc", doc(&[("n", Value::i64(3))])),
        ]);
        c.update_one(by_id("a"), upd).unwrap();
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(13)));
        // reversed: 5 -> +3 = 8 -> set 10
        let upd = uop(&[
            ("$inc", doc(&[("n", Value::i64(3))])),
            ("$set", doc(&[("n", Value::i64(10))])),
        ]);
        c.update_one(by_id("a"), upd).unwrap();
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(10)));
    }

    #[test]
    fn update_rejects_id_mutation() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        // $set / $inc / $unset of _id are all rejected; the doc keeps its _id
        for spec in [
            uop(&[("$set", doc(&[("_id", Value::str("zzz"))]))]),
            uop(&[("$inc", doc(&[("_id", Value::i64(1))]))]),
            uop(&[("$unset", doc(&[("_id", Value::str(""))]))]),
        ] {
            let r = c.update_one(by_id("a"), spec);
            assert!(
                matches!(r, Err(StoreError::InvalidUpdate(_))),
                "id mutation must fail"
            );
            assert_eq!(c.get("a").unwrap().get("_id"), Some(&Value::str("a")));
        }
    }

    #[test]
    fn update_one_malformed_specs_are_errors() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let before = c.get("a").unwrap().clone();
        let bad = [
            Value::i64(5),                                    // non-object spec
            Value::array_from(vec![Value::i64(1)]),           // non-object spec
            uop(&[("$bogus", doc(&[("n", Value::i64(1))]))]), // unknown operator
            uop(&[("$set", Value::i64(1))]),                  // $set non-object operand
            uop(&[("$inc", Value::str("x"))]),                // $inc non-object operand
            uop(&[("$unset", Value::i64(1))]),                // $unset neither object nor array
        ];
        for (i, spec) in bad.iter().enumerate() {
            let r = c.update_one(by_id("a"), spec.clone());
            assert!(
                matches!(r, Err(StoreError::InvalidUpdate(_))),
                "spec {i} must be InvalidUpdate"
            );
        }
        assert_eq!(
            c.get("a").unwrap(),
            &before,
            "store untouched after bad specs"
        );
    }

    #[test]
    fn update_many_changes_all_matches_and_counts() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(55))]))
            .unwrap();
        let n = c
            .update_many(
                doc(&[]),
                uop(&[("$set", doc(&[("flag", Value::bool(true))]))]),
            )
            .unwrap();
        assert_eq!(n, 3);
        for id in ["a", "b", "c"] {
            assert_eq!(c.get(id).unwrap().get("flag"), Some(&Value::bool(true)));
        }
        // a no-match update_many returns 0 (NOT an error, unlike update_one)
        let n = c
            .update_many(
                doc(&[("age", Value::i64(9999))]),
                uop(&[("$set", doc(&[("flag", Value::bool(false))]))]),
            )
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn update_many_refreshes_index() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.create_index("age").unwrap();
        let n = c
            .update_many(
                doc(&[("age", doc(&[("$gte", Value::i64(30))]))]),
                uop(&[("$inc", doc(&[("age", Value::i64(100))]))]),
            )
            .unwrap();
        assert_eq!(n, 1, "only b (40) matches $gte 30");
        assert_eq!(c.get("a").unwrap().get("age"), Some(&Value::i64(25)));
        assert_eq!(c.get("b").unwrap().get("age"), Some(&Value::i64(140)));
        // index: b moved 40 -> 140
        let age = c.index("age").unwrap();
        assert_eq!(age.ids_equal(&Value::i64(25)), vec!["a"]);
        assert_eq!(age.ids_equal(&Value::i64(140)), vec!["b"]);
        assert!(age.ids_equal(&Value::i64(40)).is_empty());
    }

    #[test]
    fn update_many_empty_collection_is_zero() {
        let c = Collection::new("t");
        let mut c = c;
        let n = c
            .update_many(doc(&[]), uop(&[("$set", doc(&[("x", Value::i64(1))]))]))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn update_many_malformed_spec_changes_nothing() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("n", Value::i64(2))]))
            .unwrap();
        let before = c.len();
        // an unknown operator is a shape error: validated up front, nothing applied
        let r = c.update_many(doc(&[]), uop(&[("$bogus", doc(&[("n", Value::i64(1))]))]));
        assert!(matches!(r, Err(StoreError::InvalidUpdate(_))));
        assert_eq!(c.len(), before);
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(1)));
        assert_eq!(c.get("b").unwrap().get("n"), Some(&Value::i64(2)));
    }
    // -- replace_one ------------------------------------------------------------

    #[test]
    fn replace_one_replaces_wholesale_and_preserves_id() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("age", Value::i64(25)),
            ("name", Value::str("moo")),
        ]))
        .unwrap();
        // new doc has a different shape: old fields gone, new fields added
        let r = c
            .replace_one(
                by_id("a"),
                doc(&[("age", Value::i64(40)), ("weight", Value::i64(3))]),
            )
            .unwrap();
        assert_eq!(r, 1);
        let d = c.get("a").unwrap();
        assert_eq!(d.get("_id"), Some(&Value::str("a")), "_id preserved");
        assert_eq!(d.get("age"), Some(&Value::i64(40)));
        assert_eq!(d.get("weight"), Some(&Value::i64(3)));
        assert_eq!(d.get("name"), None, "old field not in new_doc is dropped");
    }

    #[test]
    fn replace_one_refreshes_index() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.create_index("age").unwrap();
        // replace a's whole doc, dropping `age` entirely -> it becomes a Null entry
        c.replace_one(by_id("a"), doc(&[("other", Value::i64(9))]))
            .unwrap();
        let age = c.index("age").unwrap();
        assert_eq!(age.ids_equal(&Value::i64(40)), vec!["b"]);
        assert!(
            age.ids_equal(&Value::i64(25)).is_empty(),
            "old value deindexed"
        );
        // a now carries a Null age entry (field missing)
        assert_eq!(age.ids_equal(&Value::Null), vec!["a"]);
        assert_eq!(c.get("a").unwrap().get("age"), None);
    }

    #[test]
    fn replace_one_replaces_only_the_first_match() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("b")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        let r = c
            .replace_one(
                doc(&[("kind", Value::str("moo"))]),
                doc(&[("kind", Value::str("bee"))]),
            )
            .unwrap();
        assert_eq!(r, 1);
        // storage order: only one doc changed, the other stays moo
        let kinds: Vec<_> = c.iter().map(|d| d.get("kind").unwrap()).collect();
        let mut sorted = kinds
            .iter()
            .map(|v| match v {
                Value::Str(s) => s.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(sorted, vec!["bee", "moo"]);
        assert_eq!(c.count(doc(&[])), 2, "no doc was inserted or dropped");
    }

    #[test]
    fn replace_one_no_match_is_an_error() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let r = c.replace_one(by_id("ghost"), doc(&[("n", Value::i64(2))]));
        assert_eq!(r, Err(StoreError::NoMatch));
        assert_eq!(
            c.get("a").unwrap().get("n"),
            Some(&Value::i64(1)),
            "store untouched"
        );
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn replace_one_explicit_matching_id_keeps_position() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("z", Value::i64(1))]))
            .unwrap();
        // explicit _id equal to the matched one: position preserved
        c.replace_one(
            by_id("a"),
            doc(&[("x", Value::i64(2)), ("_id", Value::str("a"))]),
        )
        .unwrap();
        let d = c.get("a").unwrap();
        assert_eq!(d.get("_id"), Some(&Value::str("a")));
        assert_eq!(d.get("x"), Some(&Value::i64(2)));
        assert_eq!(d.get("z"), None);
        let keys: Vec<_> = d.keys().collect();
        assert_eq!(keys, vec!["x", "_id"], "user _id position preserved");
    }

    #[test]
    fn replace_one_missing_id_restored_as_first_key() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("z", Value::i64(1))]))
            .unwrap();
        // new doc omits _id: the matched doc's _id is restored as the first key
        c.replace_one(by_id("a"), doc(&[("x", Value::i64(2))]))
            .unwrap();
        let d = c.get("a").unwrap();
        assert_eq!(d.get("_id"), Some(&Value::str("a")));
        assert_eq!(d.keys().next(), Some("_id"));
    }

    #[test]
    fn replace_one_rejects_id_change() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let r = c.replace_one(
            by_id("a"),
            doc(&[("_id", Value::str("zzz")), ("n", Value::i64(9))]),
        );
        assert_eq!(
            r,
            Err(StoreError::IdMismatch {
                expected: "a".into(),
                found: "zzz".into()
            })
        );
        // store untouched: still id "a", content unchanged
        assert_eq!(c.len(), 1);
        assert!(c.contains("a"));
        assert!(!c.contains("zzz"));
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(1)));
    }

    #[test]
    fn replace_one_rejects_non_object_doc() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        for bad in [Value::array(), Value::i64(5), Value::str("x"), Value::Null] {
            let r = c.replace_one(by_id("a"), bad.clone());
            assert_eq!(r, Err(StoreError::NotAnObject), "non-object {bad}");
        }
        assert_eq!(
            c.get("a").unwrap().get("n"),
            Some(&Value::i64(1)),
            "untouched"
        );
    }

    // -- delete_one / delete_many ------------------------------------------------

    #[test]
    fn delete_one_removes_one_match() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("b")),
            ("kind", Value::str("bee")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("c")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        let deleted = c.delete_one(doc(&[("kind", Value::str("moo"))]));
        assert!(deleted, "delete_one returns true when a doc was removed");
        assert_eq!(c.len(), 2, "exactly one doc removed");
        // the bee doc always survives
        assert!(c.contains("b"));
        // exactly one of the two moo docs was deleted (HashMap iteration order)
        let moo_survivors = ["a", "c"].iter().filter(|id| c.contains(id)).count();
        assert_eq!(moo_survivors, 1, "exactly one moo doc remains");
    }

    #[test]
    fn delete_one_no_match_returns_false() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let deleted = c.delete_one(by_id("ghost"));
        assert!(!deleted, "delete_one returns false on no match");
        assert_eq!(c.len(), 1, "store untouched");
        assert!(c.contains("a"));
    }

    #[test]
    fn delete_one_empty_collection_returns_false() {
        let mut c = Collection::new("t");
        let deleted = c.delete_one(doc(&[]));
        assert!(!deleted);
        assert!(c.is_empty());
    }

    #[test]
    fn delete_one_refreshes_index() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(25))]))
            .unwrap();
        c.create_index("age").unwrap();
        // delete a (age=25): the age index should no longer have "a" under 25
        assert!(c.delete_one(by_id("a")));
        let age = c.index("age").unwrap();
        assert_eq!(
            age.ids_equal(&Value::i64(25)),
            vec!["c"],
            "only c remains at age 25"
        );
        assert_eq!(age.ids_equal(&Value::i64(40)), vec!["b"]);
        assert_eq!(age.len(), 2, "index shrank by one");
    }

    #[test]
    fn delete_one_removes_from_all_indexes() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("age", Value::i64(25)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("b")),
            ("age", Value::i64(40)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.create_index("age").unwrap();
        c.create_index("tag").unwrap();
        assert!(c.delete_one(by_id("a")));
        assert!(
            c.index("age")
                .unwrap()
                .ids_equal(&Value::i64(25))
                .is_empty()
        );
        assert_eq!(
            c.index("tag").unwrap().ids_equal(&Value::str("x")),
            vec!["b"]
        );
        // primary _id index also updated
        assert!(
            c.index("_id")
                .unwrap()
                .ids_equal(&Value::str("a"))
                .is_empty()
        );
    }

    #[test]
    fn delete_many_removes_all_matches_and_counts() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("b")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("c")),
            ("kind", Value::str("bee")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("d")),
            ("kind", Value::str("moo")),
        ]))
        .unwrap();
        let n = c.delete_many(doc(&[("kind", Value::str("moo"))]));
        assert_eq!(n, 3, "three moo docs deleted");
        assert_eq!(c.len(), 1);
        assert!(c.contains("c"), "bee doc survived");
        assert!(!c.contains("a") && !c.contains("b") && !c.contains("d"));
    }

    #[test]
    fn delete_many_no_match_returns_zero() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let n = c.delete_many(by_id("ghost"));
        assert_eq!(n, 0, "no match -> zero deleted, no error");
        assert_eq!(c.len(), 1);
        assert!(c.contains("a"));
    }

    #[test]
    fn delete_many_empty_collection_returns_zero() {
        let mut c = Collection::new("t");
        let n = c.delete_many(doc(&[]));
        assert_eq!(n, 0);
        assert!(c.is_empty());
    }

    #[test]
    fn delete_many_empty_filter_removes_all() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a"))])).unwrap();
        c.insert(doc(&[("_id", Value::str("b"))])).unwrap();
        c.insert(doc(&[("_id", Value::str("c"))])).unwrap();
        let n = c.delete_many(doc(&[]));
        assert_eq!(n, 3);
        assert!(c.is_empty());
    }

    #[test]
    fn delete_many_refreshes_index() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(40))]))
            .unwrap();
        c.create_index("age").unwrap();
        let n = c.delete_many(doc(&[("age", Value::i64(25))]));
        assert_eq!(n, 2);
        let age = c.index("age").unwrap();
        assert!(
            age.ids_equal(&Value::i64(25)).is_empty(),
            "all age-25 entries gone"
        );
        assert_eq!(age.ids_equal(&Value::i64(40)), vec!["c"]);
        assert_eq!(age.len(), 1);
    }

    #[test]
    fn delete_then_insert_reuses_index_slot() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(10))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(20))]))
            .unwrap();
        c.create_index("age").unwrap();
        assert_eq!(c.index("age").unwrap().len(), 2);
        assert!(c.delete_one(by_id("a")));
        assert_eq!(c.index("age").unwrap().len(), 1);
        // re-insert with the same id but different value: index is consistent
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(99))]))
            .unwrap();
        let age = c.index("age").unwrap();
        assert_eq!(age.len(), 2);
        assert_eq!(age.ids_equal(&Value::i64(99)), vec!["a"]);
        assert_eq!(age.ids_equal(&Value::i64(20)), vec!["b"]);
    }

    #[test]
    fn delete_one_by_range_filter() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(10))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(30))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(50))]))
            .unwrap();
        // range: age >= 20 and age < 40 -> only b
        let filter = doc(&[(
            "age",
            doc(&[("$gte", Value::i64(20)), ("$lt", Value::i64(40))]),
        )]);
        assert!(c.delete_one(filter));
        assert!(!c.contains("b"));
        assert!(c.contains("a") && c.contains("c"));
        assert_eq!(c.len(), 2);
    }

    // -- atomic batch (transaction) ------------------------------------------

    #[test]
    fn transaction_commit_applies_all_writes_and_indexes() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(25))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(40))]))
            .unwrap();
        c.create_index("age").unwrap();
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(30))]))
            .unwrap();
        assert_eq!(
            tx.update(
                by_id("a"),
                uop(&[("$set", doc(&[("age", Value::i64(99))]))])
            ),
            Ok(1)
        );
        assert_eq!(tx.delete(by_id("b")), 1);
        // reads inside the tx still see the pre-batch state
        assert_eq!(tx.len(), 2, "pre-batch doc count (c not yet applied)");
        assert!(!tx.contains("c"), "inserted doc invisible until commit");
        assert_eq!(tx.get("a").unwrap().get("age"), Some(&Value::i64(25)));
        assert!(tx.contains("b"), "deleted doc still visible pre-batch");
        tx.commit().unwrap();
        // all writes applied at once
        assert_eq!(c.len(), 2); // a (updated), c (inserted); b deleted
        assert!(c.contains("a") && c.contains("c"));
        assert!(!c.contains("b"));
        assert_eq!(c.get("a").unwrap().get("age"), Some(&Value::i64(99)));
        assert_eq!(c.get("c").unwrap().get("age"), Some(&Value::i64(30)));
        // index maintained: 25 and 40 gone, 99 and 30 present
        let age = c.index("age").unwrap();
        assert!(age.ids_equal(&Value::i64(25)).is_empty());
        assert!(age.ids_equal(&Value::i64(40)).is_empty());
        assert_eq!(age.ids_equal(&Value::i64(99)), vec!["a"]);
        assert_eq!(age.ids_equal(&Value::i64(30)), vec!["c"]);
        // a fresh query sees the batched result
        assert_eq!(c.count(doc(&[("age", Value::i64(99))])), 1);
    }

    #[test]
    fn transaction_rolls_back_on_duplicate_insert() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("keep")), ("n", Value::i64(1))]))
            .unwrap();
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("fresh")), ("n", Value::i64(2))]))
            .unwrap();
        let err = tx
            .insert(doc(&[("_id", Value::str("keep")), ("n", Value::i64(3))]))
            .unwrap_err();
        assert_eq!(err, StoreError::DuplicateId("keep".into()));
        assert!(tx.is_failed());
        assert_eq!(tx.error(), Some(&StoreError::DuplicateId("keep".into())));
        // commit is a no-op: nothing staged (incl. the valid earlier insert) applied
        assert_eq!(tx.commit(), Err(StoreError::DuplicateId("keep".into())));
        assert_eq!(c.len(), 1);
        assert!(!c.contains("fresh"));
        assert_eq!(c.get("keep").unwrap().get("n"), Some(&Value::i64(1)));
    }

    #[test]
    fn transaction_rolls_back_on_malformed_update() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("s", Value::str("hi"))]))
            .unwrap();
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(9))]))
            .unwrap();
        // $inc on a string field -> InvalidUpdate -> fails the whole tx
        let r = tx.update(by_id("a"), uop(&[("$inc", doc(&[("s", Value::i64(1))]))]));
        assert!(matches!(r, Err(StoreError::InvalidUpdate(_))));
        assert!(tx.is_failed());
        tx.rollback();
        assert_eq!(c.len(), 1);
        assert!(!c.contains("x"));
        assert_eq!(c.get("a").unwrap().get("s"), Some(&Value::str("hi")));
    }

    #[test]
    fn transaction_drop_without_commit_is_rollback() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        {
            let mut tx = c.begin();
            tx.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(9))]))
                .unwrap();
            tx.delete(by_id("a"));
            // reads inside see the pre-batch state
            assert_eq!(tx.len(), 1);
            assert!(tx.contains("a"));
            assert!(!tx.contains("x"));
            // tx dropped here without commit
        }
        // nothing applied
        assert_eq!(c.len(), 1);
        assert!(c.contains("a"));
        assert!(!c.contains("x"));
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(1)));
    }

    #[test]
    fn transaction_explicit_rollback_discards_writes() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(9))]))
            .unwrap();
        tx.rollback();
        assert_eq!(c.len(), 1);
        assert!(!c.contains("x"));
        assert!(c.contains("a"));
    }

    #[test]
    fn transaction_empty_commit_is_noop() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a"))])).unwrap();
        let tx = c.begin();
        assert_eq!(tx.commit(), Ok(()));
        assert_eq!(c.len(), 1);
        assert!(c.contains("a"));
    }

    #[test]
    fn transaction_intra_batch_duplicate_insert_fails() {
        let mut c = Collection::new("t");
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("twin")), ("n", Value::i64(1))]))
            .unwrap();
        let err = tx
            .insert(doc(&[("_id", Value::str("twin")), ("n", Value::i64(2))]))
            .unwrap_err();
        assert_eq!(err, StoreError::DuplicateId("twin".into()));
        assert!(tx.is_failed());
        tx.rollback();
        assert!(c.is_empty());
    }

    #[test]
    fn transaction_later_write_overwrites_earlier_on_same_id() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(0))]))
            .unwrap();
        let mut tx = c.begin();
        // Two $inc on the same field in one batch: each recomputes from the
        // pre-batch snapshot (n=0). The later op overwrites the earlier one on
        // the same id, so the committed value is 0 + 5 = 5 (NOT 0 + 1 + 5).
        assert_eq!(
            tx.update(by_id("a"), uop(&[("$inc", doc(&[("n", Value::i64(1))]))])),
            Ok(1)
        );
        assert_eq!(
            tx.update(by_id("a"), uop(&[("$inc", doc(&[("n", Value::i64(5))]))])),
            Ok(1)
        );
        tx.commit().unwrap();
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(5)));
    }

    #[test]
    fn transaction_update_then_delete_same_id_net_delete() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(1))]))
            .unwrap();
        let mut tx = c.begin();
        // update x (stages a put), then delete x (composes the put into a
        // Delete) -> net: x is removed
        assert_eq!(
            tx.update(by_id("x"), uop(&[("$set", doc(&[("n", Value::i64(9))]))])),
            Ok(1)
        );
        assert_eq!(tx.delete(by_id("x")), 1);
        tx.commit().unwrap();
        assert!(
            c.is_empty(),
            "update+delete of the same pre-batch doc nets to deletion"
        );
    }

    #[test]
    fn transaction_delete_ignores_batch_inserted_ids() {
        let mut c = Collection::new("t");
        let mut tx = c.begin();
        // a doc inserted only in this batch is invisible to pre-batch reads,
        // so a filter-based delete cannot target it (it stays staged)
        tx.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(9))]))
            .unwrap();
        assert_eq!(tx.delete(by_id("x")), 0, "x is not in the pre-batch store");
        tx.commit().unwrap();
        assert!(
            c.contains("x"),
            "the batch insert remains (delete could not see it)"
        );
    }

    #[test]
    fn transaction_mixed_ops_compose_by_id_and_refresh_indexes() {
        let mut c = Collection::new("t");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("age", Value::i64(10)),
            ("tag", Value::str("moo")),
        ]))
        .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("b")),
            ("age", Value::i64(20)),
            ("tag", Value::str("bee")),
        ]))
        .unwrap();
        c.create_index("age").unwrap();
        c.create_index("tag").unwrap();
        let mut tx = c.begin();
        // replace a wholesale (drops `tag`)
        assert_eq!(
            tx.replace(by_id("a"), doc(&[("age", Value::i64(99))])),
            Ok(1)
        );
        // update b's age
        assert_eq!(
            tx.update(by_id("b"), uop(&[("$inc", doc(&[("age", Value::i64(5))]))])),
            Ok(1)
        );
        // insert a fresh doc
        tx.insert(doc(&[
            ("_id", Value::str("c")),
            ("age", Value::i64(50)),
            ("tag", Value::str("moo")),
        ]))
        .unwrap();
        tx.commit().unwrap();
        assert_eq!(c.len(), 3);
        // a was replaced wholesale: `tag` gone, age now 99
        assert_eq!(c.get("a").unwrap().get("age"), Some(&Value::i64(99)));
        assert_eq!(
            c.get("a").unwrap().get("tag"),
            None,
            "wholesale replace dropped tag"
        );
        // b incremented
        assert_eq!(c.get("b").unwrap().get("age"), Some(&Value::i64(25)));
        // age index: 10 and 20 gone; 99(a), 25(b), 50(c) present
        let age = c.index("age").unwrap();
        assert!(age.ids_equal(&Value::i64(10)).is_empty());
        assert!(age.ids_equal(&Value::i64(20)).is_empty());
        assert_eq!(age.ids_equal(&Value::i64(99)), vec!["a"]);
        assert_eq!(age.ids_equal(&Value::i64(25)), vec!["b"]);
        assert_eq!(age.ids_equal(&Value::i64(50)), vec!["c"]);
        // tag index: a dropped to Null, b=bee, c=moo
        let tag = c.index("tag").unwrap();
        assert_eq!(tag.ids_equal(&Value::Null), vec!["a"]);
        assert_eq!(tag.ids_equal(&Value::str("bee")), vec!["b"]);
        assert_eq!(tag.ids_equal(&Value::str("moo")), vec!["c"]);
        // both indexes hold exactly one entry per doc
        assert_eq!(age.len(), 3);
        assert_eq!(tag.len(), 3);
    }

    #[test]
    fn transaction_replace_id_mismatch_rolls_back() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("n", Value::i64(1))]))
            .unwrap();
        let mut tx = c.begin();
        tx.insert(doc(&[("_id", Value::str("x")), ("n", Value::i64(9))]))
            .unwrap();
        // replace a with a doc whose _id differs -> IdMismatch -> rollback
        let r = tx.replace(
            by_id("a"),
            doc(&[("_id", Value::str("zzz")), ("n", Value::i64(9))]),
        );
        assert_eq!(
            r,
            Err(StoreError::IdMismatch {
                expected: "a".into(),
                found: "zzz".into()
            })
        );
        assert!(tx.is_failed());
        tx.rollback();
        assert_eq!(c.len(), 1);
        assert!(c.contains("a"));
        assert!(!c.contains("x"));
        assert_eq!(c.get("a").unwrap().get("n"), Some(&Value::i64(1)));
    }

    #[test]
    fn transaction_pre_batch_reads_ignore_staged_writes() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("a")), ("k", Value::str("moo"))]))
            .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("k", Value::str("bee"))]))
            .unwrap();
        let mut tx = c.begin();
        // staged writes that would change the filtered result
        tx.update(
            doc(&[("k", Value::str("moo"))]),
            uop(&[("$set", doc(&[("k", Value::str("bee"))]))]),
        )
        .unwrap();
        tx.insert(doc(&[("_id", Value::str("c")), ("k", Value::str("bee"))]))
            .unwrap();
        // pre-batch reads: still 2 docs, only one bee, a is still moo
        assert_eq!(tx.len(), 2);
        assert_eq!(tx.count(doc(&[("k", Value::str("bee"))])), 1);
        assert_eq!(tx.get("a").unwrap().get("k"), Some(&Value::str("moo")));
        tx.commit().unwrap();
        // post-commit: a->bee, c inserted -> three docs, three bee
        assert_eq!(c.len(), 3);
        assert_eq!(c.count(doc(&[("k", Value::str("bee"))])), 3);
    }

    #[test]
    fn transaction_delete_many_several_and_refresh_index() {
        let mut c = Collection::new("t");
        for (id, age) in [("a", 25), ("b", 25), ("c", 40), ("d", 25)] {
            c.insert(doc(&[("_id", Value::str(id)), ("age", Value::i64(age))]))
                .unwrap();
        }
        c.create_index("age").unwrap();
        let mut tx = c.begin();
        assert_eq!(tx.delete(doc(&[("age", Value::i64(25))])), 3);
        tx.commit().unwrap();
        assert_eq!(c.len(), 1);
        assert!(c.contains("c"));
        assert!(!c.contains("a") && !c.contains("b") && !c.contains("d"));
        let age = c.index("age").unwrap();
        assert!(age.ids_equal(&Value::i64(25)).is_empty());
        assert_eq!(age.ids_equal(&Value::i64(40)), vec!["c"]);
        assert_eq!(age.len(), 1);
    }

    #[test]
    fn transaction_insert_many_stages_all_or_nothing() {
        let mut c = Collection::new("t");
        c.insert(doc(&[("_id", Value::str("keep"))])).unwrap();
        let mut tx = c.begin();
        // third doc collides with the store -> the whole insert_many (and tx) fails
        let r = tx.insert_many([
            doc(&[("_id", Value::str("p"))]),
            doc(&[("_id", Value::str("q"))]),
            doc(&[("_id", Value::str("keep"))]),
        ]);
        assert_eq!(r, Err(StoreError::DuplicateId("keep".into())));
        assert!(tx.is_failed());
        tx.rollback();
        assert_eq!(c.len(), 1, "no partial insert_many landed");
        assert!(!c.contains("p") && !c.contains("q"));
    }

    // -- stats / reindex ------------------------------------------------------------------

    /// Three-doc fixture: a(age=30 i64, tag="x"), b(age=30.0 f64, no tag),
    /// c(age=40 i64, tag="x") with `age` + `tag` field indexes created.
    fn stats_herd() -> Collection {
        let mut c = Collection::new("stats");
        c.insert(doc(&[
            ("_id", Value::str("a")),
            ("age", Value::i64(30)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.insert(doc(&[("_id", Value::str("b")), ("age", Value::f64(30.0))]))
            .unwrap();
        c.insert(doc(&[
            ("_id", Value::str("c")),
            ("age", Value::i64(40)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.create_index("age").unwrap();
        c.create_index("tag").unwrap();
        c
    }

    #[test]
    fn stats_empty_collection() {
        let c = Collection::new("empty");
        let s = c.stats();
        assert_eq!(s.docs, 0);
        assert_eq!(s.indexes, 1, "only the primary _id index exists");
        assert_eq!(s.per_index.len(), 1);
        let ix = &s.per_index[0];
        assert_eq!(ix.field, "_id");
        assert_eq!(ix.entries, 0);
        assert_eq!(ix.distinct, 0);
        assert!(ix.memory > 0);
        assert!(s.docs_memory > 0);
        assert_eq!(s.total_memory, s.docs_memory + ix.memory);
    }

    #[test]
    fn stats_counts_and_deterministic_order() {
        let c = stats_herd();
        let s = c.stats();
        assert_eq!(s.docs, 3);
        assert_eq!(s.indexes, 3);
        // Field-name byte order: "_id" < "age" < "tag".
        assert_eq!(
            s.per_index
                .iter()
                .map(|i| i.field.as_str())
                .collect::<Vec<_>>(),
            vec!["_id", "age", "tag"]
        );
        for ix in &s.per_index {
            assert_eq!(ix.entries, 3, "one entry per doc in every index");
            assert!(ix.memory > 0);
        }
        // Cross-numeric total order: I64(30) == F64(30.0), so 30/30.0/40 -> 2.
        assert_eq!(s.per_index[0].distinct, 3, "_id is always distinct");
        assert_eq!(s.per_index[1].distinct, 2, "age: 30(i64), 30.0(f64), 40");
        assert_eq!(s.per_index[2].distinct, 2, "tag: x + Null (missing)");
        assert_eq!(
            s.total_memory,
            s.docs_memory + s.per_index.iter().map(|i| i.memory).sum::<usize>()
        );
    }

    #[test]
    fn stats_memory_scales_with_data() {
        let small = Collection::new("small");
        let s1 = small.stats();
        let mut big = Collection::new("big");
        let mut payload = String::new();
        for i in 0..200 {
            payload.push_str(&format!("doc-{i}-{}", "moo".repeat(4)));
            big.insert(doc(&[
                ("_id", Value::str(format!("d-{i}"))),
                ("n", Value::i64(i as i64)),
                ("blob", Value::str(payload.clone())),
            ]))
            .unwrap();
        }
        big.create_index("n").unwrap();
        big.create_index("blob").unwrap();
        let s2 = big.stats();
        assert_eq!(s2.docs, 200);
        assert!(
            s2.total_memory > s1.total_memory,
            "bigger store + more indexes must estimate more bytes"
        );
        assert!(s2.docs_memory > s1.docs_memory);
        let n_idx = s2.per_index.iter().find(|i| i.field == "n").unwrap();
        assert_eq!(n_idx.entries, 200);
        assert_eq!(n_idx.distinct, 200);
    }

    #[test]
    fn stats_reflect_drop_index() {
        let mut c = stats_herd();
        let before = c.stats();
        c.drop_index("tag").unwrap();
        let after = c.stats();
        assert_eq!(after.indexes, 2);
        assert!(!after.per_index.iter().any(|i| i.field == "tag"));
        assert!(after.total_memory < before.total_memory);
        assert_eq!(after.docs, before.docs);
    }

    #[test]
    fn reindex_returns_index_count() {
        let mut c = stats_herd();
        assert_eq!(c.reindex(), 3);
        let mut empty = Collection::new("e");
        assert_eq!(empty.reindex(), 1, "primary only on an empty store");
    }

    #[test]
    fn reindex_stays_in_lockstep_after_mixed_writes() {
        let mut c = Collection::new("mixed");
        for (id, age) in [("a", 10), ("b", 20), ("c", 30), ("d", 40)] {
            c.insert(doc(&[("_id", Value::str(id)), ("age", Value::i64(age))]))
                .unwrap();
        }
        c.create_index("age").unwrap();
        // Mixed mutations: bump, delete, insert-with-missing-field, replace.
        c.update_many(Value::object(), uop_inc(5)).unwrap();
        c.delete_one(Value::Object(
            std::iter::once(("_id".to_string(), Value::str("a"))).collect(),
        ))
        .then_some(());
        c.insert(doc(&[("_id", Value::str("e"))])).unwrap(); // no `age` -> Null entry
        c.reindex();
        let age = c.index("age").unwrap();
        assert_eq!(age.len(), c.len(), "every doc has exactly one entry");
        // ages: b=25, c=35, d=45, e=Null -> range [20, 50) = [b, c, d]
        use std::ops::Bound::{Excluded, Included};
        assert_eq!(
            age.ids_range(Included(&Value::i64(20)), Excluded(&Value::i64(50))),
            vec!["b", "c", "d"]
        );
        assert_eq!(age.ids_equal(&Value::Null), vec!["e"]);
        // Full-scan agreement after the rebuild.
        assert_eq!(
            c.count(Value::object_from(vec![(
                "age".to_string(),
                Value::i64(35)
            )])),
            1
        );
    }

    #[test]
    fn stats_unchanged_by_reindex() {
        let mut c = stats_herd();
        let before = c.stats();
        c.reindex();
        assert_eq!(
            c.stats(),
            before,
            "deterministic rebuild leaves stats intact"
        );
    }

    /// `{"$inc": {"age": n}}` update spec for the mixed-writes test.
    fn uop_inc(n: i64) -> Value {
        Value::Object(
            std::iter::once((
                "$inc".to_string(),
                Value::Object(std::iter::once(("age".to_string(), Value::i64(n))).collect()),
            ))
            .collect(),
        )
    }
}
