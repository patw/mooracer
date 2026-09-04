//! Indexes — the ordered index layer over [`Collection`](crate::Collection).
//!
//! Design notes (perf posture, see spec "Indexes" / "Performance posture"):
//!
//! - [`FieldIndex`] is a **sorted `Vec` of `(value, _id)` entries**, not a
//!   `BTreeMap`: equality and range scans are two `partition_point` binary
//!   searches over a contiguous array (cache-friendly, no per-node pointer
//!   chasing or allocation), and updates are `O(n)` memmoves — cheap at the
//!   entry counts this engine targets. Entries order by
//!   (`Value::Ord` — the engine's total order: exact cross-numeric, total
//!   NaN —, then `_id` byte order), so equal-value slices are contiguous and
//!   deterministic (ids in ascending byte order).
//! - [`IndexSet`] owns one `FieldIndex` per indexed field. The primary
//!   `_id` index is always present and cannot be dropped. A **missing field
//!   is indexed as `Null`** (MongoDB convention: `{"f": null}` matches both
//!   explicit `null` and absence), so every document has exactly one entry
//!   in every field index.
//! - `index_doc` / `deindex_doc` are the add/remove primitives the
//!   collection's insert / remove / set paths call; a document's entries are
//!   added or removed as a unit, so the write model (`&mut self`) never
//!   observes a half-maintained index.
//! - No `unsafe` yet: nothing has been measured that would justify it, and
//!   the sorted-vec layout is already the cache-friendly choice. If a
//!   profile ever shows the memmove cost dominating on very large indexes,
//!   the next step is a packed (value, id-offset) arena layout.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Bound;

use crate::collection::{ID_KEY, StoreError};
use crate::value::Value;

/// One index entry: the indexed value and the owning document's `_id`.
#[derive(Clone, Debug)]
struct Entry {
    value: Value,
    id: String,
}

// ---------------------------------------------------------------------------
// FieldIndex
// ---------------------------------------------------------------------------

/// A single-field ordered index: a sorted array of `(value, _id)` entries.
///
/// Invariant: `entries` is sorted by `value` (the engine's total order),
/// ties by `id` (byte order). All lookups exploit that invariant with
/// `Vec::partition_point` (binary search, no allocation).
pub struct FieldIndex {
    field: String,
    entries: Vec<Entry>,
}

impl FieldIndex {
    /// Create an empty index for `field`.
    pub fn new(field: &str) -> Self {
        FieldIndex {
            field: field.to_string(),
            entries: Vec::new(),
        }
    }

    /// The indexed field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Number of entries (one per indexed document).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no document is indexed.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of **distinct** values (equal under the engine's total order:
    /// `I64(1)` and `F64(1.0)` count as one).
    pub fn distinct(&self) -> usize {
        let mut n = 0usize;
        let mut prev: Option<&Value> = None;
        for e in &self.entries {
            match prev {
                Some(p) if *p == e.value => {}
                _ => {
                    n += 1;
                    prev = Some(&e.value);
                }
            }
        }
        n
    }

    /// Insert `(value, id)` keeping the sort invariant.
    pub fn insert(&mut self, value: Value, id: String) {
        let pos = self.entries.partition_point(|e| {
            match e.value.cmp(&value) {
                Ordering::Less => true,
                Ordering::Equal => e.id < id,
                Ordering::Greater => false,
            }
        });
        self.entries.insert(pos, Entry { value, id });
    }

    /// Remove the entry `(value, id)`. Returns `true` if an entry was found.
    pub fn remove(&mut self, id: &str, value: &Value) -> bool {
        let start = self.entries.partition_point(|e| e.value.cmp(value) == Ordering::Less);
        let end = self.entries.partition_point(|e| e.value.cmp(value) != Ordering::Greater);
        let slice = &self.entries[start..end];
        let i = slice.partition_point(|e| e.id.as_str() < id);
        if i < slice.len() && slice[i].id == id {
            self.entries.remove(start + i);
            true
        } else {
            false
        }
    }

    /// `true` when some entry's value equals `value` (engine total order).
    pub fn contains_value(&self, value: &Value) -> bool {
        let s = self.entries.partition_point(|e| e.value.cmp(value) == Ordering::Less);
        s < self.entries.len() && self.entries[s].value == *value
    }

    /// Document ids whose indexed value equals `value`, in `_id` order.
    pub fn ids_equal(&self, value: &Value) -> Vec<&str> {
        let s = self.entries.partition_point(|e| e.value.cmp(value) == Ordering::Less);
        let e = self.entries.partition_point(|e| e.value.cmp(value) != Ordering::Greater);
        self.entries[s..e].iter().map(|en| en.id.as_str()).collect()
    }

    /// Document ids whose indexed value falls in the bound range
    /// `[lo, hi]` (both inclusive/exclusive as given), in index order
    /// (ascending value, ties by `_id`).
    pub fn ids_range(&self, lo: Bound<&Value>, hi: Bound<&Value>) -> Vec<&str> {
        let s = lower_bound(&self.entries, lo);
        let e = upper_bound(&self.entries, hi);
        debug_assert!(s <= e, "bounds crossed: {s} > {e}");
        self.entries[s..e].iter().map(|en| en.id.as_str()).collect()
    }

    /// All entries in index order (ascending value, ties by `_id`).
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&Value, &str)> + '_ {
        self.entries.iter().map(|e| (&e.value, e.id.as_str()))
    }

    /// Deterministic rebuild: replace the contents with `pairs`
    /// `(_id, value)` in sorted order (insertion order of the input is
    /// irrelevant). Used by `create_index` backfill and rebuilds.
    pub fn load(&mut self, pairs: impl IntoIterator<Item = (String, Value)>) {
        let mut entries: Vec<Entry> =
            pairs.into_iter().map(|(id, value)| Entry { value, id }).collect();
        entries.sort_by(|a, b| {
            a.value
                .cmp(&b.value)
                .then_with(|| a.id.cmp(&b.id))
        });
        self.entries = entries;
    }

    /// Rough memory footprint: struct + array capacity + id string
    /// capacities + recursive heap of the indexed values. An *estimate*
    /// (capacity-based), used by `stats()`.
    pub fn memory_size(&self) -> usize {
        let mut s = std::mem::size_of::<FieldIndex>()
            + self.entries.capacity() * std::mem::size_of::<Entry>();
        for e in &self.entries {
            s += e.id.capacity() + value_heap(&e.value);
        }
        s
    }
}

/// Capacity of the first (inclusive) entry whose value is `> bound`
/// (`>= bound` when excluded) — the start of the `[lo, …]` range.
fn lower_bound(entries: &[Entry], lo: Bound<&Value>) -> usize {
    match lo {
        Bound::Unbounded => 0,
        Bound::Included(v) => entries.partition_point(|e| e.value.cmp(v) == Ordering::Less),
        Bound::Excluded(v) => entries.partition_point(|e| e.value.cmp(v) != Ordering::Greater),
    }
}

/// Capacity of the first (inclusive) entry whose value is `> bound`
/// (`>= bound` when excluded) — the end of the `[…, hi]` range.
fn upper_bound(entries: &[Entry], hi: Bound<&Value>) -> usize {
    match hi {
        Bound::Unbounded => entries.len(),
        Bound::Included(v) => entries.partition_point(|e| e.value.cmp(v) != Ordering::Greater),
        Bound::Excluded(v) => entries.partition_point(|e| e.value.cmp(v) == Ordering::Less),
    }
}

/// Recursive capacity of a value's heap (estimate). Used by
/// [`FieldIndex::memory_size`] and `Collection::stats`.
pub fn value_heap(v: &Value) -> usize {
    match v {
        Value::Str(s) => s.capacity(),
        Value::Array(a) => {
            a.capacity() * std::mem::size_of::<Value>() + a.iter().map(value_heap).sum::<usize>()
        }
        Value::Object(o) => o.capacity() * std::mem::size_of::<(String, Value)>()
            + o.iter().map(|(k, val)| k.capacity() + value_heap(val)).sum::<usize>(),
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// IndexSet
// ---------------------------------------------------------------------------

/// The collection's index set: one [`FieldIndex`] per indexed field, plus
/// the always-present primary `_id` index.
pub struct IndexSet {
    fields: HashMap<String, FieldIndex>,
}

impl IndexSet {
    /// New set with only the primary `_id` index.
    pub(crate) fn new() -> Self {
        let mut fields = HashMap::new();
        fields.insert(ID_KEY.to_string(), FieldIndex::new(ID_KEY));
        IndexSet { fields }
    }

    /// Insert (or replace) a built index for `field`.
    pub fn insert_index(&mut self, field: String, idx: FieldIndex) {
        self.fields.insert(field, idx);
    }

    /// Drop a field index. The primary `_id` index cannot be dropped, and
    /// dropping an index that was never created is an error.
    pub fn drop(&mut self, field: &str) -> Result<(), StoreError> {
        if field == ID_KEY {
            return Err(StoreError::PrimaryIndex);
        }
        match self.fields.remove(field) {
            Some(_) => Ok(()),
            None => Err(StoreError::NoIndex(field.to_string())),
        }
    }

    /// Borrow the index on `field` (the primary `_id` index is always
    /// present).
    pub fn get(&self, field: &str) -> Option<&FieldIndex> {
        self.fields.get(field)
    }

    pub fn get_mut(&mut self, field: &str) -> Option<&mut FieldIndex> {
        self.fields.get_mut(field)
    }

    /// `true` when `field` is indexed (always `true` for `_id`).
    pub fn contains(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    /// Number of indexes (always `>= 1` — the primary).
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// All indexed field names, sorted (deterministic; includes `_id`).
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.fields.keys().cloned().collect();
        v.sort();
        v
    }

    /// Register one entry of `doc` in **every** index: the value of the
    /// indexed field, or `Null` when the field is missing (MongoDB
    /// convention — see module docs). `_id` is always present on a stored
    /// document, so the primary index always gets the real id.
    pub fn index_doc(&mut self, id: &str, doc: &Value) {
        // Names are cloned (not borrowed) so the `&mut self` below can take
        // the map while the loop runs — there are always few indexes.
        let fields: Vec<String> = self.fields.keys().cloned().collect();
        for field in &fields {
            let v = doc.get(field).cloned().unwrap_or(Value::Null);
            if let Some(idx) = self.fields.get_mut(field) {
                idx.insert(v, id.to_string());
            }
        }
    }

    /// Remove one entry of `doc` from **every** index (inverse of
    /// [`IndexSet::index_doc`]; missing field → the `Null` entry).
    pub fn deindex_doc(&mut self, id: &str, doc: &Value) {
        let fields: Vec<String> = self.fields.keys().cloned().collect();
        for field in &fields {
            let v = doc.get(field).cloned().unwrap_or(Value::Null);
            if let Some(idx) = self.fields.get_mut(field) {
                idx.remove(id, &v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound::*;

    fn idx() -> FieldIndex {
        FieldIndex::new("age")
    }

    fn v(a: i64) -> Value {
        Value::i64(a)
    }

    // -- ordering -------------------------------------------------------------

    #[test]
    fn total_order_across_types() {
        let mut ix = idx();
        ix.insert(Value::str("b"), "s-b".into());
        ix.insert(Value::i64(5), "i-5".into());
        ix.insert(Value::f64(-2.5), "f--2.5".into());
        ix.insert(Value::Null, "n".into());
        ix.insert(Value::bool(true), "t".into());
        ix.insert(Value::i64(-3), "i--3".into());
        ix.insert(Value::str("a"), "s-a".into());
        let got: Vec<&str> = ix.iter().map(|(_, id)| id).collect();
        assert_eq!(
            got,
            vec!["n", "t", "i--3", "f--2.5", "i-5", "s-a", "s-b"],
            "Null < Bool < Number < Str, numbers cross-numeric, strings byte-ordered"
        );
    }

    #[test]
    fn cross_numeric_equal_values_share_slice() {
        let mut ix = idx();
        ix.insert(Value::i64(1), "a".into());
        ix.insert(Value::f64(1.0), "b".into());
        ix.insert(Value::i64(1), "c".into()); // same i64, different doc
        assert_eq!(ix.len(), 3);
        assert_eq!(ix.ids_equal(&Value::i64(1)), vec!["a", "b", "c"]);
        assert_eq!(ix.ids_equal(&Value::f64(1.0)), vec!["a", "b", "c"]);
        assert_eq!(ix.distinct(), 1);
    }

    #[test]
    fn nan_is_total_and_orders_after_inf() {
        let mut ix = idx();
        ix.insert(Value::f64(f64::INFINITY), "inf".into());
        ix.insert(Value::f64(f64::NAN), "nan1".into());
        ix.insert(Value::f64(f64::NAN), "nan2".into());
        ix.insert(Value::f64(f64::NEG_INFINITY), "ninf".into());
        let got: Vec<&str> = ix.iter().map(|(_, id)| id).collect();
        assert_eq!(got, vec!["ninf", "inf", "nan1", "nan2"]);
        // NaN == NaN in the total order: both NaNs are one equality slice.
        assert_eq!(ix.ids_equal(&Value::f64(f64::NAN)), vec!["nan1", "nan2"]);
        assert!(ix.contains_value(&Value::f64(f64::NAN)));
    }

    // -- equality ---------------------------------------------------------------

    #[test]
    fn equality_missing_value_is_empty() {
        let mut ix = idx();
        ix.insert(v(1), "a".into());
        assert!(ix.ids_equal(&v(2)).is_empty());
        assert!(!ix.contains_value(&v(2)));
    }

    #[test]
    fn duplicates_keep_id_tiebreak() {
        let mut ix = idx();
        ix.insert(v(7), "z".into());
        ix.insert(v(7), "m".into());
        ix.insert(v(7), "a".into());
        assert_eq!(ix.ids_equal(&v(7)), vec!["a", "m", "z"]);
        assert_eq!(ix.distinct(), 1);
    }

    // -- range -------------------------------------------------------------------

    fn ages(ix: &mut FieldIndex, pairs: &[(i64, &str)]) {
        for (a, id) in pairs {
            ix.insert(v(*a), (*id).into());
        }
    }

    #[test]
    fn range_bounds_included_excluded() {
        let mut ix = idx();
        ages(&mut ix, &[(1, "a1"), (2, "a2"), (3, "a3"), (4, "a4"), (5, "a5"), (6, "a6")]);
        // [2, 5)
        assert_eq!(ix.ids_range(Included(&v(2)), Excluded(&v(5))), vec!["a2", "a3", "a4"]);
        // (1, 5]
        assert_eq!(ix.ids_range(Excluded(&v(1)), Included(&v(5))), vec!["a2", "a3", "a4", "a5"]);
        // unbounded both ends = everything, in index order
        assert_eq!(
            ix.ids_range(Unbounded, Unbounded),
            vec!["a1", "a2", "a3", "a4", "a5", "a6"]
        );
        // [1, 1) is empty
        assert!(ix.ids_range(Included(&v(1)), Excluded(&v(1))).is_empty());
        // [2, 2] single hit
        assert_eq!(ix.ids_range(Included(&v(2)), Included(&v(2))), vec!["a2"]);
        // out-of-window
        assert!(ix.ids_range(Included(&v(9)), Included(&v(10))).is_empty());
    }

    #[test]
    fn range_with_duplicates_orders_by_id() {
        let mut ix = idx();
        ix.insert(v(3), "c".into());
        ix.insert(v(3), "a".into());
        ix.insert(v(2), "b".into());
        assert_eq!(ix.ids_range(Included(&v(2)), Included(&v(3))), vec!["b", "a", "c"]);
    }

    // -- remove --------------------------------------------------------------------

    #[test]
    fn remove_finds_and_removes() {
        let mut ix = idx();
        ix.insert(v(1), "a".into());
        ix.insert(v(2), "b".into());
        ix.insert(v(1), "c".into());
        assert!(ix.remove("a", &v(1)));
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.ids_equal(&v(1)), vec!["c"]);
        assert!(!ix.remove("a", &v(1)), "second remove of same id fails");
        assert!(!ix.remove("nope", &v(2)));
        assert!(ix.remove("b", &v(2)));
        assert_eq!(ix.len(), 1);
        assert!(ix.ids_range(Unbounded, Unbounded) == vec!["c"]);
    }

    #[test]
    fn remove_cross_numeric_uses_total_order() {
        let mut ix = idx();
        ix.insert(Value::f64(2.0), "f".into());
        ix.insert(Value::i64(2), "i".into());
        // 2 == 2.0 in the total order: removing by i64(2) must find i64 doc "i"
        assert!(ix.remove("i", &Value::i64(2)));
        assert_eq!(ix.ids_equal(&Value::i64(2)), vec!["f"]);
        assert!(ix.remove("f", &Value::i64(2))); // still found via 2 == 2.0
        assert!(ix.is_empty());
    }

    // -- load / stats ---------------------------------------------------------------

    #[test]
    fn load_sorts_deterministically() {
        let mut ix = idx();
        ix.load(vec![
            ("d".into(), v(9)),
            ("b".into(), v(9)),
            ("a".into(), v(1)),
            ("c".into(), v(5)),
        ]);
        let got: Vec<&str> = ix.iter().map(|(_, id)| id).collect();
        // by value: a=1, c=5, b=9, d=9 (tie on 9 broken by id)
        assert_eq!(got, vec!["a", "c", "b", "d"], "input order must not matter");
        assert_eq!(ix.len(), 4);
        assert_eq!(ix.distinct(), 3);
    }

    #[test]
    fn memory_size_grows() {
        let mut ix = idx();
        let base = ix.memory_size();
        assert!(base > 0);
        for i in 0..100 {
            ix.insert(Value::str(format!("payload-{i:04}-xxxxxxxxxxxxxxxx")), format!("id-{i}").into());
        }
        assert!(ix.memory_size() > base + 100 * 30, "estimate must reflect id + value heap");
    }

    // -- IndexSet ----------------------------------------------------------------------

    #[test]
    fn indexset_always_has_primary() {
        let s = IndexSet::new();
        assert!(s.contains(ID_KEY));
        assert_eq!(s.len(), 1);
        assert_eq!(s.names(), vec![ID_KEY.to_string()]);
        assert!(s.get(ID_KEY).unwrap().is_empty());
    }

    #[test]
    fn indexset_doc_entries_and_null_for_missing() {
        let mut s = IndexSet::new();
        s.insert_index("age".into(), FieldIndex::new("age"));
        let d1 = Value::Object(vec![
            (ID_KEY.into(), Value::str("d1")),
            ("age".into(), v(30)),
        ]);
        let d2 = Value::Object(vec![(ID_KEY.into(), Value::str("d2"))]); // no "age"
        let d3 = Value::Object(vec![
            (ID_KEY.into(), Value::str("d3")),
            ("age".into(), Value::Null), // explicit null
        ]);
        for d in [&d1, &d2, &d3] {
            let id = d.get(ID_KEY).unwrap().as_str().unwrap();
            s.index_doc(id, d);
        }
        let age = s.get("age").unwrap();
        assert_eq!(age.len(), 3);
        assert_eq!(age.ids_equal(&v(30)), vec!["d1"]);
        // missing and explicit null both land in the Null slice (MongoDB rule)
        assert_eq!(age.ids_equal(&Value::Null), vec!["d2", "d3"]);
        assert_eq!(s.get(ID_KEY).unwrap().len(), 3, "primary gets every doc");
    }

    #[test]
    fn indexset_deindex_removes_every_entry() {
        let mut s = IndexSet::new();
        s.insert_index("a".into(), FieldIndex::new("a"));
        let d = Value::Object(vec![
            (ID_KEY.into(), Value::str("x")),
            ("a".into(), v(42)),
        ]);
        s.index_doc("x", &d);
        assert_eq!(s.get("a").unwrap().len(), 1);
        s.deindex_doc("x", &d);
        assert!(s.get("a").unwrap().is_empty());
        assert!(s.get(ID_KEY).unwrap().is_empty());
        // deindexing a doc that was never indexed is a harmless no-op
        s.deindex_doc("ghost", &d);
        assert!(s.get("a").unwrap().is_empty());
    }

    #[test]
    fn indexset_drop_rules() {
        let mut s = IndexSet::new();
        assert_eq!(s.drop(ID_KEY), Err(StoreError::PrimaryIndex));
        assert_eq!(s.drop("nope"), Err(StoreError::NoIndex("nope".into())));
        s.insert_index("f".into(), FieldIndex::new("f"));
        assert_eq!(s.drop("f"), Ok(()));
        assert!(!s.contains("f"));
        assert!(s.contains(ID_KEY), "primary survives");
        assert_eq!(s.names(), vec![ID_KEY.to_string()]);
    }
}
