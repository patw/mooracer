//! `Query` — the lazy Mongo-style query builder over a [`Collection`].
//!
//! Design notes (perf posture, see spec "Query API" / "Performance posture"):
//!
//! - A [`Query<'c>`] **borrows** the collection and **owns** its `Value`
//!   filter. Nothing is scanned until a terminal (`.to_list()`, `.first()`,
//!   `.count()`) runs. This is the "evaluated lazily" contract from the spec;
//!   the later `.sort`/`.skip`/`.limit` subtask records intent on the same
//!   struct and the terminals still perform the single scan.
//! - The filter is a `Value` **object** in Mongo syntax. `{}` (the empty
//!   object) matches every document; a *non-object* filter is malformed and
//!   matches nothing (MongoDB filters are always objects — this is defensive,
//!   not a panic path).
//! - **Matching semantics** (comparison + set subtasks). `doc_matches`
//!   evaluates a filter as an implicit `$and` over its top-level keys. A
//!   condition that is an **operator object** (a non-empty object whose
//!   keys *all* start with `$`) dispatches to the comparison operators
//!   `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, the set operators
//!   `$in`, `$nin`, and the element operator `$exists`; several operators
//!   on one field are **AND-ed** (that is how range combos read:
//!   `{"age": {"$gte": 25, "$lt": 40}}`). `$exists` is *presence-only*:
//!   `{f: {$exists: true}}` matches a field whose key exists (an explicit
//!   `null` counts as present), `{f: {$exists: false}}` matches a missing
//!   field; its operand must be a boolean (anything else is malformed and
//!   matches nothing).
//!   Any other condition is a **direct value** (implicit `$eq`): the
//!   document's field must equal it under the engine's `Value` equality
//!   (exact cross-numeric, total NaN, canonical objects). Values compare
//!   with the engine **total order** (`Value::Ord`). Missing-field rules
//!   (MongoDB): a comparison operator requires the field to be **present**;
//!   `$ne` matches a missing field unless its operand is `null`; `$eq`
//!   matches a missing field only when its operand is `null`; `$in` is an
//!   **OR of `$eq` over its array operand** (so a missing field matches it
//!   only when the list contains `null`) and `$nin` is its exact complement.
//!   A non-array `$in`/`$nin` operand matches nothing (defensive). An
//!   unknown `$` operator matches nothing (defensive — malformed filters
//!   never panic). `$exists` never narrows an index scan (see the index-
//!   driven-scans note): it is always verified on the candidates. Array-
//!   element matching is `$elemMatch`'s job: `{"f": {$elemMatch:
//!   {<criteria>}}}` matches when at least one element of the array field
//!   satisfies `criteria` (a direct value → element equality, an operator
//!   object → element-level operators, a sub-document → the full filter on
//!   each element); it is field-level only and never drives an index scan.
//!   Direct values and `$in`/`$nin` stay exact whole-value equality (no array
//!   containment).
//! - **Logical operators**. `$and`/`$or` are **top-level-only**: their
//!   operand is an **array of sub-filters** (filter objects), each evaluated
//!   with the full filter semantics — `$and` requires all (an empty list is
//!   vacuous truth: matches everything), `$or` requires at least one (an
//!   empty list matches nothing). Because elements are sub-filters, the
//!   operators **nest** (`{$or: [{$and: [...]}]}`) and every operator works
//!   inside them. `$not` is **field-level**: `{"f": {$not: {<operator
//!   expression>}}}` is the negation of that whole expression (all its
//!   operators AND-ed, presence rules included), so `{$not: {$gt: 5}}`
//!   matches "f ≤ 5 *or f missing*". A non-array `$and`/`$or` operand, a
//!   non-object `$and` element, a non-operator-object `$not` operand, a
//!   field-level `$and`/`$or`, or a top-level `$not` is malformed and matches
//!   nothing (defensive — malformed filters never panic).
//! - **Index-driven scans**: when the first indexable condition in filter
//!   entry order is on an **indexed** field and carries an indexable
//!   candidate set (a bound from `$eq`/`$gt`/`$gte`/`$lt`/`$lte`, a direct
//!   value, or an `$in` list), the terminal fetches the candidate ids from
//!   that field's index (`ids_range` — two binary searches per range, one
//!   id-vector allocation) and **verifies every candidate against the full
//!   filter**. The verification keeps the index-driven and full-scan
//!   results *identical*; the index only narrows candidates. Entry order
//!   walks top-level keys first, then — because a conjunct narrows — the
//!   elements of a top-level `$and` (looked through recursively). An `$in`
//!   drives as the **union of its list's point ranges**, walked in
//!   total-order ascending so the result comes back in index order; an
//!   empty `$in` list is the empty candidate set (no scan at all). A
//!   condition carrying only `$ne`/`$nin`/`$not` never drives a scan (it
//!   would return almost everything — a plain scan is just as good), a
//!   top-level `$or`/`$not` never drives (an `$or` is a union no single
//!   condition's candidates contain), and a condition containing an unknown
//!   operator — or an `$in` with a non-array operand — falls back to the
//!   scan (which verifies it to false for every document).
//! - **Result order / pipeline**: `.sort(field, desc)`, `.skip(n)`,
//!   `.limit(m)` are the Mongo pipeline — **filter → sort → skip → limit**,
//!   applied identically by every terminal (`.to_list()`, `.first()`,
//!   `.count()`). `limit(0)` means *no limit* (the Mongo cursor convention).
//!   Without a sort the order is the underlying scan order (storage order
//!   for a full scan, index order for an index-driven scan: field value
//!   ascending per the total order, ties by `_id`), with skip/limit applied
//!   in that order. With a sort, the field value orders by the **total
//!   order** and ties break by `_id`; `descending = true` reverses the
//!   **whole** (value, `_id`) order. A **missing** sort field sorts like
//!   `Null` (the index convention) — first in ascending, last in descending.
//!   When the **sort field is indexed**, the sort streams the index itself
//!   in order (a double-ended walk of a contiguous array: zero allocation,
//!   zero sort step) and every entry is verified against the full filter, so
//!   `skip`/`limit` stop the walk after the last returned document —
//!   `sort + limit` never materializes more than `skip + limit` documents.
//!   With an **unindexed** sort field the matches are collected (clone cost
//!   unavoidable: an unsorted stream cannot be skipped) and sorted with a
//!   total-order comparator (field value, then `_id`), then sliced. `skip`
//!   is applied to the *filtered* stream (skipped docs are counted, just
//!   not returned), and `limit` caps the returned docs — so
//!   `.skip(1).limit(2)` on a 4-match query returns matches #2 and #3.
//! - No `unsafe`: index lookups are binary searches over a contiguous array
//!   and the filter pass is a flat enum tree; nothing measured justifies it.

use std::cmp::Ordering;
use std::ops::Bound;

use crate::agg::GroupQuery;
use crate::collection::Collection;
use crate::index::FieldIndex;
use crate::value::Value;

/// A lazy query over a collection, built by [`Collection::find`].
///
/// Cheap to hold (a `&Collection` plus an owned filter `Value`); the scan is
/// deferred to a terminal. Cloning is possible but pointless — build a fresh
/// `Query` per terminal.
pub struct Query<'c> {
    col: &'c Collection,
    filter: Value,
    sort: Option<String>,
    desc: bool,
    skip: usize,
    limit: usize,
}

impl<'c> Query<'c> {
    /// Build a query (called by [`Collection::find`]).
    pub(crate) fn new(col: &'c Collection, filter: Value) -> Self {
        Query {
            col,
            filter,
            sort: None,
            desc: false,
            skip: 0,
            limit: 0, // 0 = no limit (Mongo convention)
        }
    }

    /// The filter this query applies (Mongo object syntax).
    pub fn filter(&self) -> &Value {
        &self.filter
    }

    // -- pipeline (sort / skip / limit) ---------------------------------------

    /// Sort the results by `field`'s value in the engine total order; ties
    /// break by `_id`. `desc` reverses the whole (value, `_id`) order. A
    /// missing field sorts like `Null`. Replaces any earlier sort (single
    /// field, as in the spec's `.sort(field, descending)`).
    pub fn sort(mut self, field: impl Into<String>, desc: bool) -> Self {
        self.sort = Some(field.into());
        self.desc = desc;
        self
    }

    /// Skip the first `n` documents of the (sorted) result stream — applied
    /// *after* the sort, in pipeline order filter → sort → skip → limit.
    pub fn skip(mut self, n: usize) -> Self {
        self.skip = n;
        self
    }

    /// Return at most `m` documents of the (sorted) result stream. `0` means
    /// **no limit** (the Mongo cursor convention).
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// The sort field (`None` = unsorted: scan order, storage or index).
    pub fn sort_field(&self) -> Option<&str> {
        self.sort.as_deref()
    }

    /// `true` when the query sorts in descending order.
    pub fn sort_descending(&self) -> bool {
        self.desc
    }

    /// The configured skip count (getter; the builder is `skip(n)`).
    pub fn skip_count(&self) -> usize {
        self.skip
    }

    /// The configured limit (`0` = no limit).
    pub fn limit_count(&self) -> usize {
        self.limit
    }

    // -- terminals -----------------------------------------------------------

    /// Evaluate: all documents through the full pipeline (filter → sort →
    /// skip → limit), as owned clones, in the query's result order (storage
    /// order for an unsorted full scan, index order for an index-driven or
    /// sorted-on-indexed-field query, sorted order otherwise).
    pub fn to_list(self) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        self.for_each_pipelined(|doc| {
            out.push(doc.clone());
            true
        });
        out
    }

    /// Evaluate: the first document of the pipelined result stream, or
    /// `None`. Stops at the first returned document — no full scan.
    pub fn first(self) -> Option<Value> {
        let mut hit: Option<Value> = None;
        self.for_each_pipelined(|doc| {
            hit = Some(doc.clone());
            false // stop: the first document is all this terminal wants
        });
        hit
    }

    /// Evaluate: the number of documents the pipeline would return (filter →
    /// sort → skip → limit), counting only; clones nothing and stops as
    /// soon as `limit` documents have been counted.
    pub fn count(self) -> usize {
        let mut n = 0usize;
        self.for_each_pipelined(|_| {
            n += 1;
            true
        });
        n
    }

    /// Group this query's pipelined document stream by the value of
    /// `field` (a missing field groups under `Null`) and hand the
    /// [`GroupQuery`] to an aggregation terminal
    /// ([`GroupQuery::agg`]). The query's own sort/skip/limit apply to the
    /// *documents* (and hence define first/last/collect order within a
    /// group); the grouped result gets its own
    /// [`GroupQuery::sort`]/[`GroupQuery::limit`].
    pub fn group(self, field: impl Into<String>) -> GroupQuery<'c> {
        GroupQuery::new(self, field.into())
    }

    // -- pipeline driver -----------------------------------------------------

    /// Run the query's full pipeline — filter, then optional sort, then
    /// `skip`, then `limit` — invoking `f` on each **returned** document
    /// until `f` returns `false` (stop) or the pipeline is exhausted. The
    /// walk also stops as soon as `limit` documents have been returned
    /// (or `skip + limit` in scan order), so `sort + limit` on an indexed
    /// sort field never touches more of the index than it must.
    ///
    /// `pub(crate)`: the aggregation stage ([`crate::agg::GroupQuery`])
    /// groups the *same* pipelined stream.
    pub(crate) fn for_each_pipelined(&self, mut f: impl FnMut(&Value) -> bool) {
        if let Some(field) = self.sort_field() {
            if let Some(idx) = self.col.index(field) {
                // Fast path: stream the sort field's index in (reverse)
                // order — a double-ended walk of a contiguous array, zero
                // allocation, zero sort step — verifying each entry against
                // the full filter. `seen` counts every doc that matched the
                // filter (skipped ones included); the walk stops after the
                // `limit`-th returned doc.
                let skip = self.skip;
                let limit = self.limit;
                let mut seen = 0usize;
                let mut returned = 0usize;
                let mut emit = |id: &str| -> bool {
                    if let Some(doc) = self.col.get(id)
                        && doc_matches(doc, &self.filter)
                    {
                        seen += 1;
                        if seen > skip {
                            if !f(doc) {
                                return true; // the caller asked to stop
                            }
                            returned += 1;
                            if limit > 0 && returned >= limit {
                                return true;
                            }
                        }
                    }
                    false
                };
                let entries = idx.iter();
                for (_, id) in if self.desc {
                    Box::new(entries.rev()) as Box<dyn Iterator<Item = (&Value, &str)>>
                } else {
                    Box::new(entries) as Box<dyn Iterator<Item = (&Value, &str)>>
                } {
                    if emit(id) {
                        return;
                    }
                }
                return;
            }
            // Unindexed sort: collect the matches (clones unavoidable — an
            // unsorted stream cannot be skipped), sort by (field value,
            // `_id`) in the total order, reverse for descending, slice off
            // skip/limit.
            let docs = self.sorted_matches(field, self.desc);
            let start = self.skip.min(docs.len());
            let end = start
                .saturating_add(if self.limit == 0 {
                    usize::MAX
                } else {
                    self.limit
                })
                .min(docs.len());
            for doc in &docs[start..end] {
                if !f(doc) {
                    break;
                }
            }
            return;
        }
        // No sort: scan in plan order, apply skip/limit in stream order
        // (early stop once `limit` docs are out).
        let skip = self.skip;
        let limit = self.limit;
        let mut seen = 0usize;
        let mut returned = 0usize;
        let mut handle = |doc: &Value| -> bool {
            seen += 1;
            if seen > skip {
                if !f(doc) {
                    return true; // the caller asked to stop
                }
                returned += 1;
                if limit > 0 && returned >= limit {
                    return true;
                }
            }
            false
        };
        match self.plan() {
            Plan::Scan => {
                for doc in self.col.iter() {
                    if doc_matches(doc, &self.filter) && handle(doc) {
                        return;
                    }
                }
            }
            Plan::Index { idx, lo, hi } => {
                for id in idx.ids_range(bref(&lo), bref(&hi)) {
                    if let Some(doc) = self.col.get(id)
                        && doc_matches(doc, &self.filter)
                        && handle(doc)
                    {
                        return;
                    }
                }
            }
            Plan::In { idx, values } => {
                for v in values.iter() {
                    for id in idx.ids_range(Bound::Included(v), Bound::Included(v)) {
                        if let Some(doc) = self.col.get(id)
                            && doc_matches(doc, &self.filter)
                            && handle(doc)
                        {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// The matches sorted by (field value, `_id`) in the engine total order
    /// (`desc` reverses the whole order). The comparator is a *total* order
    /// (`_id` is unique), so an unstable sort is correct and faster.
    fn sorted_matches(&self, field: &str, desc: bool) -> Vec<Value> {
        let mut docs = self
            .col
            .iter()
            .filter(|d| doc_matches(d, &self.filter))
            .cloned()
            .collect::<Vec<Value>>();
        docs.sort_unstable_by(|a, b| {
            let av = a.get(field).unwrap_or(&Value::Null);
            let bv = b.get(field).unwrap_or(&Value::Null);
            av.cmp(bv).then_with(|| match (a.get("_id"), b.get("_id")) {
                (Some(x), Some(y)) => x.cmp(y),
                _ => Ordering::Equal,
            })
        });
        if desc {
            docs.reverse();
        }
        docs
    }

    /// Choose the scan strategy (cheap: one pass over the filter's top-level
    /// keys, at most a couple of bound clones). See the module docs for the
    /// rules; in short — the first indexable condition in entry order
    /// (top-level keys, then the elements of a top-level `$and`, looked
    /// through recursively) drives the query; everything else is verified
    /// on the candidate documents.
    fn plan(&self) -> Plan<'_> {
        let Some(entries) = self.filter.as_object() else {
            return Plan::Scan; // malformed filter: scan, matches nothing
        };
        plan_from_entries(self.col, entries).unwrap_or(Plan::Scan)
    }
}

// ---------------------------------------------------------------------------
// Scan plan
// ---------------------------------------------------------------------------

/// The terminal's scan strategy (see [`Query::plan`]).
enum Plan<'a> {
    /// Verify every stored document (storage order).
    Scan,
    /// Candidate ids from the field index, each verified against the full
    /// filter (index order: value ascending per the total order, ties by
    /// `_id`).
    Index {
        idx: &'a FieldIndex,
        lo: Bound<Value>,
        hi: Bound<Value>,
    },
    /// Candidate ids from the union of the point ranges of an `$in` list
    /// (distinct values, total-order ascending — walking them in order
    /// walks the index in order), each verified against the full filter.
    In {
        idx: &'a FieldIndex,
        values: Vec<Value>,
    },
}

/// `Bound<Value>` → `Bound<&Value>` for the index lookup (which takes
/// borrowed bounds). Cheap: no clone, no allocation.
fn bref(b: &Bound<Value>) -> Bound<&Value> {
    match b {
        Bound::Included(v) => Bound::Included(v),
        Bound::Excluded(v) => Bound::Excluded(v),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// The candidate set derivable from one condition (see
/// [`index_plan_for`]).
#[derive(Debug, PartialEq)]
enum IndexPlan {
    /// One contiguous value range (a direct value, or the intersection of
    /// the bound operators `$eq`/`$gt`/`$gte`/`$lt`/`$lte`).
    Range { lo: Bound<Value>, hi: Bound<Value> },
    /// The union of the point ranges of an `$in` list: the list's values,
    /// deduped and total-order ascending, so fetching the point ranges in
    /// order walks the index in order.
    Points { values: Vec<Value> },
}

/// Find the first condition in a filter-entry list (entry order) that can
/// drive an index scan: a field condition on an indexed field, or — for a
/// top-level `$and` — the entries of its sub-filter elements, looked
/// through **recursively and in order**. A conjunct narrows the result, so
/// any indexable condition inside it is a sound driver; every candidate is
/// re-verified against the full filter, so index-driven and scan results
/// stay identical. Top-level `$or`/`$not`/unknown-`$` keys never drive: an
/// `$or` is a *union*, and no single condition's candidate set contains it
/// (a scan verifies it). A malformed `$and` operand (non-array) or element
/// (non-object) simply cannot drive — the scan handles the matching (and
/// matches nothing, as [`doc_matches`] defines).
fn plan_from_entries<'a>(col: &'a Collection, entries: &'a [(String, Value)]) -> Option<Plan<'a>> {
    for (field, cond) in entries {
        let key = field.as_str();
        if key == "$and" {
            if let Some(list) = cond.as_array() {
                for elem in list {
                    if let Some(sub) = elem.as_object()
                        && let Some(p) = plan_from_entries(col, sub)
                    {
                        return Some(p);
                    }
                }
            }
            continue;
        }
        if key.starts_with('$') {
            continue; // $or / $not / unknown top-level operator: never a driver
        }
        let Some(idx) = col.index(key) else {
            continue;
        };
        let Some(plan) = index_plan_for(cond) else {
            continue; // not indexable: try the next condition
        };
        return Some(match plan {
            IndexPlan::Range { lo, hi } => Plan::Index { idx, lo, hi },
            IndexPlan::Points { values } => Plan::In { idx, values },
        });
    }
    None
}

/// Derive an index-driven candidate plan from one condition, or `None`
/// when the condition cannot drive an index scan:
///
/// - a **direct value** (including a nested document) is an implicit `$eq`
///   → the point range `[v, v]`;
/// - an **operator object with an `$in`** (array operand) → the union of
///   its list's point ranges, deduped and total-order ascending. An
///   **empty list** is the empty candidate set (the query matches nothing
///   and the scan short-circuits). Other operators in the same condition do
///   not narrow the point set — the candidates are re-verified against the
///   full filter anyway. A **non-array `$in` operand** yields `None`
///   (the condition matches nothing; the plain scan verifies it to false).
///   If several `$in` operators are listed, the first one drives and the
///   rest verify;
/// - an **operator object without `$in`** → the intersection of its bound
///   operators: `$eq`/`$gte`/`$lte` give `Included` bounds, `$gt`/`$lt`
///   give `Excluded` ones, several bounds on the same side tighten to the
///   stricter, and `$ne`/`$nin`/`$not` contribute no bound (verification
///   only). A condition with no bound operator at all (a bare
///   `$ne`/`$nin`/`$not`) yields `None` — it would return almost every
///   document, so the plain scan is just as good.
/// - an **unknown operator** yields `None`: the scan verifies it to false
///   for every document (the result is empty either way), and no candidate
///   narrowing is possible.
fn index_plan_for(cond: &Value) -> Option<IndexPlan> {
    let to_bound = |(v, inclusive): (Value, bool)| {
        if inclusive {
            Bound::Included(v)
        } else {
            Bound::Excluded(v)
        }
    };
    match cond {
        Value::Object(entries) => {
            if !is_operator_object(entries) {
                return Some(IndexPlan::Range {
                    lo: Bound::Included(cond.clone()),
                    hi: Bound::Included(cond.clone()),
                });
            }
            let mut lo: Option<(Value, bool)> = None; // (value, inclusive)
            let mut hi: Option<(Value, bool)> = None;
            let mut in_values: Option<Vec<Value>> = None;
            for (op, arg) in entries {
                match op.as_str() {
                    "$eq" => {
                        lo = tighten_lower(lo, (arg.clone(), true));
                        hi = tighten_upper(hi, (arg.clone(), true));
                    }
                    "$gt" => lo = tighten_lower(lo, (arg.clone(), false)),
                    "$gte" => lo = tighten_lower(lo, (arg.clone(), true)),
                    "$lt" => hi = tighten_upper(hi, (arg.clone(), false)),
                    "$lte" => hi = tighten_upper(hi, (arg.clone(), true)),
                    "$ne" => {}  // verified per candidate; contributes no bound
                    "$nin" => {} // complement of $in: verified per candidate, never a driver
                    "$not" => {} // negated expression: verified per candidate, contributes no bound
                    // `$exists` cannot drive: missing and explicit-null docs share
                    // the index's `Null` slot, so the index can't split presence
                    // from absence — verification on the (full) scan is the only
                    // correct path.
                    "$exists" => {} // presence check: verified per candidate, no bound
                    // `$elemMatch` cannot drive: a field index stores the whole
                    // array as a single value (elements are not indexed), so the
                    // index cannot split candidates by element — verify on the scan.
                    "$elemMatch" => {} // element match: verified per candidate, no bound
                    "$in" => match arg.as_array() {
                        Some(list) => {
                            // the first $in drives; later ones verify
                            if in_values.is_none() {
                                in_values = Some(list.clone());
                            }
                        }
                        None => return None, // non-array operand: matches nothing
                    },
                    _ => return None, // unknown operator: verify-on-scan
                }
            }
            if let Some(list) = in_values {
                return Some(IndexPlan::Points {
                    values: distinct_sorted(list),
                });
            }
            match (lo, hi) {
                (Some(l), Some(h)) => Some(IndexPlan::Range {
                    lo: to_bound(l),
                    hi: to_bound(h),
                }),
                (Some(l), None) => Some(IndexPlan::Range {
                    lo: to_bound(l),
                    hi: Bound::Unbounded,
                }),
                (None, Some(h)) => Some(IndexPlan::Range {
                    lo: Bound::Unbounded,
                    hi: to_bound(h),
                }),
                (None, None) => None, // bare $ne/$nin: not indexable
            }
        }
        _ => Some(IndexPlan::Range {
            lo: Bound::Included(cond.clone()),
            hi: Bound::Included(cond.clone()),
        }),
    }
}

/// Dedupe (engine total order) and sort ascending — the `$in` candidate
/// set's walk order, so its point ranges are fetched in index order.
fn distinct_sorted(mut values: Vec<Value>) -> Vec<Value> {
    values.sort();
    values.dedup();
    values
}

/// Tighten a lower bound: keep the stricter of the two. `inclusive` marks
/// `$gte` (Included) vs `$gt` (Excluded). Equal values: the exclusive bound
/// is stricter.
fn tighten_lower(cur: Option<(Value, bool)>, new: (Value, bool)) -> Option<(Value, bool)> {
    match cur {
        None => Some(new),
        Some((cv, cinc)) => {
            let (nv, ninc) = new;
            match cv.cmp(&nv) {
                Ordering::Greater => Some((cv, cinc)),
                Ordering::Less => Some((nv, ninc)),
                Ordering::Equal => {
                    if cinc && !ninc {
                        Some((nv, ninc))
                    } else {
                        Some((cv, cinc))
                    }
                }
            }
        }
    }
}

/// Tighten an upper bound: keep the stricter of the two (the *smaller*
/// upper bound). `inclusive` marks `$lte` vs `$lt`; equal values: exclusive
/// is stricter.
fn tighten_upper(cur: Option<(Value, bool)>, new: (Value, bool)) -> Option<(Value, bool)> {
    match cur {
        None => Some(new),
        Some((cv, cinc)) => {
            let (nv, ninc) = new;
            match cv.cmp(&nv) {
                Ordering::Less => Some((cv, cinc)),
                Ordering::Greater => Some((nv, ninc)),
                Ordering::Equal => {
                    if cinc && !ninc {
                        Some((nv, ninc))
                    } else {
                        Some((cv, cinc))
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matching (the single extension point the operator subtasks refine)
// ---------------------------------------------------------------------------

/// Does `doc` match `filter` (Mongo object syntax)? See the module docs for
/// the matching semantics. The filter is an implicit `$and` over its
/// top-level keys; `{}` (the empty object) matches every document, and a
/// malformed (non-object) filter matches nothing. Top-level `$and`/`$or` are
/// the logical operators (their operands are arrays of sub-filters); a
/// top-level `$not` is malformed (it is a field-level operator) and matches
/// nothing.
fn doc_matches(doc: &Value, filter: &Value) -> bool {
    let Some(entries) = filter.as_object() else {
        return false;
    };
    for (field, cond) in entries {
        let key = field.as_str();
        let ok = match key {
            "$and" => and_match(doc, cond),
            "$or" => or_match(doc, cond),
            // top-level `$not` is not a Mongo query shape (it is a
            // field-level operator): match nothing, never panic
            "$not" => false,
            _ => condition_match(doc, key, cond),
        };
        if !ok {
            return false;
        }
    }
    true
}

/// `$and: [expr1, expr2, ...]` (top-level only): **every** expression must
/// match. Each expression is a **sub-filter** (a filter object) evaluated
/// with the full filter semantics, so `$and` nests (`{$and: [{$or: [...]}]}`)
/// and every operator is available inside. An **empty list** is vacuous
/// truth and matches every document; a **non-array** operand, or any
/// **non-object element**, is malformed and matches nothing (defensive —
/// malformed filters never panic).
fn and_match(doc: &Value, operand: &Value) -> bool {
    let Some(list) = operand.as_array() else {
        return false;
    };
    list.iter().all(|elem| doc_matches(doc, elem))
}

/// `$or: [expr1, expr2, ...]` (top-level only): **at least one** expression
/// must match. Same sub-filter semantics as [`and_match`] (nesting, all
/// operators). An **empty list** matches nothing; a non-array operand
/// matches nothing; a non-object element simply fails its own disjunct
/// (each element is a sub-filter and a non-object sub-filter matches
/// nothing).
fn or_match(doc: &Value, operand: &Value) -> bool {
    let Some(list) = operand.as_array() else {
        return false;
    };
    list.iter().any(|elem| doc_matches(doc, elem))
}

/// Evaluate one top-level `(field, condition)` against a document: an
/// operator object dispatches to the comparison operators; any other
/// condition is a direct value (implicit `$eq`) — the field must be present
/// and equal under the engine's `Value` equality (exact cross-numeric, total
/// NaN, canonical objects).
fn condition_match(doc: &Value, field: &str, cond: &Value) -> bool {
    if let Value::Object(entries) = cond
        && is_operator_object(entries)
    {
        return op_match(doc, field, entries);
    }
    match doc.get(field) {
        Some(v) => *v == *cond,
        None => false,
    }
}

/// The field-level operators on one field: **every** listed operator must
/// hold (AND — that is how range combos work). Values compare with the
/// engine total order (`Value::Ord`). Missing-field rules (MongoDB):
///
/// - `$gt`/`$gte`/`$lt`/`$lte` require the field to be **present**;
/// - `$ne` matches a missing field unless its operand is `null`;
/// - `$eq` matches a missing field only when its operand is `null`;
/// - `$in` matches a missing field only when its list contains `null` (an
///   OR of `$eq` over the list, whole-value equality — no array
///   containment); `$nin` is the exact complement;
/// - `$exists` is presence-only and ignores the value: `$exists: true`
///   matches a present field (an explicit `null` counts as present),
///   `$exists: false` matches an absent field; a non-boolean operand is
///   malformed and matches nothing.
/// - `$not` negates its operand's **operator expression** (`{$not:
///   {$gt: 5}}` is "not (age > 5)") — so it *inverts* the presence rules:
///   a missing field satisfies `{$not: {$gt: 5}}` because the un-negated
///   `$gt` is false for an absent field. Its operand must be an operator
///   object (all-`$` keys, non-empty); anything else matches nothing.
///
/// `$and`/`$or` are top-level-only operators: seen at field level they make
/// the condition match nothing (defensive — malformed filters never panic).
/// A non-array `$in`/`$nin` operand, a non-operator-object `$not` operand,
/// like an unknown `$` operator, makes the whole condition match nothing.
fn op_match(doc: &Value, field: &str, entries: &[(String, Value)]) -> bool {
    op_match_present(doc.get(field), entries)
}

/// The shared per-value operator matcher behind [`op_match`]: every listed
/// operator must hold (AND), values compared with the engine total order,
/// missing-field rules as in [`op_match`]'s docs. [`op_match`] is
/// `op_match_present(doc.get(field), entries)`; `$elemMatch` reuses it with
/// `present` always `Some` (every array element is present).
fn op_match_present(present: Option<&Value>, entries: &[(String, Value)]) -> bool {
    for (op, arg) in entries {
        let ok = match op.as_str() {
            "$eq" => match present {
                Some(v) => *v == *arg,
                None => arg.is_null(),
            },
            "$ne" => match present {
                Some(v) => *v != *arg,
                None => !arg.is_null(),
            },
            "$gt" => present.is_some_and(|v| v.cmp(arg) == Ordering::Greater),
            "$gte" => present.is_some_and(|v| v.cmp(arg) != Ordering::Less),
            "$lt" => present.is_some_and(|v| v.cmp(arg) == Ordering::Less),
            "$lte" => present.is_some_and(|v| v.cmp(arg) != Ordering::Greater),
            "$in" => match arg.as_array() {
                Some(list) => in_list(present, list),
                None => false, // non-array operand: matches nothing
            },
            "$nin" => match arg.as_array() {
                Some(list) => !in_list(present, list),
                None => false, // non-array operand: matches nothing
            },
            // `{$exists: b}` is an *element* operator: it asks only whether the
            // field is present, never its value. Present means the key exists —
            // including an explicit `null` (so `{f: null}` satisfies
            // `{$exists: true}`); absent means the key is missing. A non-boolean
            // operand is malformed and matches nothing (defensive, no panic).
            "$exists" => match arg {
                Value::Bool(b) => *b == present.is_some(),
                _ => return false,
            },
            // `{$not: {<operator expression>}}`: the negation of the whole
            // operand expression (all its operators AND-ed, presence rules
            // included), so `$not` flips them. The operand must be an
            // operator object; a direct value / plain object / array / empty
            // object is malformed and matches nothing (MongoDB: "$not needs
            // a regex or an operator expression" — no error channel here).
            "$not" => match arg {
                Value::Object(es) if is_operator_object(es) => !op_match_present(present, es),
                _ => false,
            },
            // `{$elemMatch: {<criteria>}}` (array operator): matches a doc when
            // at least one element of the array field satisfies `criteria`.
            // `criteria` is classified: a direct value -> element equality,
            // an operator object -> element-level operators (presence rules
            // degenerate: an element is always present), a sub-document -> the
            // full filter evaluated against each element. See `elem_match`.
            "$elemMatch" => elem_match(present, arg),
            "$and" | "$or" => {
                // top-level-only operators at field level: malformed
                false
            }
            _ => return false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// The kind of a `$elemMatch` operand (classified once, then applied to
/// every element). See [`elem_match`] for the matching semantics.
enum ElemCriteria<'a> {
    /// A direct value (non-object, or an object not all-`$`): implicit `$eq`
    /// on each element — the element must equal the operand under the engine
    /// total order.
    Direct(&'a Value),
    /// An operator object (all keys start with `$`): the element (always
    /// present) must satisfy all its operators — `{$gt: 5}`, `{$gte: 1,
    /// $lt: 10}`, ...
    Op(&'a [(String, Value)]),
    /// A sub-document (a non-operator object): the full filter evaluated
    /// against each element (each element is a document).
    SubDoc(&'a Value),
}

fn elem_criteria<'a>(c: &'a Value) -> ElemCriteria<'a> {
    match c {
        Value::Object(e) if is_operator_object(e) => ElemCriteria::Op(e),
        Value::Object(_) => ElemCriteria::SubDoc(c),
        _ => ElemCriteria::Direct(c),
    }
}

/// `$elemMatch`: matches `doc` when **at least one** element of the array
/// field satisfies `criteria`. The field must be present **and** an array —
/// a missing field or a non-array field has no elements and matches nothing.
/// The `criteria` operand is classified once ([`elem_criteria`]):
///
/// - a **direct value** → element equality (the field's array holds an element
///   that equals the operand under the engine total order, e.g.
///   `{"sizes": {$elemMatch: "L"}}`);
/// - an **operator object** (all-`$` keys, non-empty) → the element must
///   satisfy all its operators, with the missing-field rules degenerated:
///   every element is present, so `$eq` is equality, `$ne` is inequality, and
///   the comparison operators are plain total-order comparisons (e.g.
///   `{"sizes": {$elemMatch: {$gt: 5}}}`);
/// - a **sub-document** (a non-operator object) → the full filter is
///   evaluated against each element, which is treated as a document (e.g.
///   `{"instock": {$elemMatch: {qty: {$gt: 5}, warehouse: "A"}}}`).
///
/// `$elemMatch` never narrows an index scan (a field index stores the whole
/// array as a single value; the elements are not indexed), so it is
/// verified on the (full) scan like `$exists`/`$ne`/`$nin`/`$not`.
fn elem_match(present: Option<&Value>, criteria: &Value) -> bool {
    let Some(elements) = present.and_then(Value::as_array) else {
        return false;
    };
    match elem_criteria(criteria) {
        ElemCriteria::Direct(c) => elements.contains(c),
        ElemCriteria::Op(entries) => elements.iter().any(|e| op_match_present(Some(e), entries)),
        ElemCriteria::SubDoc(c) => elements.iter().any(|e| doc_matches(e, c)),
    }
}

/// `$in` membership: the field's **whole value** equals some list element
/// (engine total order — cross-numeric, total NaN, canonical objects; no
/// array containment, so a stored array matches only a list element that is
/// exactly that array). Missing-field rule inherited from `$eq` (this is an
/// OR of `$eq` over the list): a **missing** field matches only when the
/// list contains `null`, so `{$in: [null]}` behaves like `{$eq: null}`
/// (explicit null *and* absence). `$nin` is the exact complement.
fn in_list(present: Option<&Value>, list: &[Value]) -> bool {
    match present {
        Some(v) => list.contains(v),
        None => list.iter().any(Value::is_null),
    }
}

/// `true` when `entries` is a non-empty object **every** key of which starts
/// with `$` — the MongoDB marker for an operator object (vs. a nested
/// document being matched literally). A mixed key set is not an operator
/// object and is left to direct-value matching (which matches nothing, since
/// no document field literally equals such an object).
fn is_operator_object(entries: &[(String, Value)]) -> bool {
    !entries.is_empty() && entries.iter().all(|(k, _)| k.starts_with('$'))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::Collection;

    fn col_with(pairs: &[(Value, Option<Value>)]) -> Collection {
        // pairs: (field value, _id or None = auto). Built as full docs.
        let mut c = Collection::new("t");
        for (val, id) in pairs {
            let mut entries: Vec<(String, Value)> = Vec::new();
            if let Some(id) = id {
                entries.push(("_id".to_string(), id.clone()));
            }
            if let Value::Object(es) = val {
                entries.extend(es.iter().map(|(k, v)| (k.clone(), v.clone())));
            } else {
                entries.push(("v".to_string(), val.clone()));
            }
            c.insert(Value::object_from(entries)).unwrap();
        }
        c
    }

    fn obj(pairs: &[(&str, Value)]) -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    /// The ids of a result list, sorted (storage order is unspecified).
    fn sorted_ids(docs: &[Value]) -> Vec<String> {
        let mut ids: Vec<String> = docs
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        ids.sort();
        ids
    }

    // -- {} = all -----------------------------------------------------------

    #[test]
    fn empty_filter_matches_everything_on_nonempty() {
        let c = col_with(&[
            (
                Value::object_from(vec![("k".into(), Value::i64(1))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("k".into(), Value::i64(2))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("k".into(), Value::i64(3))]),
                Some(Value::str("c")),
            ),
        ]);
        let all = Value::object();
        assert_eq!(c.find(all.clone()).to_list().len(), 3);
        assert_eq!(c.find(all.clone()).count(), 3);
        assert!(c.find(all.clone()).first().is_some());
        assert_eq!(c.count(all.clone()), 3);
        assert!(c.find_one(all.clone()).is_some());
        assert!(c.exists(all));
    }

    #[test]
    fn empty_filter_on_empty_collection() {
        let c = Collection::new("t");
        let all = Value::object();
        assert!(c.find(all.clone()).to_list().is_empty());
        assert_eq!(c.find(all.clone()).count(), 0);
        assert!(c.find(all).first().is_none());
        assert_eq!(c.count(Value::object()), 0);
        assert!(c.find_one(Value::object()).is_none());
        assert!(!c.exists(Value::object()));
    }

    #[test]
    fn find_to_list_returns_all_distinct_docs() {
        let c = col_with(&[
            (
                Value::object_from(vec![("name".into(), Value::str("bess"))]),
                Some(Value::str("bess")),
            ),
            (
                Value::object_from(vec![("name".into(), Value::str("moo"))]),
                Some(Value::str("moo")),
            ),
        ]);
        let got = sorted_ids(&c.find(Value::object()).to_list());
        assert_eq!(got, vec!["bess".to_string(), "moo".to_string()]);
    }

    // -- direct-value (implicit $eq) top-level matching ---------------------

    #[test]
    fn equality_filter_selects_matching_subset() {
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("young")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("mid")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("young2")),
            ),
        ]);
        let got = c.find(obj(&[("age", Value::i64(25))])).to_list();
        assert_eq!(
            sorted_ids(&got),
            vec!["young".to_string(), "young2".to_string()]
        );
    }

    #[test]
    fn cross_numeric_equality_uses_total_order() {
        // i64(1) and f64(1.0) are equal under the engine total order, so a
        // filter of f64(1.0) must match a doc storing i64(1) and vice versa.
        let c = col_with(&[
            (
                Value::object_from(vec![("n".into(), Value::i64(1))]),
                Some(Value::str("i")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::f64(1.0))]),
                Some(Value::str("f")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::i64(2))]),
                Some(Value::str("other")),
            ),
        ]);
        assert_eq!(
            sorted_ids(&c.find(obj(&[("n", Value::f64(1.0))])).to_list()),
            vec!["f".to_string(), "i".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(obj(&[("n", Value::i64(1))])).to_list()),
            vec!["f".to_string(), "i".to_string()]
        );
    }

    #[test]
    fn missing_field_does_not_match_direct_value() {
        let c = col_with(&[
            (
                Value::object_from(vec![("a".into(), Value::i64(1))]),
                Some(Value::str("has")),
            ),
            (Value::object_from(vec![]), Some(Value::str("nope"))),
        ]);
        assert_eq!(c.find(obj(&[("a", Value::i64(1))])).count(), 1);
        // filtering a field nobody has
        assert_eq!(c.find(obj(&[("zzz", Value::i64(9))])).count(), 0);
    }

    #[test]
    fn nested_object_condition_matches_literally() {
        // A condition that is a *nested* object (keys not all `$`) is a direct
        // document match — an **exact** subdocument equality (MongoDB
        // semantics), not a subset match: an extra field on the stored side
        // disqualifies the document.
        let c = col_with(&[
            // exact match: addr == {city: NYC}
            (
                Value::object_from(vec![("addr".into(), obj(&[("city", Value::str("NYC"))]))]),
                Some(Value::str("exact")),
            ),
            // superset: addr has an extra field -> NOT equal
            (
                Value::object_from(vec![(
                    "addr".into(),
                    obj(&[("city", Value::str("NYC")), ("zip", Value::str("10001"))]),
                )]),
                Some(Value::str("superset")),
            ),
            // different value
            (
                Value::object_from(vec![("addr".into(), obj(&[("city", Value::str("LA"))]))]),
                Some(Value::str("la")),
            ),
        ]);
        let got = c
            .find(obj(&[("addr", obj(&[("city", Value::str("NYC"))]))]))
            .to_list();
        assert_eq!(sorted_ids(&got), vec!["exact".to_string()]);
    }

    // -- comparison operators ($eq/$ne/$gt/$gte/$lt/$lte) -------------------

    #[test]
    fn comparison_operators_bound_the_field() {
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(29))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(41))]),
                Some(Value::str("d")),
            ),
        ]);
        let f = |pairs: &[(&str, Value)]| obj(&[("age", obj(pairs))]);
        assert_eq!(
            sorted_ids(&c.find(f(&[("$lt", Value::i64(25))])).to_list()),
            Vec::<String>::new()
        );
        assert_eq!(
            sorted_ids(&c.find(f(&[("$lte", Value::i64(25))])).to_list()),
            vec!["a".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gt", Value::i64(25))])).to_list()),
            vec!["b".to_string(), "c".to_string(), "d".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gte", Value::i64(40))])).to_list()),
            vec!["c".to_string(), "d".to_string()]
        );
        // contradictory bounds match nothing
        assert_eq!(
            c.find(f(&[("$gt", Value::i64(25)), ("$lt", Value::i64(25))]))
                .count(),
            0
        );
    }

    #[test]
    fn range_combos_and_tightening() {
        // ages: 10, 24, 25, 30, 33, 39, 40, 41
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(10))]),
                Some(Value::str("a10")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(24))]),
                Some(Value::str("a24")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a25")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("a30")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(33))]),
                Some(Value::str("a33")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(39))]),
                Some(Value::str("a39")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("a40")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(41))]),
                Some(Value::str("a41")),
            ),
        ]);
        let f = |pairs: &[(&str, Value)]| obj(&[("age", obj(pairs))]);
        // the classic range combo from the spec
        assert_eq!(
            sorted_ids(
                &c.find(f(&[("$gte", Value::i64(25)), ("$lt", Value::i64(40))]))
                    .to_list()
            ),
            vec![
                "a25".to_string(),
                "a30".to_string(),
                "a33".to_string(),
                "a39".to_string()
            ]
        );
        // same-value bounds: the exclusive one wins ($gt 30 + $gte 30 -> > 30)
        assert_eq!(
            sorted_ids(
                &c.find(f(&[("$gte", Value::i64(30)), ("$gt", Value::i64(30))]))
                    .to_list()
            ),
            vec![
                "a33".to_string(),
                "a39".to_string(),
                "a40".to_string(),
                "a41".to_string()
            ]
        );
        // different values: the stronger bound wins, order in the object irrelevant
        assert_eq!(
            sorted_ids(
                &c.find(f(&[("$gte", Value::i64(25)), ("$gt", Value::i64(30))]))
                    .to_list()
            ),
            vec![
                "a33".to_string(),
                "a39".to_string(),
                "a40".to_string(),
                "a41".to_string()
            ]
        );
        assert_eq!(
            sorted_ids(
                &c.find(f(&[("$lt", Value::i64(33)), ("$lte", Value::i64(39))]))
                    .to_list()
            ),
            vec![
                "a10".to_string(),
                "a24".to_string(),
                "a25".to_string(),
                "a30".to_string()
            ]
        );
        // $eq and $ne AND together
        assert_eq!(
            sorted_ids(
                &c.find(f(&[("$eq", Value::i64(30)), ("$ne", Value::i64(31))]))
                    .to_list()
            ),
            vec!["a30".to_string()]
        );
        assert_eq!(
            c.find(f(&[("$eq", Value::i64(30)), ("$ne", Value::i64(30))]))
                .count(),
            0
        );
        // one-sided open ranges
        assert_eq!(
            sorted_ids(&c.find(f(&[("$lte", Value::i64(25))])).to_list()),
            vec!["a10".to_string(), "a24".to_string(), "a25".to_string()]
        );
    }

    #[test]
    fn comparisons_are_exact_across_numeric_types() {
        let c = col_with(&[
            (
                Value::object_from(vec![("n".into(), Value::i64(6))]),
                Some(Value::str("i6")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::f64(5.5))]),
                Some(Value::str("f5.5")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::i64(5))]),
                Some(Value::str("i5")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::f64(6.0))]),
                Some(Value::str("f6.0")),
            ),
        ]);
        let f = |pairs: &[(&str, Value)]| obj(&[("n", obj(pairs))]);
        // an i64(5) bound against f64 values: 5.5 > 5, 6 > 5, 6.0 > 5;
        // i64(5) itself is not > 5
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gt", Value::i64(5))])).to_list()),
            vec!["f5.5".to_string(), "f6.0".to_string(), "i6".to_string()]
        );
        // f64(6.0) == i64(6) exactly: $gte catches both, $gt catches neither
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gte", Value::f64(6.0))])).to_list()),
            vec!["f6.0".to_string(), "i6".to_string()]
        );
        assert_eq!(c.find(f(&[("$gt", Value::f64(6.0))])).count(), 0);
        // explicit $eq is cross-numeric, like the implicit direct value
        assert_eq!(
            sorted_ids(&c.find(f(&[("$eq", Value::f64(6.0))])).to_list()),
            vec!["f6.0".to_string(), "i6".to_string()]
        );
    }

    #[test]
    fn missing_field_rules_for_operators() {
        let c = col_with(&[
            (
                Value::object_from(vec![("a".into(), Value::i64(1))]),
                Some(Value::str("one")),
            ),
            (
                Value::object_from(vec![("a".into(), Value::Null)]),
                Some(Value::str("null")),
            ),
            (Value::object_from(vec![]), Some(Value::str("absent"))),
        ]);
        let f = |pairs: &[(&str, Value)]| obj(&[("a", obj(pairs))]);
        // comparison operators require the field to be present
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gt", Value::i64(0))])).to_list()),
            vec!["one".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(f(&[("$lt", Value::i64(2))])).to_list()),
            vec!["null".to_string(), "one".to_string()]
        );
        // $ne matches a missing field unless the operand is null
        assert_eq!(
            sorted_ids(&c.find(f(&[("$ne", Value::i64(1))])).to_list()),
            vec!["absent".to_string(), "null".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(f(&[("$ne", Value::Null)])).to_list()),
            vec!["one".to_string()]
        );
        // $eq matches a missing field only for a null operand
        assert_eq!(
            sorted_ids(&c.find(f(&[("$eq", Value::Null)])).to_list()),
            vec!["absent".to_string(), "null".to_string()]
        );
        assert_eq!(c.find(f(&[("$eq", Value::i64(2))])).count(), 0);
        // null is the minimum rank: nothing is below it; everything present is >= it
        assert_eq!(c.find(f(&[("$lt", Value::Null)])).count(), 0);
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gte", Value::Null)])).to_list()),
            vec!["null".to_string(), "one".to_string()]
        );
    }

    #[test]
    fn comparisons_order_across_type_ranks() {
        let c = col_with(&[
            (Value::object_from(vec![]), Some(Value::str("no-v"))),
            (Value::str("b"), Some(Value::str("s-b"))),
            (Value::str("c"), Some(Value::str("s-c"))),
            (Value::bool(true), Some(Value::str("t"))),
            (Value::bool(false), Some(Value::str("f"))),
            (Value::i64(5), Some(Value::str("n5"))),
        ]);
        let f = |pairs: &[(&str, Value)]| obj(&[("v", obj(pairs))]);
        // strings compare in byte order
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gt", Value::str("b"))])).to_list()),
            vec!["s-c".to_string()]
        );
        // ranks: Null < Bool < Number < Str, so $gte true catches bool/number/str
        assert_eq!(
            sorted_ids(&c.find(f(&[("$gte", Value::bool(true))])).to_list()),
            vec![
                "n5".to_string(),
                "s-b".to_string(),
                "s-c".to_string(),
                "t".to_string()
            ]
        );
        // $lt 5 catches only the bools; the doc without a `v` field never matches
        assert_eq!(
            sorted_ids(&c.find(f(&[("$lt", Value::i64(5))])).to_list()),
            vec!["f".to_string(), "t".to_string()]
        );
    }

    #[test]
    fn unknown_operators_match_nothing() {
        let mut c = col_with(&[(
            Value::object_from(vec![("age".into(), Value::i64(30))]),
            Some(Value::str("x")),
        )]);
        let bogus = obj(&[("age", obj(&[("$bogus", Value::i64(1))]))]);
        assert_eq!(c.find(bogus.clone()).count(), 0, "scan path");
        c.create_index("age").unwrap();
        assert_eq!(c.find(bogus.clone()).count(), 0, "index-fallback path");
        // a never-real operator name is unknown: matches nothing on both
        // paths (set/logical/element/array operators have their own subtasks)
        assert_eq!(
            c.find(obj(&[("age", obj(&[("$fakeOp", Value::i64(1))]))]))
                .count(),
            0
        );
        // an unknown operator kills the condition even next to real ones
        assert_eq!(
            c.find(obj(&[(
                "age",
                obj(&[("$bogus", Value::i64(1)), ("$lt", Value::i64(40))])
            )]))
            .count(),
            0
        );
    }

    #[test]
    fn index_plan_for_derives_plans() {
        use std::ops::Bound::*;
        let i = |n: i64| Value::i64(n);
        let range = |lo: Bound<Value>, hi: Bound<Value>| IndexPlan::Range { lo, hi };
        let points = |values: Vec<Value>| IndexPlan::Points { values };
        let in_arr = |list: Vec<Value>| Value::array_from(list);
        // direct value -> point range
        assert_eq!(
            index_plan_for(&i(5)),
            Some(range(Included(i(5)), Included(i(5))))
        );
        // single operators
        assert_eq!(
            index_plan_for(&obj(&[("$gt", i(5))])),
            Some(range(Excluded(i(5)), Unbounded))
        );
        assert_eq!(
            index_plan_for(&obj(&[("$lte", i(5))])),
            Some(range(Unbounded, Included(i(5))))
        );
        // combos intersect
        assert_eq!(
            index_plan_for(&obj(&[("$gte", i(2)), ("$lt", i(5))])),
            Some(range(Included(i(2)), Excluded(i(5))))
        );
        // same-side bounds tighten: stronger value, exclusive wins ties
        assert_eq!(
            index_plan_for(&obj(&[("$gte", i(25)), ("$gt", i(30))])),
            Some(range(Excluded(i(30)), Unbounded))
        );
        assert_eq!(
            index_plan_for(&obj(&[("$gt", i(30)), ("$gte", i(25))])),
            Some(range(Excluded(i(30)), Unbounded))
        );
        assert_eq!(
            index_plan_for(&obj(&[("$gte", i(30)), ("$gt", i(30))])),
            Some(range(Excluded(i(30)), Unbounded))
        );
        assert_eq!(
            index_plan_for(&obj(&[("$lt", i(33)), ("$lte", i(39))])),
            Some(range(Unbounded, Excluded(i(33))))
        );
        // $eq is a point on both sides and intersects ranges
        assert_eq!(
            index_plan_for(&obj(&[("$eq", i(7))])),
            Some(range(Included(i(7)), Included(i(7))))
        );
        assert_eq!(
            index_plan_for(&obj(&[("$eq", i(5)), ("$lt", i(9))])),
            // the point intersected with the loose range stays the point
            Some(range(Included(i(5)), Included(i(5))))
        );
        // $in -> distinct, total-order ascending point set (list order and
        // duplicates don't leak into the plan)
        assert_eq!(
            index_plan_for(&obj(&[("$in", in_arr(vec![i(3), i(1), i(3)]))])),
            Some(points(vec![i(1), i(3)]))
        );
        // $in with bounds: the points drive, the bounds only verify
        assert_eq!(
            index_plan_for(&obj(&[("$in", in_arr(vec![i(1)])), ("$gt", i(5))])),
            Some(points(vec![i(1)]))
        );
        // empty $in -> the empty candidate set (result: nothing, no scan)
        assert_eq!(
            index_plan_for(&obj(&[("$in", in_arr(Vec::new()))])),
            Some(points(Vec::new()))
        );
        // non-array $in operand: the condition matches nothing -> plain scan
        assert_eq!(index_plan_for(&obj(&[("$in", i(5))])), None);
        // bare $ne / bare $nin: not indexable (would return almost everything)
        assert_eq!(index_plan_for(&obj(&[("$ne", i(5))])), None);
        assert_eq!(index_plan_for(&obj(&[("$nin", in_arr(vec![i(5)]))])), None);
        // $nin plus a bound: the bound drives, $nin only verifies
        assert_eq!(
            index_plan_for(&obj(&[("$nin", in_arr(vec![i(5)])), ("$gte", i(2))])),
            Some(range(Included(i(2)), Unbounded))
        );
        // unknown operators: not indexable
        assert_eq!(
            index_plan_for(&obj(&[("$ne", i(5)), ("$bogus", i(1))])),
            None
        );
        assert_eq!(
            index_plan_for(&obj(&[("$in", in_arr(vec![i(5)])), ("$bogus", i(1))])),
            None
        );
        // a nested direct object is a point range on the whole object
        let city = obj(&[("city", Value::str("NYC"))]);
        assert_eq!(
            index_plan_for(&city.clone()),
            Some(range(Included(city.clone()), Included(city)))
        );
    }

    // -- set operators ($in / $nin) ------------------------------------------

    #[test]
    fn in_matches_any_list_element() {
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(50))]),
                Some(Value::str("d")),
            ),
        ]);
        let f = |list: Vec<Value>| obj(&[("age", obj(&[("$in", Value::array_from(list))]))]);
        assert_eq!(
            sorted_ids(&c.find(f(vec![Value::i64(25), Value::i64(40)])).to_list()),
            vec!["a".to_string(), "c".to_string()]
        );
        // list order irrelevant, duplicates harmless
        assert_eq!(
            sorted_ids(
                &c.find(f(vec![Value::i64(40), Value::i64(25), Value::i64(40)]))
                    .to_list()
            ),
            vec!["a".to_string(), "c".to_string()]
        );
        // nothing in the list: no match
        assert_eq!(c.find(f(vec![Value::i64(99)])).count(), 0);
        // cross-numeric: f64(30.0) in the list matches the stored i64(30)
        assert_eq!(
            sorted_ids(&c.find(f(vec![Value::f64(30.0)])).to_list()),
            vec!["b".to_string()]
        );
        // $nin is the complement on the same set
        let g = |list: Vec<Value>| obj(&[("age", obj(&[("$nin", Value::array_from(list))]))]);
        assert_eq!(
            sorted_ids(&c.find(g(vec![Value::i64(25), Value::i64(40)])).to_list()),
            vec!["b".to_string(), "d".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(g(vec![Value::i64(99)])).to_list()),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn in_nin_missing_and_null_rules() {
        let c = col_with(&[
            (
                Value::object_from(vec![("a".into(), Value::i64(1))]),
                Some(Value::str("one")),
            ),
            (
                Value::object_from(vec![("a".into(), Value::Null)]),
                Some(Value::str("null")),
            ),
            (Value::object_from(vec![]), Some(Value::str("absent"))),
        ]);
        let f = |list: Vec<Value>| obj(&[("a", obj(&[("$in", Value::array_from(list))]))]);
        let g = |list: Vec<Value>| obj(&[("a", obj(&[("$nin", Value::array_from(list))]))]);
        // present value: plain whole-value membership
        assert_eq!(
            sorted_ids(&c.find(f(vec![Value::i64(1)])).to_list()),
            vec!["one".to_string()]
        );
        // missing field: matches $in only when the list contains null
        // (the $eq rule — OR over $eq)
        assert_eq!(
            sorted_ids(&c.find(f(vec![Value::Null])).to_list()),
            vec!["absent".to_string(), "null".to_string()]
        );
        assert_eq!(
            c.find(f(vec![Value::i64(1)])).count(),
            1,
            "missing never matches $in [1]"
        );
        assert_eq!(
            sorted_ids(&c.find(f(vec![Value::Null, Value::i64(1)])).to_list()),
            vec!["absent".to_string(), "null".to_string(), "one".to_string()]
        );
        // $nin is the exact complement, missing included
        assert_eq!(
            sorted_ids(&c.find(g(vec![Value::i64(1)])).to_list()),
            vec!["absent".to_string(), "null".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(g(vec![Value::Null])).to_list()),
            vec!["one".to_string()]
        );
        // empty list: $in matches nothing, $nin matches everything (incl. missing)
        assert_eq!(c.find(f(vec![])).count(), 0);
        assert_eq!(
            sorted_ids(&c.find(g(vec![])).to_list()),
            vec!["absent".to_string(), "null".to_string(), "one".to_string()]
        );
    }

    #[test]
    fn in_nin_non_array_operand_matches_nothing() {
        let c = col_with(&[(
            Value::object_from(vec![("a".into(), Value::i64(1))]),
            Some(Value::str("x")),
        )]);
        for bad in [
            Value::i64(1),
            Value::str("a"),
            Value::object(),
            Value::bool(true),
            Value::Null,
        ] {
            assert_eq!(
                c.find(obj(&[("a", obj(&[("$in", bad.clone())]))])).count(),
                0,
                "non-array $in operand matches nothing"
            );
            assert_eq!(
                c.find(obj(&[("a", obj(&[("$nin", bad.clone())]))])).count(),
                0,
                "malformed $nin also matches nothing (defensive, no panic)"
            );
        }
    }

    #[test]
    fn in_has_no_array_containment() {
        // A stored *array* is matched as a whole value (engine convention —
        // no array containment; element-level matching is $elemMatch's
        // territory).
        let c = col_with(&[
            (
                Value::object_from(vec![(
                    "tags".into(),
                    Value::array_from(vec![Value::str("moo"), Value::str("loud")]),
                )]),
                Some(Value::str("arr")),
            ),
            (
                Value::object_from(vec![("tags".into(), Value::str("moo"))]),
                Some(Value::str("plain")),
            ),
        ]);
        // $in ["moo"]: only the scalar doc; the array [moo, loud] is not == "moo"
        assert_eq!(
            c.find(obj(&[(
                "tags",
                obj(&[("$in", Value::array_from(vec![Value::str("moo")]))])
            )]))
            .count(),
            1
        );
        // the whole array IS a member when listed exactly
        let list = vec![Value::array_from(vec![
            Value::str("moo"),
            Value::str("loud"),
        ])];
        assert_eq!(
            sorted_ids(
                &c.find(obj(&[("tags", obj(&[("$in", Value::array_from(list))]))]))
                    .to_list()
            ),
            vec!["arr".to_string()]
        );
    }

    #[test]
    fn in_matches_object_list_members_exactly() {
        // object list members are exact subdocument equality (not subset)
        let c = col_with(&[
            (
                Value::object_from(vec![("addr".into(), obj(&[("city", Value::str("NYC"))]))]),
                Some(Value::str("nyc")),
            ),
            (
                Value::object_from(vec![("addr".into(), obj(&[("city", Value::str("LA"))]))]),
                Some(Value::str("la")),
            ),
            (
                Value::object_from(vec![(
                    "addr".into(),
                    obj(&[("city", Value::str("NYC")), ("zip", Value::str("10001"))]),
                )]),
                Some(Value::str("nyczip")),
            ),
        ]);
        let list = vec![obj(&[("city", Value::str("NYC"))])];
        assert_eq!(
            sorted_ids(
                &c.find(obj(&[("addr", obj(&[("$in", Value::array_from(list))]))]))
                    .to_list()
            ),
            vec!["nyc".to_string()]
        );
    }

    #[test]
    fn in_nin_combine_with_other_operators() {
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a25")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("a30")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(33))]),
                Some(Value::str("a33")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("a40")),
            ),
        ]);
        // $in AND $ne
        let f = obj(&[(
            "age",
            obj(&[
                (
                    "$in",
                    Value::array_from(vec![Value::i64(25), Value::i64(30), Value::i64(40)]),
                ),
                ("$ne", Value::i64(30)),
            ]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["a25".to_string(), "a40".to_string()]
        );
        // $in AND range
        let f = obj(&[(
            "age",
            obj(&[
                (
                    "$in",
                    Value::array_from(vec![Value::i64(30), Value::i64(33), Value::i64(40)]),
                ),
                ("$gte", Value::i64(33)),
            ]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["a33".to_string(), "a40".to_string()]
        );
        // $nin AND range
        let f = obj(&[(
            "age",
            obj(&[
                (
                    "$nin",
                    Value::array_from(vec![Value::i64(30), Value::i64(33)]),
                ),
                ("$gte", Value::i64(30)),
            ]),
        )]);
        assert_eq!(sorted_ids(&c.find(f).to_list()), vec!["a40".to_string()]);
        // an unknown operator kills the condition even next to $in
        let f = obj(&[(
            "age",
            obj(&[
                ("$in", Value::array_from(vec![Value::i64(30)])),
                ("$bogus", Value::i64(1)),
            ]),
        )]);
        assert_eq!(c.find(f).count(), 0);
    }

    #[test]
    fn in_drives_index_and_matches_scan() {
        let mut c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::f64(40.0))]),
                Some(Value::str("c2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::Null)]),
                Some(Value::str("n")),
            ),
            (Value::object_from(vec![]), Some(Value::str("m"))), // field absent
        ]);
        let filters: Vec<Value> = vec![
            obj(&[(
                "age",
                obj(&[(
                    "$in",
                    Value::array_from(vec![Value::i64(40), Value::i64(25)]),
                )]),
            )]), // cross-numeric point set
            obj(&[(
                "age",
                obj(&[("$in", Value::array_from(vec![Value::Null, Value::i64(40)]))]),
            )]), // Null slice: explicit null + absent
            obj(&[("age", obj(&[("$in", Value::array_from(Vec::new()))]))]), // empty list
            obj(&[(
                "age",
                obj(&[
                    ("$in", Value::array_from(vec![Value::i64(30)])),
                    ("$ne", Value::i64(30)),
                ]),
            )]), // points drive, $ne verifies everything out
            obj(&[(
                "age",
                obj(&[
                    ("$in", Value::array_from(vec![Value::i64(30)])),
                    ("$gte", Value::i64(30)),
                ]),
            )]),
            obj(&[(
                "age",
                obj(&[(
                    "$nin",
                    Value::array_from(vec![Value::i64(25), Value::i64(30)]),
                )]),
            )]), // bare $nin: plain scan
            obj(&[(
                "age",
                obj(&[("$nin", Value::array_from(vec![Value::Null]))]),
            )]), // $nin [null]: present non-null only
            obj(&[(
                "age",
                obj(&[
                    ("$nin", Value::array_from(vec![Value::i64(30)])),
                    ("$gte", Value::i64(30)),
                ]),
            )]), // bound drives, $nin verifies
            obj(&[("age", obj(&[("$in", Value::i64(30))]))]), // non-array operand: scan, matches nothing
            obj(&[(
                "age",
                obj(&[
                    ("$in", Value::array_from(vec![Value::i64(30)])),
                    ("$bogus", Value::i64(1)),
                ]),
            )]),
        ];
        let scan: Vec<Vec<String>> = filters
            .iter()
            .map(|f| sorted_ids(&c.find(f.clone()).to_list()))
            .collect();
        c.create_index("age").unwrap();
        for (i, f) in filters.iter().enumerate() {
            assert_eq!(
                sorted_ids(&c.find(f.clone()).to_list()),
                scan[i],
                "filter {i} with an `age` index must equal the scan result"
            );
        }
        // spot-check the interesting ones
        assert_eq!(
            sorted_ids(&c.find(filters[1].clone()).to_list()),
            vec![
                "c".to_string(),
                "c2".to_string(),
                "m".to_string(),
                "n".to_string()
            ],
            "$in [null, 40]: explicit null + absent + the cross-numeric 40s"
        );
        assert_eq!(
            sorted_ids(&c.find(filters[6].clone()).to_list()),
            vec![
                "a".to_string(),
                "b".to_string(),
                "b2".to_string(),
                "c".to_string(),
                "c2".to_string()
            ],
            "$nin [null]: only present non-null docs"
        );
    }

    #[test]
    fn in_index_results_come_in_index_order() {
        let mut c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b2")),
            ),
        ]);
        c.create_index("age").unwrap();
        // list order (40, 25, 30) must not leak into the result: index order
        // is value ascending per the total order, ties by `_id` -> a, b, b2, c
        let got = c
            .find(obj(&[(
                "age",
                obj(&[(
                    "$in",
                    Value::array_from(vec![Value::i64(40), Value::i64(25), Value::i64(30)]),
                )]),
            )]))
            .to_list();
        let order: Vec<&str> = got
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(order, vec!["a", "b", "b2", "c"]);
    }

    #[test]
    fn mixed_operator_and_plain_keys_is_not_an_operator_object() {
        // {"$gte":25, "x":1} has a non-$ key -> treated as a direct value,
        // which matches nothing (no doc field literally equals that object).
        let c = col_with(&[(
            Value::object_from(vec![("age".into(), Value::i64(30))]),
            Some(Value::str("x")),
        )]);
        assert_eq!(
            c.find(obj(&[(
                "age",
                obj(&[("$gte", Value::i64(25)), ("x", Value::i64(1))])
            )]))
            .count(),
            0
        );
    }
    // -- logical operators ($and / $or / $not) -------------------------------

    /// A five-doc field exercising `$not` presence rules: 3 (low), 5 (mid5),
    /// 7 (high), explicit null (nul), field absent (mis).
    fn not_herd() -> Collection {
        col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(3))]),
                Some(Value::str("low")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(5))]),
                Some(Value::str("mid5")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(7))]),
                Some(Value::str("high")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::Null)]),
                Some(Value::str("nul")),
            ),
            (Value::object_from(vec![]), Some(Value::str("mis"))),
        ])
    }

    fn cow_trio() -> Collection {
        col_with(&[
            (
                Value::object_from(vec![
                    ("age".into(), Value::i64(5)),
                    ("tag".into(), Value::str("loud")),
                ]),
                Some(Value::str("moo")),
            ),
            (
                Value::object_from(vec![
                    ("age".into(), Value::i64(5)),
                    ("tag".into(), Value::str("milky")),
                ]),
                Some(Value::str("hilde")),
            ),
            (
                Value::object_from(vec![
                    ("age".into(), Value::i64(9)),
                    ("tag".into(), Value::str("loud")),
                ]),
                Some(Value::str("daisy")),
            ),
        ])
    }
    /// `{"age": {"$not": {<op>: n}}}` — the common `$not` shape in the
    /// logical-operator tests, lifted into one binding so the nesting stays
    /// trivially bracket-checkable.
    fn not_age(op: &str, n: i64) -> Value {
        let operand = obj(&[(op, Value::i64(n))]);
        let cond = obj(&[("$not", operand)]);
        obj(&[("age", cond)])
    }

    #[test]
    fn and_requires_all_sub_filters_to_match() {
        let c = cow_trio();
        let f = obj(&[(
            "$and",
            Value::array_from(vec![
                obj(&[("age", obj(&[("$gte", Value::i64(5))]))]),
                obj(&[("tag", Value::str("loud"))]),
            ]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["daisy".to_string(), "moo".to_string()]
        );
        // an empty list is vacuous truth: matches every document
        assert_eq!(
            c.find(obj(&[("$and", Value::array_from(Vec::new()))]))
                .count(),
            3
        );
        // a single element is just the sub-filter
        assert_eq!(
            c.find(obj(&[(
                "$and",
                Value::array_from(vec![obj(&[("tag", Value::str("milky"))])]),
            )]))
            .count(),
            1
        );
    }

    #[test]
    fn or_requires_any_sub_filter_to_match() {
        let c = cow_trio();
        let f = obj(&[(
            "$or",
            Value::array_from(vec![
                obj(&[("age", Value::i64(9))]),
                obj(&[("tag", Value::str("milky"))]),
            ]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["daisy".to_string(), "hilde".to_string()]
        );
        // an empty list matches nothing
        assert_eq!(
            c.find(obj(&[("$or", Value::array_from(Vec::new()))]))
                .count(),
            0
        );
        // a non-object element fails only its own disjunct
        let f = obj(&[(
            "$or",
            Value::array_from(vec![Value::i64(9), obj(&[("tag", Value::str("milky"))])]),
        )]);
        assert_eq!(sorted_ids(&c.find(f).to_list()), vec!["hilde".to_string()]);
    }

    #[test]
    fn and_or_sub_filters_use_full_filter_semantics_and_nest() {
        let c = col_with(&[
            (
                Value::object_from(vec![
                    ("age".into(), Value::i64(5)),
                    ("tag".into(), Value::str("loud")),
                ]),
                Some(Value::str("moo")),
            ),
            (
                Value::object_from(vec![
                    ("age".into(), Value::i64(2)),
                    ("tag".into(), Value::str("loud")),
                ]),
                Some(Value::str("bess")),
            ),
        ]);
        // $or of [$and(range combo + tag), $in]: the inner $and gives moo
        // (5 <= age < 10 AND loud), the $in disjunct gives bess (age == 2).
        let f = obj(&[(
            "$or",
            Value::array_from(vec![
                obj(&[(
                    "$and",
                    Value::array_from(vec![
                        obj(&[(
                            "age",
                            obj(&[("$gte", Value::i64(5)), ("$lt", Value::i64(10))]),
                        )]),
                        obj(&[("tag", Value::str("loud"))]),
                    ]),
                )]),
                obj(&[(
                    "age",
                    obj(&[("$in", Value::array_from(vec![Value::i64(2)]))]),
                )]),
            ]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["bess".to_string(), "moo".to_string()]
        );
    }

    #[test]
    fn and_or_malformed_shapes_match_nothing() {
        let c = not_herd();
        let bad = [
            obj(&[("$and", Value::i64(5))]),   // non-array operand
            obj(&[("$and", Value::object())]), // an object is not an array
            obj(&[("$and", Value::str("x"))]),
            obj(&[("$and", Value::array_from(vec![Value::i64(5)]))]), // non-object element
            obj(&[(
                "$and",
                Value::array_from(vec![Value::array_from(Vec::new())]),
            )]),
            obj(&[("$or", Value::i64(5))]), // non-array operand
            obj(&[("$or", Value::object_from(vec![]))]), // empty list: no disjunct
        ];
        for (i, f) in bad.iter().enumerate() {
            assert_eq!(c.find(f.clone()).count(), 0, "malformed $and/$or shape {i}");
        }
    }

    #[test]
    fn field_level_and_or_match_nothing() {
        let c = not_herd();
        // Lift the deep nesting into named bindings so the bracket count stays
        // trivially checkable (see AGENTS.md — deeply nested `obj(&[...])`
        // literals are where the "missing `]`" parse errors live).
        let inner = obj(&[("age", Value::i64(5))]);
        for key in ["$and", "$or"] {
            let op_cond = obj(&[(key, Value::array_from(vec![inner.clone()]))]);
            let f = obj(&[("age", op_cond)]);
            assert_eq!(c.find(f).count(), 0, "field-level {key} is malformed");
        }
    }

    #[test]
    fn top_level_not_is_malformed_and_matches_nothing() {
        let c = not_herd();
        assert_eq!(
            c.find(obj(&[("$not", obj(&[("$gt", Value::i64(5))]))]))
                .count(),
            0
        );
        assert_eq!(c.find(obj(&[("$not", Value::i64(5))])).count(), 0);
        assert_eq!(c.find(obj(&[("$not", Value::object())])).count(), 0);
    }

    #[test]
    fn not_negates_the_operator_expression_including_presence() {
        let c = not_herd();
        // {$not: {$gt: 5}} == "not (age > 5)": age <= 5 (null included, it is
        // the minimum rank) plus the absent field ($gt is false when missing).
        let f = not_age("$gt", 5);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec![
                "low".to_string(),
                "mid5".to_string(),
                "mis".to_string(),
                "nul".to_string()
            ]
        );
        // {$not: {$ne: 5}} == "not (age != 5)": the presence rules flip too —
        // missing ($ne is true when absent) and null (null != 5) are excluded,
        // leaving exactly the docs whose age is 5.
        let f = not_age("$ne", 5);
        assert_eq!(sorted_ids(&c.find(f).to_list()), vec!["mid5".to_string()]);
        // {$not: {$eq: 5}} == "age != 5": null and missing included.
        let f = not_age("$eq", 5);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec![
                "high".to_string(),
                "low".to_string(),
                "mis".to_string(),
                "nul".to_string()
            ]
        );
    }

    #[test]
    fn not_of_a_multi_operator_expression_negates_the_and() {
        // {$not: {$gt: 3, $lt: 7}} == "not (3 < age < 7)": the inner operators
        // AND first, then the whole expression is negated.
        let c = not_herd();
        // Lift the nested literals into bindings so the bracket count stays
        // trivially checkable (see AGENTS.md).
        let inner = obj(&[("$gt", Value::i64(3)), ("$lt", Value::i64(7))]);
        let cond = obj(&[("$not", inner)]);
        let f = obj(&[("age", cond)]);
        // 3 fails $gt and 7 fails $lt -> inner false -> negation true;
        // 5 passes both -> inner true -> excluded; null/missing fail $gt -> in.
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec![
                "high".to_string(),
                "low".to_string(),
                "mis".to_string(),
                "nul".to_string()
            ]
        );
    }

    #[test]
    fn not_combines_with_other_operators_by_and() {
        let c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(2))]),
                Some(Value::str("a2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(3))]),
                Some(Value::str("a3")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(4))]),
                Some(Value::str("a4")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(5))]),
                Some(Value::str("a5")),
            ),
            (Value::object_from(vec![]), Some(Value::str("mis"))),
        ]);
        // {$gte: 3} AND {$not: {$gte: 5}} == 3 <= age < 5, presence required.
        let inner = obj(&[("$gte", Value::i64(5))]);
        let cond = obj(&[("$gte", Value::i64(3)), ("$not", inner)]);
        let f = obj(&[("age", cond)]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["a3".to_string(), "a4".to_string()]
        );
    }

    #[test]
    fn not_with_non_operator_object_operand_matches_nothing() {
        let c = not_herd();
        // Lift the four malformed-operand shapes into bindings so the bracket
        // count stays trivially checkable (see AGENTS.md).
        let inner_direct = obj(&[("$not", Value::i64(5))]); // {"$not": 5}
        let inner_plain = obj(&[("$not", obj(&[("x", Value::i64(1))]))]); // {"$not": {"x":1}}
        let inner_arr = obj(&[("$not", Value::array_from(vec![Value::i64(5)]))]); // {"$not": [5]}
        let inner_empty = obj(&[("$not", Value::object())]); // {"$not": {}}
        let bad = [
            obj(&[("age", inner_direct)]),
            obj(&[("age", inner_plain)]),
            obj(&[("age", inner_arr)]),
            obj(&[("age", inner_empty)]),
        ];
        for (i, f) in bad.iter().enumerate() {
            assert_eq!(
                c.find(f.clone()).count(),
                0,
                "non-operator-object $not operand {i}"
            );
        }
        // Degenerate corner: the operand IS an operator object, but the inner
        // expression matches nothing (unknown operator) -> its negation
        // matches everything. Consistent with "negation of the whole
        // expression", recorded as the spec's contract.
        let f = not_age("$bogus", 1);
        assert_eq!(
            c.find(f).count(),
            5,
            "negation of a nothing-matching expression"
        );
    }

    #[test]
    fn top_level_and_or_keys_are_reserved_not_field_names() {
        // A doc storing a literal "$and" field is not reachable through a
        // top-level {"$and": ...} filter (the key is reserved), and a literal
        // "$not" field likewise (top-level $not is always malformed).
        let c = col_with(&[(
            Value::object_from(vec![
                ("$and".to_string(), Value::i64(7)),
                ("x".into(), Value::i64(1)),
            ]),
            Some(Value::str("lit")),
        )]);
        assert_eq!(
            c.find(obj(&[("$and", Value::i64(7))])).count(),
            0,
            "non-array $and operand"
        );
        let c2 = col_with(&[(
            Value::object_from(vec![("$not".to_string(), Value::i64(7))]),
            Some(Value::str("lit2")),
        )]);
        assert_eq!(c2.find(obj(&[("$not", Value::i64(7))])).count(), 0);
    }

    #[test]
    fn logical_plans_andor_not() {
        let mut c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(45))]),
                Some(Value::str("d")),
            ),
        ]);
        c.create_index("age").unwrap();

        // a $and element with a bound drives
        let sub = obj(&[("age", obj(&[("$gte", Value::i64(30))]))]);
        let q = c.find(obj(&[("$and", Value::array_from(vec![sub]))]));
        assert!(
            matches!(q.plan(), Plan::Index { .. }),
            "a $and element with a bound must drive"
        );

        // nested $and is looked through
        let leaf = obj(&[("age", Value::i64(30))]);
        let inner_and = obj(&[("$and", Value::array_from(vec![leaf]))]);
        let q = c.find(obj(&[("$and", Value::array_from(vec![inner_and]))]));
        assert!(
            matches!(q.plan(), Plan::Index { .. }),
            "nested $and must be looked through"
        );

        // a top-level $or never drives (a union no condition's candidates contain)
        let d1 = obj(&[("age", Value::i64(30))]);
        let d2 = obj(&[("age", Value::i64(40))]);
        let q = c.find(obj(&[("$or", Value::array_from(vec![d1, d2]))]));
        assert!(matches!(q.plan(), Plan::Scan), "a $or must not drive");

        // a bare $not never drives
        let q = c.find(not_age("$lt", 45));
        assert!(matches!(q.plan(), Plan::Scan), "a bare $not must not drive");

        // $not next to a bound: the bound drives, $not verifies
        let lt = obj(&[("$lt", Value::i64(45))]);
        let cond = obj(&[("$not", lt), ("$gte", Value::i64(30))]);
        let q = c.find(obj(&[("age", cond)]));
        assert!(
            matches!(q.plan(), Plan::Index { .. }),
            "a bound next to $not drives"
        );

        // a malformed $and operand cannot drive; a later top-level key can
        let q = c.find(obj(&[
            ("$and", Value::i64(5)),
            ("age", obj(&[("$gte", Value::i64(30))])),
        ]));
        assert!(
            matches!(q.plan(), Plan::Index { .. }),
            "malformed $and is skipped"
        );

        // a malformed $and element cannot drive
        let q = c.find(obj(&[("$and", Value::array_from(vec![Value::i64(5)]))]));
        assert!(matches!(q.plan(), Plan::Scan));

        // a top-level $not is skipped; a later bound drives
        let q = c.find(obj(&[
            ("$not", obj(&[("$gt", Value::i64(5))])),
            ("age", obj(&[("$gte", Value::i64(30))])),
        ]));
        assert!(
            matches!(q.plan(), Plan::Index { .. }),
            "top-level $not is skipped"
        );
    }

    #[test]
    fn logical_filters_index_driven_match_full_scan() {
        let mut c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(45))]),
                Some(Value::str("d")),
            ),
            (Value::object_from(vec![]), Some(Value::str("m"))), // age absent
        ]);
        // Lift every filter into a named binding so the bracket count stays
        // trivially checkable (see AGENTS.md). Each is a single `obj(...)` line.
        let gte30 = obj(&[("age", obj(&[("$gte", Value::i64(30))]))]);
        let lt45 = obj(&[("age", obj(&[("$lt", Value::i64(45))]))]);
        let and_range = obj(&[("$and", Value::array_from(vec![gte30, lt45]))]);
        let lt40 = obj(&[("age", obj(&[("$lt", Value::i64(40))]))]);
        let inner_and = obj(&[("$and", Value::array_from(vec![lt40]))]);
        let nested_and = obj(&[("$and", Value::array_from(vec![inner_and]))]);
        let age30 = obj(&[("age", Value::i64(30))]);
        let age45 = obj(&[("age", Value::i64(45))]);
        let or_point = obj(&[("$or", Value::array_from(vec![age30, age45]))]);
        let not_lt30 = obj(&[("age", obj(&[("$not", obj(&[("$lt", Value::i64(30))]))]))]);
        let age25 = obj(&[("age", Value::i64(25))]);
        let or_not = obj(&[("$or", Value::array_from(vec![not_lt30, age25]))]);
        let not_lt45 = obj(&[("age", obj(&[("$not", obj(&[("$lt", Value::i64(45))]))]))]);
        let lt40op = obj(&[("$lt", Value::i64(40))]);
        let bound_not = obj(&[("age", obj(&[("$not", lt40op), ("$gte", Value::i64(30))]))]);
        let filters: Vec<Value> = vec![
            and_range,                                                // 30 <= age < 45
            nested_and,                                               // nested $and: age < 40
            or_point,                        // $or never drives: age == 30 | age == 45
            or_not,                          // $not inside an $or disjunct
            not_lt45,                        // bare $not: not(age < 45)
            bound_not,                       // bound drives, $not verifies: 30 <= age, age >= 40
            obj(&[("$and", Value::i64(5))]), // malformed operand
            obj(&[("$or", Value::array_from(Vec::new()))]), // empty list
            obj(&[("$and", Value::array_from(vec![Value::i64(1)]))]), // non-object element
            obj(&[("$not", obj(&[("$gt", Value::i64(5))]))]), // top-level $not
        ];
        let scan: Vec<Vec<String>> = filters
            .iter()
            .map(|f| sorted_ids(&c.find(f.clone()).to_list()))
            .collect();
        c.create_index("age").unwrap();
        for (i, f) in filters.iter().enumerate() {
            assert_eq!(
                sorted_ids(&c.find(f.clone()).to_list()),
                scan[i],
                "filter {i} with an `age` index must equal the scan result"
            );
        }
        // spot-check the answers (derived from the fixture)
        assert_eq!(
            scan[0],
            vec!["b".to_string(), "b2".to_string(), "c".to_string()]
        );
        assert_eq!(
            scan[1],
            vec!["a".to_string(), "b".to_string(), "b2".to_string()]
        );
        assert_eq!(
            scan[2],
            vec!["b".to_string(), "b2".to_string(), "d".to_string()]
        );
        assert_eq!(
            scan[3],
            vec![
                "a".to_string(),
                "b".to_string(),
                "b2".to_string(),
                "c".to_string(),
                "d".to_string(),
                "m".to_string()
            ],
            "not(age < 30) covers 30..45 plus the absent doc; age 25 adds a -> all"
        );
        assert_eq!(
            scan[4],
            vec!["d".to_string(), "m".to_string()],
            "not(age < 45): 45 and absent"
        );
        assert_eq!(
            scan[5],
            vec!["c".to_string(), "d".to_string()],
            "30 <= age < 40: only 40 and 45"
        );
        for (i, v) in scan.iter().enumerate().skip(6) {
            assert_eq!(
                v,
                &Vec::<String>::new(),
                "malformed shape {i} matches nothing"
            );
        }
    }

    #[test]
    fn and_drives_index_results_come_in_index_order() {
        let mut c = col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(40))]),
                Some(Value::str("c")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(25))]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(30))]),
                Some(Value::str("b2")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(45))]),
                Some(Value::str("d")),
            ),
        ]);
        c.create_index("age").unwrap();
        // the $and element's bound drives: index order (value ascending per
        // the total order, ties by _id) is b, b2, c — a full scan would
        // return storage order instead, so the exact order proves the plan.
        let gte30 = obj(&[("age", obj(&[("$gte", Value::i64(30))]))]);
        let lt45 = obj(&[("age", obj(&[("$lt", Value::i64(45))]))]);
        let f = obj(&[("$and", Value::array_from(vec![gte30, lt45]))]);
        let order: Vec<String> = c
            .find(f)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            order,
            vec!["b".to_string(), "b2".to_string(), "c".to_string()]
        );
        // first() on the index path is the smallest matching value
        let only = obj(&[("age", obj(&[("$gte", Value::i64(30))]))]);
        let first = c
            .find(obj(&[("$and", Value::array_from(vec![only]))]))
            .first()
            .expect("a match exists");
        assert_eq!(first.get("_id"), Some(&Value::str("b")));
    }

    // -- element operator ($exists) ------------------------------------------

    /// A field `n` that is present (non-null) on two docs, explicitly `null`
    /// on one, and absent on one — the `$exists` presence matrix.
    fn exists_herd() -> Collection {
        col_with(&[
            (
                Value::object_from(vec![("n".into(), Value::i64(1))]),
                Some(Value::str("one")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::i64(5))]),
                Some(Value::str("five")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::Null)]),
                Some(Value::str("nul")),
            ),
            (Value::object_from(vec![]), Some(Value::str("abs"))),
        ])
    }

    fn exists_cond(field: &str, b: bool) -> Value {
        let inner = obj(&[("$exists", Value::bool(b))]);
        obj(&[(field, inner)])
    }

    #[test]
    fn exists_true_matches_present_fields_including_null() {
        let c = exists_herd();
        // the explicit-null doc (nul) is present: $exists true includes it
        assert_eq!(
            sorted_ids(&c.find(exists_cond("n", true)).to_list()),
            vec!["five".to_string(), "nul".to_string(), "one".to_string()]
        );
    }

    #[test]
    fn exists_false_matches_only_absent_fields() {
        let c = exists_herd();
        // explicit null (nul) is present, so $exists false excludes it; only
        // the truly-absent doc matches
        assert_eq!(
            sorted_ids(&c.find(exists_cond("n", false)).to_list()),
            vec!["abs".to_string()]
        );
    }

    #[test]
    fn exists_true_and_false_partition_the_collection() {
        let c = exists_herd();
        let t = sorted_ids(&c.find(exists_cond("n", true)).to_list());
        let f = sorted_ids(&c.find(exists_cond("n", false)).to_list());
        let mut all = t;
        all.extend(f);
        all.sort();
        assert_eq!(
            all,
            vec![
                "abs".to_string(),
                "five".to_string(),
                "nul".to_string(),
                "one".to_string()
            ]
        );
    }

    #[test]
    fn exists_on_a_field_nobody_has_is_vacuously_absent() {
        let c = exists_herd();
        // filtering a field no doc has: every doc is "absent" -> $exists false
        // matches all, $exists true matches none
        assert_eq!(c.find(exists_cond("zzz", false)).count(), 4);
        assert_eq!(c.find(exists_cond("zzz", true)).count(), 0);
    }

    #[test]
    fn exists_non_boolean_operand_matches_nothing() {
        let c = exists_herd();
        let bads = [
            Value::i64(1),
            Value::f64(1.0),
            Value::str("true"),
            Value::Null,
            Value::array(),
            Value::object(),
        ];
        for (i, bad) in bads.iter().enumerate() {
            let f = obj(&[("n", obj(&[("$exists", bad.clone())]))]);
            assert_eq!(
                c.find(f).count(),
                0,
                "non-boolean $exists operand {i} matches nothing"
            );
        }
    }

    #[test]
    fn exists_combines_with_other_operators_by_and() {
        let c = exists_herd();
        // $exists true is redundant next to a presence-requiring operator
        let f = obj(&[(
            "n",
            obj(&[("$exists", Value::bool(true)), ("$gte", Value::i64(1))]),
        )]);
        assert_eq!(
            sorted_ids(&c.find(f).to_list()),
            vec!["five".to_string(), "one".to_string()]
        );
        // $exists false contradicts a presence-requiring operator: nothing
        let f = obj(&[(
            "n",
            obj(&[("$exists", Value::bool(false)), ("$gte", Value::i64(1))]),
        )]);
        assert_eq!(c.find(f).count(), 0);
        // a non-boolean $exists operand kills the condition even next to a
        // valid operator
        let f = obj(&[(
            "n",
            obj(&[("$exists", Value::i64(1)), ("$gte", Value::i64(1))]),
        )]);
        assert_eq!(c.find(f).count(), 0);
    }

    #[test]
    fn exists_never_drives_an_index_scan() {
        let mut c = exists_herd();
        c.create_index("n").unwrap();
        // a bare $exists contributes no bound -> the plan is a full scan even
        // though `n` is indexed (missing and explicit-null share the index's
        // Null slot, so the index cannot split presence from absence)
        assert!(matches!(c.find(exists_cond("n", true)).plan(), Plan::Scan));
        assert!(matches!(c.find(exists_cond("n", false)).plan(), Plan::Scan));
        // ...and the results are identical with and without the index
        let t = sorted_ids(&c.find(exists_cond("n", true)).to_list());
        let f = sorted_ids(&c.find(exists_cond("n", false)).to_list());
        assert_eq!(
            t,
            vec!["five".to_string(), "nul".to_string(), "one".to_string()]
        );
        assert_eq!(f, vec!["abs".to_string()]);
    }

    // -- array operator ($elemMatch) ----------------------------------------

    /// A collection of cows, each with a `sizes` array (of i64) and an
    /// `instock` array of subdocuments `{qty, warehouse}`. Also a scalar
    /// `sizes` (non-array) and a missing `sizes` to exercise the no-match
    /// corners. Exercises all three `$elemMatch` operand kinds.
    fn elem_herd() -> Collection {
        col_with(&[
            (
                Value::object_from(vec![(
                    "sizes".into(),
                    Value::array_from(vec![Value::i64(1), Value::i64(4)]),
                )]),
                Some(Value::str("a")),
            ),
            (
                Value::object_from(vec![(
                    "sizes".into(),
                    Value::array_from(vec![Value::i64(5), Value::i64(9)]),
                )]),
                Some(Value::str("b")),
            ),
            (
                Value::object_from(vec![(
                    "sizes".into(),
                    Value::array_from(vec![Value::i64(7), Value::i64(7)]),
                )]),
                Some(Value::str("c")),
            ),
            // non-array field: a plain scalar, never matches $elemMatch
            (
                Value::object_from(vec![("sizes".into(), Value::i64(5))]),
                Some(Value::str("d")),
            ),
            // missing field
            (Value::object_from(vec![]), Some(Value::str("e"))),
            // subdocuments: instock = [{qty:3,wh:A},{qty:8,wh:A}]
            (
                Value::object_from(vec![(
                    "instock".into(),
                    Value::array_from(vec![
                        obj(&[("qty", Value::i64(3)), ("warehouse", Value::str("A"))]),
                        obj(&[("qty", Value::i64(8)), ("warehouse", Value::str("A"))]),
                    ]),
                )]),
                Some(Value::str("sd1")),
            ),
            // subdocuments: instock = [{qty:10,wh:B}]
            (
                Value::object_from(vec![(
                    "instock".into(),
                    Value::array_from(vec![obj(&[
                        ("qty", Value::i64(10)),
                        ("warehouse", Value::str("B")),
                    ])]),
                )]),
                Some(Value::str("sd2")),
            ),
        ])
    }

    fn em(field: &str, criteria: Value) -> Value {
        obj(&[(field, obj(&[("$elemMatch", criteria)]))])
    }

    #[test]
    fn elem_match_direct_value_is_element_equality() {
        let c = elem_herd();
        // a: [1,4] has 4; b: [5,9] has 5; c: [7,7]; d: scalar 5 (not an array)
        assert_eq!(
            sorted_ids(&c.find(em("sizes", Value::i64(4))).to_list()),
            vec!["a".to_string()]
        );
        assert_eq!(
            sorted_ids(&c.find(em("sizes", Value::i64(5))).to_list()),
            vec!["b".to_string()]
        );
        // a value no array contains
        assert_eq!(c.find(em("sizes", Value::i64(99))).count(), 0);
        // a scalar field is not an array -> no elements -> no match (even
        // though d's scalar is 5, $elemMatch requires an array)
        assert_eq!(
            c.find(em("sizes", Value::i64(5))).count(),
            1,
            "d (scalar 5) must NOT match; only b"
        );
        // missing field matches nothing
        assert_eq!(c.find(em("nope", Value::i64(5))).count(), 0);
        // cross-numeric element equality: f64(9.0) matches i64(9) in b
        assert_eq!(
            sorted_ids(&c.find(em("sizes", Value::f64(9.0))).to_list()),
            vec!["b".to_string()]
        );
    }

    #[test]
    fn elem_match_direct_value_matches_strings_and_objects() {
        // string elements: build a tag array doc
        let mut c = Collection::new("t");
        c.insert(obj(&[
            ("_id", Value::str("x")),
            (
                "tags",
                Value::array_from(vec![Value::str("moo"), Value::str("loud")]),
            ),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("y")),
            ("tags", Value::array_from(vec![Value::str("moo")])),
        ]))
        .unwrap();
        // "loud" is only in x
        assert_eq!(
            sorted_ids(&c.find(em("tags", Value::str("loud"))).to_list()),
            vec!["x".to_string()]
        );
        // "moo" is in both
        assert_eq!(
            sorted_ids(&c.find(em("tags", Value::str("moo"))).to_list()),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn elem_match_operator_object_is_element_level() {
        let c = elem_herd();
        // sizes: a [1,4], b [5,9], c [7,7]; d scalar 5; e absent
        // {$elemMatch: {$gt: 4}} -> an element > 4: b (5,9), c (7,7). d is not
        // an array so it never matches.
        assert_eq!(
            sorted_ids(
                &c.find(em("sizes", obj(&[("$gt", Value::i64(4))])))
                    .to_list()
            ),
            vec!["b".to_string(), "c".to_string()]
        );
        // {$elemMatch: {$gte: 7, $lt: 8}} -> an element in [7,8): only c
        let op = obj(&[("$gte", Value::i64(7)), ("$lt", Value::i64(8))]);
        assert_eq!(
            sorted_ids(&c.find(em("sizes", op)).to_list()),
            vec!["c".to_string()]
        );
        // {$elemMatch: {$eq: 5, $ne: 9}} -> an element == 5 and != 9: only b
        let op = obj(&[("$eq", Value::i64(5)), ("$ne", Value::i64(9))]);
        assert_eq!(
            sorted_ids(&c.find(em("sizes", op)).to_list()),
            vec!["b".to_string()]
        );
        // {$elemMatch: {$in: [1, 7]}} -> an element in {1,7}: a (1), c (7)
        let op = obj(&[("$in", Value::array_from(vec![Value::i64(1), Value::i64(7)]))]);
        assert_eq!(
            sorted_ids(&c.find(em("sizes", op)).to_list()),
            vec!["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn elem_match_subdocument_uses_full_filter() {
        let c = elem_herd();
        // sd1: [{qty:3,wh:A},{qty:8,wh:A}]; sd2: [{qty:10,wh:B}]
        // {$elemMatch: {qty: {$gt: 5}, warehouse: "A"}}: an element with qty
        // > 5 AND warehouse A -> only sd1 (qty 8, wh A). sd2's single element
        // has wh B.
        let sub = obj(&[
            ("qty", obj(&[("$gt", Value::i64(5))])),
            ("warehouse", Value::str("A")),
        ]);
        assert_eq!(
            sorted_ids(&c.find(em("instock", sub)).to_list()),
            vec!["sd1".to_string()]
        );
        // {$elemMatch: {qty: 3, warehouse: "A"}}: exact subdocument match on
        // one element -> sd1 (first element is exactly {qty:3, warehouse:A})
        let sub = obj(&[("qty", Value::i64(3)), ("warehouse", Value::str("A"))]);
        assert_eq!(
            sorted_ids(&c.find(em("instock", sub)).to_list()),
            vec!["sd1".to_string()]
        );
        // {$elemMatch: {warehouse: "B"}} -> only sd2
        let sub = obj(&[("warehouse", Value::str("B"))]);
        assert_eq!(
            sorted_ids(&c.find(em("instock", sub)).to_list()),
            vec!["sd2".to_string()]
        );
        // a subdocument filter that matches no element
        let sub = obj(&[("qty", Value::i64(99)), ("warehouse", Value::str("A"))]);
        assert_eq!(c.find(em("instock", sub)).count(), 0);
        // an array of scalars with a subdocument filter: elements aren't
        // objects -> doc_matches fails on each -> no match
        assert_eq!(
            c.find(em("sizes", obj(&[("qty", Value::i64(1))]))).count(),
            0
        );
    }

    #[test]
    fn elem_match_missing_and_non_array_match_nothing() {
        let c = elem_herd();
        // e has no `sizes` field; d has a scalar (not an array). Neither
        // matches $elemMatch regardless of operand.
        for crit in [
            Value::i64(5),
            obj(&[("$gt", Value::i64(0))]),
            obj(&[("x", Value::i64(1))]),
        ] {
            assert_eq!(
                c.find(em("missing", crit.clone())).count(),
                0,
                "missing field"
            );
        }
        // d's scalar sizes=5 has no elements: only the three real arrays
        // a/b/c match $gt 0 (d and the missing e never match)
        assert_eq!(
            c.find(em("sizes", obj(&[("$gt", Value::i64(0))]))).count(),
            3,
            "a, b, c (d is a scalar, e is missing)"
        );
    }

    #[test]
    fn elem_match_combines_with_top_level_and() {
        // AND a $elemMatch condition with a second direct-value condition on a
        // separate field: build a small col with both `sizes` and `top`.
        let mut c = Collection::new("t");
        c.insert(obj(&[
            ("_id", Value::str("p")),
            ("sizes", Value::array_from(vec![Value::i64(6)])),
            ("top", Value::i64(9)),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("q")),
            ("sizes", Value::array_from(vec![Value::i64(6)])),
            ("top", Value::i64(2)),
        ]))
        .unwrap();
        // {sizes: {$elemMatch: {$gt: 4}}, top: 9} -> only p
        let f = obj(&[
            (
                "sizes",
                obj(&[("$elemMatch", obj(&[("$gt", Value::i64(4))]))]),
            ),
            ("top", Value::i64(9)),
        ]);
        assert_eq!(sorted_ids(&c.find(f).to_list()), vec!["p".to_string()]);
    }

    #[test]
    fn elem_match_never_drives_an_index_scan() {
        let mut c = elem_herd();
        c.create_index("sizes").unwrap();
        // a bare $elemMatch contributes no bound -> the plan is a full scan
        // even though `sizes` is indexed (the index stores the whole array as
        // one value; the elements are not indexed)
        assert!(matches!(
            c.find(em("sizes", Value::i64(4))).plan(),
            Plan::Scan
        ));
        assert!(matches!(
            c.find(em("sizes", obj(&[("$gt", Value::i64(4))]))).plan(),
            Plan::Scan
        ));
        // ...and the results are identical with and without the index
        let direct = sorted_ids(&c.find(em("sizes", Value::i64(4))).to_list());
        let op = sorted_ids(
            &c.find(em("sizes", obj(&[("$gt", Value::i64(4))])))
                .to_list(),
        );
        assert_eq!(direct, vec!["a".to_string()]);
        assert_eq!(op, vec!["b".to_string(), "c".to_string()]);
    }

    // -- malformed filters --------------------------------------------------

    #[test]
    fn non_object_filter_matches_nothing() {
        let c = col_with(&[(
            Value::object_from(vec![("a".into(), Value::i64(1))]),
            Some(Value::str("x")),
        )]);
        for bad in [
            Value::i64(5),
            Value::str("x"),
            Value::array(),
            Value::bool(true),
        ] {
            assert_eq!(
                c.find(bad).count(),
                0,
                "non-object filter must match nothing"
            );
        }
    }

    // -- sort / skip / limit pipeline ----------------------------------------

    /// Six cows with ages: bess 2, butch 3, hilde 5, moo 5, clara 7, daisy 9.
    /// (hilde and moo tie at 5 — the `_id` tiebreak is observable.)
    fn sort_herd() -> Collection {
        col_with(&[
            (
                Value::object_from(vec![("age".into(), Value::i64(2))]),
                Some(Value::str("bess")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(5))]),
                Some(Value::str("moo")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(5))]),
                Some(Value::str("hilde")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(9))]),
                Some(Value::str("daisy")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(3))]),
                Some(Value::str("butch")),
            ),
            (
                Value::object_from(vec![("age".into(), Value::i64(7))]),
                Some(Value::str("clara")),
            ),
        ])
    }

    #[test]
    fn sort_ascending_uses_total_order_and_id_tiebreak() {
        let c = sort_herd();
        // 2(bess) 3(butch) 5(hilde) 5(moo) 7(clara) 9(daisy) — hilde < moo
        // is the `_id` byte tiebreak
        let order: Vec<String> = c
            .find(Value::object())
            .sort("age", false)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            order,
            vec!["bess", "butch", "hilde", "moo", "clara", "daisy"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sort_descending_reverses_value_and_id_ties() {
        let c = sort_herd();
        // descending reverses the WHOLE (value, _id) order: the 5-tie comes
        // back as moo, hilde (id descending inside the equal-value slice)
        let order: Vec<String> = c
            .find(Value::object())
            .sort("age", true)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            order,
            vec![
                "daisy".to_string(),
                "clara".to_string(),
                "moo".to_string(),
                "hilde".to_string(),
                "butch".to_string(),
                "bess".to_string()
            ]
        );
    }

    #[test]
    fn sort_missing_field_sorts_like_null() {
        let c = col_with(&[
            (
                Value::object_from(vec![("v".into(), Value::i64(5))]),
                Some(Value::str("five")),
            ),
            (
                Value::object_from(vec![("v".into(), Value::Null)]),
                Some(Value::str("nul")),
            ),
            (Value::object_from(vec![]), Some(Value::str("abs"))), // field absent
            (
                Value::object_from(vec![("v".into(), Value::i64(1))]),
                Some(Value::str("one")),
            ),
        ]);
        // Null < Number: both the explicit-null doc and the absent doc sort
        // first (ties by `_id`), then the numbers
        let asc: Vec<String> = c
            .find(Value::object())
            .sort("v", false)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            asc,
            vec![
                "abs".to_string(),
                "nul".to_string(),
                "one".to_string(),
                "five".to_string()
            ]
        );
        // descending: the Null slice lands last
        let desc: Vec<String> = c
            .find(Value::object())
            .sort("v", true)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            desc,
            vec![
                "five".to_string(),
                "one".to_string(),
                "nul".to_string(),
                "abs".to_string()
            ]
        );
    }

    #[test]
    fn sort_is_cross_numeric_total_order() {
        let c = col_with(&[
            (
                Value::object_from(vec![("n".into(), Value::f64(2.5))]),
                Some(Value::str("f25")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::i64(2))]),
                Some(Value::str("i2")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::i64(-1))]),
                Some(Value::str("n1")),
            ),
            (
                Value::object_from(vec![("n".into(), Value::str("z"))]),
                Some(Value::str("s")),
            ),
        ]);
        let asc: Vec<String> = c
            .find(Value::object())
            .sort("n", false)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            asc,
            vec![
                "n1".to_string(),
                "i2".to_string(),
                "f25".to_string(),
                "s".to_string()
            ]
        );
        let desc: Vec<String> = c
            .find(Value::object())
            .sort("n", true)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            desc,
            vec![
                "s".to_string(),
                "f25".to_string(),
                "i2".to_string(),
                "n1".to_string()
            ]
        );
    }

    #[test]
    fn skip_and_limit_apply_to_filtered_stream() {
        let c = sort_herd();
        let f = |pairs: &[(&str, Value)]| obj(&[("age", obj(pairs))]);
        let f5 = f(&[("$gte", Value::i64(5))]); // matches: hilde 5, moo 5, clara 7, daisy 9
        // sorted pipeline: 5(hilde) 5(moo) 7(clara) 9(daisy)
        assert_eq!(
            sorted_ids(
                &c.find(f5.clone())
                    .sort("age", false)
                    .skip(1)
                    .limit(2)
                    .to_list()
            ),
            vec!["clara".to_string(), "moo".to_string()]
        );
        // skip alone
        assert_eq!(
            sorted_ids(&c.find(f5.clone()).sort("age", false).skip(2).to_list()),
            vec!["clara".to_string(), "daisy".to_string()]
        );
        // limit alone (first N of the sorted stream)
        assert_eq!(
            sorted_ids(&c.find(f5.clone()).sort("age", false).limit(2).to_list()),
            vec!["hilde".to_string(), "moo".to_string()]
        );
        // skip past the end: empty
        assert!(
            c.find(f5.clone())
                .sort("age", false)
                .skip(4)
                .to_list()
                .is_empty()
        );
        assert!(
            c.find(f5.clone())
                .sort("age", false)
                .skip(5)
                .to_list()
                .is_empty()
        );
        // limit 0 = no limit (the Mongo cursor convention)
        assert_eq!(
            sorted_ids(&c.find(f5.clone()).sort("age", false).limit(0).to_list()),
            vec![
                "clara".to_string(),
                "daisy".to_string(),
                "hilde".to_string(),
                "moo".to_string()
            ]
        );
        // unsorted: skip/limit apply in scan (storage) order — the exact
        // doc is unspecified, but the count is deterministic
        assert_eq!(c.find(f5.clone()).skip(1).limit(1).count(), 1);
        assert_eq!(c.find(f5.clone()).skip(3).count(), 1);
        assert_eq!(c.find(f5).limit(2).count(), 2, "count honors limit");
    }

    #[test]
    fn sort_on_indexed_field_streams_the_index() {
        let mut c = sort_herd();
        c.create_index("age").unwrap();
        // ascending: index order (value ascending, ties by `_id`)
        let asc: Vec<String> = c
            .find(Value::object())
            .sort("age", false)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            asc,
            vec![
                "bess".to_string(),
                "butch".to_string(),
                "hilde".to_string(),
                "moo".to_string(),
                "clara".to_string(),
                "daisy".to_string()
            ]
        );
        // descending: reverse index walk
        let desc: Vec<String> = c
            .find(Value::object())
            .sort("age", true)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            desc,
            vec![
                "daisy".to_string(),
                "clara".to_string(),
                "moo".to_string(),
                "hilde".to_string(),
                "butch".to_string(),
                "bess".to_string()
            ]
        );
        // indexed sort with a filter: candidates verified in sort order
        let f = obj(&[("age", obj(&[("$gte", Value::i64(5))]))]);
        let got: Vec<String> = c
            .find(f)
            .sort("age", true)
            .limit(2)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        // 9(daisy) first, then 7(clara) — the 5-tie (moo, hilde) is #3/#4
        assert_eq!(got, vec!["daisy".to_string(), "clara".to_string()]);
        // a limit that reaches the equal-value slice shows the tiebreak:
        // descending reverses the tie too, so moo (id > hilde) comes first
        let got: Vec<String> = c
            .find(obj(&[(
                "age",
                obj(&[("$gte", Value::i64(5)), ("$lt", Value::i64(9))]),
            )]))
            .sort("age", true)
            .limit(3)
            .to_list()
            .iter()
            .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
            .collect();
        // 7(clara) first, then the 5-tie reversed: moo before hilde
        assert_eq!(
            got,
            vec!["clara".to_string(), "moo".to_string(), "hilde".to_string()]
        );
    }

    #[test]
    fn sort_on_indexed_field_matches_unindexed_sort() {
        let mut c = sort_herd();
        let f = obj(&[(
            "age",
            obj(&[("$gte", Value::i64(3)), ("$lt", Value::i64(9))]),
        )]);
        let asc_scan = c.find(f.clone()).sort("age", false).to_list();
        let desc_scan = c.find(f.clone()).sort("age", true).to_list();
        c.create_index("age").unwrap();
        assert_eq!(c.find(f.clone()).sort("age", false).to_list(), asc_scan);
        assert_eq!(c.find(f).sort("age", true).to_list(), desc_scan);
    }

    #[test]
    fn first_and_count_honor_skip_and_limit() {
        let c = sort_herd();
        let f = obj(&[("age", obj(&[("$gte", Value::i64(5))]))]); // 4 matches
        // first() is the first doc of the pipelined stream
        let first = c
            .find(f.clone())
            .sort("age", true)
            .skip(1)
            .first()
            .expect("a match exists");
        // sorted desc: daisy 9 (skipped), clara 7 -> clara
        assert_eq!(first.get("_id"), Some(&Value::str("clara")));
        let first = c
            .find(f.clone())
            .sort("age", false)
            .skip(1)
            .limit(1)
            .first()
            .expect("a match exists");
        // sorted asc: hilde 5 (skipped), moo 5 -> moo
        assert_eq!(first.get("_id"), Some(&Value::str("moo")));
        // count honors skip/limit
        assert_eq!(c.find(f.clone()).sort("age", false).count(), 4);
        assert_eq!(c.find(f.clone()).sort("age", false).skip(2).count(), 2);
        assert_eq!(c.find(f.clone()).sort("age", false).limit(2).count(), 2);
        assert_eq!(
            c.find(f.clone())
                .sort("age", false)
                .skip(1)
                .limit(2)
                .count(),
            2
        );
        assert_eq!(c.find(f.clone()).skip(6).count(), 0);
        assert_eq!(c.find(f.clone()).skip(2).limit(10).count(), 2);
        assert_eq!(c.find(f).limit(0).count(), 4, "limit(0) is no limit");
    }

    #[test]
    fn pipeline_accessors_expose_settings() {
        let c = sort_herd();
        let all = Value::object();
        let q = c.find(all.clone());
        assert_eq!(q.sort_field(), None);
        assert!(!q.sort_descending());
        assert_eq!(q.skip_count(), 0);
        assert_eq!(q.limit_count(), 0);
        let q = c.find(all.clone()).sort("age", true).skip(2).limit(5);
        assert_eq!(q.sort_field(), Some("age"));
        assert!(q.sort_descending());
        assert_eq!(q.skip_count(), 2);
        assert_eq!(q.limit_count(), 5);
        // re-sorting replaces the previous sort
        let q = c.find(all).sort("age", true).sort("age", false);
        assert_eq!(q.sort_field(), Some("age"));
        assert!(!q.sort_descending());
    }

    // -- terminals agree with eager entry points ----------------------------

    #[test]
    fn eager_and_lazy_terminals_agree() {
        let c = col_with(&[
            (
                Value::object_from(vec![("s".into(), Value::str("hit"))]),
                Some(Value::str("h1")),
            ),
            (
                Value::object_from(vec![("s".into(), Value::str("no"))]),
                Some(Value::str("n1")),
            ),
            (
                Value::object_from(vec![("s".into(), Value::str("hit"))]),
                Some(Value::str("h2")),
            ),
        ]);
        let f = || obj(&[("s", Value::str("hit"))]);
        assert_eq!(c.count(f()), 2);
        assert_eq!(c.find(f()).count(), 2);
        assert_eq!(c.find(f()).to_list().len(), 2);
        assert!(c.exists(f()));
        assert!(c.find(f()).first().is_some());
        assert!(c.find_one(f()).is_some());
    }

    #[test]
    fn first_and_find_one_are_consistent_and_present() {
        let c = col_with(&[(
            Value::object_from(vec![("s".into(), Value::str("only"))]),
            Some(Value::str("only")),
        )]);
        let f = obj(&[("s", Value::str("only"))]);
        let via_find = c.find(f.clone()).first().unwrap();
        let via_find_one = c.find_one(f).unwrap();
        assert_eq!(
            via_find.get("_id"),
            Some(&Value::str("only")),
            "both terminals return the matching doc"
        );
        assert_eq!(via_find_one.get("_id"), Some(&Value::str("only")),);
    }

    #[test]
    fn query_filter_accessor_exposes_filter() {
        let c = Collection::new("t");
        let f = obj(&[("a", Value::i64(1))]);
        assert_eq!(c.find(f.clone()).filter(), &f);
    }
}
