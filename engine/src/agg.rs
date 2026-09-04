//! Aggregation: `Query::group(field)` → [`GroupQuery`] with the terminal
//! `.agg(fn, field)`.
//!
//! Contract (recorded in spec.md "Aggregation decisions"):
//!
//! - `.find(filter).group(g)` groups the query's pipelined document stream
//!   (filter → sort → skip → limit, in that stream order) by the value of
//!   group field `g`; a **missing group field groups under `Null`** (the
//!   engine's missing-field convention).
//! - The terminal `.agg(f, field)` emits one result document per group:
//!   `{ "_id": <group key value>, "<fn-name>": <result> }`.
//! - `count` ignores `field` and counts group members.
//! - `sum`/`mean` take **numeric** values (`I64`/`F64`); missing or
//!   non-numeric values are skipped. `sum` over no numerics is `I64(0)`;
//!   `mean` over no numerics is `Null`. All-`I64` sums stay `I64` and widen
//!   to `F64` only on overflow (the `$inc` rule); any `F64` operand makes
//!   the sum `F64`.
//! - `min`/`max` take the total-order extreme over **present** values
//!   (missing skipped); no present values → `Null`.
//! - `collect` appends one element per group member **in document (stream)
//!   order** — missing field contributes `Null`.
//! - `first`/`last` take the field value of the first/last group member in
//!   stream order; missing field → `Null`.
//! - Groups are deterministic: by default sorted by group key (total order);
//!   `GroupQuery::sort(field, desc)` re-sorts the *group documents* by that
//!   field's total order (missing → `Null`, ties by `_id` — the same
//!   convention as `Query::sort`), and `GroupQuery::limit(n)` truncates
//!   (`0` = no limit, the Mongo convention).
//!
//! No `unsafe`: the hot path is one stream pass + one stable sort over a
//! contiguous `(key, value)` array; nothing measured justifies more.

use crate::query::Query;
use crate::value::Value;

/// Aggregation functions for [`GroupQuery::agg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFn {
    /// Number of documents in the group (the `field` argument is ignored).
    Count,
    /// Sum of the field's numeric values.
    Sum,
    /// Mean of the field's numeric values.
    Mean,
    /// Total-order minimum of the field's present values.
    Min,
    /// Total-order maximum of the field's present values.
    Max,
    /// Array of the field's values, one per member, in document order.
    Collect,
    /// The field's value in the first member (document order).
    First,
    /// The field's value in the last member (document order).
    Last,
}

impl AggFn {
    /// The result document's field name for this function.
    pub const fn name(self) -> &'static str {
        match self {
            AggFn::Count => "count",
            AggFn::Sum => "sum",
            AggFn::Mean => "mean",
            AggFn::Min => "min",
            AggFn::Max => "max",
            AggFn::Collect => "collect",
            AggFn::First => "first",
            AggFn::Last => "last",
        }
    }
}

/// A grouping over a query's document stream: built by
/// [`Query::group`], terminated by [`GroupQuery::agg`].
///
/// Cheap to hold (it owns the source `Query` plus a couple of `String`s);
/// the single document-stream pass runs only at `agg`.
pub struct GroupQuery<'c> {
    q: Query<'c>,
    group_field: String,
    /// Sort field on the *group documents* (`None` = default: key order).
    sort: Option<String>,
    desc: bool,
    /// Group limit (`0` = no limit).
    limit: usize,
}

impl<'c> GroupQuery<'c> {
    /// Build a grouped query (called by [`Query::group`]).
    pub(crate) fn new(q: Query<'c>, group_field: String) -> Self {
        GroupQuery {
            q,
            group_field,
            sort: None,
            desc: false,
            limit: 0,
        }
    }

    /// Sort the **group result documents** by `field`'s value in the total
    /// order (missing → `Null`, ties by `_id`). Replaces any earlier sort.
    pub fn sort(mut self, field: impl Into<String>, desc: bool) -> Self {
        self.sort = Some(field.into());
        self.desc = desc;
        self
    }

    /// Return at most `m` groups. `0` means **no limit**.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Evaluate: the grouped aggregate, one document per group
    /// `{ "_id": <key>, "<fn-name>": <result> }`, in group order
    /// (default: group key total order; re-sortable via
    /// [`GroupQuery::sort`], truncatable via [`GroupQuery::limit`]).
    pub fn agg(self, f: AggFn, field: impl Into<String>) -> Vec<Value> {
        let field = field.into();
        // One stream pass: capture (group key, field value) per document.
        // Missing group field → Null key; missing field → None (the
        // functions decide how to treat absence).
        let mut pairs: Vec<(Value, Option<Value>)> = Vec::new();
        self.q.for_each_pipelined(|doc| {
            let key = doc
                .get(&self.group_field)
                .cloned()
                .unwrap_or(Value::Null);
            let fv = if f == AggFn::Count {
                None
            } else {
                doc.get(&field).cloned()
            };
            pairs.push((key, fv));
            true
        });
        // Stable sort by key: groups become contiguous runs and the
        // within-run document (stream) order — needed by first/last/collect
        // — is preserved.
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        // Walk the runs, building one result doc per group.
        let mut out: Vec<Value> = Vec::new();
        let mut i = 0usize;
        while i < pairs.len() {
            let mut j = i + 1;
            while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                j += 1;
            }
            out.push(result_doc(&pairs[i].0, f, &pairs[i..j]));
            i = j;
        }

        // Optional sort on the group documents (same convention as
        // Query::sort: total order, missing → Null, ties by `_id`).
        if let Some(sf) = &self.sort {
            let desc = self.desc;
            out.sort_unstable_by(|a, b| {
                let av = a.get(sf).unwrap_or(&Value::Null);
                let bv = b.get(sf).unwrap_or(&Value::Null);
                let o = av.cmp(bv).then_with(|| match (a.get("_id"), b.get("_id")) {
                    (Some(x), Some(y)) => x.cmp(y),
                    _ => std::cmp::Ordering::Equal,
                });
                if desc {
                    o.reverse()
                } else {
                    o
                }
            });
        }
        if self.limit > 0 {
            out.truncate(self.limit);
        }
        out
    }
}

/// Build the result document for one group (one run of `pairs`).
fn result_doc(key: &Value, f: AggFn, run: &[(Value, Option<Value>)]) -> Value {
    let res: Value = match f {
        AggFn::Count => Value::i64(run.len() as i64),
        AggFn::Sum => sum_values(run),
        AggFn::Mean => {
            let mut n = 0usize;
            let mut acc = (0i128, 0f64, false);
            for &(_, ref fv) in run {
                if let Some(v) = fv {
                    if add_numeric(v, &mut acc) {
                        n += 1;
                    }
                }
            }
            if n == 0 {
                Value::Null
            } else {
                let total = if acc.2 { acc.1 } else { acc.0 as f64 };
                Value::f64(total / n as f64)
            }
        }
        AggFn::Min => run
            .iter()
            .filter_map(|&(_, ref fv)| fv.clone())
            .reduce(|m, v| if m <= v { m } else { v })
            .unwrap_or(Value::Null),
        AggFn::Max => run
            .iter()
            .filter_map(|&(_, ref fv)| fv.clone())
            .reduce(|m, v| if m <= v { v } else { m })
            .unwrap_or(Value::Null),
        AggFn::Collect => {
            Value::array_from(run.iter().map(|&(_, ref fv)| fv.clone().unwrap_or(Value::Null)).collect())
        }
        AggFn::First => run.first().and_then(|&(_, ref fv)| fv.clone()).unwrap_or(Value::Null),
        AggFn::Last => run
            .last()
            .and_then(|&(_, ref fv)| fv.clone())
            .unwrap_or(Value::Null),
    };
    Value::object_from(vec![(
        String::from("_id"),
        key.clone(),
    ), (
        String::from(f.name()),
        res,
    )])
}

/// Sum a run's numeric field values (missing / non-numeric skipped).
fn sum_values(run: &[(Value, Option<Value>)]) -> Value {
    let mut acc = (0i128, 0f64, false); // (i128 int acc, f64 acc, float-seen)
    let mut any = false;
    for &(_, ref fv) in run {
        if let Some(v) = fv {
            any = true;
            add_numeric(v, &mut acc);
        }
    }
    if !any || !acc.2 {
        // No numerics, or all-integer sum: stay integer unless it overflowed
        // i64 (the `$inc` widening rule).
        let n = acc.0;
        if !any {
            Value::i64(0)
        } else if n >= i64::MIN as i128 && n <= i64::MAX as i128 {
            Value::i64(n as i64)
        } else {
            Value::f64(n as f64)
        }
    } else {
        Value::f64(acc.1)
    }
}

/// Add a numeric `v` into the `(i128, f64, float-seen)` accumulator.
/// Returns `true` when `v` was numeric (non-numeric values are skipped).
fn add_numeric(v: &Value, acc: &mut (i128, f64, bool)) -> bool {
    match *v {
        Value::I64(n) => {
            acc.0 += n as i128;
            if acc.2 {
                acc.1 += n as f64;
            }
            true
        }
        Value::F64(x) => {
            acc.2 = true;
            if acc.1 == 0.0 && acc.0 != 0 {
                // First float after integer accumulation: seed the f64 lane
                // with the accumulated integer so far.
                acc.1 = acc.0 as f64;
            }
            acc.1 += x;
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::Collection;
    use crate::value::Value;

    fn obj(pairs: &[(&str, Value)]) -> Value {
        Value::object_from(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }

    fn herd() -> Collection {
        let mut c = Collection::new("t");
        // east: a(10) b(20) e(15.5, f64); west: c(5) d(30, tag missing)
        c.insert(obj(&[
            ("_id", Value::str("a")),
            ("region", Value::str("east")),
            ("price", Value::i64(10)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("b")),
            ("region", Value::str("east")),
            ("price", Value::i64(20)),
            ("tag", Value::str("x")),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("c")),
            ("region", Value::str("west")),
            ("price", Value::i64(5)),
            ("tag", Value::str("y")),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("d")),
            ("region", Value::str("west")),
            ("price", Value::i64(30)),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("e")),
            ("region", Value::str("east")),
            ("price", Value::f64(15.5)),
        ]))
        .unwrap();
        c
    }

    /// The `count` of a group result doc.
    fn count_of(doc: &Value) -> i64 {
        doc.get("count").unwrap().as_i64().unwrap()
    }

    #[test]
    fn count_groups() {
        let c = herd();
        let g = c.find(Value::object()).group("region").agg(AggFn::Count, "");
        assert_eq!(g.len(), 2);
        // default group order: key total order ("east" < "west")
        assert_eq!(g[0].get("_id"), Some(&Value::str("east")));
        assert_eq!(g[1].get("_id"), Some(&Value::str("west")));
        assert_eq!(count_of(&g[0]), 3);
        assert_eq!(count_of(&g[1]), 2);
    }

    #[test]
    fn missing_group_field_groups_under_null() {
        let c = herd();
        // "tag" is missing on d and e → the Null group has 2 members.
        let g = c.find(Value::object()).group("tag").agg(AggFn::Count, "");
        assert_eq!(g.len(), 3);
        // total order: Null < Str
        assert_eq!(g[0].get("_id"), Some(&Value::Null));
        assert_eq!(count_of(&g[0]), 2);
        let rest: Vec<i64> = g[1..].iter().map(count_of).collect();
        assert_eq!(rest, vec![2, 1]); // "x" (a,b), "y" (c)
    }

    #[test]
    fn sum_stays_i64_and_widens_on_float() {
        let c = herd();
        let g = c.find(Value::object()).group("region").agg(AggFn::Sum, "price");
        // west: 5 + 30 = 35 (all i64 → i64)
        assert_eq!(g[1].get("sum"), Some(&Value::i64(35)));
        // east: 10 + 20 + 15.5 → f64 45.5
        assert_eq!(g[0].get("sum"), Some(&Value::f64(45.5)));
        // type check: 45.5 is genuinely an F64 (cross-numeric equality would
        // otherwise let a wrong type slip through)
        assert!(matches!(g[0].get("sum"), Some(Value::F64(_))));
    }

    #[test]
    fn sum_no_numerics_is_zero_and_skips_non_numeric() {
        let c = herd();
        // "tag": west has "y" (non-numeric) + missing → no numerics → 0
        let g = c.find(Value::object()).group("region").agg(AggFn::Sum, "tag");
        assert_eq!(g[0].get("sum"), Some(&Value::i64(0)));
        assert_eq!(g[1].get("sum"), Some(&Value::i64(0)));
    }

    #[test]
    fn sum_i64_overflow_widens_to_f64() {
        let mut c = Collection::new("t");
        c.insert(obj(&[
            ("_id", Value::str("a")),
            ("k", Value::str("g")),
            ("price", Value::i64(i64::MAX)),
        ]))
        .unwrap();
        c.insert(obj(&[
            ("_id", Value::str("b")),
            ("k", Value::str("g")),
            ("price", Value::i64(1)),
        ]))
        .unwrap();
        let g = c.find(Value::object()).group("k").agg(AggFn::Sum, "price");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].get("sum"), Some(&Value::f64(i64::MAX as f64 + 1.0)));
    }

    #[test]
    fn mean_of_numerics_and_null_when_none() {
        let c = herd();
        let g = c.find(Value::object()).group("region").agg(AggFn::Mean, "price");
        let east = g[0].get("mean").unwrap().as_f64().unwrap();
        assert!((east - 45.5 / 3.0).abs() < 1e-12);
        let west = g[1].get("mean").unwrap().as_f64().unwrap();
        assert!((west - 17.5).abs() < 1e-12);
        // mean of a non-numeric field → Null
        let g = c.find(Value::object()).group("region").agg(AggFn::Mean, "tag");
        assert_eq!(g[0].get("mean"), Some(&Value::Null));
    }

    #[test]
    fn min_max_total_order_and_missing_skipped() {
        let c = herd();
        let g = c.find(Value::object()).group("region").agg(AggFn::Min, "price");
        // east min = 10 (10 < 15.5 < 20 in the total order)
        assert_eq!(g[0].get("min"), Some(&Value::i64(10)));
        assert_eq!(g[1].get("min"), Some(&Value::i64(5)));
        let g = c.find(Value::object()).group("region").agg(AggFn::Max, "price");
        assert_eq!(g[0].get("max"), Some(&Value::i64(20)));
        assert_eq!(g[1].get("max"), Some(&Value::i64(30)));
        // no present values → Null
        let mut c = Collection::new("t");
        c.insert(obj(&[("_id", Value::str("a")), ("k", Value::str("g"))]))
            .unwrap();
        let g = c.find(Value::object()).group("k").agg(AggFn::Min, "nope");
        assert_eq!(g[0].get("min"), Some(&Value::Null));
    }

    #[test]
    fn collect_first_last_follow_stream_order() {
        let c = herd();
        // Sort by price so the stream (and hence within-group) order is
        // deterministic: east = a(10), e(15.5), b(20); west = c(5), d(30).
        let q = c
            .find(Value::object())
            .sort("price", false)
            .group("region");
        let g = q.agg(AggFn::Collect, "tag");
        assert_eq!(
            g[0].get("collect"),
            Some(&Value::array_from(vec![
                Value::str("x"),
                Value::Null,
                Value::str("x"),
            ]))
        );
        let g = c
            .find(Value::object())
            .sort("price", false)
            .group("region")
            .agg(AggFn::First, "tag");
        assert_eq!(g[0].get("first"), Some(&Value::str("x")));
        assert_eq!(g[1].get("first"), Some(&Value::str("y")));
        let g = c
            .find(Value::object())
            .sort("price", false)
            .group("region")
            .agg(AggFn::Last, "tag");
        assert_eq!(g[0].get("last"), Some(&Value::str("x")));
        assert_eq!(g[1].get("last"), Some(&Value::Null));
        // first/last of a never-present field → Null
        let g = c
            .find(Value::object())
            .group("region")
            .agg(AggFn::First, "nope");
        assert_eq!(g[0].get("first"), Some(&Value::Null));
    }

    #[test]
    fn group_sort_and_limit_apply_to_groups() {
        let c = herd();
        // Sort groups by their count descending → west? no: east(3) first.
        let g = c
            .find(Value::object())
            .group("region")
            .sort("count", true)
            .agg(AggFn::Count, "");
        assert_eq!(g[0].get("_id"), Some(&Value::str("east")));
        assert_eq!(g[1].get("_id"), Some(&Value::str("west")));
        // ...and ascending puts west first
        let g = c
            .find(Value::object())
            .group("region")
            .sort("count", false)
            .agg(AggFn::Count, "");
        assert_eq!(g[0].get("_id"), Some(&Value::str("west")));
        // limit(1) keeps only the first group; limit(0) = no limit
        let g = c
            .find(Value::object())
            .group("region")
            .limit(1)
            .agg(AggFn::Count, "");
        assert_eq!(g.len(), 1);
        let g = c
            .find(Value::object())
            .group("region")
            .limit(0)
            .agg(AggFn::Count, "");
        assert_eq!(g.len(), 2);
        // sort by the sum result field
        let g = c
            .find(Value::object())
            .group("region")
            .sort("sum", false)
            .agg(AggFn::Sum, "price");
        // west sum 35 < east sum 45.5
        assert_eq!(g[0].get("_id"), Some(&Value::str("west")));
    }

    #[test]
    fn filter_narrowing_and_query_pipeline_precedence() {
        let c = herd();
        // Filter: only price >= 20 → east: b(20); west: d(30)
        let f = obj(&[("price", obj(&[("$gte", Value::i64(20))]))]);
        let g = c.find(f).group("region").agg(AggFn::Count, "");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].get("_id"), Some(&Value::str("east")));
        assert_eq!(count_of(&g[0]), 1);
        assert_eq!(count_of(&g[1]), 1);
        // The query's own limit applies to the *document* stream before
        // grouping (filter → sort → skip → limit, then group).
        let g = c
            .find(Value::object())
            .sort("price", false)
            .limit(2)
            .group("region")
            .agg(AggFn::Count, "");
        // only the two cheapest docs (a=10 east, c=5 west) reach the group
        assert_eq!(g.len(), 2);
        assert_eq!(count_of(&g[0]), 1);
        assert_eq!(count_of(&g[1]), 1);
    }

    #[test]
    fn empty_stream_yields_no_groups() {
        let c = herd();
        let g = c
            .find(obj(&[("price", Value::i64(999))]))
            .group("region")
            .agg(AggFn::Count, "");
        assert!(g.is_empty());
    }

    #[test]
    fn cross_numeric_keys_merge_into_one_group() {
        // The total order treats I64(1) == F64(1.0): both docs land in the
        // same group.
        let mut c = Collection::new("t");
        c.insert(obj(&[("_id", Value::str("a")), ("k", Value::i64(1)), ("v", Value::i64(1))]))
            .unwrap();
        c.insert(obj(&[("_id", Value::str("b")), ("k", Value::f64(1.0)), ("v", Value::i64(2))]))
            .unwrap();
        let g = c.find(Value::object()).group("k").agg(AggFn::Count, "");
        assert_eq!(g.len(), 1);
        assert_eq!(count_of(&g[0]), 2);
        let g = c.find(Value::object()).group("k").agg(AggFn::Sum, "v");
        assert_eq!(g[0].get("sum"), Some(&Value::i64(3)));
    }
}
