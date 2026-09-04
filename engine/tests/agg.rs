//! Integration tests for aggregation:
//! `Collection::find(filter).group(field).agg(fn, field)` with optional
//! group sort/limit.

use mooracer_engine::{AggFn, Collection, Value};

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::object_from(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

/// The sales herd: 6 docs, two regions.
///
/// east:  a(10) b(20) e(15.5 f64)     west:  c(5) d(30, tag missing)
fn sales() -> Collection {
    let mut c = Collection::new("sales");
    c.insert(obj(&[
        ("_id", Value::str("a")),
        ("region", Value::str("east")),
        ("price", Value::i64(10)),
        ("tag", Value::str("x")),
        ("qty", Value::i64(2)),
    ]))
    .unwrap();
    c.insert(obj(&[
        ("_id", Value::str("b")),
        ("region", Value::str("east")),
        ("price", Value::i64(20)),
        ("tag", Value::str("x")),
        ("qty", Value::i64(1)),
    ]))
    .unwrap();
    c.insert(obj(&[
        ("_id", Value::str("c")),
        ("region", Value::str("west")),
        ("price", Value::i64(5)),
        ("tag", Value::str("y")),
        ("qty", Value::i64(4)),
    ]))
    .unwrap();
    c.insert(obj(&[
        ("_id", Value::str("d")),
        ("region", Value::str("west")),
        ("price", Value::i64(30)),
        ("qty", Value::i64(3)),
    ]))
    .unwrap();
    c.insert(obj(&[
        ("_id", Value::str("e")),
        ("region", Value::str("east")),
        ("price", Value::f64(15.5)),
        ("qty", Value::i64(5)),
    ]))
    .unwrap();
    c.insert(obj(&[
        ("_id", Value::str("f")),
        ("price", Value::i64(1)),
        // region missing → Null group
    ]))
    .unwrap();
    c
}

fn key(g: &Value) -> &str {
    match g.get("_id").unwrap() {
        Value::Str(s) => s.as_str(),
        Value::Null => "",
        _ => panic!("unexpected key"),
    }
}

#[test]
fn group_count_shapes_and_default_key_order() {
    let c = sales();
    let g = c.find(Value::object()).group("region").agg(AggFn::Count, "");
    // three groups: Null (f), "east" (a,b,e), "west" (c,d); default order is
    // key total order: Null < "east" < "west"
    assert_eq!(g.len(), 3);
    assert_eq!(key(&g[0]), "");
    assert_eq!(g[0].get("count"), Some(&Value::i64(1)));
    assert_eq!(key(&g[1]), "east");
    assert_eq!(g[1].get("count"), Some(&Value::i64(3)));
    assert_eq!(key(&g[2]), "west");
    assert_eq!(g[2].get("count"), Some(&Value::i64(2)));
    // the result document carries exactly _id + count
    assert_eq!(g[1].len(), 2);
}

#[test]
fn group_sum_mean_per_region() {
    let c = sales();
    let g = c.find(Value::object()).group("region").agg(AggFn::Sum, "price");
    // east: 10 + 20 + 15.5 = 45.5 (F64), west: 35 (I64), Null: 1 (I64)
    assert_eq!(g[0].get("sum"), Some(&Value::i64(1)));
    assert_eq!(g[1].get("sum"), Some(&Value::f64(45.5)));
    assert!(matches!(g[1].get("sum"), Some(Value::F64(_))));
    assert_eq!(g[2].get("sum"), Some(&Value::i64(35)));
    assert!(matches!(g[2].get("sum"), Some(Value::I64(35))));

    let g = c.find(Value::object()).group("region").agg(AggFn::Mean, "price");
    let east = g[1].get("mean").unwrap().as_f64().unwrap();
    assert!((east - 45.5 / 3.0).abs() < 1e-12);
    let west = g[2].get("mean").unwrap().as_f64().unwrap();
    assert!((west - 17.5).abs() < 1e-12);
}

#[test]
fn group_min_max_and_collect() {
    let c = sales();
    let g = c.find(Value::object()).group("region").agg(AggFn::Min, "price");
    assert_eq!(g[1].get("min"), Some(&Value::i64(10)));
    assert_eq!(g[2].get("min"), Some(&Value::i64(5)));
    let g = c.find(Value::object()).group("region").agg(AggFn::Max, "price");
    assert_eq!(g[1].get("max"), Some(&Value::i64(20)));
    assert_eq!(g[2].get("max"), Some(&Value::i64(30)));

    // collect in a deterministic stream order (sorted by price):
    // east = a(10), e(15.5), b(20) → tags x, (missing → Null), x
    let g = c
        .find(Value::object())
        .sort("price", false)
        .group("region")
        .agg(AggFn::Collect, "tag");
    assert_eq!(
        g[1].get("collect"),
        Some(&Value::array_from(vec![
            Value::str("x"),
            Value::Null,
            Value::str("x"),
        ]))
    );
    // west = c(5), d(30) → y, Null
    assert_eq!(
        g[2].get("collect"),
        Some(&Value::array_from(vec![Value::str("y"), Value::Null]))
    );
}

#[test]
fn group_first_last_follow_sorted_stream() {
    let c = sales();
    let g = c
        .find(Value::object())
        .sort("price", false)
        .group("region")
        .agg(AggFn::First, "tag");
    assert_eq!(g[1].get("first"), Some(&Value::str("x"))); // a
    assert_eq!(g[2].get("first"), Some(&Value::str("y"))); // c
    let g = c
        .find(Value::object())
        .sort("price", true)
        .group("region")
        .agg(AggFn::Last, "tag");
    // desc: east last = a(10) → "x"; west last = c(5) → "y"
    assert_eq!(g[1].get("last"), Some(&Value::str("x")));
    assert_eq!(g[2].get("last"), Some(&Value::str("y")));
    // missing field → Null
    let g = c
        .find(Value::object())
        .group("region")
        .agg(AggFn::First, "region_nope");
    assert_eq!(g[1].get("first"), Some(&Value::Null));
}

#[test]
fn filter_narrows_the_groups() {
    let c = sales();
    let f = obj(&[("qty", obj(&[("$gte", Value::i64(3))]))]);
    let g = c.find(f).group("region").agg(AggFn::Count, "");
    // qty ≥ 3: c(4), d(3), e(5) → west 2, east 1 (f has qty 1)
    assert_eq!(g.len(), 2);
    // default group order is key ascending: "east" < "west"
    assert_eq!(key(&g[0]), "east");
    assert_eq!(key(&g[1]), "west");
    assert_eq!(g[0].get("count"), Some(&Value::i64(1)));
    assert_eq!(g[1].get("count"), Some(&Value::i64(2)));
}

#[test]
fn group_sort_and_limit_order_groups() {
    let c = sales();
    // sort groups by count descending → east(3), west(2), Null(1)
    let g = c
        .find(Value::object())
        .group("region")
        .sort("count", true)
        .agg(AggFn::Count, "");
    assert_eq!(key(&g[0]), "east");
    assert_eq!(key(&g[1]), "west");
    assert_eq!(key(&g[2]), "");
    // limit(2) → top two
    let g = c
        .find(Value::object())
        .group("region")
        .sort("count", true)
        .limit(2)
        .agg(AggFn::Count, "");
    assert_eq!(g.len(), 2);
    assert_eq!(key(&g[1]), "west");
    // sort by the sum result: 1 < 35 < 45.5 → Null, west, east
    let g = c
        .find(Value::object())
        .group("region")
        .sort("sum", false)
        .agg(AggFn::Sum, "price");
    assert_eq!(key(&g[0]), "");
    assert_eq!(key(&g[1]), "west");
    assert_eq!(key(&g[2]), "east");
}

#[test]
fn query_limit_applies_before_grouping() {
    let c = sales();
    // only the two cheapest docs reach the group stage: c(west), a(east)
    let g = c
        .find(Value::object())
        .sort("price", false)
        .limit(2)
        .group("region")
        .agg(AggFn::Count, "");
    assert_eq!(g.len(), 2);
    for doc in &g {
        assert_eq!(doc.get("count"), Some(&Value::i64(1)));
    }
}

#[test]
fn empty_filter_result_yields_no_groups() {
    let c = sales();
    let g = c
        .find(obj(&[("price", Value::i64(99999))]))
        .group("region")
        .agg(AggFn::Count, "");
    assert!(g.is_empty());
}

#[test]
fn large_group_counts_are_exact() {
    // 3000 docs over 300 groups (id/10), 10 members each.
    let mut c = Collection::new("bulk");
    for i in 0..3000i64 {
        c.insert(obj(&[
            ("_id", Value::str(format!("d{i:05}"))),
            ("g", Value::i64(i / 10)),
            ("v", Value::i64(i % 7)),
        ]))
        .unwrap();
    }
    let g = c.find(Value::object()).group("g").agg(AggFn::Count, "");
    assert_eq!(g.len(), 300);
    assert_eq!(
        g.iter()
            .map(|d| d.get("count").unwrap().as_i64().unwrap())
            .sum::<i64>(),
        3000
    );
    for d in &g {
        assert_eq!(d.get("count"), Some(&Value::i64(10)));
    }
    // default group order: numeric key ascending 0..299
    for (i, d) in g.iter().enumerate() {
        assert_eq!(d.get("_id"), Some(&Value::i64(i as i64)));
    }
    // group limit top-5 keeps keys 0..5
    let g = c
        .find(Value::object())
        .group("g")
        .limit(5)
        .agg(AggFn::Count, "");
    assert_eq!(g.len(), 5);
    assert_eq!(g[4].get("_id"), Some(&Value::i64(4)));
}
