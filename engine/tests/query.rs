//! Integration tests: the lazy Mongo-style query API as external users
//! (server, clients) will consume it.
//!
//! Covers the lazy query API: `Collection::find` / `find_one` / `count` /
//! `exists` and the `Query` terminals `.to_list()` / `.first()` / `.count()`,
//! with `{}` = all, top-level direct-value (implicit `$eq`) matching,
//! comparison operators `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte` (incl. range
//! combos), set operators `$in`/`$nin`, and index-accelerated lookups
//! (results identical with and without a field index; index-driven order is
//! deterministic). Logical / element / array operators land in the
//! following subtasks.

use mooracer_engine::{Collection, Query, Value};

/// Build an object from ordered `(key, value)` pairs (no `_id` prepended).
fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::object_from(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

/// A small deterministic dataset: six cows with `name`, `age` (i64), and a
/// `tag` (str). Returns the collection.
fn cow_herd() -> Collection {
    let mut c = Collection::new("cows");
    let rows: &[(&str, i64, &str)] = &[
        ("bess", 2, "milky"),
        ("moo", 5, "loud"),
        ("hilde", 5, "milky"),
        ("daisy", 9, "loud"),
        ("butch", 3, "loud"),
        ("clara", 7, "quiet"),
    ];
    for &(name, age, tag) in rows {
        let doc = obj(&[
            ("_id", Value::str(name)),
            ("name", Value::str(name)),
            ("age", Value::i64(age)),
            ("tag", Value::str(tag)),
        ]);
        c.insert(doc).unwrap();
    }
    c
}

/// Sorted `_id`s of a result list (storage order is unspecified).
fn ids(docs: &[Value]) -> Vec<String> {
    let mut v: Vec<String> = docs
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

// -- {} = all ---------------------------------------------------------------

#[test]
fn empty_filter_selects_everything() {
    let c = cow_herd();
    let all = Value::object();
    assert_eq!(c.find(all.clone()).to_list().len(), 6);
    assert_eq!(c.find(all.clone()).count(), 6);
    assert_eq!(c.count(all.clone()), 6);
    assert!(c.exists(all.clone()));
    assert!(c.find_one(all.clone()).is_some());
}

#[test]
fn empty_filter_on_empty_collection() {
    let c = Collection::new("empty");
    assert!(c.find(Value::object()).to_list().is_empty());
    assert_eq!(c.count(Value::object()), 0);
    assert!(!c.exists(Value::object()));
    assert!(c.find_one(Value::object()).is_none());
}

// -- direct-value (implicit $eq) top-level matching -------------------------

#[test]
fn equality_on_single_field() {
    let c = cow_herd();
    let got = c.find(obj(&[("tag", Value::str("milky"))])).to_list();
    assert_eq!(ids(&got), vec!["bess".to_string(), "hilde".to_string()]);
}

#[test]
fn two_top_level_fields_are_implicit_and() {
    // {age: 5, tag: "loud"} -> only "moo" (hilde is 5 but milky).
    let c = cow_herd();
    let got = c
        .find(obj(&[("age", Value::i64(5)), ("tag", Value::str("loud"))]))
        .to_list();
    assert_eq!(ids(&got), vec!["moo".to_string()]);
}

#[test]
fn id_field_queries_match() {
    let c = cow_herd();
    let got = c.find(obj(&[("_id", Value::str("daisy"))])).to_list();
    assert_eq!(ids(&got), vec!["daisy".to_string()]);
}

#[test]
fn no_match_returns_empty() {
    let c = cow_herd();
    assert!(
        c.find(obj(&[("tag", Value::str("nope"))]))
            .to_list()
            .is_empty()
    );
    assert_eq!(c.count(obj(&[("age", Value::i64(99))])), 0);
    assert!(!c.exists(obj(&[("tag", Value::str("nope"))])));
    assert!(c.find_one(obj(&[("tag", Value::str("nope"))])).is_none());
}

#[test]
fn cross_numeric_equality_matches() {
    // A filter of f64(5.0) must match docs storing i64(5) (total order).
    let c = cow_herd();
    let got = c.find(obj(&[("age", Value::f64(5.0))])).to_list();
    assert_eq!(ids(&got), vec!["hilde".to_string(), "moo".to_string()]);
}

// -- terminals agree with eager entry points --------------------------------

#[test]
fn terminals_and_eager_methods_agree() {
    let c = cow_herd();
    let mk = || obj(&[("tag", Value::str("loud"))]);
    // three loud cows: moo, daisy, butch
    assert_eq!(c.find(mk()).count(), 3);
    assert_eq!(c.count(mk()), 3);
    assert_eq!(c.find(mk()).to_list().len(), 3);
    assert!(c.exists(mk()));
    assert!(c.find(mk()).first().is_some());
    assert!(c.find_one(mk()).is_some());
}

#[test]
fn first_returns_a_full_document() {
    let c = cow_herd();
    let first = c
        .find(obj(&[("_id", Value::str("butch"))]))
        .first()
        .expect("butch exists");
    assert_eq!(first.get("age"), Some(&Value::i64(3)));
    assert_eq!(first.get("tag"), Some(&Value::str("loud")));
    assert_eq!(first.get("_id"), Some(&Value::str("butch")));
}

// -- the Query is a lazy, chainable builder ---------------------------------

#[test]
fn query_borrows_collection_and_defers_scan() {
    let c = cow_herd();
    // Build the query; nothing has run yet. Inspect the filter.
    let q: Query<'_> = c.find(obj(&[("tag", Value::str("quiet"))]));
    assert_eq!(q.filter().get("tag"), Some(&Value::str("quiet")));
    // Terminal runs the single scan and returns owned docs.
    let docs = q.to_list();
    assert_eq!(ids(&docs), vec!["clara".to_string()]);
}

#[test]
fn multiple_terminals_run_independently() {
    let c = cow_herd();
    // Each terminal consumes its own Query; the collection is shared read-only.
    let n = c.find(obj(&[("age", Value::i64(5))])).count();
    let list = c.find(obj(&[("age", Value::i64(5))])).to_list();
    let first = c.find(obj(&[("age", Value::i64(5))])).first();
    assert_eq!(n, 2);
    assert_eq!(list.len(), 2);
    assert!(first.is_some());
}

// -- having a field index present must not change results ------------------

#[test]
fn index_presence_does_not_change_results() {
    // A query on an indexed field routes through the index; the results must
    // be identical to the full-scan answer (the index narrows candidates,
    // every candidate is re-verified against the full filter).
    let mut c = cow_herd();
    let scan = ids(&c.find(obj(&[("age", Value::i64(5))])).to_list());
    c.create_index("age").unwrap();
    let with_index = ids(&c.find(obj(&[("age", Value::i64(5))])).to_list());
    assert_eq!(scan, with_index);
    assert_eq!(with_index, vec!["hilde".to_string(), "moo".to_string()]);
}

// -- comparison operators over the public API ------------------------------

#[test]
fn comparison_operators_public_api() {
    let c = cow_herd();
    // ages: bess 2, butch 3, hilde 5, moo 5, clara 7, daisy 9
    let f = |pairs: &[(&str, Value)]| obj(&[("age", obj(pairs))]);
    assert_eq!(
        ids(&c
            .find(f(&[("$gte", Value::i64(5)), ("$lt", Value::i64(9))]))
            .to_list()),
        vec!["clara".to_string(), "hilde".to_string(), "moo".to_string()]
    );
    assert_eq!(
        ids(&c.find(f(&[("$ne", Value::i64(5))])).to_list()),
        vec![
            "bess".to_string(),
            "butch".to_string(),
            "clara".to_string(),
            "daisy".to_string()
        ]
    );
    // $ne combined with a bound (both verified): clara(7) fails $ne,
    // bess(2)/butch(3) fail $gte, daisy(9) passes both
    assert_eq!(
        ids(&c
            .find(f(&[("$gte", Value::i64(5)), ("$ne", Value::i64(7))]))
            .to_list()),
        vec!["daisy".to_string(), "hilde".to_string(), "moo".to_string()]
    );
}

// -- set operators ($in / $nin) ---------------------------------------------

#[test]
fn set_operators_public_api() {
    let c = cow_herd();
    // ages: bess 2, butch 3, hilde 5, moo 5, clara 7, daisy 9
    let f = |list: Vec<Value>| obj(&[("age", obj(&[("$in", Value::array_from(list))]))]);
    let g = |list: Vec<Value>| obj(&[("age", obj(&[("$nin", Value::array_from(list))]))]);
    assert_eq!(
        ids(&c.find(f(vec![Value::i64(5), Value::i64(9)])).to_list()),
        vec!["daisy".to_string(), "hilde".to_string(), "moo".to_string()]
    );
    // $nin is the exact complement on the same list
    assert_eq!(
        ids(&c.find(g(vec![Value::i64(5), Value::i64(9)])).to_list()),
        vec!["bess".to_string(), "butch".to_string(), "clara".to_string()]
    );
    // list order and duplicates are irrelevant
    assert_eq!(
        ids(&c
            .find(f(vec![
                Value::i64(9),
                Value::i64(2),
                Value::i64(9),
                Value::i64(3)
            ]))
            .to_list()),
        vec!["bess".to_string(), "butch".to_string(), "daisy".to_string()]
    );
    // empty list: $in matches nothing, $nin matches everything
    assert_eq!(c.find(f(Vec::new())).count(), 0);
    assert_eq!(c.find(g(Vec::new())).count(), 6);
    // cross-numeric membership: f64(7.0) matches the stored i64(7)
    assert_eq!(
        ids(&c.find(f(vec![Value::f64(7.0)])).to_list()),
        vec!["clara".to_string()]
    );
    // string fields work too
    let t = |list: Vec<Value>| obj(&[("tag", obj(&[("$in", Value::array_from(list))]))]);
    assert_eq!(
        ids(&c
            .find(t(vec![Value::str("milky"), Value::str("quiet")]))
            .to_list()),
        vec!["bess".to_string(), "clara".to_string(), "hilde".to_string()]
    );
}

#[test]
fn set_operators_combine_with_comparison() {
    let c = cow_herd();
    // $in [5, 9] AND $ne 5 -> only daisy(9)
    let f = obj(&[(
        "age",
        obj(&[
            ("$in", Value::array_from(vec![Value::i64(5), Value::i64(9)])),
            ("$ne", Value::i64(5)),
        ]),
    )]);
    assert_eq!(ids(&c.find(f).to_list()), vec!["daisy".to_string()]);
    // $in [5, 7, 9] AND $gte 7 -> clara(7), daisy(9)
    let f = obj(&[(
        "age",
        obj(&[
            (
                "$in",
                Value::array_from(vec![Value::i64(5), Value::i64(7), Value::i64(9)]),
            ),
            ("$gte", Value::i64(7)),
        ]),
    )]);
    assert_eq!(
        ids(&c.find(f).to_list()),
        vec!["clara".to_string(), "daisy".to_string()]
    );
    // top-level AND: $in on age combined with a direct value on tag
    let f = obj(&[
        (
            "age",
            obj(&[("$in", Value::array_from(vec![Value::i64(5), Value::i64(3)]))]),
        ),
        ("tag", Value::str("loud")),
    ]);
    // moo (5, loud) and butch (3, loud); hilde (5, milky) is excluded
    assert_eq!(
        ids(&c.find(f).to_list()),
        vec!["butch".to_string(), "moo".to_string()]
    );
}

// -- index-accelerated lookups ----------------------------------------------

/// `_id`s in result order (index order for index-driven queries).
fn ordered_ids(docs: &[Value]) -> Vec<String> {
    docs.iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect()
}

/// A dataset for the index + missing-field interaction: `score` is present
/// on five docs (two of them the cross-numeric pair i64(5) / f64(5.0)),
/// explicitly `null` on one, and absent on one.
fn score_herd() -> Collection {
    let mut c = Collection::new("scores");
    let rows: &[(&str, Option<Value>)] = &[
        ("s01", Some(Value::i64(3))),
        ("s02", Some(Value::i64(7))),
        ("s03", Some(Value::f64(5.0))),
        ("s04", Some(Value::i64(5))),
        ("s05", Some(Value::i64(11))),
        ("s06", Some(Value::Null)), // explicit null
        ("s07", None),              // field absent
    ];
    for (id, score) in rows {
        let mut pairs: Vec<(String, Value)> = vec![("_id".to_string(), Value::str(*id))];
        if let Some(s) = score {
            pairs.push(("score".to_string(), s.clone()));
        }
        c.insert(Value::object_from(pairs)).unwrap();
    }
    c
}

#[test]
fn index_driven_results_match_full_scan() {
    let mut c = score_herd();
    let filters: Vec<Value> = vec![
        obj(&[("score", Value::i64(5))]), // point; cross-numeric (s03 f64 + s04 i64)
        obj(&[("score", obj(&[("$gt", Value::i64(5))]))]),
        obj(&[("score", obj(&[("$gte", Value::f64(5.0))]))]),
        obj(&[("score", obj(&[("$lt", Value::f64(5.0))]))]),
        obj(&[("score", obj(&[("$lte", Value::i64(3))]))]),
        obj(&[(
            "score",
            obj(&[("$gte", Value::i64(5)), ("$lt", Value::i64(11))]),
        )]),
        obj(&[("score", obj(&[("$ne", Value::i64(7))]))]), // bare $ne: not a driver
        obj(&[("score", obj(&[("$ne", Value::Null)]))]),
        obj(&[("score", obj(&[("$eq", Value::Null)]))]),
        obj(&[("score", Value::Null)]),
        obj(&[("score", obj(&[("$gt", Value::Null)]))]),
        obj(&[(
            "_id",
            obj(&[("$gte", Value::str("s03")), ("$lt", Value::str("s06"))]),
        )]), // primary range
        obj(&[("_id", Value::str("s05"))]),
        obj(&[
            ("score", obj(&[("$lt", Value::i64(8))])),
            ("_id", Value::str("s05")),
        ]), // multi-condition: index drives, rest verifies
        obj(&[
            ("score", obj(&[("$ne", Value::i64(7))])),
            ("_id", obj(&[("$lt", Value::str("s05"))])),
        ]), // first condition not indexable -> next one drives
    ];
    // Baseline before the `score` index exists.
    let scan: Vec<Vec<String>> = filters
        .iter()
        .map(|f| ids(&c.find(f.clone()).to_list()))
        .collect();
    c.create_index("score").unwrap();
    for (i, f) in filters.iter().enumerate() {
        let got = ids(&c.find(f.clone()).to_list());
        assert_eq!(
            got, scan[i],
            "filter {i} with a `score` index must equal the scan result"
        );
    }
}

#[test]
fn index_driven_missing_and_null_entries() {
    let mut c = score_herd();
    c.create_index("score").unwrap();
    // missing and explicit null are indexed as Null; comparison operators
    // must not match the absent doc, and $eq/$ne handle the Null slice:
    assert_eq!(
        ids(&c
            .find(obj(&[("score", obj(&[("$gt", Value::i64(5))]))]))
            .to_list()),
        vec!["s02".to_string(), "s05".to_string()],
        "$gt: only present values above the bound"
    );
    assert_eq!(
        ids(&c
            .find(obj(&[("score", obj(&[("$ne", Value::i64(7))]))]))
            .to_list()),
        vec![
            "s01".to_string(),
            "s03".to_string(),
            "s04".to_string(),
            "s05".to_string(),
            "s06".to_string(),
            "s07".to_string()
        ],
        "$ne: matches explicit null AND the missing doc"
    );
    assert_eq!(
        ids(&c
            .find(obj(&[("score", obj(&[("$eq", Value::Null)]))]))
            .to_list()),
        vec!["s06".to_string(), "s07".to_string()],
        "$eq null: explicit null AND missing"
    );
    assert_eq!(
        ids(&c
            .find(obj(&[("score", obj(&[("$ne", Value::Null)]))]))
            .to_list()),
        vec![
            "s01".to_string(),
            "s02".to_string(),
            "s03".to_string(),
            "s04".to_string(),
            "s05".to_string()
        ],
        "$ne null: only present non-null docs"
    );
    // cross-numeric through the index: i64(5) and f64(5.0) are one slice
    assert_eq!(
        ids(&c.find(obj(&[("score", Value::i64(5))])).to_list()),
        vec!["s03".to_string(), "s04".to_string()]
    );
}

#[test]
fn index_driven_results_come_in_index_order() {
    let mut c = score_herd();
    c.create_index("score").unwrap();
    // an unbounded-from-below range over the index covers every entry; the
    // result comes back in index order (value ascending per the total order,
    // ties by `_id`): null(s06) < 3(s01) < 5(s03, s04) < 7(s02) < 11(s05).
    // The absent doc (s07) has a Null entry too but fails verification
    // ($gte requires presence) and is dropped.
    let f = obj(&[("score", obj(&[("$gte", Value::Null)]))]);
    assert_eq!(
        ordered_ids(&c.find(f).to_list()),
        vec!["s06", "s01", "s03", "s04", "s02", "s05"]
    );
    // first() on the index path is the smallest value, not a storage-order doc
    let first = c
        .find(obj(&[("score", obj(&[("$gt", Value::Null)]))]))
        .first()
        .expect("non-null scores exist");
    assert_eq!(first.get("_id"), Some(&Value::str("s01")));
}

#[test]
fn primary_id_index_drives_range_and_point() {
    let c = cow_herd();
    // no create_index needed: `_id` is always indexed
    let range = obj(&[("$gte", Value::str("bess")), ("$lt", Value::str("daisy"))]);
    let got = c.find(obj(&[("_id", range)])).to_list();
    assert_eq!(
        ids(&got),
        vec!["bess".to_string(), "butch".to_string(), "clara".to_string()]
    );
    assert_eq!(
        ids(&c
            .find(obj(&[("_id", obj(&[("$ne", Value::str("moo"))]))]))
            .to_list()),
        vec![
            "bess".to_string(),
            "butch".to_string(),
            "clara".to_string(),
            "daisy".to_string(),
            "hilde".to_string()
        ]
    );
}

#[test]
fn in_nin_index_drives_and_matches_scan() {
    let mut c = score_herd();
    let filters: Vec<Value> = vec![
        obj(&[(
            "score",
            obj(&[(
                "$in",
                Value::array_from(vec![Value::i64(7), Value::i64(3), Value::i64(7)]),
            )]),
        )]),
        obj(&[(
            "score",
            obj(&[(
                "$in",
                Value::array_from(vec![Value::f64(5.0), Value::i64(11)]),
            )]),
        )]), // cross-numeric point set
        obj(&[(
            "score",
            obj(&[("$in", Value::array_from(vec![Value::Null]))]),
        )]), // Null slice: explicit null + missing
        obj(&[("score", obj(&[("$in", Value::array_from(Vec::new()))]))]), // empty list: nothing, no scan
        obj(&[(
            "score",
            obj(&[
                ("$in", Value::array_from(vec![Value::i64(5), Value::i64(7)])),
                ("$ne", Value::i64(5)), // cross-numeric: excludes f64(5.0) too
            ]),
        )]),
        obj(&[(
            "score",
            obj(&[(
                "$nin",
                Value::array_from(vec![Value::i64(3), Value::i64(7)]),
            )]),
        )]), // bare $nin: plain scan
        obj(&[(
            "score",
            obj(&[("$nin", Value::array_from(vec![Value::Null]))]),
        )]), // only present non-null
        obj(&[(
            "score",
            obj(&[
                ("$nin", Value::array_from(vec![Value::i64(11)])),
                ("$gte", Value::i64(3)), // the bound drives, $nin verifies
            ]),
        )]),
        obj(&[("score", obj(&[("$in", Value::i64(5))]))]), // non-array operand: matches nothing
    ];
    // Baseline before the `score` index exists.
    let scan: Vec<Vec<String>> = filters
        .iter()
        .map(|f| ids(&c.find(f.clone()).to_list()))
        .collect();
    c.create_index("score").unwrap();
    for (i, f) in filters.iter().enumerate() {
        let got = ids(&c.find(f.clone()).to_list());
        assert_eq!(
            got, scan[i],
            "filter {i} with a `score` index must equal the scan result"
        );
    }
    // Spot-check the expected answers (derived from the fixture):
    assert_eq!(scan[0], vec!["s01".to_string(), "s02".to_string()]); // 3, 7 (dups ignored)
    assert_eq!(
        scan[1],
        vec!["s03".to_string(), "s04".to_string(), "s05".to_string()]
    ); // f64(5.0)+i64(5) is one slice, plus 11
    assert_eq!(scan[2], vec!["s06".to_string(), "s07".to_string()]); // explicit null + missing
    assert_eq!(scan[3], Vec::<String>::new());
    assert_eq!(scan[4], vec!["s02".to_string()]); // 7 only (5/5.0 excluded by $ne)
    assert_eq!(
        scan[5],
        vec![
            "s03".to_string(),
            "s04".to_string(),
            "s05".to_string(),
            "s06".to_string(),
            "s07".to_string()
        ]
    ); // null and missing are NOT in [3, 7]
    assert_eq!(
        scan[6],
        vec![
            "s01".to_string(),
            "s02".to_string(),
            "s03".to_string(),
            "s04".to_string(),
            "s05".to_string()
        ]
    );
    assert_eq!(
        scan[7],
        vec![
            "s01".to_string(),
            "s02".to_string(),
            "s03".to_string(),
            "s04".to_string()
        ]
    ); // >= 3 minus 11
    assert_eq!(scan[8], Vec::<String>::new());
}

#[test]
fn in_index_results_come_in_index_order() {
    let mut c = score_herd();
    c.create_index("score").unwrap();
    // list order (11, 3) must not leak: the point set is walked in
    // total-order ascending, so the result is in index order: 3 (s01) then
    // 11 (s05).
    let f = obj(&[(
        "score",
        obj(&[(
            "$in",
            Value::array_from(vec![Value::i64(11), Value::i64(3)]),
        )]),
    )]);
    assert_eq!(
        ordered_ids(&c.find(f).to_list()),
        vec!["s01".to_string(), "s05".to_string()]
    );
    // a null in the list lands on the Null slice, which is walked first
    // (Null is the minimum rank); ties inside the slice by `_id`.
    let f = obj(&[(
        "score",
        obj(&[(
            "$in",
            Value::array_from(vec![Value::i64(11), Value::Null, Value::i64(3)]),
        )]),
    )]);
    assert_eq!(
        ordered_ids(&c.find(f).to_list()),
        vec![
            "s06".to_string(),
            "s07".to_string(),
            "s01".to_string(),
            "s05".to_string()
        ]
    );
}

// -- element operator ($exists) ---------------------------------------------

/// A collection where `opt` is present (non-null) on two docs, explicitly
/// `null` on one, and absent on one — the `$exists` presence matrix as
/// external users (server/clients) will hit it.
fn opt_herd() -> Collection {
    let mut c = Collection::new("opts");
    let rows: &[(&str, Option<Value>)] = &[
        ("a", Some(Value::i64(1))),
        ("b", Some(Value::i64(5))),
        ("c", Some(Value::Null)), // explicit null
        ("d", None),              // field absent
    ];
    for (id, opt) in rows {
        let mut pairs: Vec<(String, Value)> = vec![("_id".to_string(), Value::str(*id))];
        if let Some(v) = opt {
            pairs.push(("opt".to_string(), v.clone()));
        }
        c.insert(Value::object_from(pairs)).unwrap();
    }
    c
}

#[test]
fn exists_public_api_true_and_false() {
    let c = opt_herd();
    let t = obj(&[("opt", obj(&[("$exists", Value::bool(true))]))]);
    let f = obj(&[("opt", obj(&[("$exists", Value::bool(false))]))]);
    // present (incl. explicit null): a, b, c
    assert_eq!(
        ids(&c.find(t.clone()).to_list()),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
    // absent only: d (the explicit-null doc c is present, so excluded)
    assert_eq!(ids(&c.find(f.clone()).to_list()), vec!["d".to_string()]);
    // eager entry points agree with the lazy terminals
    assert_eq!(c.count(t.clone()), 3);
    assert_eq!(c.count(f.clone()), 1);
    assert!(c.exists(t.clone()));
    assert!(c.exists(f.clone()));
    assert!(c.find_one(t).is_some());
    assert!(c.find_one(f).is_some());
}

#[test]
fn exists_non_boolean_operand_matches_nothing_public_api() {
    let c = opt_herd();
    for bad in [Value::i64(1), Value::str("true"), Value::Null] {
        let f = obj(&[("opt", obj(&[("$exists", bad.clone())]))]);
        assert_eq!(c.find(f.clone()).count(), 0);
        assert!(!c.exists(f.clone()));
        assert!(c.find_one(f).is_none());
    }
}

#[test]
fn exists_combines_with_other_conditions_public_api() {
    let c = opt_herd();
    // {opt: {$exists: true, $gte: 5}} -> only b (opt 5); a is 1, c is null
    let f = obj(&[(
        "opt",
        obj(&[("$exists", Value::bool(true)), ("$gte", Value::i64(5))]),
    )]);
    assert_eq!(ids(&c.find(f).to_list()), vec!["b".to_string()]);
    // {opt: {$exists: false, $gte: 5}}: absent vs present is a contradiction
    let f = obj(&[(
        "opt",
        obj(&[("$exists", Value::bool(false)), ("$gte", Value::i64(5))]),
    )]);
    assert_eq!(c.find(f).count(), 0);
}

#[test]
fn exists_results_identical_with_or_without_an_index() {
    let mut c = opt_herd();
    let t = obj(&[("opt", obj(&[("$exists", Value::bool(true))]))]);
    let f = obj(&[("opt", obj(&[("$exists", Value::bool(false))]))]);
    let scan_t = ids(&c.find(t.clone()).to_list());
    let scan_f = ids(&c.find(f.clone()).to_list());
    c.create_index("opt").unwrap();
    // $exists never drives; the (full) scan verifies presence identically
    assert_eq!(ids(&c.find(t.clone()).to_list()), scan_t);
    assert_eq!(ids(&c.find(f).to_list()), scan_f);
}

// -- array operator ($elemMatch) ---------------------------------------------

/// A collection with a `sizes` array (i64) on three docs, a scalar (non-array)
/// `sizes` on one, and a missing `sizes` on one — plus an `instock` array of
/// subdocuments on two docs.
fn elem_herd() -> Collection {
    let mut c = Collection::new("elem");
    let sizes: &[(&str, Option<Value>)] = &[
        (
            "a",
            Some(Value::array_from(vec![Value::i64(1), Value::i64(4)])),
        ),
        (
            "b",
            Some(Value::array_from(vec![Value::i64(5), Value::i64(9)])),
        ),
        (
            "c",
            Some(Value::array_from(vec![Value::i64(7), Value::i64(7)])),
        ),
        ("d", Some(Value::i64(5))), // non-array scalar
        ("e", None),                // missing
    ];
    for (id, v) in sizes {
        let mut pairs = vec![("_id".to_string(), Value::str(*id))];
        if let Some(x) = v {
            pairs.push(("sizes".to_string(), x.clone()));
        }
        c.insert(Value::object_from(pairs)).unwrap();
    }
    let sd1 = Value::array_from(vec![
        obj(&[("qty", Value::i64(3)), ("warehouse", Value::str("A"))]),
        obj(&[("qty", Value::i64(8)), ("warehouse", Value::str("A"))]),
    ]);
    let sd2 = Value::array_from(vec![obj(&[
        ("qty", Value::i64(10)),
        ("warehouse", Value::str("B")),
    ])]);
    c.insert(obj(&[("_id", Value::str("sd1")), ("instock", sd1)]))
        .unwrap();
    c.insert(obj(&[("_id", Value::str("sd2")), ("instock", sd2)]))
        .unwrap();
    c
}

fn em(field: &str, criteria: Value) -> Value {
    obj(&[(field, obj(&[("$elemMatch", criteria)]))])
}

#[test]
fn elem_match_direct_value_element_equality() {
    let c = elem_herd();
    assert_eq!(
        ids(&c.find(em("sizes", Value::i64(4))).to_list()),
        vec!["a".to_string()]
    );
    assert_eq!(
        ids(&c.find(em("sizes", Value::i64(9))).to_list()),
        vec!["b".to_string()]
    );
    // the scalar `sizes` (d) is not an array, so a direct value 5 matches only b
    assert_eq!(
        ids(&c.find(em("sizes", Value::i64(5))).to_list()),
        vec!["b".to_string()]
    );
    // missing / non-matching
    assert_eq!(c.find(em("sizes", Value::i64(99))).count(), 0);
    // eager entry points agree
    assert_eq!(c.count(em("sizes", Value::i64(7))), 1);
    assert!(c.exists(em("sizes", Value::i64(4))));
    assert!(c.find_one(em("sizes", Value::i64(4))).is_some());
}

#[test]
fn elem_match_comparison_operators() {
    let c = elem_herd();
    // an element > 4: b (5,9) and c (7,7); d (scalar 5) is not an array
    assert_eq!(
        ids(&c
            .find(em("sizes", obj(&[("$gt", Value::i64(4))])))
            .to_list()),
        vec!["b".to_string(), "c".to_string()]
    );
    // an element in [7, 8): only c
    assert_eq!(
        ids(&c
            .find(em(
                "sizes",
                obj(&[("$gte", Value::i64(7)), ("$lt", Value::i64(8))])
            ))
            .to_list()),
        vec!["c".to_string()]
    );
    // cross-numeric: an element >= f64(8.0): only b (9)
    assert_eq!(
        ids(&c
            .find(em("sizes", obj(&[("$gte", Value::f64(8.0))])))
            .to_list()),
        vec!["b".to_string()]
    );
    // $in over elements: an element in {1, 7}: a (1) and c (7)
    assert_eq!(
        ids(&c
            .find(em(
                "sizes",
                obj(&[("$in", Value::array_from(vec![Value::i64(1), Value::i64(7)]))])
            ))
            .to_list()),
        vec!["a".to_string(), "c".to_string()]
    );
}

#[test]
fn elem_match_subdocument_filter() {
    let c = elem_herd();
    // an element with qty > 5 and warehouse A -> only sd1 (qty 8, wh A)
    let sub = obj(&[
        ("qty", obj(&[("$gt", Value::i64(5))])),
        ("warehouse", Value::str("A")),
    ]);
    assert_eq!(
        ids(&c.find(em("instock", sub)).to_list()),
        vec!["sd1".to_string()]
    );
    // exact subdocument on one element -> sd1
    let sub = obj(&[("qty", Value::i64(3)), ("warehouse", Value::str("A"))]);
    assert_eq!(
        ids(&c.find(em("instock", sub)).to_list()),
        vec!["sd1".to_string()]
    );
    // warehouse B -> sd2
    assert_eq!(
        ids(&c
            .find(em("instock", obj(&[("warehouse", Value::str("B"))])))
            .to_list()),
        vec!["sd2".to_string()]
    );
    // a subdocument filter that matches no element
    assert_eq!(
        c.find(em("instock", obj(&[("warehouse", Value::str("Z"))])))
            .count(),
        0
    );
    // a subdocument filter applied to an array of scalars: no match
    assert_eq!(
        c.find(em("sizes", obj(&[("qty", Value::i64(1))]))).count(),
        0
    );
}

#[test]
fn elem_match_missing_and_non_array_match_nothing() {
    let c = elem_herd();
    // no doc has a `missing` field -> $elemMatch matches nothing for any operand
    assert_eq!(c.find(em("missing", Value::i64(5))).count(), 0);
    assert_eq!(
        c.find(em("missing", obj(&[("$gt", Value::i64(0))])))
            .count(),
        0
    );
    assert_eq!(
        c.find(em("missing", obj(&[("x", Value::i64(1))]))).count(),
        0
    );
    // the scalar `sizes` (d) and the missing `sizes` (e) never match
    let f = em("sizes", obj(&[("$gt", Value::i64(0))]));
    assert_eq!(
        ids(&c.find(f).to_list()),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn elem_match_combines_with_other_conditions() {
    let c = elem_herd();
    // $elemMatch AND a direct condition on a separate field: the `sizes`
    // arrays are only on a/b/c; AND with a doc field that only some share.
    // Build: {sizes: {$elemMatch: {$gt: 4}}, _id: "c"} -> only c
    let f = obj(&[
        (
            "sizes",
            obj(&[("$elemMatch", obj(&[("$gt", Value::i64(4))]))]),
        ),
        ("_id", Value::str("c")),
    ]);
    assert_eq!(ids(&c.find(f).to_list()), vec!["c".to_string()]);
    // $or of two $elemMatch on different arrays
    let f = obj(&[(
        "$or",
        Value::array_from(vec![
            em("sizes", Value::i64(4)),
            em("instock", obj(&[("warehouse", Value::str("B"))])),
        ]),
    )]);
    // sizes has 4 -> a; instock has warehouse B -> sd2
    assert_eq!(
        ids(&c.find(f).to_list()),
        vec!["a".to_string(), "sd2".to_string()]
    );
}

// -- sort / skip / limit pipeline ------------------------------------------

#[test]
fn sort_public_api_ascending_and_descending() {
    let c = cow_herd();
    // ages: bess 2, butch 3, hilde 5, moo 5, clara 7, daisy 9; the 5-tie
    // (hilde/moo) is broken by `_id` — and REVERSED by descending
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
}

#[test]
fn sort_with_filter_and_limit_is_pipeline_order() {
    let c = cow_herd();
    // loud cows: moo 5, daisy 9, butch 3 — filter, THEN sort, THEN limit
    let f = obj(&[("tag", Value::str("loud"))]);
    let top: Vec<String> = c
        .find(f.clone())
        .sort("age", true)
        .limit(2)
        .to_list()
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(top, vec!["daisy".to_string(), "moo".to_string()]);
    // first() respects the pipeline: skip the top, take the next
    let second = c
        .find(f)
        .sort("age", true)
        .skip(1)
        .first()
        .expect("a match exists");
    assert_eq!(second.get("_id"), Some(&Value::str("moo")));
}

#[test]
fn sort_by_unindexed_field_matches_indexed_sort() {
    let mut c = cow_herd();
    // `tag` is not indexed: the sort collects + sorts; the result must be
    // identical when an index happens to exist
    let asc_scan: Vec<String> = c
        .find(Value::object())
        .sort("tag", false)
        .to_list()
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    // byte order: "loud" < "milky" < "quiet"; ties inside a tag by `_id`
    assert_eq!(
        asc_scan,
        vec![
            "butch".to_string(),
            "daisy".to_string(),
            "moo".to_string(),
            "bess".to_string(),
            "hilde".to_string(),
            "clara".to_string()
        ]
    );
    c.create_index("tag").unwrap();
    let asc_idx: Vec<String> = c
        .find(Value::object())
        .sort("tag", false)
        .to_list()
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(asc_idx, asc_scan, "indexed vs unindexed sort must agree");
}

#[test]
fn skip_limit_public_api_all_terminals() {
    let c = cow_herd();
    let f = obj(&[("age", obj(&[("$gte", Value::i64(5))]))]); // hilde, moo, clara, daisy
    // count honors skip/limit; limit(0) = no limit
    assert_eq!(c.find(f.clone()).sort("age", false).count(), 4);
    assert_eq!(
        c.find(f.clone())
            .sort("age", false)
            .skip(2)
            .limit(1)
            .count(),
        1
    );
    assert_eq!(c.find(f.clone()).sort("age", false).limit(0).count(), 4);
    assert_eq!(c.find(f.clone()).skip(4).count(), 0);
    // to_list honors the pipeline in sorted order
    let got = c
        .find(f.clone())
        .sort("age", false)
        .skip(1)
        .limit(2)
        .to_list();
    assert_eq!(
        ordered_ids(&got),
        vec!["moo".to_string(), "clara".to_string()]
    );
    // skip past the end: empty
    assert!(c.find(f).sort("age", false).skip(4).to_list().is_empty());
}

#[test]
fn sort_with_null_and_missing_scores_public_api() {
    let mut c = score_herd();
    c.create_index("score").unwrap();
    // Null (s06 explicit, s07 missing) sorts first in ascending, last in
    // descending; the cross-numeric 5-tie (f64 5.0 / i64 5) breaks by `_id`
    let asc = ordered_ids(&c.find(Value::object()).sort("score", false).to_list());
    assert_eq!(
        asc,
        vec![
            "s06".to_string(),
            "s07".to_string(),
            "s01".to_string(),
            "s03".to_string(),
            "s04".to_string(),
            "s02".to_string(),
            "s05".to_string()
        ]
    );
    let desc = ordered_ids(&c.find(Value::object()).sort("score", true).to_list());
    assert_eq!(
        desc,
        vec![
            "s05".to_string(),
            "s02".to_string(),
            "s04".to_string(),
            "s03".to_string(),
            "s01".to_string(),
            "s07".to_string(),
            "s06".to_string()
        ]
    );
    // skip/limit on the sorted stream: drop the two Null docs, take two
    let page = ordered_ids(
        &c.find(Value::object())
            .sort("score", false)
            .skip(2)
            .limit(2)
            .to_list(),
    );
    assert_eq!(page, vec!["s01".to_string(), "s03".to_string()]);
}

#[test]
fn sort_limit_is_deterministic_at_scale() {
    // 500 docs with values 0..500 (id encodes the value); sort + limit must
    // return the exact top-N in descending value order (ties: none).
    let mut c = Collection::new("scale");
    for i in 0..500u32 {
        c.insert(obj(&[
            ("_id", Value::str(format!("id-{i:04}"))),
            ("v", Value::i64(i as i64)),
        ]))
        .unwrap();
    }
    c.create_index("v").unwrap();
    let got: Vec<String> = c
        .find(Value::object())
        .sort("v", true)
        .limit(10)
        .to_list()
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    let expected: Vec<String> = (490..500).rev().map(|i| format!("id-{i:04}")).collect();
    assert_eq!(got, expected, "top-10 descending");
    // and the same query unindexed agrees (different engine, same contract)
    let mut c2 = Collection::new("scale2");
    for i in 0..500u32 {
        c2.insert(obj(&[
            ("_id", Value::str(format!("id-{i:04}"))),
            ("v", Value::i64(i as i64)),
        ]))
        .unwrap();
    }
    let got2: Vec<String> = c2
        .find(Value::object())
        .sort("v", true)
        .limit(10)
        .to_list()
        .iter()
        .map(|d| d.get("_id").unwrap().as_str().unwrap().to_string())
        .collect();
    assert_eq!(got2, expected, "unindexed sort must match the indexed sort");
    // first() is the single top doc
    let top = c.find(Value::object()).sort("v", true).first().unwrap();
    assert_eq!(top.get("v"), Some(&Value::i64(499)));
}

#[test]
fn elem_match_results_identical_with_or_without_an_index() {
    let mut c = elem_herd();
    let filters: Vec<Value> = vec![
        em("sizes", Value::i64(4)),
        em("sizes", obj(&[("$gt", Value::i64(4))])),
        em(
            "sizes",
            obj(&[("$gte", Value::i64(7)), ("$lt", Value::i64(8))]),
        ),
        em(
            "sizes",
            obj(&[("$in", Value::array_from(vec![Value::i64(1), Value::i64(7)]))]),
        ),
        em("missing", Value::i64(1)),
    ];
    let scan: Vec<Vec<String>> = filters
        .iter()
        .map(|f| ids(&c.find(f.clone()).to_list()))
        .collect();
    c.create_index("sizes").unwrap();
    for (i, f) in filters.iter().enumerate() {
        let got = ids(&c.find(f.clone()).to_list());
        assert_eq!(
            got, scan[i],
            "filter {i} with a `sizes` index must equal the scan result"
        );
    }
    // spot-checks (derived from the fixture)
    assert_eq!(scan[0], vec!["a".to_string()]);
    assert_eq!(scan[1], vec!["b".to_string(), "c".to_string()]);
    assert_eq!(scan[2], vec!["c".to_string()]);
    assert_eq!(scan[3], vec!["a".to_string(), "c".to_string()]);
    assert_eq!(scan[4], Vec::<String>::new());
}
