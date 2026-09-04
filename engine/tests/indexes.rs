//! Integration tests: the index layer of `Collection` (primary `_id`
//! index, field indexes, equality + range scans, index maintenance on
//! removal / refresh) as an external user (server, query layer) sees it.

use mooracer_engine::{Collection, StoreError, Value};
use std::ops::Bound::*;

fn doc(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

fn id_doc(id: &str, extra: &[(&str, Value)]) -> Value {
    let mut pairs: Vec<(String, Value)> = vec![("_id".to_string(), Value::str(id))];
    pairs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    Value::object_from(pairs)
}

#[test]
fn primary_id_index_always_present() {
    let mut c = Collection::new("t");
    assert_eq!(c.index_names(), vec!["_id".to_string()]);
    assert!(c.index("_id").is_some());
    assert!(c.index("age").is_none());
    assert!(c.index("_id").unwrap().is_empty());

    c.insert(doc(&[("age", Value::i64(1))])).unwrap();
    assert_eq!(c.index_names(), vec!["_id".to_string()]);
}

#[test]
fn insert_registers_primary_entries() {
    let mut c = Collection::new("t");
    let id1 = c.insert(doc(&[("a", Value::i64(1))])).unwrap();
    let id2 = c.insert(doc(&[("a", Value::i64(2))])).unwrap();
    let ix = c.index("_id").unwrap();
    assert_eq!(ix.len(), 2);
    assert_eq!(ix.ids_equal(&Value::str(&id1)), vec![id1.as_str()]);
    assert_eq!(ix.ids_equal(&Value::str(&id2)), vec![id2.as_str()]);
}

#[test]
fn field_index_backfills_existing_docs() {
    let mut c = Collection::new("t");
    c.insert(id_doc("d1", &[("age", Value::i64(30))])).unwrap();
    c.insert(id_doc("d2", &[("age", Value::i64(20))])).unwrap();
    c.insert(id_doc("d3", &[])).unwrap(); // no "age"

    c.create_index("age").unwrap();
    assert_eq!(c.index_names(), vec!["_id".to_string(), "age".to_string()]);
    let ix = c.index("age").unwrap();
    assert_eq!(ix.field(), "age");
    assert_eq!(ix.len(), 3, "every doc gets an entry, missing → Null");
    // ascending value order, ties by id
    assert_eq!(ix.ids_range(Unbounded, Unbounded), vec!["d3", "d2", "d1"]);
    assert_eq!(ix.ids_equal(&Value::i64(20)), vec!["d2"]);
    assert_eq!(ix.ids_equal(&Value::Null), vec!["d3"]);
}

#[test]
fn field_index_created_empty_registers_on_insert() {
    let mut c = Collection::new("t");
    c.create_index("age").unwrap();
    assert!(c.index("age").unwrap().is_empty());
    c.insert(id_doc("x", &[("age", Value::i64(5))])).unwrap();
    assert_eq!(c.index("age").unwrap().ids_equal(&Value::i64(5)), vec!["x"]);
}

#[test]
fn create_index_id_is_noop() {
    let c = &mut Collection::new("t");
    c.create_index("_id").unwrap();
    assert_eq!(c.index_names(), vec!["_id".to_string()]);
}

#[test]
fn create_index_rebuilds_deterministically() {
    let mut c = Collection::new("t");
    c.insert(id_doc("b", &[("v", Value::i64(2))])).unwrap();
    c.insert(id_doc("a", &[("v", Value::i64(1))])).unwrap();
    c.create_index("v").unwrap();
    // change the world, then rebuild
    c.set_doc("b", doc(&[("v", Value::i64(9))])).unwrap();
    c.create_index("v").unwrap();
    let ix = c.index("v").unwrap();
    assert_eq!(ix.ids_range(Unbounded, Unbounded), vec!["a", "b"]);
    assert_eq!(
        ix.ids_equal(&Value::i64(9)),
        vec!["b"],
        "rebuild reflects current docs"
    );
    assert_eq!(ix.len(), 2);
}

#[test]
fn drop_index_rules() {
    let mut c = Collection::new("t");
    c.insert(id_doc("d", &[("age", Value::i64(1))])).unwrap();
    assert_eq!(c.drop_index("_id"), Err(StoreError::PrimaryIndex));
    assert_eq!(
        c.drop_index("nope"),
        Err(StoreError::NoIndex("nope".into()))
    );
    c.create_index("age").unwrap();
    c.create_index("name").unwrap();
    c.drop_index("age").unwrap();
    assert_eq!(c.index_names(), vec!["_id".to_string(), "name".to_string()]);
    assert!(c.index("age").is_none());
}

#[test]
fn insert_many_registers_batch_entries() {
    let mut c = Collection::new("t");
    c.create_index("k").unwrap();
    c.insert_many([
        id_doc("m1", &[("k", Value::i64(1))]),
        id_doc("m2", &[("k", Value::i64(3))]),
        id_doc("m3", &[]), // missing → Null entry
    ])
    .unwrap();
    assert_eq!(c.index("k").unwrap().len(), 3);
    assert_eq!(
        c.index("k").unwrap().ids_range(Unbounded, Unbounded),
        vec!["m3", "m1", "m2"]
    );
    // a rejected batch must not leave index entries behind
    let err = c
        .insert_many([id_doc("m2", &[]), id_doc("fresh", &[])])
        .unwrap_err();
    assert_eq!(err, StoreError::DuplicateId("m2".into()));
    assert_eq!(
        c.index("k").unwrap().len(),
        3,
        "atomic batch: indexes untouched"
    );
}

// -- range scans ---------------------------------------------------------------

#[test]
fn range_scan_over_collection_index() {
    let mut c = Collection::new("t");
    c.create_index("age").unwrap();
    for (id, age) in [
        ("a", 10),
        ("b", 25),
        ("c", 30),
        ("d", 35),
        ("e", 40),
        ("f", 55),
    ] {
        c.insert(id_doc(id, &[("age", Value::i64(age))])).unwrap();
    }
    let ix = c.index("age").unwrap();
    assert_eq!(
        ix.ids_range(Included(&Value::i64(25)), Excluded(&Value::i64(40))),
        vec!["b", "c", "d"]
    );
    assert_eq!(
        ix.ids_range(Excluded(&Value::i64(0)), Unbounded),
        vec!["a", "b", "c", "d", "e", "f"]
    );
    assert_eq!(
        ix.ids_range(Included(&Value::i64(50)), Unbounded),
        vec!["f"]
    );
    assert!(
        ix.ids_range(Included(&Value::i64(41)), Included(&Value::i64(54)))
            .is_empty()
    );
}

#[test]
fn cross_numeric_index_sees_1_and_1_0_as_equal() {
    let mut c = Collection::new("t");
    c.create_index("n").unwrap();
    c.insert(id_doc("i", &[("n", Value::i64(1))])).unwrap();
    c.insert(id_doc("f", &[("n", Value::f64(1.0))])).unwrap();
    let ix = c.index("n").unwrap();
    // engine total order: I64(1) == F64(1.0), both in one slice, id order
    assert_eq!(ix.ids_equal(&Value::i64(1)), vec!["f", "i"]);
    assert_eq!(ix.ids_equal(&Value::f64(1.0)), vec!["f", "i"]);
    assert_eq!(ix.distinct(), 1);
    // exactness: 1 must NOT match 1.5
    assert!(ix.ids_equal(&Value::f64(1.5)).is_empty());
}

// -- removal ---------------------------------------------------------------------

#[test]
fn remove_doc_cleans_all_indexes() {
    let mut c = Collection::new("t");
    c.create_index("age").unwrap();
    c.insert(id_doc("keep", &[("age", Value::i64(1))])).unwrap();
    c.insert(id_doc("gone", &[("age", Value::i64(1))])).unwrap();
    c.insert(id_doc("noage", &[])).unwrap();

    let removed = c.remove_doc("gone").expect("doc removed");
    assert_eq!(removed.get("_id"), Some(&Value::str("gone")));
    assert!(!c.contains("gone"));
    assert_eq!(c.len(), 2);

    let age = c.index("age").unwrap();
    assert_eq!(age.len(), 2, "one entry per remaining doc");
    assert_eq!(age.ids_equal(&Value::i64(1)), vec!["keep"]);
    assert_eq!(age.ids_equal(&Value::Null), vec!["noage"]);
    assert_eq!(
        c.index("_id")
            .unwrap()
            .ids_range(Unbounded, Unbounded)
            .len(),
        2,
        "primary index shrank too"
    );
    assert_eq!(c.remove_doc("gone"), None, "second remove finds nothing");
    assert_eq!(c.remove_doc("never-there"), None);
    assert_eq!(c.len(), 2);
}

// -- refresh (set_doc) -------------------------------------------------------------

#[test]
fn set_doc_refreshes_index_entries() {
    let mut c = Collection::new("t");
    c.create_index("age").unwrap();
    c.insert(id_doc(
        "d",
        &[("age", Value::i64(20)), ("name", Value::str("old"))],
    ))
    .unwrap();

    let old = c
        .set_doc(
            "d",
            doc(&[("age", Value::i64(40)), ("name", Value::str("new"))]),
        )
        .unwrap()
        .expect("doc was present");
    assert_eq!(old.get("name"), Some(&Value::str("old")));

    let ix = c.index("age").unwrap();
    assert_eq!(ix.len(), 1, "no duplicate entries after refresh");
    assert!(ix.ids_equal(&Value::i64(20)).is_empty(), "stale value gone");
    assert_eq!(ix.ids_equal(&Value::i64(40)), vec!["d"]);
    // primary entry untouched (same _id)
    assert_eq!(
        c.index("_id").unwrap().ids_equal(&Value::str("d")),
        vec!["d"]
    );
    // missing field in the new doc → Null entry in that field's index
    assert_eq!(ix.ids_range(Unbounded, Unbounded), vec!["d"]);
}

#[test]
fn set_doc_refreshes_missing_to_present_and_back() {
    let mut c = Collection::new("t");
    c.create_index("f").unwrap();
    c.insert(id_doc("d", &[("f", Value::i64(7))])).unwrap();

    // remove the field entirely (replaced doc without it) → Null entry
    c.set_doc("d", doc(&[])).unwrap();
    assert_eq!(c.index("f").unwrap().ids_equal(&Value::Null), vec!["d"]);
    assert!(c.index("f").unwrap().ids_equal(&Value::i64(7)).is_empty());

    // back to a value
    c.set_doc("d", doc(&[("f", Value::i64(9))])).unwrap();
    assert_eq!(c.index("f").unwrap().ids_equal(&Value::i64(9)), vec!["d"]);
    assert!(c.index("f").unwrap().ids_equal(&Value::Null).is_empty());
    // stored doc now carries the preserved _id as its first key
    let d = c.get("d").unwrap();
    assert_eq!(d.get("_id"), Some(&Value::str("d")));
    assert_eq!(d.keys().next(), Some("_id"));
}

#[test]
fn set_doc_preserves_id_and_rejects_changes() {
    let mut c = Collection::new("t");
    c.create_index("v").unwrap();
    c.insert(id_doc("d", &[("v", Value::i64(1))])).unwrap();

    // explicit matching _id: kept, position preserved
    c.set_doc("d", doc(&[("v", Value::i64(2)), ("_id", Value::str("d"))]))
        .unwrap();
    let keys: Vec<_> = c.get("d").unwrap().keys().collect();
    assert_eq!(keys, vec!["v", "_id"]);

    // _id change is an error, store + indexes untouched
    let err = c
        .set_doc("d", doc(&[("_id", Value::str("other"))]))
        .unwrap_err();
    match err {
        StoreError::IdMismatch { expected, found } => {
            assert_eq!(expected, "d");
            assert_eq!(found, "other");
        }
        other => panic!("expected IdMismatch, got {other:?}"),
    }
    assert_eq!(c.get("d").unwrap().get("v"), Some(&Value::i64(2)));
    assert_eq!(c.index("v").unwrap().ids_equal(&Value::i64(2)), vec!["d"]);
    assert!(!c.contains("other"));

    // non-string _id
    assert_eq!(
        c.set_doc("d", doc(&[("_id", Value::i64(5))])),
        Err(StoreError::IdMustBeString)
    );
    // non-object
    assert_eq!(c.set_doc("d", Value::i64(1)), Err(StoreError::NotAnObject));
    // error paths left everything intact
    assert_eq!(c.index("v").unwrap().ids_equal(&Value::i64(2)), vec!["d"]);
}

#[test]
fn set_doc_missing_id_is_noop_even_for_bad_docs() {
    let mut c = Collection::new("t");
    c.create_index("v").unwrap();
    // absent id: Ok(None), and an invalid doc must NOT raise (filter-first)
    assert_eq!(c.set_doc("ghost", doc(&[])), Ok(None));
    assert_eq!(c.set_doc("ghost", Value::i64(1)), Ok(None));
    assert!(c.is_empty());
    assert!(c.index("v").unwrap().is_empty());
}
