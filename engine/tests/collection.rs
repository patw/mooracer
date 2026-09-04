//! Integration tests: the `Collection` public API as external users
//! (server, clients) will consume it.

use mooracer_engine::{Collection, StoreError, Transaction, Value};

fn doc(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

#[test]
fn external_insert_get_roundtrip() {
    let mut c = Collection::new("horses");
    let id = c
        .insert(doc(&[
            ("name", Value::str("bess")),
            ("legs", Value::i64(4)),
        ]))
        .unwrap();
    assert_eq!(id.len(), 24);
    let d = c.get(&id).expect("doc retrievable by generated id");
    assert_eq!(d.get("name"), Some(&Value::str("bess")));
    assert_eq!(d.get("_id"), Some(&Value::Str(id)));
}

#[test]
fn external_duplicate_and_type_errors() {
    let mut c = Collection::new("t");
    c.insert(doc(&[("_id", Value::str("x"))])).unwrap();
    assert_eq!(
        c.insert(doc(&[("_id", Value::str("x"))])),
        Err(StoreError::DuplicateId("x".into()))
    );
    assert_eq!(
        c.insert(doc(&[("_id", Value::i64(1))])),
        Err(StoreError::IdMustBeString)
    );
    assert_eq!(c.insert(Value::bool(true)), Err(StoreError::NotAnObject));
    // display is useful in server error messages
    assert_eq!(
        StoreError::IdMustBeString.to_string(),
        "`_id` must be a string"
    );
}

#[test]
fn external_insert_many_is_atomic() {
    let mut c = Collection::new("t");
    let before = 0usize;
    assert_eq!(c.len(), before);
    let res = c.insert_many([
        doc(&[("v", Value::i64(1))]),
        doc(&[("v", Value::i64(2))]),
        doc(&[("_id", Value::Null)]), // invalid → whole batch rejected
    ]);
    assert_eq!(res, Err(StoreError::IdMustBeString));
    assert_eq!(c.len(), before, "no partial commit");

    let n = c
        .insert_many([
            doc(&[("v", Value::i64(3))]),
            doc(&[("v", Value::i64(4))]),
            doc(&[("v", Value::i64(5))]),
        ])
        .unwrap();
    assert_eq!(n, 3);
    assert_eq!(c.len(), 3);
}

#[test]
fn external_two_collections_are_independent() {
    let mut a = Collection::new("a");
    let mut b = Collection::new("b");
    a.insert(doc(&[("_id", Value::str("same"))])).unwrap();
    b.insert(doc(&[("_id", Value::str("same"))])).unwrap();
    assert!(a.contains("same") && b.contains("same"));
    // auto ids from the two collections never collide with each other
    // (explicit ids *may* repeat across collections — uniqueness is
    // per-collection by design).
    for _ in 0..200 {
        a.insert(Value::object()).unwrap();
        b.insert(Value::object()).unwrap();
    }
    let auto: fn(&Collection) -> std::collections::HashSet<&str> = |c| {
        c.iter()
            .filter(|d| d.get("_id") != Some(&Value::Str("same".into())))
            .map(|d| d.get("_id").unwrap().as_str().unwrap())
            .collect()
    };
    let ia = auto(&a);
    let ib = auto(&b);
    assert_eq!(ia.len(), 200);
    assert_eq!(ib.len(), 200);
    assert!(
        ia.is_disjoint(&ib),
        "process-wide id counter keeps collections disjoint"
    );
}
#[test]
fn external_delete_one_returns_bool_and_removes() {
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
    assert!(c.delete_one(doc(&[("kind", Value::str("moo"))])));
    assert!(!c.contains("a"));
    assert!(c.contains("b"));
    // second call: nothing left to delete
    assert!(!c.delete_one(doc(&[("kind", Value::str("moo"))])));
    assert_eq!(c.len(), 1);
}

#[test]
fn external_delete_many_counts_and_removes() {
    let mut c = Collection::new("t");
    for i in 0..5 {
        c.insert(doc(&[
            ("_id", Value::str(format!("cow{i}"))),
            ("age", Value::i64(i as i64)),
        ]))
        .unwrap();
    }
    let n = c.delete_many(doc(&[("age", doc(&[("$gte", Value::i64(2))]))]));
    assert_eq!(n, 3, "age 2,3,4 match");
    assert_eq!(c.len(), 2);
    assert!(c.contains("cow0") && c.contains("cow1"));
}

#[test]
fn external_delete_preserves_index_invariants() {
    let mut c = Collection::new("t");
    c.insert(doc(&[("_id", Value::str("a")), ("score", Value::i64(10))]))
        .unwrap();
    c.insert(doc(&[("_id", Value::str("b")), ("score", Value::i64(20))]))
        .unwrap();
    c.insert(doc(&[("_id", Value::str("c")), ("score", Value::i64(30))]))
        .unwrap();
    c.create_index("score").unwrap();
    c.delete_many(doc(&[("score", doc(&[("$lt", Value::i64(25))]))]));
    let score = c.index("score").unwrap();
    assert_eq!(score.len(), 1);
    assert_eq!(score.ids_equal(&Value::i64(30)), vec!["c"]);
    // the primary _id index also shrank
    assert_eq!(c.index("_id").unwrap().len(), 1);
}

// -- atomic batch (transaction) public API --------------------------------

#[test]
fn external_transaction_commit_and_index_invariants() {
    let mut c = Collection::new("horses");
    c.insert(doc(&[("_id", Value::str("a")), ("age", Value::i64(3))]))
        .unwrap();
    c.insert(doc(&[("_id", Value::str("b")), ("age", Value::i64(5))]))
        .unwrap();
    c.create_index("age").unwrap();

    let tx: Transaction<'_> = c.begin();
    let mut tx = tx;
    tx.insert(doc(&[("_id", Value::str("c")), ("age", Value::i64(4))]))
        .unwrap();
    // pre-batch read through the public API: the insert is invisible
    assert_eq!(tx.count(doc(&[])), 2);
    tx.commit().unwrap();

    assert_eq!(c.len(), 3);
    let age = c.index("age").unwrap();
    assert_eq!(age.ids_equal(&Value::i64(3)), vec!["a"]);
    assert_eq!(age.ids_equal(&Value::i64(4)), vec!["c"]);
    assert_eq!(age.ids_equal(&Value::i64(5)), vec!["b"]);
    assert_eq!(c.index("_id").unwrap().len(), 3);
}

#[test]
fn external_transaction_error_rolls_back() {
    let mut c = Collection::new("t");
    c.insert(doc(&[("_id", Value::str("keep"))])).unwrap();
    let mut tx = c.begin();
    tx.insert(doc(&[("_id", Value::str("new"))])).unwrap();
    let err = tx.insert(doc(&[("_id", Value::str("keep"))])).unwrap_err();
    assert_eq!(err, StoreError::DuplicateId("keep".into()));
    assert!(tx.is_failed());
    // an errored transaction commits nothing
    assert_eq!(tx.commit(), Err(StoreError::DuplicateId("keep".into())));
    assert_eq!(c.len(), 1);
    assert!(!c.contains("new"));
}
