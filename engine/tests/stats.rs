//! Integration tests for `Collection::stats()` and `Collection::reindex()`
//! (public API, spec "Stats & reindex decisions").

use mooracer_engine::{Collection, CollectionStats, IndexStats, Value};

fn doc(pairs: &[(&str, Value)]) -> Value {
    Value::object_from(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

fn populate() -> Collection {
    let mut c = Collection::new("cattle");
    c.insert(doc(&[
        ("_id", Value::str("moo")),
        ("age", Value::i64(4)),
        ("sound", Value::str("moo")),
    ]))
    .unwrap();
    c.insert(doc(&[
        ("_id", Value::str("bess")),
        ("age", Value::i64(6)),
        ("sound", Value::str("baa")),
    ]))
    .unwrap();
    c.insert(doc(&[
        ("_id", Value::str("daisy")),
        ("age", Value::i64(4)),
        ("sound", Value::str("moo")),
    ]))
    .unwrap();
    c.create_index("age").unwrap();
    c.create_index("sound").unwrap();
    c
}

#[test]
fn stats_public_shape() {
    let c = populate();
    let s: CollectionStats = c.stats();
    assert_eq!(s.docs, 3);
    assert_eq!(s.indexes, 3);
    assert_eq!(s.per_index.len(), 3);
    assert_eq!(
        s.per_index
            .iter()
            .map(|i: &IndexStats| i.field.as_str())
            .collect::<Vec<_>>(),
        vec!["_id", "age", "sound"]
    );
    for ix in &s.per_index {
        assert_eq!(ix.entries, 3);
        assert!(ix.memory > 0);
    }
    // age: {4, 6} ; sound: {baa, moo}
    assert_eq!(s.per_index[1].distinct, 2);
    assert_eq!(s.per_index[2].distinct, 2);
    // The invariant the server layer can rely on: total = docs + per-index.
    let sum: usize = s.per_index.iter().map(|i| i.memory).sum();
    assert_eq!(s.total_memory, s.docs_memory + sum);
}

#[test]
fn reindex_then_queries_still_correct() {
    let mut c = populate();
    assert_eq!(c.reindex(), 3);
    // Indexed point + range queries agree with the full scan.
    let mut f = Vec::new();
    f.push(("age".to_string(), Value::i64(4)));
    assert_eq!(c.count(Value::object_from(f.clone())), 2);
    f.push(("sound".to_string(), Value::str("moo")));
    assert_eq!(c.count(Value::object_from(f)), 2);
    assert_eq!(
        c.index("age").unwrap().ids_equal(&Value::i64(4)),
        vec!["daisy", "moo"]
    );
    // Stats before/after are identical: the rebuild is deterministic.
    let before = c.stats();
    c.reindex();
    assert_eq!(c.stats(), before);
}

#[test]
fn stats_and_reindex_on_empty_and_primary_only() {
    let mut c = Collection::new("empty");
    let s = c.stats();
    assert_eq!(s.docs, 0);
    assert_eq!(s.indexes, 1);
    assert_eq!(s.per_index[0].field, "_id");
    assert_eq!(s.per_index[0].entries, 0);
    assert_eq!(c.reindex(), 1);
    assert_eq!(c.stats(), s, "reindex of an empty store is a no-op");
}
