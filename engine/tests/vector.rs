//! Integration tests for the vector index + brute-force cosine
//! `vector_search`, exercising the public `Collection` API end-to-end.

use mooracer_engine::{Collection, StoreError, Value};

/// Build a document with an explicit `_id` plus the given extra fields.
fn doc(id: &str, extra: &[(&str, Value)]) -> Value {
    let mut pairs: Vec<(String, Value)> = Vec::with_capacity(extra.len() + 1);
    pairs.push(("_id".to_string(), Value::str(id)));
    for (k, v) in extra {
        pairs.push((k.to_string(), v.clone()));
    }
    Value::Object(pairs)
}

/// An `f32` embedding vector as a `Value::Array` of `Value::F64` (mixed with
/// `Value::I64` where noted) — the shape the vector index accepts.
fn vec_of(xs: &[f32]) -> Value {
    Value::array_from(xs.iter().map(|x| Value::f64(*x as f64)).collect())
}

fn id_of(d: &Value) -> &str {
    d.get("_id").and_then(Value::as_str).unwrap()
}

// -- empty / no-index ------------------------------------------------------

#[test]
fn vector_search_empty_collection_returns_empty() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 3);
    let hits = c.vector_search("embedding", &[1.0, 2.0, 3.0], 10).unwrap();
    assert!(hits.is_empty(), "empty index -> empty result");
}

#[test]
fn vector_search_without_index_is_a_noindex_error() {
    let c = Collection::new("t");
    let r = c.vector_search("embedding", &[1.0], 5);
    assert_eq!(r, Err(StoreError::NoIndex("embedding".into())));
}

#[test]
fn wrong_length_query_returns_empty_not_error() {
    let mut c = Collection::new("t");
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0, 0.0]))]))
        .unwrap();
    c.create_vector_index("embedding", 3);
    // 2-length query against a dim-3 index -> no hits, no error.
    let hits = c.vector_search("embedding", &[1.0, 0.0], 5).unwrap();
    assert!(hits.is_empty());
}

// -- basic ranking ----------------------------------------------------------

#[test]
fn vector_search_ranks_by_cosine_best_first() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc("east", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    c.insert(doc("north", &[("embedding", vec_of(&[0.0, 1.0]))]))
        .unwrap();
    c.insert(doc("northeast", &[("embedding", vec_of(&[1.0, 1.0]))]))
        .unwrap();

    // Query due east: east (cos 1) > northeast (cos ~0.707) > north (cos 0).
    let hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(id_of(&hits[0].0), "east");
    assert_eq!(id_of(&hits[1].0), "northeast");
    assert_eq!(id_of(&hits[2].0), "north");
    // Strictly descending scores, all in [-1, 1].
    assert!(hits[0].1 > hits[1].1 && hits[1].1 > hits[2].1);
    for (_, s) in &hits {
        let sv = *s as f64;
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&sv),
            "score {s} in [-1,1]"
        );
    }
    // The top hit (aligned) scores exactly 1.
    assert!((hits[0].1 - 1.0).abs() < 1e-5);
    // The orthogonal hit (north) scores 0.
    assert!((hits[2].1 - 0.0).abs() < 1e-5);
}

#[test]
fn vector_search_returns_the_full_document() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc(
        "a",
        &[
            ("embedding", vec_of(&[1.0, 0.0])),
            ("name", Value::str("moo")),
            ("count", Value::i64(7)),
        ],
    ))
    .unwrap();
    let hits = c.vector_search("embedding", &[1.0, 0.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    let d = &hits[0].0;
    assert_eq!(id_of(d), "a");
    assert_eq!(d.get("name"), Some(&Value::str("moo")));
    assert_eq!(d.get("count"), Some(&Value::i64(7)));
    assert!(
        d.get("embedding").is_some(),
        "full doc clone includes the field"
    );
}

#[test]
fn vector_search_limit_is_top_k_and_zero_means_all() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    for i in 0..8 {
        let a = (i as f32) * 0.4;
        c.insert(doc(
            &format!("d{i}"),
            &[("embedding", vec_of(&[a.cos(), a.sin()]))],
        ))
        .unwrap();
    }
    assert_eq!(
        c.vector_search("embedding", &[1.0, 0.0], 3).unwrap().len(),
        3
    );
    assert_eq!(
        c.vector_search("embedding", &[1.0, 0.0], 0).unwrap().len(),
        8
    );
    assert_eq!(
        c.vector_search("embedding", &[1.0, 0.0], 50).unwrap().len(),
        8
    );
}

// -- Normalize-unit-vector correctness (scale invariance) ------------------

#[test]
fn cosine_is_invariant_to_query_and_doc_scale() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 3);
    // Two parallel (co-directional) docs, one a short unit vector and one a
    // 10x longer non-unit vector — cosine must treat them identically.
    c.insert(doc("short", &[("embedding", vec_of(&[1.0, 0.0, 0.0]))]))
        .unwrap();
    c.insert(doc("long", &[("embedding", vec_of(&[10.0, 0.0, 0.0]))]))
        .unwrap();

    // A non-unit query pointing the same way.
    let big = c.vector_search("embedding", &[5.0, 0.0, 0.0], 0).unwrap();
    // The unit query gives the identical per-doc scores.
    let small = c.vector_search("embedding", &[1.0, 0.0, 0.0], 0).unwrap();
    assert_eq!(big.len(), small.len());
    for (h1, h2) in big.iter().zip(small.iter()) {
        assert_eq!(id_of(&h1.0), id_of(&h2.0), "same ranking");
        assert!((h1.1 - h2.1).abs() < 1e-5, "same score (scale-invariant)");
    }
    // Both docs are perfectly aligned with the query -> cosine 1.
    for (_, s) in &big {
        assert!((s - 1.0).abs() < 1e-5);
    }
}

#[test]
fn stored_vectors_are_normalized_once_at_write() {
    // A non-unit doc vector must be normalized so the search treats it as its
    // unit direction: a 100x-longer doc in the same direction scores the same
    // as a unit doc.
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc("unit", &[("embedding", vec_of(&[0.0, 1.0]))]))
        .unwrap();
    c.insert(doc("scaled", &[("embedding", vec_of(&[0.0, 100.0]))]))
        .unwrap();
    let hits = c.vector_search("embedding", &[0.0, 1.0], 0).unwrap();
    assert_eq!(hits.len(), 2);
    let unit = hits.iter().find(|h| id_of(&h.0) == "unit").unwrap().1;
    let scaled = hits.iter().find(|h| id_of(&h.0) == "scaled").unwrap().1;
    // Both point straight north -> both score ~1.0...
    assert!((unit - 1.0).abs() < 1e-5, "unit score {unit}");
    assert!((scaled - 1.0).abs() < 1e-5, "scaled score {scaled}");
    // ...and the scaled doc scores identically to the unit doc (scale-invariant
    // thanks to write-time normalization).
    assert!(
        (unit - scaled).abs() < 1e-5,
        "unit {unit} vs scaled {scaled}"
    );
}

// -- mixed i64 / f64 elements ----------------------------------------------

#[test]
fn mixed_integer_and_float_elements_convert_to_f32() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 3);
    // i64 and f64 mixed in the same vector.
    let mixed = Value::array_from(vec![Value::i64(1), Value::f64(2.0), Value::i64(0)]);
    c.insert(doc("a", &[("embedding", mixed)])).unwrap();
    let hits = c.vector_search("embedding", &[1.0, 2.0, 0.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        (hits[0].1 - 1.0).abs() < 1e-5,
        "i64/f64 mix converts to f32 correctly"
    );
}

// -- missing field / dim mismatch -----------------------------------------

#[test]
fn missing_field_doc_is_not_indexed_and_does_not_error() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    // A doc without the field: fine, not searchable.
    c.insert(doc("nope", &[("x", Value::i64(1))])).unwrap();
    c.insert(doc("with", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    let hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    assert_eq!(hits.len(), 1, "only the doc with a valid vector is indexed");
    assert_eq!(id_of(&hits[0].0), "with");
}

#[test]
fn insert_with_wrong_dimension_is_an_error() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 3);
    // 2 elements against a dim-3 index -> VectorDimMismatch, store untouched.
    let r = c.insert(doc("bad", &[("embedding", vec_of(&[1.0, 0.0]))]));
    assert_eq!(
        r,
        Err(StoreError::VectorDimMismatch {
            field: "embedding".into(),
            expected: 3,
            found: 2,
        })
    );
    assert_eq!(c.len(), 0, "the bad doc was not inserted");
}

#[test]
fn insert_with_non_numeric_vector_is_an_error() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    // A non-numeric element disqualifies the vector.
    let bad = Value::array_from(vec![Value::i64(1), Value::str("x")]);
    let r = c.insert(doc("bad", &[("embedding", bad)]));
    assert!(matches!(
        r,
        Err(StoreError::VectorDimMismatch { found: 2, .. })
    ));
    assert_eq!(c.len(), 0);
}

#[test]
fn insert_missing_field_after_index_creation_is_fine() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    // Field missing -> not indexed, no error (documented behavior).
    c.insert(doc("plain", &[("n", Value::i64(1))])).unwrap();
    assert!(c.has_vector_index("embedding"));
    let hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    assert!(hits.is_empty());
}

// -- backfill ----------------------------------------------------------------

#[test]
fn create_vector_index_backfills_existing_docs() {
    let mut c = Collection::new("t");
    // Insert first, create the index after: it must pick up all valid docs.
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    c.insert(doc("b", &[("embedding", vec_of(&[0.0, 1.0]))]))
        .unwrap();
    c.insert(doc("skip", &[("other", Value::i64(1))])).unwrap(); // no embedding
    c.create_vector_index("embedding", 2);
    let hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(id_of(&hits[0].0), "a");
}

// -- maintenance (update / delete) -----------------------------------------

#[test]
fn vector_index_refreshes_on_update() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    c.insert(doc("b", &[("embedding", vec_of(&[0.0, 1.0]))]))
        .unwrap();

    // Update a's embedding to point north: now b (north) should be the top.
    let upd = Value::Object(vec![(
        "$set".to_string(),
        Value::Object(vec![("embedding".to_string(), vec_of(&[0.0, 1.0]))]),
    )]);
    c.update_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        upd,
    )
    .unwrap();

    let hits = c.vector_search("embedding", &[0.0, 1.0], 0).unwrap();
    assert_eq!(hits.len(), 2);
    // Both now point north: a (updated) moved from east to north, so both
    // score ~1.0 for a north query. (Before the update a would have scored ~0.)
    let sa = hits.iter().find(|h| id_of(&h.0) == "a").unwrap().1;
    let sb = hits.iter().find(|h| id_of(&h.0) == "b").unwrap().1;
    assert!((sa - 1.0).abs() < 1e-5, "a moved to north, score {sa}");
    assert!((sb - 1.0).abs() < 1e-5, "b was already north, score {sb}");
}

#[test]
fn vector_index_removes_on_delete() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    c.insert(doc("b", &[("embedding", vec_of(&[0.0, 1.0]))]))
        .unwrap();

    assert!(c.delete_one(Value::Object(vec![("_id".to_string(), Value::str("a"))])));

    let hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    assert_eq!(hits.len(), 1, "only b remains indexed");
    assert_eq!(id_of(&hits[0].0), "b");
}

#[test]
fn vector_index_updates_on_replace() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 2);
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0]))]))
        .unwrap();
    // Replace a wholesale, pointing its vector north.
    c.replace_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        doc("a", &[("embedding", vec_of(&[0.0, 1.0]))]),
    )
    .unwrap();
    let hits = c.vector_search("embedding", &[0.0, 1.0], 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert!((hits[0].1 - 1.0).abs() < 1e-4);
}

#[test]
fn update_to_wrong_dimension_is_an_error_and_untouched() {
    let mut c = Collection::new("t");
    c.create_vector_index("embedding", 3);
    c.insert(doc("a", &[("embedding", vec_of(&[1.0, 0.0, 0.0]))]))
        .unwrap();
    // $set the embedding to a 2-element vector -> InvalidUpdate? No: it is a
    // VectorDimMismatch from set_doc's vector check.
    let upd = Value::Object(vec![(
        "$set".to_string(),
        Value::Object(vec![("embedding".to_string(), vec_of(&[1.0, 0.0]))]),
    )]);
    let r = c.update_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        upd,
    );
    assert!(
        matches!(r, Err(StoreError::VectorDimMismatch { .. })),
        "got {r:?}"
    );
    // store untouched: the original 3-dim vector is still there.
    let hits = c.vector_search("embedding", &[1.0, 0.0, 0.0], 1).unwrap();
    assert!((hits[0].1 - 1.0).abs() < 1e-4);
}

// -- multiple vector indexes -----------------------------------------------

#[test]
fn two_vector_indexes_on_different_fields_are_independent() {
    let mut c = Collection::new("t");
    c.create_vector_index("e1", 2);
    c.create_vector_index("e2", 2);
    c.insert(doc(
        "a",
        &[("e1", vec_of(&[1.0, 0.0])), ("e2", vec_of(&[0.0, 1.0]))],
    ))
    .unwrap();
    assert_eq!(c.vector_index_names(), vec!["e1", "e2"]);
    // e1 search points east, e2 points north — both the same doc but different
    // score orientation.
    let h1 = c.vector_search("e1", &[1.0, 0.0], 1).unwrap();
    let h2 = c.vector_search("e2", &[0.0, 1.0], 1).unwrap();
    assert!((h1[0].1 - 1.0).abs() < 1e-4);
    assert!((h2[0].1 - 1.0).abs() < 1e-4);
    // Dropping one index does not affect the other.
    c.drop_vector_index("e1");
    assert!(c.vector_search("e1", &[1.0, 0.0], 1).is_err());
    assert!(c.vector_search("e2", &[0.0, 1.0], 1).is_ok());
}
