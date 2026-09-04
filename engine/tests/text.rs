//! Integration tests for the BM25 text index + `text_search`, exercising the
//! public `Collection` API end-to-end.

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

/// A string field value.
fn s(text: &str) -> Value {
    Value::str(text)
}

/// An array-of-strings field value.
fn sarr(items: &[&str]) -> Value {
    Value::array_from(items.iter().map(|t| Value::str(*t)).collect())
}

fn id_of(d: &Value) -> &str {
    d.get("_id").and_then(Value::as_str).unwrap()
}

// -- empty / no-index ------------------------------------------------------

#[test]
fn text_search_empty_collection_returns_empty() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    let hits = c.text_search("body", "moo cow", 10).unwrap();
    assert!(hits.is_empty(), "empty index -> empty result");
}

#[test]
fn text_search_without_index_is_a_noindex_error() {
    let c = Collection::new("t");
    let r = c.text_search("body", "moo", 5);
    assert_eq!(r, Err(StoreError::NoIndex("body".into())));
}

// -- basic ranking ----------------------------------------------------------

#[test]
fn text_search_ranks_by_bm25_best_first() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    // Doc with the rare term "moose" once (should rank high for it).
    c.insert(doc("a", &[("body", s("the quick moose runs"))]))
        .unwrap();
    // Doc with "moo" many times.
    c.insert(doc("b", &[("body", s("moo moo moo moo"))]))
        .unwrap();
    // Doc with "moo" once plus filler.
    c.insert(doc("c", &[("body", s("moo and lots of other words here"))]))
        .unwrap();
    // Doc without "moo".
    c.insert(doc("d", &[("body", s("a completely different topic"))]))
        .unwrap();

    // Query for "moo": a (has "moo"? no — "moose" stems to "moos", "moo"
    // stems to "moo"; "moose" and "moo" are DIFFERENT stems) ...
    let hits = c.text_search("body", "moo", 0).unwrap();
    // Only the docs containing the stem "moo" are returned (b, c); a ("moose")
    // and d are absent.
    assert_eq!(hits.len(), 2, "got {hits:?}");
    // b has tf 4 of "moo" in a short doc -> outranks c (tf 1, longer doc).
    assert_eq!(id_of(&hits[0].0), "b");
    assert_eq!(id_of(&hits[1].0), "c");
    assert!(hits[0].1 > hits[1].1, "higher tf/shorter doc scores higher");
    assert!(hits[0].1 > 0.0 && hits[1].1 > 0.0);
}

#[test]
fn text_search_returns_the_full_document() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    c.insert(doc(
        "a",
        &[
            ("body", s("moo the cow")),
            ("name", s("daisy")),
            ("count", Value::i64(7)),
        ],
    ))
    .unwrap();
    let hits = c.text_search("body", "cow", 1).unwrap();
    assert_eq!(hits.len(), 1);
    let d = &hits[0].0;
    assert_eq!(id_of(d), "a");
    assert_eq!(d.get("name"), Some(&Value::str("daisy")));
    assert_eq!(d.get("count"), Some(&Value::i64(7)));
    assert!(d.get("body").is_some(), "full doc clone includes the field");
}

#[test]
fn text_search_limit_is_top_k_and_zero_means_all() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    for i in 0..8 {
        // More "moo" in each successive doc.
        let words = std::iter::repeat_n("moo", i + 1)
            .collect::<Vec<_>>()
            .join(" ");
        c.insert(doc(&format!("d{i}"), &[("body", s(&words))]))
            .unwrap();
    }
    assert_eq!(c.text_search("body", "moo", 3).unwrap().len(), 3);
    assert_eq!(c.text_search("body", "moo", 0).unwrap().len(), 8);
    assert_eq!(c.text_search("body", "moo", 50).unwrap().len(), 8);
    // Descending scores.
    let hits = c.text_search("body", "moo", 0).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].1 >= w[1].1, "scores must be non-increasing");
    }
}

// -- stemming ---------------------------------------------------------------

#[test]
fn text_search_stems_the_query_at_search_time() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    // Indexed docs store stemmed forms via text_tokens: "running" -> "run",
    // "skies" -> "ski".
    c.insert(doc("runner", &[("body", s("the running race"))]))
        .unwrap();
    c.insert(doc("sky", &[("body", s("the bright skies"))]))
        .unwrap();

    // A raw query "running" must hit the "run" doc via stemming.
    let hits = c.text_search("body", "running", 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "runner");
    // "skied" (stem "ski") hits the "ski" doc.
    let hits = c.text_search("body", "skied", 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "sky");
    // A query that stems to something absent returns nothing.
    assert!(c.text_search("body", "zebra", 1).unwrap().is_empty());
}

// -- field shape -----------------------------------------------------------

#[test]
fn array_of_strings_field_is_indexed() {
    let mut c = Collection::new("t");
    c.create_text_index("tags");
    c.insert(doc("a", &[("tags", sarr(&["moo", "cow"]))]))
        .unwrap();
    c.insert(doc("b", &[("tags", sarr(&["horse"]))])).unwrap();
    let hits = c.text_search("tags", "moo", 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "a");
}

#[test]
fn non_text_field_is_not_indexed_but_not_an_error() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    // A numeric field is not a string / array of strings -> not indexed, but
    // the insert must succeed (a text index never rejects a write).
    c.insert(doc("num", &[("body", Value::i64(12345))]))
        .unwrap();
    c.insert(doc("str", &[("body", s("moo moo"))])).unwrap();
    assert_eq!(c.len(), 2, "the numeric doc was still inserted");
    let hits = c.text_search("body", "moo", 0).unwrap();
    assert_eq!(hits.len(), 1, "only the string doc is searchable");
    assert_eq!(id_of(&hits[0].0), "str");
}

#[test]
fn a_mixed_array_is_not_indexed() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    // An array containing a non-string element disqualifies the whole field.
    let mixed = Value::array_from(vec![Value::str("moo"), Value::i64(1)]);
    c.insert(doc("a", &[("body", mixed)])).unwrap();
    let hits = c.text_search("body", "moo", 0).unwrap();
    assert!(hits.is_empty(), "a mixed array is not tokenized");
}

// -- backfill ----------------------------------------------------------------

#[test]
fn create_text_index_backfills_existing_docs() {
    let mut c = Collection::new("t");
    c.insert(doc("a", &[("body", s("moo the first"))])).unwrap();
    c.insert(doc("b", &[("body", s("cow the second"))]))
        .unwrap();
    c.create_text_index("body");
    let hits = c.text_search("body", "moo", 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "a");
}

#[test]
fn reindex_rebuilds_the_text_index_deterministically() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    c.insert(doc("a", &[("body", s("moo moo"))])).unwrap();
    c.insert(doc("b", &[("body", s("cow"))])).unwrap();
    // Mutate a, then rebuild: results must be identical to incremental state.
    c.update_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        Value::Object(vec![(
            "$set".to_string(),
            Value::Object(vec![("body".to_string(), s("moo cow moo"))]),
        )]),
    )
    .unwrap();
    let before = c.text_search("body", "moo", 0).unwrap();
    c.reindex();
    let after = c.text_search("body", "moo", 0).unwrap();
    assert_eq!(
        before
            .iter()
            .map(|h| (id_of(&h.0).to_string(), h.1))
            .collect::<Vec<_>>(),
        after
            .iter()
            .map(|h| (id_of(&h.0).to_string(), h.1))
            .collect::<Vec<_>>(),
        "reindex must be a deterministic no-op on the text index"
    );
    assert_eq!(
        before.len(),
        1,
        "only a's updated body ('moo cow moo') has 'moo'"
    );
}

// -- maintenance (update / delete / replace) -------------------------------

#[test]
fn text_index_refreshes_on_update() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    c.insert(doc("a", &[("body", s("moo the cow"))])).unwrap();

    // Overwrite a's body so it no longer mentions "moo".
    c.update_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        Value::Object(vec![(
            "$set".to_string(),
            Value::Object(vec![("body".to_string(), s("just a horse"))]),
        )]),
    )
    .unwrap();

    assert!(
        c.text_search("body", "moo", 0).unwrap().is_empty(),
        "moo posting removed"
    );
    let hits = c.text_search("body", "horse", 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "a");
}

#[test]
fn text_index_removes_on_delete() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    c.insert(doc("a", &[("body", s("moo moo"))])).unwrap();
    c.insert(doc("b", &[("body", s("cow"))])).unwrap();

    assert!(c.delete_one(Value::Object(vec![("_id".to_string(), Value::str("a"))])));
    let hits = c.text_search("body", "moo", 0).unwrap();
    assert!(hits.is_empty(), "a's postings removed");
    let hits = c.text_search("body", "cow", 0).unwrap();
    assert_eq!(hits.len(), 1, "b survives");
    assert_eq!(id_of(&hits[0].0), "b");
}

#[test]
fn text_index_updates_on_replace() {
    let mut c = Collection::new("t");
    c.create_text_index("body");
    c.insert(doc("a", &[("body", s("moo moo"))])).unwrap();
    c.replace_one(
        Value::Object(vec![("_id".to_string(), Value::str("a"))]),
        doc("a", &[("body", s("elephant in the grass"))]),
    )
    .unwrap();
    assert!(c.text_search("body", "moo", 0).unwrap().is_empty());
    let hits = c.text_search("body", "elephant", 0).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(id_of(&hits[0].0), "a");
}

// -- multiple text indexes -------------------------------------------------

#[test]
fn two_text_indexes_on_different_fields_are_independent() {
    let mut c = Collection::new("t");
    c.create_text_index("title");
    c.create_text_index("body");
    c.insert(doc(
        "a",
        &[("title", s("moo racing")), ("body", s("the quick cow"))],
    ))
    .unwrap();
    assert_eq!(c.text_index_names(), vec!["body", "title"]);
    let h1 = c.text_search("title", "moo", 1).unwrap();
    let h2 = c.text_search("body", "cow", 1).unwrap();
    assert_eq!(id_of(&h1[0].0), "a");
    assert_eq!(id_of(&h2[0].0), "a");
    // Dropping one index leaves the other intact.
    c.drop_text_index("title");
    assert!(c.text_search("title", "moo", 1).is_err());
    assert!(c.text_search("body", "cow", 1).is_ok());
    assert!(!c.has_text_index("title"));
    assert!(c.has_text_index("body"));
}
