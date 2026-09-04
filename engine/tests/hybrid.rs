//! Integration tests for hybrid search (RRF fusion of BM25 + vector
//! rankings), exercising the public `Collection::hybrid_search` API end-to-end.

use mooracer_engine::{Collection, RRF_K, StoreError, Value};

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

/// A 2-dim numeric-array embedding value.
fn vec2(x: f64, y: f64) -> Value {
    Value::array_from(vec![Value::f64(x), Value::f64(y)])
}

fn id_of(d: &Value) -> &str {
    d.get("_id").and_then(Value::as_str).unwrap()
}

/// A small herd where the BM25 and vector rankings deliberately disagree, so
/// the fusion (and its union semantics) is actually exercised:
/// - `alpha`: "moo" + embedding (1,0)      → top in BOTH rankings
/// - `gamma`: "moo" + embedding (1,1)/√2   → 2nd in both
/// - `beta`:  "horse" + embedding (0,1)    → 3rd in vector only (no "moo")
/// - `delta`: "zebra" + NO embedding       → in neither ranking
fn build_herd() -> Collection {
    let mut c = Collection::new("hybrid");
    c.create_text_index("body");
    c.create_vector_index("embedding", 2);
    c.insert(doc(
        "alpha",
        &[
            ("body", s("moo the alpha cow")),
            ("embedding", vec2(1.0, 0.0)),
        ],
    ))
    .unwrap();
    c.insert(doc(
        "gamma",
        &[
            ("body", s("moo the gamma cow")),
            ("embedding", vec2(1.0, 1.0)),
        ],
    ))
    .unwrap();
    c.insert(doc(
        "beta",
        &[("body", s("just a horse")), ("embedding", vec2(0.0, 1.0))],
    ))
    .unwrap();
    c.insert(doc("delta", &[("body", s("a zebra grazing"))]))
        .unwrap();
    c
}

// -- no-index / empty ------------------------------------------------------

#[test]
fn hybrid_search_missing_text_index_is_a_noindex_error() {
    let mut c = Collection::new("h");
    c.create_vector_index("embedding", 2);
    let r = c.hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 5);
    assert_eq!(r, Err(StoreError::NoIndex("body".into())));
}

#[test]
fn hybrid_search_missing_vector_index_is_a_noindex_error() {
    let mut c = Collection::new("h");
    c.create_text_index("body");
    let r = c.hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 5);
    assert_eq!(r, Err(StoreError::NoIndex("embedding".into())));
}

#[test]
fn hybrid_search_empty_collection_returns_empty() {
    let mut c = Collection::new("h");
    c.create_text_index("body");
    c.create_vector_index("embedding", 2);
    let hits = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 10)
        .unwrap();
    assert!(hits.is_empty(), "no ranked docs -> empty result");
}

// -- fusion ranking --------------------------------------------------------

#[test]
fn hybrid_fusion_ranks_a_doc_in_both_rankings_at_the_top() {
    let c = build_herd();
    // Query that is strong in both signals for "alpha".
    let hits = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| id_of(&h.0)).collect();
    // alpha is #1 in the text ranking (inserted first, "moo") and #1 in the
    // vector ranking (cos(1,0)=1) -> it earns two rank-1 contributions, the
    // maximum possible, so it must be first.
    assert_eq!(ids.first(), Some(&"alpha"), "got {ids:?}");
    // gamma is 2nd in both -> second.
    assert_eq!(ids.get(1), Some(&"gamma"), "got {ids:?}");
    // Scores are strictly descending.
    for w in hits.windows(2) {
        assert!(w[0].1 > w[1].1, "fused scores must be strictly descending");
    }
}

#[test]
fn hybrid_fusion_is_a_union_of_the_two_rankings() {
    let c = build_herd();
    let hits = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| id_of(&h.0)).collect();
    // "beta" contains no "moo" (absent from the text ranking) but its
    // embedding is the worst vector match — it must STILL surface, proving
    // the fusion is a union over the document sets, not an intersection.
    assert!(
        ids.contains(&"beta"),
        "a doc ranked by only one signal must appear (union), got {ids:?}"
    );
    // "delta" is in neither ranking (no "moo", no embedding) -> absent.
    assert!(
        !ids.contains(&"delta"),
        "a doc in neither ranking is absent, got {ids:?}"
    );
    // Exactly the three docs in at least one ranking.
    assert_eq!(ids.len(), 3, "got {ids:?}");
    assert_eq!(
        ids.last(),
        Some(&"beta"),
        "beta is the worst fused doc, got {ids:?}"
    );
}

#[test]
fn hybrid_scores_are_the_rrf_sum_of_rank_contributions() {
    // Hand-compute the expected RRF for a controlled 3-doc case: all three docs
    // rank identically in BOTH signals (so a doc's fused score is
    // 1/(K+text_rank) + 1/(K+vec_rank) with the same rank in each).
    let mut c = Collection::new("h");
    c.create_text_index("body");
    c.create_vector_index("embedding", 2);
    // Each doc carries "moo" once -> tied in the text ranking; the tie breaks
    // by index (insertion) order: alpha, beta, gamma -> ranks 1, 2, 3.
    // Give them distinct, mutually-aligned embeddings so the vector ranking is
    // alpha > beta > gamma (ranks 1, 2, 3) — same order as the text tie-break.
    c.insert(doc(
        "alpha",
        &[("body", s("moo")), ("embedding", vec2(1.0, 0.0))],
    ))
    .unwrap();
    c.insert(doc(
        "beta",
        &[("body", s("moo")), ("embedding", vec2(0.9, 0.1))],
    ))
    .unwrap();
    c.insert(doc(
        "gamma",
        &[("body", s("moo")), ("embedding", vec2(0.5, 0.5))],
    ))
    .unwrap();

    let hits = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap();
    assert_eq!(hits.len(), 3);
    // alpha: 1/(K+1) twice; beta: 1/(K+2) twice; gamma: 1/(K+3) twice.
    let k = RRF_K as f64;
    let expect: [f64; 3] = [2.0 / (k + 1.0), 2.0 / (k + 2.0), 2.0 / (k + 3.0)];
    let want_ids = ["alpha", "beta", "gamma"];
    for (i, (hit, exp)) in hits.iter().zip(expect.iter()).enumerate() {
        assert_eq!(id_of(&hit.0), want_ids[i]);
        assert!(
            (hit.1 - *exp).abs() < 1e-12,
            "score {i}: {} != {}",
            hit.1,
            exp
        );
    }
}

// -- limit / shape ---------------------------------------------------------

#[test]
fn hybrid_search_limit_is_top_k_and_zero_means_all() {
    let c = build_herd();
    let all = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap();
    assert_eq!(all.len(), 3, "3 docs are in at least one ranking");
    assert_eq!(
        c.hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 2)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        c.hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 1)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        c.hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 50)
            .unwrap()
            .len(),
        3
    );
    // limit(1) is just the top fused doc.
    let top = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 1)
        .unwrap();
    assert_eq!(id_of(&top[0].0), "alpha");
}

#[test]
fn hybrid_search_returns_the_full_document() {
    let mut c = Collection::new("h");
    c.create_text_index("body");
    c.create_vector_index("embedding", 2);
    c.insert(doc(
        "a",
        &[
            ("body", s("moo the cow")),
            ("embedding", vec2(1.0, 0.0)),
            ("name", s("daisy")),
            ("count", Value::i64(7)),
        ],
    ))
    .unwrap();
    let hits = c
        .hybrid_search("body", "embedding", "cow", &[1.0, 0.0], 1)
        .unwrap();
    assert_eq!(hits.len(), 1);
    let d = &hits[0].0;
    assert_eq!(id_of(d), "a");
    assert_eq!(d.get("name"), Some(&Value::str("daisy")));
    assert_eq!(d.get("count"), Some(&Value::i64(7)));
    assert!(d.get("body").is_some());
}

#[test]
fn hybrid_search_is_deterministic_and_stable_across_repeats() {
    let c = build_herd();
    let a = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap()
        .iter()
        .map(|h| (id_of(&h.0).to_string(), h.1))
        .collect::<Vec<_>>();
    let b = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap()
        .iter()
        .map(|h| (id_of(&h.0).to_string(), h.1))
        .collect::<Vec<_>>();
    assert_eq!(a, b, "hybrid search must be deterministic");
}

#[test]
fn hybrid_search_agrees_with_each_signal_present() {
    // Sanity: the top hybrid hit is present in at least one single-signal
    // result for the same query.
    let c = build_herd();
    let hybrid = c
        .hybrid_search("body", "embedding", "moo", &[1.0, 0.0], 0)
        .unwrap();
    let text_hits = c.text_search("body", "moo", 0).unwrap();
    let vec_hits = c.vector_search("embedding", &[1.0, 0.0], 0).unwrap();
    let text: Vec<&str> = text_hits.iter().map(|h| id_of(&h.0)).collect();
    let vec: Vec<&str> = vec_hits.iter().map(|h| id_of(&h.0)).collect();
    for h in &hybrid {
        let id = id_of(&h.0);
        assert!(
            text.contains(&id) || vec.contains(&id),
            "hybrid hit {id} must be in a signal ranking"
        );
    }
}
