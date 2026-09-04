//! Integration tests for the Rust client over a **real TCP socket**.
//!
//! Each test binds an ephemeral server, seeds it, runs the accept loop in a
//! background thread, and drives the full Mongo-style chain API through the
//! [`mooracer_client::Client`] — covering every command kind, the lazy query
//! pipeline, typed errors, and the search/aggregation surfaces.

use std::thread;

use mooracer_client::{Client, Error, Stats};
use mooracer_engine::{AggFn, Value};
use mooracer_wire::Status;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

/// Build an engine `Value` from a JSON-ish inline shape using small helpers.
fn val(n: i64) -> Value {
    Value::I64(n)
}
fn str(s: &str) -> Value {
    Value::Str(s.to_string())
}
/// `{}` = the empty filter (matches all).
fn all() -> Value {
    Value::Object(vec![])
}

/// Bind an ephemeral server, seed it, run it, and return a connected client.
fn seeded_client(name: &str, docs: &[Value]) -> Client {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    if !docs.is_empty() {
        server.seed_docs(name, docs).unwrap();
    }
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    Client::connect(&addr.to_string()).unwrap()
}

fn id_of(doc: &Value) -> String {
    doc.get("_id").unwrap().as_str().unwrap().to_string()
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

#[test]
fn insert_assigns_and_returns_id() {
    let mut c = seeded_client("c", &[]);
    let mut herd = c.collection("c");

    // Explicit _id is returned verbatim.
    let id = herd
        .insert(&obj(&[("_id", str("a")), ("age", val(3))]))
        .unwrap();
    assert_eq!(id, "a");

    // No _id → an auto-generated 24-char lowercase hex id.
    let auto = herd.insert(&obj(&[("age", val(5))])).unwrap();
    assert_eq!(auto.len(), 24);
    assert!(auto.chars().all(|ch| ch.is_ascii_hexdigit()));
}

#[test]
fn insert_many_returns_ids_in_order() {
    let mut c = seeded_client("c", &[]);
    let mut herd = c.collection("c");
    let ids = herd
        .insert_many(&[
            obj(&[("_id", str("a")), ("n", val(1))]),
            obj(&[("_id", str("b")), ("n", val(2))]),
            obj(&[("n", val(3))]),
        ])
        .unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(ids[0], "a");
    assert_eq!(ids[1], "b");
    assert_eq!(ids[2].len(), 24);
}

#[test]
fn typed_error_duplicate_id_surfaces() {
    let mut c = seeded_client("c", &[obj(&[("_id", str("1"))])]);
    let mut herd = c.collection("c");
    let err = herd.insert(&obj(&[("_id", str("1"))])).unwrap_err();
    assert!(
        matches!(err, Error::Server(Status::DuplicateId, _)),
        "{err:?}"
    );
    assert_eq!(err.status(), Some(Status::DuplicateId));
    assert!(!err.to_string().is_empty());
}

// ---------------------------------------------------------------------------
// Find / find_one / count / exists
// ---------------------------------------------------------------------------

#[test]
fn find_all_and_count_and_exists() {
    let docs = [
        obj(&[("_id", str("a")), ("age", val(30))]),
        obj(&[("_id", str("b")), ("age", val(20))]),
        obj(&[("_id", str("c")), ("age", val(40))]),
    ];
    let mut c = seeded_client("cows", &docs);
    let mut herd = c.collection("cows");

    let all_docs: Vec<Value> = herd.find(all()).to_list().unwrap();
    assert_eq!(all_docs.len(), 3);

    assert_eq!(herd.count(all()).unwrap(), 3);
    assert!(herd.exists(all()).unwrap());

    // A filter that matches nothing.
    let none = herd.find(obj(&[("age", val(9999))])).to_list().unwrap();
    assert!(none.is_empty());
    assert_eq!(herd.count(obj(&[("age", val(9999))])).unwrap(), 0);
    assert!(!herd.exists(obj(&[("age", val(9999))])).unwrap());
}

#[test]
fn find_one_returns_first_or_none() {
    let docs = [
        obj(&[("_id", str("a")), ("age", val(30))]),
        obj(&[("_id", str("b")), ("age", val(20))]),
    ];
    let mut c = seeded_client("cows", &docs);
    let mut herd = c.collection("cows");

    let one = herd.find_one(obj(&[("age", val(30))])).unwrap();
    assert_eq!(one.as_ref().map(id_of), Some("a".to_string()));

    let none = herd.find_one(obj(&[("age", val(9999))])).unwrap();
    assert!(none.is_none());
}

#[test]
fn find_chain_filter_sort_skip_limit() {
    let docs = [
        obj(&[("_id", str("a")), ("age", val(1))]),
        obj(&[("_id", str("b")), ("age", val(5))]),
        obj(&[("_id", str("c")), ("age", val(3))]),
        obj(&[("_id", str("d")), ("age", val(4))]),
    ];
    let mut c = seeded_client("cows", &docs);

    // filter age>=3 (c,b,d) sorted asc → c(3),d(4),b(5); limit 2 → c,d.
    let mut herd = c.collection("cows");
    let docs_out: Vec<Value> = herd
        .find(obj(&[("age", obj(&[("\u{0024}gte", val(3))]))]))
        .sort("age", false)
        .limit(2)
        .to_list()
        .unwrap();
    let order: Vec<String> = docs_out.iter().map(id_of).collect();
    assert_eq!(order, vec!["c".to_string(), "d".to_string()]);

    // descending + skip 1: sorted desc b(5),d(4),c(3); skip 1 → d,c.
    let mut herd = c.collection("cows");
    let docs_out: Vec<Value> = herd
        .find(obj(&[("age", obj(&[("\u{0024}gte", val(3))]))]))
        .sort("age", true)
        .skip(1)
        .to_list()
        .unwrap();
    let order: Vec<String> = docs_out.iter().map(id_of).collect();
    assert_eq!(order, vec!["d".to_string(), "c".to_string()]);

    // count ignores the pipeline (counts the filtered set) = 3.
    let mut herd = c.collection("cows");
    assert_eq!(
        herd.find(obj(&[("age", obj(&[("\u{0024}gte", val(3))]))]))
            .sort("age", false)
            .count()
            .unwrap(),
        3
    );
}

#[test]
fn value_tree_roundtrips_through_client() {
    // A deeply nested document with every value kind.
    let nested = obj(&[
        ("_id", str("deep")),
        ("i", val(42)),
        ("f", Value::F64(1.5)),
        ("b", Value::Bool(true)),
        ("n", Value::Null),
        ("s", str("moo 🐄")),
        (
            "arr",
            Value::Array(vec![val(1), str("x"), Value::Bool(false)]),
        ),
        ("obj", obj(&[("k1", val(7)), ("k2", str("inner"))])),
    ]);
    let mut c = seeded_client("deep", &[]);
    let mut herd = c.collection("deep");
    herd.insert(&nested).unwrap();

    let back: Vec<Value> = herd.find(obj(&[("_id", str("deep"))])).to_list().unwrap();
    assert_eq!(back.len(), 1);
    assert_eq!(back[0], nested, "value tree must round-trip losslessly");
}

// ---------------------------------------------------------------------------
// Update / replace / delete
// ---------------------------------------------------------------------------

#[test]
fn update_one_applies_and_errors_on_no_match() {
    let docs = [
        obj(&[("_id", str("a")), ("n", val(1))]),
        obj(&[("_id", str("b")), ("n", val(2))]),
    ];
    let mut c = seeded_client("c", &docs);
    let mut herd = c.collection("c");

    // $set n=99 on a → count 1, and the change is visible.
    let set99 = obj(&[("$set", obj(&[("n", val(99))]))]);
    let n = herd.update_one(obj(&[("_id", str("a"))]), set99).unwrap();
    assert_eq!(n, 1);
    let got: Vec<Value> = herd.find(obj(&[("_id", str("a"))])).to_list().unwrap();
    assert_eq!(got[0].get("n").unwrap(), &Value::I64(99));

    // $inc across docs.
    let inc10 = obj(&[("$inc", obj(&[("n", val(10))]))]);
    let n = herd.update_many(obj(&[]), inc10).unwrap();
    assert_eq!(n, 2);

    // update_one with no match → NoMatch error.
    let set1 = obj(&[("$set", obj(&[("n", val(1))]))]);
    let err = herd
        .update_one(obj(&[("_id", str("zzz"))]), set1)
        .unwrap_err();
    assert!(matches!(err, Error::Server(Status::NoMatch, _)), "{err:?}");
}

#[test]
fn replace_one_preserves_id_and_errors_on_no_match() {
    let docs = [obj(&[("_id", str("a")), ("x", val(1)), ("y", val(2))])];
    let mut c = seeded_client("c", &docs);
    let mut herd = c.collection("c");

    // Wholesale replace: only `x` remains, `_id` preserved.
    let n = herd
        .replace_one(obj(&[("_id", str("a"))]), obj(&[("x", val(100))]))
        .unwrap();
    assert_eq!(n, 1);
    let got: Vec<Value> = herd.find(obj(&[("_id", str("a"))])).to_list().unwrap();
    assert_eq!(got[0].get("_id").unwrap(), &str("a"));
    assert_eq!(got[0].get("x").unwrap(), &val(100));
    assert!(got[0].get("y").is_none(), "replaced wholesale");

    // No match → NoMatch.
    let err = herd
        .replace_one(obj(&[("_id", str("zzz"))]), obj(&[("x", val(1))]))
        .unwrap_err();
    assert!(matches!(err, Error::Server(Status::NoMatch, _)), "{err:?}");
}

#[test]
fn delete_one_and_delete_many() {
    let docs = [
        obj(&[("_id", str("a")), ("k", str("x"))]),
        obj(&[("_id", str("b")), ("k", str("x"))]),
        obj(&[("_id", str("c")), ("k", str("y"))]),
    ];
    let mut c = seeded_client("c", &docs);
    let mut herd = c.collection("c");

    // delete_one: removes exactly one match, returns true.
    assert!(herd.delete_one(obj(&[("k", str("x"))])).unwrap());
    assert_eq!(herd.count(obj(&[("k", str("x"))])).unwrap(), 1);

    // delete_many: removes every match, returns the count.
    assert_eq!(herd.delete_many(obj(&[("k", str("x"))])).unwrap(), 1);
    assert_eq!(herd.count(all()).unwrap(), 1); // only "c" remains

    // delete_one with no match → false (not an error).
    assert!(!herd.delete_one(obj(&[("k", str("nope"))])).unwrap());
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[test]
fn vector_search_over_client() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "vec",
            &[
                obj(&[
                    ("_id", str("p")),
                    ("emb", Value::Array(vec![val(1), val(0)])),
                ]),
                obj(&[
                    ("_id", str("q")),
                    ("emb", Value::Array(vec![val(0), val(1)])),
                ]),
            ],
        )
        .unwrap();
    server
        .state()
        .write()
        .unwrap()
        .get_mut("vec")
        .unwrap()
        .create_vector_index("emb", 2);
    thread::spawn(move || {
        let _ = server.run(&listener);
    });

    let mut c = Client::connect(&addr.to_string()).unwrap();
    let mut herd = c.collection("vec");
    let hits = herd.vector_search("emb", &[1.0, 0.0], 0).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(id_of(&hits[0].0), "p");
    assert!((hits[0].1 - 1.0).abs() < 1e-5);
}

#[test]
fn vector_search_no_index_is_typed_error() {
    let mut c = seeded_client("none", &[]);
    let mut herd = c.collection("none");
    let err = herd.vector_search("emb", &[1.0], 5).unwrap_err();
    assert!(matches!(err, Error::Server(Status::NoIndex, _)), "{err:?}");
}

#[test]
fn index_management_over_client_enables_search() {
    let mut c = seeded_client("ix", &[]);
    let mut herd = c.collection("ix");

    // Without indexes, search is a typed NoIndex error.
    let err = herd.vector_search("emb", &[1.0, 0.0], 5).unwrap_err();
    assert!(matches!(err, Error::Server(Status::NoIndex, _)), "{err:?}");

    herd.insert(&obj(&[
        ("_id", str("a")),
        ("kind", str("cow")),
        ("emb", Value::Array(vec![val(1), val(0)])),
        ("body", str("mooing cow")),
    ]))
    .unwrap();
    herd.insert(&obj(&[
        ("_id", str("b")),
        ("kind", str("pig")),
        ("emb", Value::Array(vec![val(0), val(1)])),
        ("body", str("snorting pig")),
    ]))
    .unwrap();

    // Create the indexes over the wire (no server-side seeding).
    herd.create_index("kind").unwrap();
    herd.create_vector_index("emb", 2).unwrap();
    herd.create_text_index("body").unwrap();

    let hits = herd.vector_search("emb", &[1.0, 0.0], 0).unwrap();
    assert_eq!(id_of(&hits[0].0), "a");
    let thits = herd.text_search("body", "cow", 0).unwrap();
    assert_eq!(id_of(&thits[0].0), "a");
    // Value index drives a range find (no full scan needed).
    assert_eq!(
        herd.count(Value::object_from(vec![("kind".into(), str("cow"))]))
            .unwrap(),
        1
    );

    // Dropping the primary `_id` index is a typed error.
    let err = herd.drop_index("_id").unwrap_err();
    assert!(
        matches!(err, Error::Server(Status::PrimaryIndex, _)),
        "{err:?}"
    );
}

#[test]
fn text_search_over_client() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "docs",
            &[
                obj(&[
                    ("_id", str("moo")),
                    ("text", str("the quick brown cow moo moo")),
                ]),
                obj(&[
                    ("_id", str("milk")),
                    ("text", str("the cold milk of the night")),
                ]),
            ],
        )
        .unwrap();
    server
        .state()
        .write()
        .unwrap()
        .get_mut("docs")
        .unwrap()
        .create_text_index("text");
    thread::spawn(move || {
        let _ = server.run(&listener);
    });

    let mut c = Client::connect(&addr.to_string()).unwrap();
    let mut herd = c.collection("docs");
    let hits = herd.text_search("text", "moo", 0).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(id_of(&hits[0].0), "moo");
    assert!(hits[0].1 > 0.0);
}

#[test]
fn hybrid_search_over_client() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "h",
            &[
                obj(&[
                    ("_id", str("moo")),
                    ("text", str("brown cow moo")),
                    ("emb", Value::Array(vec![val(1), val(0)])),
                ]),
                obj(&[
                    ("_id", str("milk")),
                    ("text", str("cold milk night")),
                    ("emb", Value::Array(vec![val(0), val(1)])),
                ]),
            ],
        )
        .unwrap();
    {
        let mut g = server.state().write().unwrap();
        let coll = g.get_mut("h").unwrap();
        coll.create_text_index("text");
        coll.create_vector_index("emb", 2);
    }
    thread::spawn(move || {
        let _ = server.run(&listener);
    });

    let mut c = Client::connect(&addr.to_string()).unwrap();
    let mut herd = c.collection("h");
    let hits = herd
        .hybrid_search("text", "emb", "moo cow", &[1.0, 0.0], 0)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(id_of(&hits[0].0), "moo");
    assert!(hits[0].1 > 0.0);
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

#[test]
fn group_agg_over_client() {
    let docs = [
        obj(&[("_id", str("a")), ("k", str("x")), ("v", val(1))]),
        obj(&[("_id", str("b")), ("k", str("x")), ("v", val(2))]),
        obj(&[("_id", str("c")), ("k", str("y")), ("v", val(10))]),
    ];
    let mut c = seeded_client("g", &docs);
    let mut herd = c.collection("g");

    // group by k, sum v: x → 3, y → 10 (group-key order).
    let groups: Vec<Value> = herd.find(all()).group("k").agg(AggFn::Sum, "v").unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].get("_id").unwrap(), &str("x"));
    assert_eq!(groups[0].get("sum").unwrap(), &val(3));
    assert_eq!(groups[1].get("_id").unwrap(), &str("y"));
    assert_eq!(groups[1].get("sum").unwrap(), &val(10));

    // count ignores the field; groups x → 2, y → 1.
    let groups: Vec<Value> = herd.find(all()).group("k").agg(AggFn::Count, "v").unwrap();
    assert_eq!(groups[0].get("count").unwrap(), &val(2));
    assert_eq!(groups[1].get("count").unwrap(), &val(1));
}

#[test]
fn group_sort_and_limit_over_client() {
    let docs = [
        obj(&[("_id", str("a")), ("k", str("x")), ("v", val(1))]),
        obj(&[("_id", str("b")), ("k", str("y")), ("v", val(2))]),
        obj(&[("_id", str("c")), ("k", str("z")), ("v", val(3))]),
    ];
    let mut c = seeded_client("g", &docs);
    let mut herd = c.collection("g");

    // Three groups, each summing its single v: x=1,y=2,z=3. Sort by `sum`
    // ascending → x,y,z; limit 2 → x,y.
    let groups: Vec<Value> = herd
        .find(all())
        .group("k")
        .sort("sum", false)
        .limit(2)
        .agg(AggFn::Sum, "v")
        .unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].get("_id").unwrap(), &str("x"));
    assert_eq!(groups[1].get("_id").unwrap(), &str("y"));
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[test]
fn stats_over_client() {
    let docs = [
        obj(&[("_id", str("a")), ("n", val(1))]),
        obj(&[("_id", str("b")), ("n", val(2))]),
    ];
    let mut c = seeded_client("c", &docs);
    let mut herd = c.collection("c");
    let s: Stats = herd.stats().unwrap();
    assert_eq!(s.docs, 2);
    assert!(s.indexes >= 1);
    let fields: Vec<&str> = s.per_index.iter().map(|i| i.field.as_str()).collect();
    assert!(fields.contains(&"_id"));
    // The invariant the server layer relies on.
    let sum: u64 = s.per_index.iter().map(|i| i.memory).sum();
    assert_eq!(s.total_memory, s.docs_memory + sum);
}

// ---------------------------------------------------------------------------
// Multi-request session on one connection
// ---------------------------------------------------------------------------

#[test]
fn one_connection_many_requests_reuses_buffer() {
    let mut c = seeded_client("c", &[]);
    let mut herd = c.collection("c");

    herd.insert(&obj(&[("_id", str("a")), ("n", val(1))]))
        .unwrap();
    herd.insert(&obj(&[("_id", str("b")), ("n", val(2))]))
        .unwrap();
    assert_eq!(herd.count(all()).unwrap(), 2);
    let docs: Vec<Value> = herd.find(all()).to_list().unwrap();
    assert_eq!(docs.len(), 2);
    assert!(herd.exists(obj(&[("n", val(2))])).unwrap());
    herd.delete_many(obj(&[])).unwrap();
    assert_eq!(herd.count(all()).unwrap(), 0);
    // The same connection still works after a delete-all.
    herd.insert(&obj(&[("_id", str("z")), ("n", val(9))]))
        .unwrap();
    assert_eq!(herd.count(all()).unwrap(), 1);
}
