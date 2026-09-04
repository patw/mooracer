//! Integration tests for the MooRacer server over a **real TCP socket**.
//!
//! Each test binds an ephemeral listener, seeds a server, runs the
//! accept-loop + thread pool in a background thread, and drives the protocol
//! with a raw socket — writing length-prefixed FlatBuffers request frames and
//! reading the length-prefixed FlatBuffers response frames. This is the
//! "raw round-trip" the subtask asks for, plus framing, multi-request
//! sessions, typed errors, the thread pool, and the search path.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};
use mooracer_engine::Value;
use mooracer_server::{encode_frame, read_frame, write_frame};
use mooracer_wire as wire;
use wire::{Command, Status, WIRE_VERSION};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build an engine `Value` object from ordered `(key, value)` pairs.
fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

/// Encode an engine [`Value`] as a wire `Value` (for building request payloads).
fn enc<'a>(b: &mut FlatBufferBuilder<'a>, v: &Value) -> WIPOffset<wire::Value<'a>> {
    match v {
        Value::Null => wire::Value::create(
            b,
            &wire::ValueArgs {
                kind: wire::ValueKind::Null,
                ..Default::default()
            },
        ),
        Value::Bool(x) => wire::Value::create(
            b,
            &wire::ValueArgs {
                kind: wire::ValueKind::Bool,
                b: *x,
                ..Default::default()
            },
        ),
        Value::I64(x) => wire::Value::create(
            b,
            &wire::ValueArgs {
                kind: wire::ValueKind::I64,
                i: *x,
                ..Default::default()
            },
        ),
        Value::F64(x) => wire::Value::create(
            b,
            &wire::ValueArgs {
                kind: wire::ValueKind::F64,
                f: *x,
                ..Default::default()
            },
        ),
        Value::Str(s) => {
            let s = b.create_string(s);
            wire::Value::create(
                b,
                &wire::ValueArgs {
                    kind: wire::ValueKind::Str,
                    s: Some(s),
                    ..Default::default()
                },
            )
        }
        Value::Array(items) => {
            let offs: Vec<_> = items.iter().map(|it| enc(b, it)).collect();
            let arr = b.create_vector(&offs);
            wire::Value::create(
                b,
                &wire::ValueArgs {
                    kind: wire::ValueKind::Array,
                    arr: Some(arr),
                    ..Default::default()
                },
            )
        }
        Value::Object(pairs) => {
            let vals: Vec<_> = pairs.iter().map(|(_, val)| enc(b, val)).collect();
            let vals_off = b.create_vector(&vals);
            let keys: Vec<_> = pairs.iter().map(|(k, _)| b.create_string(k)).collect();
            let keys_off = b.create_vector(&keys);
            wire::Value::create(
                b,
                &wire::ValueArgs {
                    kind: wire::ValueKind::Object,
                    keys: Some(keys_off),
                    vals: Some(vals_off),
                    ..Default::default()
                },
            )
        }
    }
}

/// Decode a wire `Value` back into an engine [`Value`] (for assertions).
fn dec(v: &wire::Value) -> Value {
    match v.kind() {
        wire::ValueKind::Null => Value::Null,
        wire::ValueKind::Bool => Value::Bool(v.b()),
        wire::ValueKind::I64 => Value::I64(v.i()),
        wire::ValueKind::F64 => Value::F64(v.f()),
        wire::ValueKind::Str => Value::Str(v.s().expect("s").to_string()),
        wire::ValueKind::Array => {
            let arr = v.arr().expect("arr");
            Value::Array(arr.iter().map(|v| dec(&v)).collect())
        }
        wire::ValueKind::Object => {
            let keys = v.keys().expect("keys");
            let vals = v.vals().expect("vals");
            Value::Object(
                keys.iter()
                    .zip(vals.iter())
                    .map(|(k, val)| (k.to_string(), dec(&val)))
                    .collect(),
            )
        }
        _ => panic!("unknown ValueKind"),
    }
}

/// The decoded `_id` of a wire document, as an owned `String`.
fn doc_id(d: wire::Value) -> String {
    dec(&d).get("_id").unwrap().as_str().unwrap().to_string()
}

/// Build a length-prefixed request frame. `f` builds the command payload on
/// the builder and returns the union discriminant + offset (`None` offset for
/// the empty `Command::NONE`).
fn request_frame<F>(coll: &str, req_id: u64, version: u64, f: F) -> Vec<u8>
where
    F: FnOnce(&mut FlatBufferBuilder) -> (Command, Option<WIPOffset<UnionWIPOffset>>),
{
    let mut b = FlatBufferBuilder::new();
    let (command_type, command) = f(&mut b);
    let coll_off = b.create_string(coll);
    let req = wire::Request::create(
        &mut b,
        &wire::RequestArgs {
            version,
            req_id,
            collection: Some(coll_off),
            command_type,
            command,
        },
    );
    b.finish(req, Some(wire::FILE_IDENTIFIER));
    encode_frame(b.finished_data())
}

/// A raw framed client over a stream. The decoded response borrows from the
/// client's own buffer (reused across calls), so a `recv` result is only valid
/// until the next `send`/`recv` on the same client.
struct Client<T> {
    s: T,
    buf: Vec<u8>,
}

impl<T: Read + Write> Client<T> {
    fn new(stream: T) -> Self {
        Client {
            s: stream,
            buf: Vec::new(),
        }
    }

    fn send(&mut self, frame: &[u8]) -> std::io::Result<()> {
        // `frame` is already length-prefixed (built by `request_frame`).
        self.s.write_all(frame)?;
        self.s.flush()
    }

    fn send_payload(&mut self, payload: &[u8]) -> std::io::Result<()> {
        write_frame(&mut self.s, payload)
    }

    fn recv(&mut self) -> std::io::Result<wire::Response<'_>> {
        self.buf = read_frame(&mut self.s)?;
        flatbuffers::root::<wire::Response>(&self.buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

/// Bind an ephemeral server, run it in the background, and hand back the
/// address. Returns `addr`; the server keeps running for the process lifetime.
fn start_server() -> std::net::SocketAddr {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    addr
}

fn connect(addr: std::net::SocketAddr) -> Client<TcpStream> {
    let stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    Client::new(stream)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn raw_roundtrip_stats() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    // Seed two docs so stats reports a non-trivial count.
    server
        .seed_docs(
            "herd",
            &[
                obj(&[("_id", Value::Str("a".into())), ("n", Value::I64(1))]),
                obj(&[("_id", Value::Str("b".into())), ("n", Value::I64(2))]),
            ],
        )
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });

    let mut c = connect(addr);
    let frame = request_frame("herd", 42, WIRE_VERSION, |b| {
        let cmd = wire::StatsCmd::create(b, &wire::StatsCmdArgs::default());
        (Command::StatsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert_eq!(resp.req_id(), 42, "req_id echoed");
    assert_eq!(resp.version(), WIRE_VERSION);
    assert!(resp.status() == Status::OK, "status: {:?}", resp.status());
    let stats = resp.body_as_stats_res().expect("StatsRes body");
    assert_eq!(stats.docs(), 2);
    assert!(stats.indexes() >= 1);
    // The primary `_id` index is always present.
    let per = stats.per_index().expect("per_index");
    let fields: Vec<&str> = per.iter().map(|s| s.field().unwrap()).collect();
    assert!(fields.contains(&"_id"));
}

#[test]
fn raw_roundtrip_insert_then_find_then_count_on_one_connection() {
    let addr = start_server();
    let mut c = connect(addr);

    // 1) Insert three docs in a single batch.
    let frame = request_frame("cows", 1, WIRE_VERSION, |b| {
        let d1 = enc(
            b,
            &obj(&[("_id", Value::Str("a".into())), ("age", Value::I64(3))]),
        );
        let d2 = enc(
            b,
            &obj(&[("_id", Value::Str("b".into())), ("age", Value::I64(5))]),
        );
        let d3 = enc(
            b,
            &obj(&[("_id", Value::Str("c".into())), ("age", Value::I64(7))]),
        );
        let docs = b.create_vector(&[d1, d2, d3]);
        let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs) });
        (Command::InsertCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    let ins = resp.body_as_insert_res().unwrap();
    let ids: Vec<&str> = ins.ids().unwrap().iter().collect();
    assert_eq!(ids.len(), 3);

    // 2) Find all on the same connection.
    let frame = request_frame("cows", 2, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![])); // {} = all
        let cmd = wire::FindCmd::create(
            b,
            &wire::FindCmdArgs {
                filter: Some(f),
                ..Default::default()
            },
        );
        (Command::FindCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    let docs = resp.body_as_find_res().unwrap().docs().unwrap();
    assert_eq!(docs.len(), 3);

    // 3) Count on the same connection.
    let frame = request_frame("cows", 3, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let cmd = wire::CountCmd::create(b, &wire::CountCmdArgs { filter: Some(f) });
        (Command::CountCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    assert_eq!(resp.body_as_count_res().unwrap().count(), 3);
}

#[test]
fn find_reserves_pipeline_order_and_one_flag() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "cows",
            &[
                obj(&[("_id", Value::Str("a".into())), ("age", Value::I64(1))]),
                obj(&[("_id", Value::Str("b".into())), ("age", Value::I64(3))]),
                obj(&[("_id", Value::Str("c".into())), ("age", Value::I64(2))]),
            ],
        )
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);

    // Sort ascending by `age`: a(1), c(2), b(3).
    let frame = request_frame("cows", 1, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let sf = b.create_string("age");
        let cmd = wire::FindCmd::create(
            b,
            &wire::FindCmdArgs {
                filter: Some(f),
                sort_field: Some(sf),
                sort_desc: false,
                ..Default::default()
            },
        );
        (Command::FindCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    let docs = resp.body_as_find_res().unwrap().docs().unwrap();
    let order: Vec<String> = docs.iter().map(|d| doc_id(d)).collect();
    assert_eq!(
        order,
        vec!["a".to_string(), "c".to_string(), "b".to_string()]
    );

    // Sort descending + limit 2: b(3), c(2).
    let frame = request_frame("cows", 2, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let sf = b.create_string("age");
        let cmd = wire::FindCmd::create(
            b,
            &wire::FindCmdArgs {
                filter: Some(f),
                sort_field: Some(sf),
                sort_desc: true,
                limit: 2,
                ..Default::default()
            },
        );
        (Command::FindCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    let docs = resp.body_as_find_res().unwrap().docs().unwrap();
    let order: Vec<String> = docs.iter().map(|d| doc_id(d)).collect();
    assert_eq!(order, vec!["b".to_string(), "c".to_string()]);

    // one=true: only the first in sorted-ascending order.
    let frame = request_frame("cows", 3, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let sf = b.create_string("age");
        let cmd = wire::FindCmd::create(
            b,
            &wire::FindCmdArgs {
                filter: Some(f),
                sort_field: Some(sf),
                one: true,
                ..Default::default()
            },
        );
        (Command::FindCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    let docs = resp.body_as_find_res().unwrap().docs().unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(
        dec(&docs.get(0)).get("_id").unwrap(),
        &Value::Str("a".into())
    );
}

#[test]
fn typed_error_duplicate_id_over_wire() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs("c", &[obj(&[("_id", Value::Str("1".into()))])])
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);

    // Re-insert `_id = "1"` → DuplicateId.
    let frame = request_frame("c", 7, WIRE_VERSION, |b| {
        let d = enc(b, &obj(&[("_id", Value::Str("1".into()))]));
        let docs = b.create_vector(&[d]);
        let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs) });
        (Command::InsertCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::DuplicateId,
        "status: {:?}",
        resp.status()
    );
    assert!(resp.message().is_some() && !resp.message().unwrap().is_empty());
    assert!(resp.body().is_none(), "error responses carry no body");
}

#[test]
fn typed_error_update_no_match() {
    let addr = start_server();
    let mut c = connect(addr);
    // update_one (`many = false`) against an empty collection → NoMatch.
    let frame = request_frame("empty", 1, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let u = enc(b, &obj(&[("$set", obj(&[("x", Value::I64(1))]))]));
        let cmd = wire::UpdateCmd::create(
            b,
            &wire::UpdateCmdArgs {
                filter: Some(f),
                update: Some(u),
                many: false,
            },
        );
        (Command::UpdateCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::NoMatch,
        "status: {:?}",
        resp.status()
    );

    // The same update with `many = true` succeeds with count 0.
    let frame = request_frame("empty", 2, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let u = enc(b, &obj(&[("$set", obj(&[("x", Value::I64(1))]))]));
        let cmd = wire::UpdateCmd::create(
            b,
            &wire::UpdateCmdArgs {
                filter: Some(f),
                update: Some(u),
                many: true,
            },
        );
        (Command::UpdateCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    assert_eq!(resp.body_as_update_res().unwrap().count(), 0);
}

#[test]
fn unknown_command_returns_unknown() {
    let addr = start_server();
    let mut c = connect(addr);
    // A `Command::NONE` (0) request has no payload.
    let frame = request_frame("c", 9, WIRE_VERSION, |b| {
        let _ = b;
        (Command::NONE, None)
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::UnknownCommand,
        "status: {:?}",
        resp.status()
    );
}

#[test]
fn unsupported_version() {
    let addr = start_server();
    let mut c = connect(addr);
    let frame = request_frame("c", 5, 999, |b| {
        let cmd = wire::StatsCmd::create(b, &wire::StatsCmdArgs::default());
        (Command::StatsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert_eq!(resp.req_id(), 5);
    assert!(
        resp.status() == Status::UnsupportedVersion,
        "status: {:?}",
        resp.status()
    );
}

#[test]
fn malformed_frame_returns_malformed_request() {
    let addr = start_server();
    let mut c = connect(addr);
    // A frame whose payload is not a valid "MOOR" FlatBuffer request.
    c.send_payload(b"definitely not a mooracer frame").unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::MalformedRequest,
        "status: {:?}",
        resp.status()
    );
}

#[test]
fn search_over_wire_returns_hits_when_indexed() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "vec",
            &[
                obj(&[
                    ("_id", Value::Str("p".into())),
                    ("emb", Value::Array(vec![Value::I64(1), Value::I64(0)])),
                ]),
                obj(&[
                    ("_id", Value::Str("q".into())),
                    ("emb", Value::Array(vec![Value::I64(0), Value::I64(1)])),
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
    let mut c = connect(addr);

    // Query (1,0) → "p" is the near-perfect hit.
    let frame = request_frame("vec", 1, WIRE_VERSION, |b| {
        let field = b.create_string("emb");
        let q = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::VectorSearchCmd::create(
            b,
            &wire::VectorSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 0,
            },
        );
        (Command::VectorSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK, "status: {:?}", resp.status());
    let hits = resp.body_as_search_res().unwrap().hits().unwrap();
    assert_eq!(hits.len(), 2);
    let top = dec(hits.get(0).doc().as_ref().unwrap());
    assert_eq!(top.get("_id").unwrap(), &Value::Str("p".into()));
    assert!((hits.get(0).score() - 1.0).abs() < 1e-5);
}

#[test]
fn index_management_over_wire() {
    let addr = start_server();
    let mut c = connect(addr);

    // Insert two docs with a vector (dim 2) and a text field. The collection
    // is created on insert; indexes are created *over the wire* below.
    let frame = request_frame("ix", 1, WIRE_VERSION, |b| {
        let d1 = obj(&[
            ("_id", Value::Str("a".into())),
            ("kind", Value::Str("cow".into())),
            ("emb", Value::Array(vec![Value::I64(1), Value::I64(0)])),
            ("body", Value::Str("mooing cow".into())),
        ]);
        let d2 = obj(&[
            ("_id", Value::Str("b".into())),
            ("kind", Value::Str("pig".into())),
            ("emb", Value::Array(vec![Value::I64(0), Value::I64(1)])),
            ("body", Value::Str("snorting pig".into())),
        ]);
        let v1 = enc(b, &d1);
        let v2 = enc(b, &d2);
        let docs = b.create_vector(&[v1, v2]);
        let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs) });
        (Command::InsertCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::OK);

    // Create value + vector + text indexes via IndexCmd (no server seeding).
    for (kind, field, dim) in [
        (wire::IndexKind::CreateValue, "kind", 0),
        (wire::IndexKind::CreateVector, "emb", 2),
        (wire::IndexKind::CreateText, "body", 0),
    ] {
        let frame = request_frame("ix", 7, WIRE_VERSION, |b| {
            let f = b.create_string(field);
            let cmd = wire::IndexCmd::create(
                b,
                &wire::IndexCmdArgs {
                    kind,
                    field: Some(f),
                    dim,
                },
            );
            (Command::IndexCmd, Some(cmd.as_union_value()))
        });
        c.send(&frame).unwrap();
        let resp = c.recv().unwrap();
        assert!(resp.status() == Status::OK, "index {:?} on {field}", kind.0);
        assert!(resp.body_as_index_res().is_some());
    }

    // Vector search now works (it was NoIndex before the IndexCmd).
    let frame = request_frame("ix", 8, WIRE_VERSION, |b| {
        let field = b.create_string("emb");
        let q = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::VectorSearchCmd::create(
            b,
            &wire::VectorSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 0,
            },
        );
        (Command::VectorSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    let hits = resp.body_as_search_res().unwrap().hits().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(
        dec(hits.get(0).doc().as_ref().unwrap()).get("_id").unwrap(),
        &Value::Str("a".into())
    );

    // Text search also works.
    let frame = request_frame("ix", 9, WIRE_VERSION, |b| {
        let field = b.create_string("body");
        let q = b.create_string("cow");
        let cmd = wire::TextSearchCmd::create(
            b,
            &wire::TextSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 0,
            },
        );
        (Command::TextSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::OK);

    // Dropping the primary `_id` index is an error.
    let frame = request_frame("ix", 10, WIRE_VERSION, |b| {
        let f = b.create_string("_id");
        let cmd = wire::IndexCmd::create(
            b,
            &wire::IndexCmdArgs {
                kind: wire::IndexKind::DropValue,
                field: Some(f),
                dim: 0,
            },
        );
        (Command::IndexCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::PrimaryIndex);

    // Dropping a nonexistent field index is NoIndex.
    let frame = request_frame("ix", 11, WIRE_VERSION, |b| {
        let f = b.create_string("nope");
        let cmd = wire::IndexCmd::create(
            b,
            &wire::IndexCmdArgs {
                kind: wire::IndexKind::DropValue,
                field: Some(f),
                dim: 0,
            },
        );
        (Command::IndexCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::NoIndex);

    // Dropping the vector index makes search return NoIndex again.
    let frame = request_frame("ix", 12, WIRE_VERSION, |b| {
        let f = b.create_string("emb");
        let cmd = wire::IndexCmd::create(
            b,
            &wire::IndexCmdArgs {
                kind: wire::IndexKind::DropVector,
                field: Some(f),
                dim: 0,
            },
        );
        (Command::IndexCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::OK);

    let frame = request_frame("ix", 13, WIRE_VERSION, |b| {
        let field = b.create_string("emb");
        let q = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::VectorSearchCmd::create(
            b,
            &wire::VectorSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 5,
            },
        );
        (Command::VectorSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    assert!(c.recv().unwrap().status() == Status::NoIndex);
}

#[test]
fn search_without_index_returns_no_index() {
    let addr = start_server();
    let mut c = connect(addr);
    // No vector index was created on this (empty) collection.
    let frame = request_frame("none", 1, WIRE_VERSION, |b| {
        let field = b.create_string("emb");
        let q = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::VectorSearchCmd::create(
            b,
            &wire::VectorSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 5,
            },
        );
        (Command::VectorSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::NoIndex,
        "status: {:?}",
        resp.status()
    );
}

#[test]
fn group_over_wire() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "g",
            &[
                obj(&[
                    ("_id", Value::Str("a".into())),
                    ("k", Value::Str("x".into())),
                    ("v", Value::I64(1)),
                ]),
                obj(&[
                    ("_id", Value::Str("b".into())),
                    ("k", Value::Str("x".into())),
                    ("v", Value::I64(2)),
                ]),
                obj(&[
                    ("_id", Value::Str("c".into())),
                    ("k", Value::Str("y".into())),
                    ("v", Value::I64(10)),
                ]),
            ],
        )
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);
    let frame = request_frame("g", 1, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let gf = b.create_string("k");
        let af = b.create_string("v");
        let cmd = wire::GroupCmd::create(
            b,
            &wire::GroupCmdArgs {
                filter: Some(f),
                group_field: Some(gf),
                agg_fn: wire::AggFn::Sum,
                agg_field: Some(af),
                ..Default::default()
            },
        );
        (Command::GroupCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    let groups = resp.body_as_group_res().unwrap().groups().unwrap();
    assert_eq!(groups.len(), 2);
    // Group keys in total order: "x" then "y".
    let g0 = dec(&groups.get(0));
    let g1 = dec(&groups.get(1));
    assert_eq!(g0.get("_id").unwrap(), &Value::Str("x".into()));
    assert_eq!(g0.get("sum").unwrap(), &Value::I64(3));
    assert_eq!(g1.get("_id").unwrap(), &Value::Str("y".into()));
    assert_eq!(g1.get("sum").unwrap(), &Value::I64(10));
}

#[test]
fn thread_pool_serves_many_concurrent_connections() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = mooracer_server::Server::with_pool_size(4);
    thread::spawn(move || {
        let _ = server.run(&listener);
    });

    let n = 8usize;
    let mut handles = Vec::new();
    for t in 0..n {
        handles.push(thread::spawn(move || {
            let mut c = connect(addr);
            let id = format!("t{t}");
            let frame = request_frame("conc", 1, WIRE_VERSION, |b| {
                let d = enc(b, &obj(&[("_id", Value::Str(id.clone()))]));
                let docs = b.create_vector(&[d]);
                let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs) });
                (Command::InsertCmd, Some(cmd.as_union_value()))
            });
            c.send(&frame).unwrap();
            let resp = c.recv().unwrap();
            assert!(
                resp.status() == Status::OK,
                "thread {t}: {:?}",
                resp.status()
            );
            assert_eq!(resp.body_as_insert_res().unwrap().ids().unwrap().len(), 1);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // A final stats request sees all eight inserted docs.
    let mut c = connect(addr);
    let frame = request_frame("conc", 1, WIRE_VERSION, |b| {
        let cmd = wire::StatsCmd::create(b, &wire::StatsCmdArgs::default());
        (Command::StatsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert_eq!(resp.body_as_stats_res().unwrap().docs(), n as u64);
}

#[test]
fn delete_and_replace_over_wire() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "d",
            &[
                obj(&[("_id", Value::Str("a".into())), ("v", Value::I64(1))]),
                obj(&[("_id", Value::Str("b".into())), ("v", Value::I64(2))]),
                obj(&[("_id", Value::Str("c".into())), ("v", Value::I64(3))]),
            ],
        )
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);

    // replace_one: match a, replace wholesale.
    let frame = request_frame("d", 1, WIRE_VERSION, |b| {
        let f = enc(b, &obj(&[("_id", Value::Str("a".into()))]));
        let nd = enc(
            b,
            &obj(&[("_id", Value::Str("a".into())), ("v", Value::I64(99))]),
        );
        let cmd = wire::ReplaceCmd::create(
            b,
            &wire::ReplaceCmdArgs {
                filter: Some(f),
                new_doc: Some(nd),
            },
        );
        (Command::ReplaceCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    assert_eq!(resp.body_as_replace_res().unwrap().count(), 1);

    // delete_many with empty filter → removes all 3.
    let frame = request_frame("d", 2, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let cmd = wire::DeleteCmd::create(
            b,
            &wire::DeleteCmdArgs {
                filter: Some(f),
                many: true,
            },
        );
        (Command::DeleteCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    assert_eq!(resp.body_as_delete_res().unwrap().count(), 3);

    // count is now 0.
    let frame = request_frame("d", 3, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let cmd = wire::CountCmd::create(b, &wire::CountCmdArgs { filter: Some(f) });
        (Command::CountCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert_eq!(resp.body_as_count_res().unwrap().count(), 0);
}

#[test]
fn exists_over_wire_matches_and_no_match() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "c",
            &[
                obj(&[("_id", Value::Str("a".into())), ("age", Value::I64(30))]),
                obj(&[("_id", Value::Str("b".into())), ("age", Value::I64(20))]),
            ],
        )
        .unwrap();
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);

    // A filter that matches a doc -> exists = true.
    let frame = request_frame("c", 1, WIRE_VERSION, |b| {
        let f = enc(b, &obj(&[("age", Value::I64(30))]));
        let cmd = wire::ExistsCmd::create(b, &wire::ExistsCmdArgs { filter: Some(f) });
        (Command::ExistsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK, "status: {:?}", resp.status());
    assert!(resp.body_as_exists_res().unwrap().exists());

    // A filter that matches nothing -> exists = false.
    let frame = request_frame("c", 2, WIRE_VERSION, |b| {
        let f = enc(b, &obj(&[("age", Value::I64(9999))]));
        let cmd = wire::ExistsCmd::create(b, &wire::ExistsCmdArgs { filter: Some(f) });
        (Command::ExistsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK);
    assert!(!resp.body_as_exists_res().unwrap().exists());

    // Empty filter -> all -> true.
    let frame = request_frame("c", 3, WIRE_VERSION, |b| {
        let f = enc(b, &Value::Object(vec![]));
        let cmd = wire::ExistsCmd::create(b, &wire::ExistsCmdArgs { filter: Some(f) });
        (Command::ExistsCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.body_as_exists_res().unwrap().exists());
}

#[test]
fn text_search_over_wire_returns_hits_when_indexed() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "docs",
            &[
                obj(&[
                    ("_id", Value::Str("moo".into())),
                    ("text", Value::Str("the quick brown cow moo moo".into())),
                ]),
                obj(&[
                    ("_id", Value::Str("milk".into())),
                    ("text", Value::Str("the cold milk of the night".into())),
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
    let mut c = connect(addr);

    // Query "moo" -> the "moo" doc (contains it twice) is the top hit.
    let frame = request_frame("docs", 1, WIRE_VERSION, |b| {
        let field = b.create_string("text");
        let q = b.create_string("moo");
        let cmd = wire::TextSearchCmd::create(
            b,
            &wire::TextSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 0,
            },
        );
        (Command::TextSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK, "status: {:?}", resp.status());
    let hits = resp.body_as_search_res().unwrap().hits().unwrap();
    assert!(!hits.is_empty());
    let top = dec(hits.get(0).doc().as_ref().unwrap());
    assert_eq!(top.get("_id").unwrap(), &Value::Str("moo".into()));
    assert!(hits.get(0).score() > 0.0);
}

#[test]
fn text_search_without_index_returns_no_index() {
    let addr = start_server();
    let mut c = connect(addr);
    let frame = request_frame("none", 1, WIRE_VERSION, |b| {
        let field = b.create_string("text");
        let q = b.create_string("moo");
        let cmd = wire::TextSearchCmd::create(
            b,
            &wire::TextSearchCmdArgs {
                field: Some(field),
                query: Some(q),
                limit: 5,
            },
        );
        (Command::TextSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::NoIndex,
        "status: {:?}",
        resp.status()
    );
}

#[test]
fn hybrid_search_over_wire_returns_hits_when_both_indexed() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    server
        .seed_docs(
            "h",
            &[
                obj(&[
                    ("_id", Value::Str("moo".into())),
                    ("text", Value::Str("brown cow moo".into())),
                    ("emb", Value::Array(vec![Value::I64(1), Value::I64(0)])),
                ]),
                obj(&[
                    ("_id", Value::Str("milk".into())),
                    ("text", Value::Str("cold milk night".into())),
                    ("emb", Value::Array(vec![Value::I64(0), Value::I64(1)])),
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
    let mut c = connect(addr);

    // Query both signals; the "moo" doc wins on both -> top hit.
    let frame = request_frame("h", 1, WIRE_VERSION, |b| {
        let tf = b.create_string("text");
        let vf = b.create_string("emb");
        let qt = b.create_string("moo cow");
        let qv = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::HybridSearchCmd::create(
            b,
            &wire::HybridSearchCmdArgs {
                text_field: Some(tf),
                vec_field: Some(vf),
                query_text: Some(qt),
                query_vec: Some(qv),
                limit: 0,
            },
        );
        (Command::HybridSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(resp.status() == Status::OK, "status: {:?}", resp.status());
    let hits = resp.body_as_search_res().unwrap().hits().unwrap();
    assert_eq!(hits.len(), 2);
    let top = dec(hits.get(0).doc().as_ref().unwrap());
    assert_eq!(top.get("_id").unwrap(), &Value::Str("moo".into()));
    assert!(hits.get(0).score() > 0.0);
}

#[test]
fn hybrid_search_missing_one_index_returns_no_index() {
    let (server, listener) = mooracer_server::Server::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    // Seed a doc with a vector field but create ONLY the vector index (no text index).
    server
        .seed_docs(
            "h",
            &[obj(&[
                ("_id", Value::Str("a".into())),
                ("emb", Value::Array(vec![Value::I64(1), Value::I64(0)])),
            ])],
        )
        .unwrap();
    server
        .state()
        .write()
        .unwrap()
        .get_mut("h")
        .unwrap()
        .create_vector_index("emb", 2);
    thread::spawn(move || {
        let _ = server.run(&listener);
    });
    let mut c = connect(addr);
    let frame = request_frame("h", 1, WIRE_VERSION, |b| {
        let tf = b.create_string("text"); // no text index exists
        let vf = b.create_string("emb");
        let qt = b.create_string("moo");
        let qv = b.create_vector(&[1.0f32, 0.0]);
        let cmd = wire::HybridSearchCmd::create(
            b,
            &wire::HybridSearchCmdArgs {
                text_field: Some(tf),
                vec_field: Some(vf),
                query_text: Some(qt),
                query_vec: Some(qv),
                limit: 0,
            },
        );
        (Command::HybridSearchCmd, Some(cmd.as_union_value()))
    });
    c.send(&frame).unwrap();
    let resp = c.recv().unwrap();
    assert!(
        resp.status() == Status::NoIndex,
        "status: {:?}",
        resp.status()
    );
}
