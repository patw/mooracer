//! Round-trip tests for the MooRacer wire schema (FlatBuffers).
//!
//! These are the contract tests for `schema/mooracer.fbs`: every Value kind,
//! every command payload, every response body, and every status code must
//! encode and decode losslessly, and the numeric discriminants are pinned
//! (they are the wire format — changing them is a version bump).
//!
//! A `FlatBufferBuilder` cannot be reused after `finish()`, so every request
//! buffer is built through [`build_request`], which owns a fresh builder.

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use mooracer_wire::*;

/// A tiny test-local value tree mirroring the engine's `Value`.
#[derive(Debug, PartialEq)]
enum TV {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    Array(Vec<TV>),
    Object(Vec<(String, TV)>),
}

impl TV {
    fn obj(pairs: Vec<(&str, TV)>) -> Self {
        TV::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
}

/// Encode a test tree as a wire `Value`.
fn enc<'a>(b: &mut FlatBufferBuilder<'a>, v: &TV) -> WIPOffset<Value<'a>> {
    match v {
        TV::Null => Value::create(
            b,
            &ValueArgs {
                kind: ValueKind::Null,
                ..Default::default()
            },
        ),
        TV::Bool(x) => Value::create(
            b,
            &ValueArgs {
                kind: ValueKind::Bool,
                b: *x,
                ..Default::default()
            },
        ),
        TV::I64(x) => Value::create(
            b,
            &ValueArgs {
                kind: ValueKind::I64,
                i: *x,
                ..Default::default()
            },
        ),
        TV::F64(x) => Value::create(
            b,
            &ValueArgs {
                kind: ValueKind::F64,
                f: *x,
                ..Default::default()
            },
        ),
        TV::Str(s) => {
            let s = b.create_string(s);
            Value::create(
                b,
                &ValueArgs {
                    kind: ValueKind::Str,
                    s: Some(s),
                    ..Default::default()
                },
            )
        }
        TV::Array(items) => {
            let offs: Vec<_> = items.iter().map(|it| enc(b, it)).collect();
            let arr = b.create_vector(&offs);
            Value::create(
                b,
                &ValueArgs {
                    kind: ValueKind::Array,
                    arr: Some(arr),
                    ..Default::default()
                },
            )
        }
        TV::Object(pairs) => {
            let vals: Vec<_> = pairs.iter().map(|(_, v)| enc(b, v)).collect();
            let vals_off = b.create_vector(&vals);
            let keys: Vec<_> = pairs.iter().map(|(k, _)| b.create_string(k)).collect();
            let keys_off = b.create_vector(&keys);
            Value::create(
                b,
                &ValueArgs {
                    kind: ValueKind::Object,
                    keys: Some(keys_off),
                    vals: Some(vals_off),
                    ..Default::default()
                },
            )
        }
    }
}

/// Decode a wire `Value` back to a test tree. Unknown kinds (future protocol
/// versions) must not silently decode as garbage — they panic loudly.
fn dec(v: &Value) -> TV {
    match v.kind() {
        ValueKind::Null => TV::Null,
        ValueKind::Bool => TV::Bool(v.b()),
        ValueKind::I64 => TV::I64(v.i()),
        ValueKind::F64 => TV::F64(v.f()),
        ValueKind::Str => TV::Str(v.s().expect("Str kind carries s").to_string()),
        ValueKind::Array => {
            let arr = v.arr().expect("Array kind carries arr");
            TV::Array(arr.iter().map(|v| dec(&v)).collect())
        }
        ValueKind::Object => {
            let keys = v.keys().expect("Object kind carries keys");
            let vals = v.vals().expect("Object kind carries vals");
            let mut out = Vec::with_capacity(keys.len());
            for (k, val) in keys.iter().zip(vals.iter()) {
                out.push((k.to_string(), dec(&val)));
            }
            TV::Object(out)
        }
        ValueKind(k) => panic!("unknown ValueKind on wire: {k} (protocol break?)"),
    }
}

/// Encode one Request buffer. Owns a fresh `FlatBufferBuilder` (a builder
/// cannot be reused after `finish`), so `f` builds the command payload on it
/// and returns the union discriminant + offset.
fn build_request<F>(collection: &str, f: F) -> Vec<u8>
where
    F: FnOnce(&mut FlatBufferBuilder) -> (Command, WIPOffset<flatbuffers::UnionWIPOffset>),
{
    let mut b = FlatBufferBuilder::new();
    let (command_type, command) = f(&mut b);
    let coll = b.create_string(collection);
    let req = Request::create(
        &mut b,
        &RequestArgs {
            req_id: 1234,
            collection: Some(coll),
            command_type,
            command: Some(command),
            ..Default::default()
        },
    );
    b.finish(req, Some(FILE_IDENTIFIER));
    b.finished_data().to_vec()
}

#[test]
fn value_tree_roundtrips_all_kinds_losslessly() {
    // Carry the tree through a FindCmd filter (a real Value slot).
    let tree = TV::obj(vec![
        ("null", TV::Null),
        ("t", TV::Bool(true)),
        ("f", TV::Bool(false)),
        ("i", TV::I64(i64::MIN)),
        ("big_i", TV::I64(4_611_686_018_427_387_903)), // i64::MAX
        ("f64", TV::F64(-0.125)),
        ("s", TV::Str("moo 🐄".to_string())),
        (
            "nested",
            TV::obj(vec![(
                "deep",
                TV::Array(vec![TV::I64(1), TV::Null, TV::Str("x".into())]),
            )]),
        ),
        ("empty_arr", TV::Array(vec![])),
        ("empty_obj", TV::Object(vec![])),
    ]);
    let buf = build_request("herd", |b| {
        let filter = enc(b, &tree);
        let cmd = FindCmd::create(
            b,
            &FindCmdArgs {
                filter: Some(filter),
                ..Default::default()
            },
        );
        (Command::FindCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert!(r.command_type() == Command::FindCmd);
    let f = r.command_as_find_cmd().expect("FindCmd payload");
    assert_eq!(dec(&f.filter().expect("filter Value")), tree);
}

#[test]
fn value_object_preserves_key_order() {
    // Insertion order (not sorted order) must survive the round-trip: the
    // engine's Object is an ordered Vec of pairs.
    let tree = TV::Object(vec![
        ("zeta".to_string(), TV::I64(1)),
        ("alpha".to_string(), TV::I64(2)),
        ("mid".to_string(), TV::I64(3)),
    ]);
    let buf = build_request("c", |b| {
        let filter = enc(b, &tree);
        let cmd = CountCmd::create(
            b,
            &CountCmdArgs {
                filter: Some(filter),
            },
        );
        (Command::CountCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let c = r.command_as_count_cmd().unwrap();
    let keys = c.filter().unwrap().keys().unwrap();
    let got: Vec<&str> = keys.iter().collect();
    assert_eq!(got, vec!["zeta", "alpha", "mid"]);
}

#[test]
fn insert_command_roundtrips_multiple_docs() {
    let d1 = TV::obj(vec![("name", TV::Str("daisy".into())), ("age", TV::I64(9))]);
    let d2 = TV::obj(vec![
        ("name", TV::Str("hilde".into())),
        (
            "tags",
            TV::Array(vec![TV::Str("a".into()), TV::Str("b".into())]),
        ),
    ]);
    let buf = build_request("cows", |b| {
        let v1 = enc(b, &d1);
        let v2 = enc(b, &d2);
        let docs = b.create_vector(&[v1, v2]);
        let cmd = InsertCmd::create(b, &InsertCmdArgs { docs: Some(docs) });
        (Command::InsertCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let ins = r.command_as_insert_cmd().unwrap();
    let docs = ins.docs().unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(dec(&docs.get(0)), d1);
    assert_eq!(dec(&docs.get(1)), d2);
}

#[test]
fn find_command_roundtrips_pipeline() {
    let filter = TV::obj(vec![("age", TV::obj(vec![("$gte", TV::I64(5))]))]);
    let buf = build_request("c", |b| {
        let f = enc(b, &filter);
        let sf = b.create_string("age");
        let cmd = FindCmd::create(
            b,
            &FindCmdArgs {
                filter: Some(f),
                sort_field: Some(sf),
                sort_desc: true,
                skip: 10,
                limit: 25,
                one: true,
            },
        );
        (Command::FindCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let fnd = r.command_as_find_cmd().unwrap();
    assert_eq!(dec(&fnd.filter().unwrap()), filter);
    assert_eq!(fnd.sort_field().unwrap(), "age");
    assert!(fnd.sort_desc());
    assert_eq!(fnd.skip(), 10);
    assert_eq!(fnd.limit(), 25);
    assert!(fnd.one());
}

#[test]
fn find_command_defaults_are_unsorted_unlimited() {
    let buf = build_request("c", |b| {
        let f = enc(b, &TV::Object(vec![])); // {} = all
        let cmd = FindCmd::create(
            b,
            &FindCmdArgs {
                filter: Some(f),
                ..Default::default()
            },
        );
        (Command::FindCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let fnd = r.command_as_find_cmd().unwrap();
    assert_eq!(dec(&fnd.filter().unwrap()), TV::Object(vec![]));
    assert!(fnd.sort_field().is_none());
    assert!(!fnd.sort_desc());
    assert_eq!(fnd.skip(), 0);
    assert_eq!(fnd.limit(), 0); // 0 = no limit
    assert!(!fnd.one());
}

#[test]
fn update_replace_delete_commands_roundtrip() {
    let filter = TV::obj(vec![("kind", TV::Str("cow".into()))]);
    let update = TV::obj(vec![
        ("$set", TV::obj(vec![("name", TV::Str("moo".into()))])),
        ("$inc", TV::obj(vec![("age", TV::I64(1))])),
    ]);
    let new_doc = TV::obj(vec![("name", TV::Str("bess".into()))]);

    let buf1 = build_request("c", |b| {
        let f = enc(b, &filter);
        let u = enc(b, &update);
        let up = UpdateCmd::create(
            b,
            &UpdateCmdArgs {
                filter: Some(f),
                update: Some(u),
                many: false,
            },
        );
        (Command::UpdateCmd, up.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf1).unwrap();
    let uc = r.command_as_update_cmd().unwrap();
    assert_eq!(dec(&uc.filter().unwrap()), filter);
    assert_eq!(dec(&uc.update().unwrap()), update);
    assert!(!uc.many());

    let buf2 = build_request("c", |b| {
        let f = enc(b, &filter);
        let nd = enc(b, &new_doc);
        let rp = ReplaceCmd::create(
            b,
            &ReplaceCmdArgs {
                filter: Some(f),
                new_doc: Some(nd),
            },
        );
        (Command::ReplaceCmd, rp.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf2).unwrap();
    let rc = r.command_as_replace_cmd().unwrap();
    assert_eq!(dec(&rc.new_doc().unwrap()), new_doc);

    let buf3 = build_request("c", |b| {
        let f = enc(b, &filter);
        let dl = DeleteCmd::create(
            b,
            &DeleteCmdArgs {
                filter: Some(f),
                many: true,
            },
        );
        (Command::DeleteCmd, dl.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf3).unwrap();
    let dc = r.command_as_delete_cmd().unwrap();
    assert!(dc.many());
}

#[test]
fn search_commands_roundtrip() {
    let buf = build_request("c", |b| {
        let field = b.create_string("emb");
        let qvec = b.create_vector(&[0.25f32, -1.5, 3.0, 0.0]);
        let vc = VectorSearchCmd::create(
            b,
            &VectorSearchCmdArgs {
                field: Some(field),
                query: Some(qvec),
                limit: 5,
            },
        );
        (Command::VectorSearchCmd, vc.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let vs = r.command_as_vector_search_cmd().unwrap();
    assert_eq!(vs.field().unwrap(), "emb");
    let q = vs.query().unwrap();
    assert_eq!(q.len(), 4);
    assert_eq!(q.get(0), 0.25f32);
    assert_eq!(q.get(1), -1.5);
    assert_eq!(q.get(2), 3.0);
    assert_eq!(vs.limit(), 5);

    let buf = build_request("c", |b| {
        let tf = b.create_string("body");
        let tq = b.create_string("mooing loudly");
        let tc = TextSearchCmd::create(
            b,
            &TextSearchCmdArgs {
                field: Some(tf),
                query: Some(tq),
                limit: 0,
            },
        );
        (Command::TextSearchCmd, tc.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let ts = r.command_as_text_search_cmd().unwrap();
    assert_eq!(ts.field().unwrap(), "body");
    assert_eq!(ts.query().unwrap(), "mooing loudly");
    assert_eq!(ts.limit(), 0);

    let buf = build_request("c", |b| {
        let tf2 = b.create_string("body");
        let vf2 = b.create_string("emb");
        let qt = b.create_string("loud");
        let qv = b.create_vector(&[1.0f32, 0.0]);
        let hc = HybridSearchCmd::create(
            b,
            &HybridSearchCmdArgs {
                text_field: Some(tf2),
                vec_field: Some(vf2),
                query_text: Some(qt),
                query_vec: Some(qv),
                limit: 3,
            },
        );
        (Command::HybridSearchCmd, hc.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let hs = r.command_as_hybrid_search_cmd().unwrap();
    assert_eq!(hs.text_field().unwrap(), "body");
    assert_eq!(hs.vec_field().unwrap(), "emb");
    assert_eq!(hs.query_text().unwrap(), "loud");
    assert_eq!(hs.query_vec().unwrap().get(0), 1.0f32);
    assert_eq!(hs.limit(), 3);
}

#[test]
fn group_command_roundtrips_full_pipeline() {
    let filter = TV::obj(vec![("region", TV::Str("north".into()))]);
    let buf = build_request("c", |b| {
        let f = enc(b, &filter);
        let sf = b.create_string("age");
        let gf = b.create_string("kind");
        let af = b.create_string("age");
        let gsf = b.create_string("_id");
        let cmd = GroupCmd::create(
            b,
            &GroupCmdArgs {
                filter: Some(f),
                sort_field: Some(sf),
                sort_desc: true,
                skip: 2,
                limit: 100,
                group_field: Some(gf),
                agg_fn: AggFn::Mean,
                agg_field: Some(af),
                group_sort_field: Some(gsf),
                group_sort_desc: false,
                group_limit: 10,
            },
        );
        (Command::GroupCmd, cmd.as_union_value())
    });

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let g = r.command_as_group_cmd().unwrap();
    assert_eq!(dec(&g.filter().unwrap()), filter);
    assert_eq!(g.sort_field().unwrap(), "age");
    assert!(g.sort_desc());
    assert_eq!(g.skip(), 2);
    assert_eq!(g.limit(), 100);
    assert_eq!(g.group_field().unwrap(), "kind");
    assert!(g.agg_fn() == AggFn::Mean);
    assert_eq!(g.agg_field().unwrap(), "age");
    assert_eq!(g.group_sort_field().unwrap(), "_id");
    assert!(!g.group_sort_desc());
    assert_eq!(g.group_limit(), 10);
}

#[test]
fn index_command_roundtrips_all_kinds() {
    // create value index (dim ignored)
    let buf = build_request("c", |b| {
        let f = b.create_string("age");
        let cmd = IndexCmd::create(
            b,
            &IndexCmdArgs {
                kind: IndexKind::CreateValue,
                field: Some(f),
                dim: 0,
            },
        );
        (Command::IndexCmd, cmd.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert!(r.command_type() == Command::IndexCmd);
    let ic = r.command_as_index_cmd().unwrap();
    assert!(ic.kind() == IndexKind::CreateValue);
    assert_eq!(ic.field().unwrap(), "age");
    assert_eq!(ic.dim(), 0);

    // create vector index (dim carried)
    let buf = build_request("c", |b| {
        let f = b.create_string("embedding");
        let cmd = IndexCmd::create(
            b,
            &IndexCmdArgs {
                kind: IndexKind::CreateVector,
                field: Some(f),
                dim: 8,
            },
        );
        (Command::IndexCmd, cmd.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    let ic = r.command_as_index_cmd().unwrap();
    assert!(ic.kind() == IndexKind::CreateVector);
    assert_eq!(ic.field().unwrap(), "embedding");
    assert_eq!(ic.dim(), 8);

    // create text index
    let buf = build_request("c", |b| {
        let f = b.create_string("body");
        let cmd = IndexCmd::create(
            b,
            &IndexCmdArgs {
                kind: IndexKind::CreateText,
                field: Some(f),
                dim: 0,
            },
        );
        (Command::IndexCmd, cmd.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert!(r.command_as_index_cmd().unwrap().kind() == IndexKind::CreateText);
}

#[test]
fn count_exists_stats_commands_roundtrip() {
    let filter = TV::obj(vec![("a", TV::I64(1))]);

    let buf = build_request("c", |b| {
        let f = enc(b, &filter);
        let cc = CountCmd::create(b, &CountCmdArgs { filter: Some(f) });
        (Command::CountCmd, cc.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert_eq!(
        dec(&r.command_as_count_cmd().unwrap().filter().unwrap()),
        filter
    );

    let buf = build_request("c", |b| {
        let f = enc(b, &filter);
        let ec = ExistsCmd::create(b, &ExistsCmdArgs { filter: Some(f) });
        (Command::ExistsCmd, ec.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert!(r.command_as_exists_cmd().is_some());

    let buf = build_request("c", |b| {
        let sc = StatsCmd::create(b, &StatsCmdArgs::default());
        (Command::StatsCmd, sc.as_union_value())
    });
    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert!(r.command_type() == Command::StatsCmd);
}

#[test]
fn request_envelope_carries_version_reqid_collection_identifier() {
    let buf = build_request("moo_col", |b| {
        let sc = StatsCmd::create(b, &StatsCmdArgs::default());
        (Command::StatsCmd, sc.as_union_value())
    });

    // The buffer carries the "MOOR" file identifier (frame sanity check).
    assert!(flatbuffers::buffer_has_identifier(
        &buf,
        FILE_IDENTIFIER,
        false
    ));

    let r = flatbuffers::root::<Request>(&buf).unwrap();
    assert_eq!(r.version(), WIRE_VERSION); // default = 1
    assert_eq!(r.req_id(), 1234);
    assert_eq!(r.collection().unwrap(), "moo_col");

    // The generated convenience root decoder agrees.
    let r2 = root_as_request(&buf).unwrap();
    assert_eq!(r2.req_id(), 1234);
}

#[test]
fn request_buffer_passes_full_verifier() {
    let tree = TV::obj(vec![(
        "a",
        TV::Array(vec![TV::obj(vec![("b", TV::I64(7))])]),
    )]);
    let buf = build_request("c", |b| {
        let f = enc(b, &tree);
        let cmd = FindCmd::create(
            b,
            &FindCmdArgs {
                filter: Some(f),
                ..Default::default()
            },
        );
        (Command::FindCmd, cmd.as_union_value())
    });

    use flatbuffers::Verifiable;
    let opts = flatbuffers::VerifierOptions::default();
    let mut v = flatbuffers::Verifier::new(&opts, &buf);
    // The 25.x crate's `flatbuffers::root` verifies through the root-offset
    // wrapper at position 0 — that is the correct verifier entry point.
    // (Passing `Request::run_verifier(v, 0)` treats 0 as a *table* position,
    // which is wrong for a root-offset-prefixed buffer.)
    <flatbuffers::ForwardsUOffset<Request>>::run_verifier(&mut v, 0)
        .expect("well-formed buffer verifies");
}

#[test]
fn response_ok_roundtrips_every_body() {
    let doc1 = TV::obj(vec![("_id", TV::Str("a".into())), ("n", TV::I64(1))]);
    let doc2 = TV::obj(vec![("_id", TV::Str("b".into())), ("n", TV::I64(2))]);
    let group = TV::obj(vec![("_id", TV::Str("x".into())), ("count", TV::I64(3))]);

    let mut b = FlatBufferBuilder::new();
    let id1 = b.create_string("id1");
    let id2 = b.create_string("id2");
    let ids = b.create_vector(&[id1, id2]);
    let ins = InsertRes::create(&mut b, &InsertResArgs { ids: Some(ids) });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            req_id: 9,
            status: Status::OK,
            body_type: ResponseBody::InsertRes,
            body: Some(ins.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, Some(FILE_IDENTIFIER));
    let buf = b.finished_data().to_vec();
    let r = flatbuffers::root::<Response>(&buf).unwrap();
    assert_eq!(r.req_id(), 9);
    assert_eq!(r.version(), WIRE_VERSION);
    assert!(r.status() == Status::OK);
    assert!(r.message().is_none());
    let body = r.body_as_insert_res().unwrap();
    let got: Vec<&str> = body.ids().unwrap().iter().collect();
    assert_eq!(got, vec!["id1", "id2"]);

    // FindRes with two docs.
    let mut b = FlatBufferBuilder::new();
    let d1 = enc(&mut b, &doc1);
    let d2 = enc(&mut b, &doc2);
    let docs = b.create_vector(&[d1, d2]);
    let fr = FindRes::create(&mut b, &FindResArgs { docs: Some(docs) });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            req_id: 1,
            body_type: ResponseBody::FindRes,
            body: Some(fr.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    let r = flatbuffers::root::<Response>(b.finished_data()).unwrap();
    let docs = r.body_as_find_res().unwrap().docs().unwrap();
    assert_eq!(dec(&docs.get(0)), doc1);
    assert_eq!(dec(&docs.get(1)), doc2);

    // Scalar bodies.
    let mut b = FlatBufferBuilder::new();
    let cr = CountRes::create(&mut b, &CountResArgs { count: 42 });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::CountRes,
            body: Some(cr.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert_eq!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_count_res()
            .unwrap()
            .count(),
        42
    );

    let mut b = FlatBufferBuilder::new();
    let er = ExistsRes::create(&mut b, &ExistsResArgs { exists: true });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::ExistsRes,
            body: Some(er.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_exists_res()
            .unwrap()
            .exists()
    );

    let mut b = FlatBufferBuilder::new();
    let ur = UpdateRes::create(&mut b, &UpdateResArgs { count: 1 });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::UpdateRes,
            body: Some(ur.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert_eq!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_update_res()
            .unwrap()
            .count(),
        1
    );

    let mut b = FlatBufferBuilder::new();
    let rpr = ReplaceRes::create(&mut b, &ReplaceResArgs { count: 1 });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::ReplaceRes,
            body: Some(rpr.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_replace_res()
            .is_some()
    );

    let mut b = FlatBufferBuilder::new();
    let dl = DeleteRes::create(&mut b, &DeleteResArgs { count: 7 });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::DeleteRes,
            body: Some(dl.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert_eq!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_delete_res()
            .unwrap()
            .count(),
        7
    );

    // SearchRes: two scored hits, best-first order preserved.
    let mut b = FlatBufferBuilder::new();
    let h1d = enc(&mut b, &doc1);
    let h1 = SearchHit::create(
        &mut b,
        &SearchHitArgs {
            doc: Some(h1d),
            score: 0.87,
        },
    );
    let h2d = enc(&mut b, &doc2);
    let h2 = SearchHit::create(
        &mut b,
        &SearchHitArgs {
            doc: Some(h2d),
            score: 1.5,
        },
    );
    let hits = b.create_vector(&[h2, h1]);
    let sr = SearchRes::create(&mut b, &SearchResArgs { hits: Some(hits) });
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::SearchRes,
            body: Some(sr.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    let hits = flatbuffers::root::<Response>(b.finished_data())
        .unwrap()
        .body_as_search_res()
        .unwrap()
        .hits()
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert!((hits.get(0).score() - 1.5).abs() < 1e-12);
    assert!((hits.get(1).score() - 0.87).abs() < 1e-12);
    assert_eq!(dec(hits.get(1).doc().as_ref().unwrap()), doc1);

    // GroupRes.
    let mut b = FlatBufferBuilder::new();
    let g0 = enc(&mut b, &group);
    let groups = b.create_vector(&[g0]);
    let gr = GroupRes::create(
        &mut b,
        &GroupResArgs {
            groups: Some(groups),
        },
    );
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::GroupRes,
            body: Some(gr.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    let groups = flatbuffers::root::<Response>(b.finished_data())
        .unwrap()
        .body_as_group_res()
        .unwrap()
        .groups()
        .unwrap();
    assert_eq!(dec(&groups.get(0)), group);

    // StatsRes with per-index table.
    let mut b = FlatBufferBuilder::new();
    let f1s = b.create_string("_id");
    let s1 = IndexStat::create(
        &mut b,
        &IndexStatArgs {
            field: Some(f1s),
            entries: 100,
            distinct: 100,
            memory: 512,
        },
    );
    let f2s = b.create_string("age");
    let s2 = IndexStat::create(
        &mut b,
        &IndexStatArgs {
            field: Some(f2s),
            entries: 100,
            distinct: 40,
            memory: 777,
        },
    );
    let per = b.create_vector(&[s1, s2]);
    let st = StatsRes::create(
        &mut b,
        &StatsResArgs {
            docs: 100,
            docs_memory: 2048,
            indexes: 2,
            total_memory: 3337,
            per_index: Some(per),
        },
    );
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::StatsRes,
            body: Some(st.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    let st = flatbuffers::root::<Response>(b.finished_data())
        .unwrap()
        .body_as_stats_res()
        .unwrap();
    assert_eq!(st.docs(), 100);
    assert_eq!(st.docs_memory(), 2048);
    assert_eq!(st.indexes(), 2);
    assert_eq!(st.total_memory(), 3337);
    let per = st.per_index().unwrap();
    assert_eq!(per.get(0).field().unwrap(), "_id");
    assert_eq!(per.get(1).field().unwrap(), "age");
    assert_eq!(per.get(1).distinct(), 40);

    // IndexRes (a success marker for IndexCmd).
    let mut b = FlatBufferBuilder::new();
    let ir = IndexRes::create(&mut b, &IndexResArgs::default());
    let resp = Response::create(
        &mut b,
        &ResponseArgs {
            body_type: ResponseBody::IndexRes,
            body: Some(ir.as_union_value()),
            ..Default::default()
        },
    );
    b.finish(resp, None);
    assert!(
        flatbuffers::root::<Response>(b.finished_data())
            .unwrap()
            .body_as_index_res()
            .is_some()
    );
}

#[test]
fn response_error_status_roundtrips_every_code() {
    let all = [
        Status::OK,
        Status::NotAnObject,
        Status::IdMustBeString,
        Status::DuplicateId,
        Status::IdMismatch,
        Status::NoIndex,
        Status::PrimaryIndex,
        Status::NoMatch,
        Status::InvalidUpdate,
        Status::VectorDimMismatch,
        Status::MalformedRequest,
        Status::UnknownCommand,
        Status::UnsupportedVersion,
        Status::InternalError,
    ];
    for s in all {
        let mut b = FlatBufferBuilder::new();
        let msg = b.create_string(&format!("detail for {}", s.0));
        let resp = Response::create(
            &mut b,
            &ResponseArgs {
                req_id: 55,
                status: s,
                message: Some(msg),
                ..Default::default()
            },
        );
        b.finish(resp, Some(FILE_IDENTIFIER));
        let r = flatbuffers::root::<Response>(b.finished_data()).unwrap();
        assert_eq!(r.req_id(), 55, "req_id echo for {:?}", s.0);
        assert!(r.status() == s, "status round-trip for {}", s.0);
        assert_eq!(r.message().unwrap(), &format!("detail for {}", s.0));
        assert!(r.body().is_none(), "error responses carry no body");
    }
}

#[test]
fn discriminant_numbers_are_the_wire_contract() {
    // These raw values are the on-wire format. Pinning them catches
    // accidental re-orderings in the .fbs (which would be a protocol
    // break, not a code change).
    assert_eq!(ValueKind::Null.0, 0);
    assert_eq!(ValueKind::Bool.0, 1);
    assert_eq!(ValueKind::I64.0, 2);
    assert_eq!(ValueKind::F64.0, 3);
    assert_eq!(ValueKind::Str.0, 4);
    assert_eq!(ValueKind::Array.0, 5);
    assert_eq!(ValueKind::Object.0, 6);

    assert_eq!(Command::NONE.0, 0);
    assert_eq!(Command::InsertCmd.0, 1);
    assert_eq!(Command::FindCmd.0, 2);
    assert_eq!(Command::CountCmd.0, 3);
    assert_eq!(Command::ExistsCmd.0, 4);
    assert_eq!(Command::UpdateCmd.0, 5);
    assert_eq!(Command::ReplaceCmd.0, 6);
    assert_eq!(Command::DeleteCmd.0, 7);
    assert_eq!(Command::VectorSearchCmd.0, 8);
    assert_eq!(Command::TextSearchCmd.0, 9);
    assert_eq!(Command::HybridSearchCmd.0, 10);
    assert_eq!(Command::GroupCmd.0, 11);
    assert_eq!(Command::StatsCmd.0, 12);
    assert_eq!(Command::IndexCmd.0, 13); // added later; must stay appended

    assert_eq!(ResponseBody::NONE.0, 0);
    assert_eq!(ResponseBody::InsertRes.0, 1);
    assert_eq!(ResponseBody::FindRes.0, 2);
    assert_eq!(ResponseBody::CountRes.0, 3);
    assert_eq!(ResponseBody::ExistsRes.0, 4);
    assert_eq!(ResponseBody::UpdateRes.0, 5);
    assert_eq!(ResponseBody::ReplaceRes.0, 6);
    assert_eq!(ResponseBody::DeleteRes.0, 7);
    assert_eq!(ResponseBody::SearchRes.0, 8);
    assert_eq!(ResponseBody::GroupRes.0, 9);
    assert_eq!(ResponseBody::StatsRes.0, 10);
    assert_eq!(ResponseBody::IndexRes.0, 11); // added later; must stay appended

    // IndexKind discriminants (the index-management command's op code).
    assert_eq!(IndexKind::CreateValue.0, 0);
    assert_eq!(IndexKind::DropValue.0, 1);
    assert_eq!(IndexKind::CreateVector.0, 2);
    assert_eq!(IndexKind::DropVector.0, 3);
    assert_eq!(IndexKind::CreateText.0, 4);
    assert_eq!(IndexKind::DropText.0, 5);

    // `Status` is a `byte` enum — the generated wrapper is *signed* i8, so
    // the numeric contract is pinned in i8.
    assert_eq!(Status::OK.0, 0i8);
    assert_eq!(Status::NotAnObject.0, 1i8);
    assert_eq!(Status::IdMustBeString.0, 2i8);
    assert_eq!(Status::DuplicateId.0, 3i8);
    assert_eq!(Status::IdMismatch.0, 4i8);
    assert_eq!(Status::NoIndex.0, 5i8);
    assert_eq!(Status::PrimaryIndex.0, 6i8);
    assert_eq!(Status::NoMatch.0, 7i8);
    assert_eq!(Status::InvalidUpdate.0, 8i8);
    assert_eq!(Status::VectorDimMismatch.0, 9i8);
    assert_eq!(Status::MalformedRequest.0, 10i8);
    assert_eq!(Status::UnknownCommand.0, 11i8);
    assert_eq!(Status::UnsupportedVersion.0, 12i8);
    assert_eq!(Status::InternalError.0, 13i8);

    assert_eq!(AggFn::Count.0, 0);
    assert_eq!(AggFn::Sum.0, 1);
    assert_eq!(AggFn::Mean.0, 2);
    assert_eq!(AggFn::Min.0, 3);
    assert_eq!(AggFn::Max.0, 4);
    assert_eq!(AggFn::Collect.0, 5);
    assert_eq!(AggFn::First.0, 6);
    assert_eq!(AggFn::Last.0, 7);
}

#[test]
fn status_maps_1to1_to_store_error_variants() {
    // Every engine StoreError variant has a wire status (and the transport
    // codes are distinct from the store codes). The server subtask maps
    // StoreError -> Status; this pins the set it must cover.
    let store_codes = [
        Status::NotAnObject,
        Status::IdMustBeString,
        Status::DuplicateId,
        Status::IdMismatch,
        Status::NoIndex,
        Status::PrimaryIndex,
        Status::NoMatch,
        Status::InvalidUpdate,
        Status::VectorDimMismatch,
    ];
    let transport_codes = [
        Status::MalformedRequest,
        Status::UnknownCommand,
        Status::UnsupportedVersion,
        Status::InternalError,
    ];
    let mut all: Vec<i8> = store_codes
        .iter()
        .map(|s| s.0)
        .chain(transport_codes.iter().map(|s| s.0))
        .collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 13, "store + transport codes are disjoint");
    assert_eq!(all.iter().min().copied(), Some(Status::NotAnObject.0));
    assert_eq!(all.iter().max().copied(), Some(Status::InternalError.0));
}
