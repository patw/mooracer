//! mooracer-server — the network layer.
//!
//! A TCP server speaking length-prefixed FlatBuffers frames over a small,
//! configurable thread pool. A single `RwLock<HashMap<String, Collection>>`
//! guards every collection: read-only commands take a **shared** lock (so
//! many readers proceed concurrently) and mutating commands take the
//! **exclusive** lock (so writes are serialized and see a consistent store).
//!
//! Framing: each frame is a 4-byte little-endian `u32` payload length followed
//! by exactly that many bytes of FlatBuffer data (a [`Request`] client→server,
//! a [`Response`] server→client). The protocol (schema, Value tree, command /
//! response unions, `Status` codes) lives in the `mooracer-wire` crate; this
//! crate owns the I/O, the pool, and the request/response loop.
//!
//! Perf posture: the hot path clones nothing except the matched docs (the
//! engine's own clones); frames are read/written with buffered I/O into a
//! reused per-connection buffer; a `u32` length prefix + a sane max-frame cap
//! keeps a hostile length from allocating unbounded memory.

use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use mooracer_engine::{AggFn, Collection, StoreError, Value};
use mooracer_wire as wire;
use wire::{Status, WIRE_VERSION};

/// A single shared store: collection name → `Collection`, guarded by one
/// `RwLock`. Readers share; writers take it exclusive.
type Store = RwLock<HashMap<String, Collection>>;

/// Maximum frame payload size (bytes). Anything larger is treated as a
/// corrupt/hostile length prefix and the connection is dropped — this bounds
/// the allocation a bad `u32` could otherwise force.
pub const MAX_FRAME: u32 = 256 * 1024 * 1024; // 256 MiB

/// Default thread-pool size when the caller does not configure one.
pub const DEFAULT_POOL_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Value conversion (wire <-> engine)
// ---------------------------------------------------------------------------

/// Decode a wire `Value` tree into a native engine [`Value`].
///
/// Wire tables are `Copy` (an offset + lifetime), and field accessors return
/// them by value, so this takes `wire::Value` by value. Unknown kinds (a
/// future protocol version) fail loudly rather than decoding as garbage —
/// mirrors the wire crate's contract.
fn value_from_wire(v: wire::Value) -> Value {
    match v.kind() {
        wire::ValueKind::Null => Value::Null,
        wire::ValueKind::Bool => Value::Bool(v.b()),
        wire::ValueKind::I64 => Value::I64(v.i()),
        wire::ValueKind::F64 => Value::F64(v.f()),
        wire::ValueKind::Str => Value::Str(v.s().expect("Str kind carries s").to_string()),
        wire::ValueKind::Array => {
            let arr = v.arr().expect("Array kind carries arr");
            Value::Array(arr.iter().map(value_from_wire).collect())
        }
        wire::ValueKind::Object => {
            let keys = v.keys().expect("Object kind carries keys");
            let vals = v.vals().expect("Object kind carries vals");
            let pairs = keys
                .iter()
                .zip(vals.iter())
                .map(|(k, val)| (k.to_string(), value_from_wire(val)))
                .collect();
            Value::Object(pairs)
        }
        wire::ValueKind(k) => panic!("unknown ValueKind on wire: {k} (protocol break?)"),
    }
}

/// Encode a native engine [`Value`] tree as a wire `Value` on `b`.
///
/// Objects preserve insertion order (the engine's `Object` is an ordered pair
/// vector, and the wire stores parallel `keys`/`vals` vectors in the same
/// order).
fn value_to_wire<'a>(b: &mut FlatBufferBuilder<'a>, v: &Value) -> WIPOffset<wire::Value<'a>> {
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
            let offs: Vec<_> = items.iter().map(|it| value_to_wire(b, it)).collect();
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
            let vals: Vec<_> = pairs.iter().map(|(_, val)| value_to_wire(b, val)).collect();
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

/// Convert a wire [`AggFn`] to the engine's [`AggFn`]. An unknown discriminant
/// (future protocol version) yields `InternalError` rather than a panic.
fn agg_fn_from_wire(f: wire::AggFn) -> Option<AggFn> {
    Some(match f {
        wire::AggFn::Count => AggFn::Count,
        wire::AggFn::Sum => AggFn::Sum,
        wire::AggFn::Mean => AggFn::Mean,
        wire::AggFn::Min => AggFn::Min,
        wire::AggFn::Max => AggFn::Max,
        wire::AggFn::Collect => AggFn::Collect,
        wire::AggFn::First => AggFn::First,
        wire::AggFn::Last => AggFn::Last,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Framing (length-prefixed FlatBuffers)
// ---------------------------------------------------------------------------

/// Read one length-prefixed frame from `r`.
///
/// Returns the FlatBuffer payload bytes. A clean end-of-stream (the client
/// closed the connection) is surfaced as an [`io::Error`] of kind
/// [`io::ErrorKind::UnexpectedEof`], which the connection loop treats as a
/// normal close. A payload length above [`MAX_FRAME`] is `InvalidData`.
pub fn read_frame(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut lenbuf = [0u8; 4];
    r.read_exact(&mut lenbuf)?;
    let len = u32::from_le_bytes(lenbuf) as usize;
    if len > MAX_FRAME as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds MAX_FRAME ({MAX_FRAME})"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Write one length-prefixed frame to `w` (4-byte little-endian length +
/// payload). The length prefix and payload are issued as two `write_all`
/// calls so no scratch buffer of the whole frame is needed on the wire.
pub fn write_frame(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)
}

/// Wrap a FlatBuffer payload in a length-prefixed frame (test helper).
pub fn encode_frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 4);
    let len = u32::try_from(payload.len()).expect("payload within u32");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

// ---------------------------------------------------------------------------
// Response encoding
// ---------------------------------------------------------------------------

/// Build a [`wire::Response`] table on the caller's builder. `body` (when
/// `Some`) is the already-built typed result table (its union discriminant +
/// offset, created on the *same* builder). Errors pass `body = None` + a
/// `message`.
fn resp<'a>(
    b: &mut FlatBufferBuilder<'a>,
    req_id: u64,
    status: Status,
    message: Option<&str>,
    body: Option<(wire::ResponseBody, WIPOffset<flatbuffers::UnionWIPOffset>)>,
) -> WIPOffset<wire::Response<'a>> {
    let body_type = body.map(|(t, _)| t).unwrap_or(wire::ResponseBody::NONE);
    let body_off = body.map(|(_, o)| o);
    let msg = message.map(|m| b.create_string(m));
    wire::Response::create(
        b,
        &wire::ResponseArgs {
            req_id,
            status,
            message: msg,
            body_type,
            body: body_off,
            ..Default::default()
        },
    )
}

/// Finish a builder with a Response table and the "MOOR" identifier, returning
/// the frame payload bytes.
fn finish_response(b: &mut FlatBufferBuilder, r: WIPOffset<wire::Response>) -> Vec<u8> {
    b.finish(r, Some(wire::FILE_IDENTIFIER));
    b.finished_data().to_vec()
}

/// A short error response (no body). Used for malformed requests, unknown
/// commands, version mismatches, and engine `StoreError`s.
fn err_response(req_id: u64, status: Status, message: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let r = resp(&mut b, req_id, status, Some(message), None);
    finish_response(&mut b, r)
}

/// Finish a builder with a successful (`Status::OK`) response carrying `body`.
/// The body table and all its strings/vectors must already have been built on
/// `b`. Kept separate from [`finish_response`] so the `resp` + `finish` borrows
/// of the builder are sequential (not nested).
fn finish_ok(
    b: &mut FlatBufferBuilder,
    req_id: u64,
    body: Option<(wire::ResponseBody, WIPOffset<flatbuffers::UnionWIPOffset>)>,
) -> Vec<u8> {
    let r = resp(b, req_id, Status::OK, None, body);
    finish_response(b, r)
}

/// Map an engine [`StoreError`] to its wire [`Status`].
fn status_for_error(e: &StoreError) -> Status {
    use StoreError::*;
    match e {
        NotAnObject => Status::NotAnObject,
        IdMustBeString => Status::IdMustBeString,
        DuplicateId(_) => Status::DuplicateId,
        IdMismatch { .. } => Status::IdMismatch,
        NoIndex(_) => Status::NoIndex,
        PrimaryIndex => Status::PrimaryIndex,
        NoMatch => Status::NoMatch,
        InvalidUpdate(_) => Status::InvalidUpdate,
        VectorDimMismatch { .. } => Status::VectorDimMismatch,
    }
}

/// The `_id` of a document, if it is a string (`None` for a missing or
/// non-string `_id`).
fn id_of(doc: &Value) -> Option<String> {
    doc.get("_id").and_then(Value::as_str).map(str::to_string)
}

/// Reconstruct the ids of the docs just inserted by `insert_many` **in
/// request order**.
///
/// `Collection::insert_many` returns only a count, so the server recovers the
/// ids by diffing the store before/after the write: an explicit string `_id`
/// is returned verbatim, while an auto-generated id is the freshly-created,
/// zero-padded 24-hex counter value — those sort ascending in exactly their
/// (monotonic) assignment order, so the batch's generated ids line up with the
/// request positions that lacked an `_id`. `before` MUST be the id-set
/// captured **before** the insert (the write mutates the store in place).
fn inserted_ids(collection: &Collection, docs: &[Value], before: &HashSet<String>) -> Vec<String> {
    let explicit: Vec<String> = docs.iter().filter_map(id_of).collect();
    let mut generated: Vec<String> = collection
        .iter()
        .filter_map(id_of)
        .filter(|id| !before.contains(id))
        .filter(|id| !explicit.contains(id))
        .collect();
    generated.sort();
    let mut gen_iter = generated.into_iter();
    let mut out = Vec::with_capacity(docs.len());
    for d in docs {
        match d.get("_id") {
            Some(Value::Str(s)) => out.push(s.clone()),
            _ => out.push(
                gen_iter
                    .next()
                    .expect("a generated id for every auto-id doc"),
            ),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The MooRacer server: a thread pool over a single shared, `RwLock`-guarded
/// collection store. Build with [`Server::bind`], then call [`Server::run`]
/// (blocking accept loop + pool) or drive connections manually.
///
/// `Clone` is cheap (the store is behind an `Arc`): `run` hands each pool
/// worker an owned clone so the workers never borrow the calling `&self` for
/// longer than the accept loop.
#[derive(Clone)]
pub struct Server {
    state: Arc<Store>,
    pool_size: usize,
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

impl Server {
    /// New server with the default pool size ([`DEFAULT_POOL_SIZE`]).
    pub fn new() -> Self {
        Self::with_pool_size(DEFAULT_POOL_SIZE)
    }

    /// New server with an explicit pool size (clamped to at least 1).
    pub fn with_pool_size(size: usize) -> Self {
        Server {
            state: Arc::new(RwLock::new(HashMap::new())),
            pool_size: size.max(1),
        }
    }

    /// Bind a TCP listener at `addr`. The returned server shares the store
    /// with the listener's eventual `run`; seed collections through
    /// [`Server::state`] before serving.
    pub fn bind(addr: &str) -> io::Result<(Server, TcpListener)> {
        Ok((Self::new(), TcpListener::bind(addr)?))
    }

    /// The configured thread-pool size.
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// The shared store (for seeding collections in tests / setup).
    pub fn state(&self) -> &RwLock<HashMap<String, Collection>> {
        &self.state
    }

    /// Number of collections currently registered.
    pub fn collection_count(&self) -> usize {
        self.state.read().unwrap().len()
    }

    /// Seed a collection with some documents before serving (test/setup
    /// helper; takes the exclusive lock).
    pub fn seed_docs(&self, name: &str, docs: &[Value]) -> Result<usize, StoreError> {
        let mut guard = self.state.write().unwrap();
        let coll = guard
            .entry(name.to_string())
            .or_insert_with(|| Collection::new(name));
        let mut n = 0;
        for d in docs {
            coll.insert(d.clone())?;
            n += 1;
        }
        Ok(n)
    }

    /// Accept loop + thread pool: pull each accepted connection onto the
    /// shared pool and serve it with [`Server::handle_connection`]. Returns on
    /// a listener error. This is the blocking entry point for the binary.
    ///
    /// The pool is `pool_size` worker threads sharing one `mpsc` receiver (this
    /// toolchain's `mpsc::Receiver` does not resolve `Clone`, so the receiver
    /// is shared through an `Arc<Mutex<_>>`; each worker locks, pulls one
    /// connection, then releases the lock to serve it). The accept loop enqueues
    /// into an unbounded channel, so `send` never blocks.
    pub fn run(&self, listener: &TcpListener) -> io::Result<()> {
        let (tx, rx) = mpsc::channel::<TcpStream>();
        let rx = Arc::new(Mutex::new(rx));
        let n = self.pool_size;
        let workers: Vec<thread::JoinHandle<()>> = (0..n)
            .map(|_| {
                let rx = rx.clone();
                let server = self.clone();
                thread::spawn(move || {
                    loop {
                        // Pull the next queued connection (serialize the pull,
                        // then serve without holding the lock).
                        let stream = rx.lock().unwrap().recv();
                        match stream {
                            Ok(stream) => {
                                let mut conn = stream;
                                if let Err(e) = server.handle_connection(&mut conn) {
                                    // A per-connection I/O error (client
                                    // abort, bad frame) is logged and swallowed:
                                    // one bad client must not take down the pool.
                                    eprintln!("mooracer-server: connection ended: {e}");
                                }
                            }
                            Err(_) => break, // channel closed + drained
                        }
                    }
                })
            })
            .collect();

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    if tx.send(stream).is_err() {
                        // All workers are gone (should not happen while the
                        // handles are alive); break to avoid spinning.
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        drop(tx);
        for h in workers {
            let _ = h.join();
        }
        Ok(())
    }

    /// Run the request/response loop on a single connection until the client
    /// closes it or a fatal I/O error occurs. A clean EOF is `Ok(())`.
    pub fn handle_connection(&self, stream: &mut TcpStream) -> io::Result<()> {
        loop {
            match read_frame(stream) {
                Err(e)
                    if e.kind() == io::ErrorKind::UnexpectedEof
                        || e.kind() == io::ErrorKind::ConnectionAborted =>
                {
                    // Client closed the connection: a normal end of session.
                    return Ok(());
                }
                Err(e) => return Err(e),
                Ok(payload) => {
                    let response = self.process_request(&payload);
                    write_frame(stream, &response)?;
                }
            }
        }
    }

    /// Decode one request buffer and produce the response buffer. Pure with
    /// respect to I/O — the only side effect is the (locked) engine call.
    fn process_request(&self, buf: &[u8]) -> Vec<u8> {
        // Frame sanity: a request buffer must carry the "MOOR" identifier.
        if !flatbuffers::buffer_has_identifier(buf, wire::FILE_IDENTIFIER, false) {
            return err_response(
                0,
                Status::MalformedRequest,
                "missing/invalid file identifier",
            );
        }
        let req = match flatbuffers::root::<wire::Request>(buf) {
            Ok(r) => r,
            Err(_) => {
                return err_response(0, Status::MalformedRequest, "invalid FlatBuffer request");
            }
        };
        let req_id = req.req_id();
        if req.version() != WIRE_VERSION {
            let msg = format!(
                "unsupported wire version {} (server speaks {})",
                req.version(),
                WIRE_VERSION
            );
            return err_response(req_id, Status::UnsupportedVersion, &msg);
        }
        let coll = req.collection().unwrap_or_default().to_string();
        match req.command_type() {
            wire::Command::InsertCmd => {
                let c = req.command_as_insert_cmd().expect("payload");
                self.do_insert(req_id, coll, &c)
            }
            wire::Command::FindCmd => {
                let c = req.command_as_find_cmd().expect("payload");
                self.do_find(req_id, coll, &c)
            }
            wire::Command::CountCmd => {
                let c = req.command_as_count_cmd().expect("payload");
                self.do_count(req_id, coll, &c)
            }
            wire::Command::ExistsCmd => {
                let c = req.command_as_exists_cmd().expect("payload");
                self.do_exists(req_id, coll, &c)
            }
            wire::Command::UpdateCmd => {
                let c = req.command_as_update_cmd().expect("payload");
                self.do_update(req_id, coll, &c)
            }
            wire::Command::ReplaceCmd => {
                let c = req.command_as_replace_cmd().expect("payload");
                self.do_replace(req_id, coll, &c)
            }
            wire::Command::DeleteCmd => {
                let c = req.command_as_delete_cmd().expect("payload");
                self.do_delete(req_id, coll, &c)
            }
            wire::Command::VectorSearchCmd => {
                let c = req.command_as_vector_search_cmd().expect("payload");
                self.do_vector_search(req_id, coll, &c)
            }
            wire::Command::TextSearchCmd => {
                let c = req.command_as_text_search_cmd().expect("payload");
                self.do_text_search(req_id, coll, &c)
            }
            wire::Command::HybridSearchCmd => {
                let c = req.command_as_hybrid_search_cmd().expect("payload");
                self.do_hybrid_search(req_id, coll, &c)
            }
            wire::Command::GroupCmd => {
                let c = req.command_as_group_cmd().expect("payload");
                self.do_group(req_id, coll, &c)
            }
            wire::Command::StatsCmd => {
                let c = req.command_as_stats_cmd().expect("payload");
                self.do_stats(req_id, coll, &c)
            }
            wire::Command::IndexCmd => {
                let c = req.command_as_index_cmd().expect("payload");
                self.do_index(req_id, coll, &c)
            }
            wire::Command::NONE | wire::Command(_) => err_response(
                req_id,
                Status::UnknownCommand,
                "unknown or unimplemented command",
            ),
        }
    }

    // -- read commands (shared lock) --------------------------------------

    fn do_stats(&self, req_id: u64, coll: String, _c: &wire::StatsCmd) -> Vec<u8> {
        let guard = self.state.read().unwrap();
        let stats = guard
            .get(&coll)
            .map(|col| col.stats())
            .unwrap_or_else(|| Collection::new(&coll).stats());
        let mut b = FlatBufferBuilder::new();
        let per: Vec<_> = stats
            .per_index
            .iter()
            .map(|s| {
                let field = b.create_string(&s.field);
                wire::IndexStat::create(
                    &mut b,
                    &wire::IndexStatArgs {
                        field: Some(field),
                        entries: s.entries as u64,
                        distinct: s.distinct as u64,
                        memory: s.memory as u64,
                    },
                )
            })
            .collect();
        let per_off = b.create_vector(&per);
        let st = wire::StatsRes::create(
            &mut b,
            &wire::StatsResArgs {
                docs: stats.docs as u64,
                docs_memory: stats.docs_memory as u64,
                indexes: stats.indexes as u64,
                total_memory: stats.total_memory as u64,
                per_index: Some(per_off),
            },
        );
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::StatsRes, st.as_union_value())),
        )
    }

    fn do_count(&self, req_id: u64, coll: String, c: &wire::CountCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let guard = self.state.read().unwrap();
        let n = guard.get(&coll).map(|col| col.count(filter)).unwrap_or(0);
        let mut b = FlatBufferBuilder::new();
        let r = wire::CountRes::create(&mut b, &wire::CountResArgs { count: n as u64 });
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::CountRes, r.as_union_value())),
        )
    }

    fn do_exists(&self, req_id: u64, coll: String, c: &wire::ExistsCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let guard = self.state.read().unwrap();
        let e = guard
            .get(&coll)
            .map(|col| col.exists(filter))
            .unwrap_or(false);
        let mut b = FlatBufferBuilder::new();
        let r = wire::ExistsRes::create(&mut b, &wire::ExistsResArgs { exists: e });
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::ExistsRes, r.as_union_value())),
        )
    }

    fn do_find(&self, req_id: u64, coll: String, c: &wire::FindCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let guard = self.state.read().unwrap();
        let docs: Vec<Value> = match guard.get(&coll) {
            None => Vec::new(),
            Some(col) => {
                let mut q = col.find(filter);
                if let Some(sf) = c.sort_field() {
                    q = q.sort(sf, c.sort_desc());
                }
                if c.skip() > 0 {
                    q = q.skip(c.skip() as usize);
                }
                if c.limit() > 0 {
                    q = q.limit(c.limit() as usize);
                }
                if c.one() {
                    q.first().into_iter().collect()
                } else {
                    q.to_list()
                }
            }
        };
        let mut b = FlatBufferBuilder::new();
        let offs: Vec<_> = docs.iter().map(|d| value_to_wire(&mut b, d)).collect();
        let docs_off = b.create_vector(&offs);
        let r = wire::FindRes::create(
            &mut b,
            &wire::FindResArgs {
                docs: Some(docs_off),
            },
        );
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::FindRes, r.as_union_value())),
        )
    }

    // -- write commands (exclusive lock) ----------------------------------

    fn do_insert(&self, req_id: u64, coll: String, c: &wire::InsertCmd) -> Vec<u8> {
        let docs: Vec<Value> = c
            .docs()
            .map(|v| v.iter().map(value_from_wire).collect())
            .unwrap_or_default();
        let mut guard = self.state.write().unwrap();
        let collection = guard
            .entry(coll.clone())
            .or_insert_with(|| Collection::new(&coll));
        let before: HashSet<String> = collection.iter().filter_map(id_of).collect();
        match collection.insert_many(docs.iter().cloned()) {
            Ok(_) => {
                let ids = inserted_ids(collection, &docs, &before);
                let mut b = FlatBufferBuilder::new();
                let id_strs: Vec<_> = ids.iter().map(|s| b.create_string(s)).collect();
                let ids_off = b.create_vector(&id_strs);
                let r =
                    wire::InsertRes::create(&mut b, &wire::InsertResArgs { ids: Some(ids_off) });
                finish_ok(
                    &mut b,
                    req_id,
                    Some((wire::ResponseBody::InsertRes, r.as_union_value())),
                )
            }
            Err(e) => err_response(req_id, status_for_error(&e), &e.to_string()),
        }
    }

    fn do_update(&self, req_id: u64, coll: String, c: &wire::UpdateCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let update = c.update().map(value_from_wire).unwrap_or_default();
        let many = c.many();
        let mut guard = self.state.write().unwrap();
        let collection = guard
            .entry(coll.clone())
            .or_insert_with(|| Collection::new(&coll));
        match collection.update_many(filter, update) {
            Ok(n) if !many && n == 0 => {
                // update_one with no match is an error (NoMatch), per the
                // engine's write-API convention.
                err_response(req_id, Status::NoMatch, "update matched no document")
            }
            Ok(n) => {
                let mut b = FlatBufferBuilder::new();
                let r = wire::UpdateRes::create(&mut b, &wire::UpdateResArgs { count: n as u64 });
                finish_ok(
                    &mut b,
                    req_id,
                    Some((wire::ResponseBody::UpdateRes, r.as_union_value())),
                )
            }
            Err(e) => err_response(req_id, status_for_error(&e), &e.to_string()),
        }
    }

    fn do_replace(&self, req_id: u64, coll: String, c: &wire::ReplaceCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let new_doc = c.new_doc().map(value_from_wire).unwrap_or_default();
        let mut guard = self.state.write().unwrap();
        let collection = guard
            .entry(coll.clone())
            .or_insert_with(|| Collection::new(&coll));
        match collection.replace_one(filter, new_doc) {
            Ok(n) => {
                let mut b = FlatBufferBuilder::new();
                let r = wire::ReplaceRes::create(&mut b, &wire::ReplaceResArgs { count: n as u64 });
                finish_ok(
                    &mut b,
                    req_id,
                    Some((wire::ResponseBody::ReplaceRes, r.as_union_value())),
                )
            }
            Err(e) => err_response(req_id, status_for_error(&e), &e.to_string()),
        }
    }

    fn do_delete(&self, req_id: u64, coll: String, c: &wire::DeleteCmd) -> Vec<u8> {
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let many = c.many();
        let mut guard = self.state.write().unwrap();
        let collection = guard
            .entry(coll.clone())
            .or_insert_with(|| Collection::new(&coll));
        let n = if many {
            collection.delete_many(filter)
        } else {
            collection.delete_one(filter) as usize
        };
        let mut b = FlatBufferBuilder::new();
        let r = wire::DeleteRes::create(&mut b, &wire::DeleteResArgs { count: n as u64 });
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::DeleteRes, r.as_union_value())),
        )
    }

    // -- index management (exclusive lock: mutates indexes) ----------------

    fn do_index(&self, req_id: u64, coll: String, c: &wire::IndexCmd) -> Vec<u8> {
        use wire::IndexKind;
        let field = c.field().unwrap_or_default().to_string();
        let mut guard = self.state.write().unwrap();
        let collection = guard
            .entry(coll.clone())
            .or_insert_with(|| Collection::new(&coll));
        let result: Result<(), StoreError> = match c.kind() {
            IndexKind::CreateValue => collection.create_index(&field),
            IndexKind::DropValue => collection.drop_index(&field),
            IndexKind::CreateVector => {
                collection.create_vector_index(&field, c.dim() as usize);
                Ok(())
            }
            IndexKind::DropVector => {
                collection.drop_vector_index(&field);
                Ok(())
            }
            IndexKind::CreateText => {
                collection.create_text_index(&field);
                Ok(())
            }
            IndexKind::DropText => {
                collection.drop_text_index(&field);
                Ok(())
            }
            IndexKind(_) => {
                return err_response(
                    req_id,
                    Status::InternalError,
                    &format!("unknown IndexKind discriminant {}", c.kind().0),
                );
            }
        };
        match result {
            Ok(()) => {
                let mut b = FlatBufferBuilder::new();
                let r = wire::IndexRes::create(&mut b, &wire::IndexResArgs::default());
                finish_ok(
                    &mut b,
                    req_id,
                    Some((wire::ResponseBody::IndexRes, r.as_union_value())),
                )
            }
            Err(e) => err_response(req_id, status_for_error(&e), &e.to_string()),
        }
    }

    // -- search commands (shared lock: reads) ------------------------------

    fn do_vector_search(&self, req_id: u64, coll: String, c: &wire::VectorSearchCmd) -> Vec<u8> {
        let field = c.field().unwrap_or_default().to_string();
        let query: Vec<f32> = c.query().map(|v| v.iter().collect()).unwrap_or_default();
        let limit = c.limit() as usize;
        let guard = self.state.read().unwrap();
        match guard
            .get(&coll)
            .map(|col| col.vector_search(&field, &query, limit))
        {
            Some(Ok(hits)) => self.search_response(req_id, hits),
            Some(Err(e)) => err_response(req_id, status_for_error(&e), &e.to_string()),
            None => err_response(
                req_id,
                Status::NoIndex,
                &format!("no vector index on `{field}`"),
            ),
        }
    }

    fn do_text_search(&self, req_id: u64, coll: String, c: &wire::TextSearchCmd) -> Vec<u8> {
        let field = c.field().unwrap_or_default().to_string();
        let query = c.query().unwrap_or_default().to_string();
        let limit = c.limit() as usize;
        let guard = self.state.read().unwrap();
        match guard
            .get(&coll)
            .map(|col| col.text_search(&field, &query, limit))
        {
            Some(Ok(hits)) => self.search_response(req_id, hits),
            Some(Err(e)) => err_response(req_id, status_for_error(&e), &e.to_string()),
            None => err_response(
                req_id,
                Status::NoIndex,
                &format!("no text index on `{field}`"),
            ),
        }
    }

    fn do_hybrid_search(&self, req_id: u64, coll: String, c: &wire::HybridSearchCmd) -> Vec<u8> {
        let text_field = c.text_field().unwrap_or_default().to_string();
        let vec_field = c.vec_field().unwrap_or_default().to_string();
        let query_text = c.query_text().unwrap_or_default().to_string();
        let query_vec: Vec<f32> = c
            .query_vec()
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        let limit = c.limit() as usize;
        let guard = self.state.read().unwrap();
        match guard
            .get(&coll)
            .map(|col| col.hybrid_search(&text_field, &vec_field, &query_text, &query_vec, limit))
        {
            Some(Ok(hits)) => self.search_response(req_id, hits),
            Some(Err(e)) => err_response(req_id, status_for_error(&e), &e.to_string()),
            None => err_response(
                req_id,
                Status::NoIndex,
                "hybrid search requires both a text and a vector index",
            ),
        }
    }

    /// Shared encoder for the three search kinds. Each returns `(doc, score)`
    /// where the score is `f32` (vector) or `f64` (text/hybrid); both widen to
    /// the `f64` on the wire.
    fn search_response<T, S>(&self, req_id: u64, hits: T) -> Vec<u8>
    where
        T: IntoIterator<Item = (Value, S)>,
        S: Into<f64>,
    {
        let mut b = FlatBufferBuilder::new();
        let offs: Vec<WIPOffset<wire::SearchHit>> = hits
            .into_iter()
            .map(|(doc, score)| {
                let d = value_to_wire(&mut b, &doc);
                wire::SearchHit::create(
                    &mut b,
                    &wire::SearchHitArgs {
                        doc: Some(d),
                        score: score.into(),
                    },
                )
            })
            .collect();
        let hits_off = b.create_vector(&offs);
        let r = wire::SearchRes::create(
            &mut b,
            &wire::SearchResArgs {
                hits: Some(hits_off),
            },
        );
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::SearchRes, r.as_union_value())),
        )
    }

    fn do_group(&self, req_id: u64, coll: String, c: &wire::GroupCmd) -> Vec<u8> {
        let fn_ = match agg_fn_from_wire(c.agg_fn()) {
            Some(f) => f,
            None => {
                return err_response(
                    req_id,
                    Status::InternalError,
                    &format!("unknown AggFn discriminant {}", c.agg_fn().0),
                );
            }
        };
        let filter = c.filter().map(value_from_wire).unwrap_or_default();
        let agg_field = c.agg_field().unwrap_or_default().to_string();
        let group_field = c.group_field().unwrap_or_default().to_string();
        let guard = self.state.read().unwrap();
        let groups: Vec<Value> = match guard.get(&coll) {
            None => Vec::new(),
            Some(col) => {
                let mut q = col.find(filter);
                if let Some(sf) = c.sort_field() {
                    q = q.sort(sf, c.sort_desc());
                }
                if c.skip() > 0 {
                    q = q.skip(c.skip() as usize);
                }
                if c.limit() > 0 {
                    q = q.limit(c.limit() as usize);
                }
                let mut gq = q.group(group_field);
                if let Some(gsf) = c.group_sort_field() {
                    gq = gq.sort(gsf, c.group_sort_desc());
                }
                if c.group_limit() > 0 {
                    gq = gq.limit(c.group_limit() as usize);
                }
                gq.agg(fn_, agg_field)
            }
        };
        let mut b = FlatBufferBuilder::new();
        let offs: Vec<_> = groups.iter().map(|g| value_to_wire(&mut b, g)).collect();
        let groups_off = b.create_vector(&offs);
        let r = wire::GroupRes::create(
            &mut b,
            &wire::GroupResArgs {
                groups: Some(groups_off),
            },
        );
        finish_ok(
            &mut b,
            req_id,
            Some((wire::ResponseBody::GroupRes, r.as_union_value())),
        )
    }
}

/// Run the server to completion, blocking (used by `main`). Returns on a
/// listener error.
pub fn serve(listener: TcpListener) -> io::Result<()> {
    let server = Server::new();
    server.run(&listener)
}

// ---------------------------------------------------------------------------
// Tests (unit; the TCP integration tests live in `server/tests/tcp.rs`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_roundtrips() {
        let payload = b"hello mooracer frame";
        let frame = encode_frame(payload);
        let mut r = frame.as_slice();
        let decoded = read_frame(&mut r).unwrap();
        assert_eq!(decoded, payload.to_vec());
    }

    #[test]
    fn framing_rejects_overlong_length_prefix() {
        // A 5-byte frame whose 4-byte prefix claims MAX_FRAME+1 bytes: the
        // guard must refuse before allocating.
        let mut frame = (MAX_FRAME + 1).to_le_bytes().to_vec();
        frame.extend_from_slice(b"0123456789");
        let mut r = frame.as_slice();
        let err = read_frame(&mut r).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn status_maps_every_store_error() {
        assert_eq!(
            status_for_error(&StoreError::NotAnObject),
            Status::NotAnObject
        );
        assert_eq!(
            status_for_error(&StoreError::IdMustBeString),
            Status::IdMustBeString
        );
        assert_eq!(
            status_for_error(&StoreError::DuplicateId("x".into())),
            Status::DuplicateId
        );
        assert_eq!(
            status_for_error(&StoreError::IdMismatch {
                expected: "a".into(),
                found: "b".into()
            }),
            Status::IdMismatch
        );
        assert_eq!(
            status_for_error(&StoreError::NoIndex("f".into())),
            Status::NoIndex
        );
        assert_eq!(
            status_for_error(&StoreError::PrimaryIndex),
            Status::PrimaryIndex
        );
        assert_eq!(status_for_error(&StoreError::NoMatch), Status::NoMatch);
        assert_eq!(
            status_for_error(&StoreError::InvalidUpdate("bad".into())),
            Status::InvalidUpdate
        );
        assert_eq!(
            status_for_error(&StoreError::VectorDimMismatch {
                field: "e".into(),
                expected: 2,
                found: 3
            }),
            Status::VectorDimMismatch
        );
    }

    #[test]
    fn value_roundtrips_engine_to_wire_to_engine() {
        let v = Value::Object(vec![
            ("_id".to_string(), Value::Str("a".into())),
            ("n".to_string(), Value::I64(7)),
            ("f".to_string(), Value::F64(1.5)),
            ("b".to_string(), Value::Bool(true)),
            ("s".to_string(), Value::Null),
            (
                "arr".to_string(),
                Value::Array(vec![Value::I64(1), Value::Str("x".into())]),
            ),
        ]);
        let mut b = FlatBufferBuilder::new();
        let off = value_to_wire(&mut b, &v);
        b.finish_minimal(off);
        let w = flatbuffers::root::<wire::Value>(b.finished_data()).unwrap();
        let back = value_from_wire(w);
        assert_eq!(back, v);
    }

    #[test]
    fn server_seeds_and_counts() {
        let s = Server::with_pool_size(2);
        let d = Value::Object(vec![
            ("_id".to_string(), Value::Str("1".into())),
            ("n".to_string(), Value::I64(1)),
        ]);
        s.seed_docs("c", &[d.clone(), d.clone()]).unwrap_err(); // dup id
        assert_eq!(s.collection_count(), 1);
    }

    #[test]
    fn server_rejects_bad_frame_with_malformed() {
        let s = Server::new();
        let resp = s.process_request(b"this is not a mooracer frame");
        let r = flatbuffers::root::<wire::Response>(&resp).unwrap();
        assert!(r.status() == Status::MalformedRequest);
    }
}
