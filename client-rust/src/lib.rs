//! mooracer-client — a synchronous Rust client for a MooRacer server.
//!
//! The client speaks the same length-prefixed FlatBuffers wire protocol the
//! server does (defined once in `mooracer-wire`) and exposes a **Mongo-style
//! chain API** over TCP that returns native engine [`Value`] trees and typed
//! [`Error`]s.
//!
//! ## Shape
//!
//! ```ignore
//! let client = mooracer_client::Client::connect("127.0.0.1:4000")?;
//! let herd = client.collection("cows");
//! herd.insert_many(&[doc_a, doc_b])?;
//!
//! // Lazy query chain: filter -> sort -> skip -> limit, terminal runs the RPC.
//! let docs: Vec<Value> = herd.find(filter).sort("age", false).limit(10).to_list()?;
//!
//! // Eager entry points:
//! let one: Option<Value> = herd.find_one(filter)?;
//! let n: u64 = herd.count(filter)?;
//! let has: bool = herd.exists(filter)?;
//!
//! // Writes / search / aggregation:
//! herd.update_one(filter, update)?;
//! herd.vector_search("emb", &[1.0, 0.0], 5)?;
//! let groups = herd.find(filter).group("kind").agg(AggFn::Count, "age")?;
//! ```
//!
//! One `Client` owns one TCP connection and reuses a single response buffer
//! across calls, so it is not `Sync` (use one client per thread, or a pool).
//! `tokio` async support is intentionally out of scope for this subtask — the
//! blocking API is the contract; an async wrapper would just wrap this one.

use std::io::{self, Read, Write};
use std::net::TcpStream;

use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};
use mooracer_engine::{AggFn, Value};
use mooracer_wire as wire;
use wire::{Command, Status, WIRE_VERSION};

pub use mooracer_wire::Status as WireStatus;

/// Maximum frame payload size (bytes), mirroring the server's cap so a hostile
/// or corrupt length prefix can never force an unbounded allocation.
const MAX_FRAME: u32 = 256 * 1024 * 1024; // 256 MiB

/// A client [`Result`] — an `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// A typed client error.
///
/// * [`Error::Io`] — a socket-level failure (connect, read, write).
/// * [`Error::Protocol`] — the server sent something this client could not
///   decode (bad frame, invalid FlatBuffer, or a body union it does not
///   recognize). The connection is considered poisoned after this.
/// * [`Error::Server`] — the server answered with a typed non-`OK` [`Status`]
///   (the engine's `StoreError`s plus the transport codes). `message` carries
///   the human-readable detail the server sent.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Protocol(String),
    Server(Status, String),
}

impl Error {
    /// The wire [`Status`] when this is a typed server error, else `None`.
    pub fn status(&self) -> Option<Status> {
        match self {
            Error::Server(s, _) => Some(*s),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Protocol(msg) => write!(f, "protocol error: {msg}"),
            Error::Server(status, msg) => write!(f, "server error {:?}: {msg}", status),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Framing (length-prefixed FlatBuffers). Local to the client so it stays
// self-contained against the protocol crate (no dependency on the server).
// ---------------------------------------------------------------------------

/// Read one length-prefixed frame: a 4-byte little-endian `u32` length, then
/// exactly that many payload bytes.
fn read_frame(r: &mut impl Read) -> io::Result<Vec<u8>> {
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

/// Write one length-prefixed frame (4-byte LE length + payload).
fn write_frame(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)
}

// ---------------------------------------------------------------------------
// Value conversion (engine <-> wire)
// ---------------------------------------------------------------------------

/// Decode a wire `Value` tree into a native engine [`Value`].
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
        wire::ValueKind(k) => {
            panic!("unknown ValueKind on wire: {k} (protocol break?)")
        }
    }
}

/// Encode a native engine [`Value`] tree as a wire `Value` on `b`. Objects
/// preserve insertion order (parallel `keys`/`vals` vectors).
fn value_to_wire<'a>(b: &mut FlatBufferBuilder<'a>, v: &Value) -> WIPOffset<wire::Value<'a>> {
    match v {
        Value::Null => wire::Value::create(
            b,
            &wire::ValueArgs { kind: wire::ValueKind::Null, ..Default::default() },
        ),
        Value::Bool(x) => wire::Value::create(
            b,
            &wire::ValueArgs { kind: wire::ValueKind::Bool, b: *x, ..Default::default() },
        ),
        Value::I64(x) => wire::Value::create(
            b,
            &wire::ValueArgs { kind: wire::ValueKind::I64, i: *x, ..Default::default() },
        ),
        Value::F64(x) => wire::Value::create(
            b,
            &wire::ValueArgs { kind: wire::ValueKind::F64, f: *x, ..Default::default() },
        ),
        Value::Str(s) => {
            let s = b.create_string(s);
            wire::Value::create(
                b,
                &wire::ValueArgs { kind: wire::ValueKind::Str, s: Some(s), ..Default::default() },
            )
        }
        Value::Array(items) => {
            let offs: Vec<_> = items.iter().map(|it| value_to_wire(b, it)).collect();
            let arr = b.create_vector(&offs);
            wire::Value::create(
                b,
                &wire::ValueArgs { kind: wire::ValueKind::Array, arr: Some(arr), ..Default::default() },
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

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Build a finished request FlatBuffer (not yet length-prefixed). `f` builds
/// the command payload on the builder and returns the union discriminant +
/// offset.
fn build_request<F>(coll: &str, req_id: u64, f: F) -> Vec<u8>
where
    F: FnOnce(&mut FlatBufferBuilder) -> (Command, Option<WIPOffset<UnionWIPOffset>>),
{
    let mut b = FlatBufferBuilder::new();
    let (command_type, command) = f(&mut b);
    let coll_off = b.create_string(coll);
    let req = wire::Request::create(
        &mut b,
        &wire::RequestArgs {
            version: WIRE_VERSION,
            req_id,
            collection: Some(coll_off),
            command_type,
            command,
            ..Default::default()
        },
    );
    b.finish(req, Some(wire::FILE_IDENTIFIER));
    b.finished_data().to_vec()
}

// ---------------------------------------------------------------------------
// Decoded response payloads (owned, decoupled from the reused response buffer)
// ---------------------------------------------------------------------------

/// A per-index statistics record decoded from the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStat {
    pub field: String,
    pub entries: u64,
    pub distinct: u64,
    pub memory: u64,
}

/// Collection statistics decoded from the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    pub docs: u64,
    pub docs_memory: u64,
    pub indexes: u64,
    pub total_memory: u64,
    pub per_index: Vec<IndexStat>,
}

/// The owned payload of a successful (`Status::OK`) response, keyed by the
/// command that produced it. Every value is fully copied out of the wire
/// buffer here, so the client may reuse it on the next call.
#[derive(Debug)]
pub enum Response {
    Insert(Vec<String>),
    Find(Vec<Value>),
    Count(u64),
    Exists(bool),
    Update(u64),
    Replace(u64),
    Delete(u64),
    /// The three search kinds, best-first, `(doc, score)` with the score
    /// widened to `f64`.
    Search(Vec<(Value, f64)>),
    Group(Vec<Value>),
    Stats(Stats),
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

/// A MooRacer client over one TCP connection.
///
/// Build with [`Client::connect`]. It is cheap to keep around (one socket +
/// one reused response buffer) but not thread-safe: a single client is not
/// `Sync`. For concurrent access, open one client per thread.
pub struct Client {
    stream: TcpStream,
    buf: Vec<u8>,
    next_id: u64,
}

impl Client {
    /// Open a TCP connection to a MooRacer server at `addr`.
    pub fn connect(addr: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true).ok();
        Ok(Client { stream, buf: Vec::new(), next_id: 0 })
    }

    /// The next correlation id, wrapping.
    fn take_id(&mut self) -> u64 {
        self.next_id = self.next_id.wrapping_add(1);
        self.next_id
    }

    /// A [`Collection`] handle scoped to this connection and named `name`.
    ///
    /// Reborrowed `&mut`: a `Client` may drive one in-flight request at a
    /// time (a request/response loop), which is exactly the wire's model.
    pub fn collection(&mut self, name: &str) -> Collection<'_> {
        Collection { client: self, name: name.to_string() }
    }

    /// Send one finished request frame, read the response, and decode its
    /// body into an owned [`Response`]. A non-`OK` status becomes
    /// [`Error::Server`]; a decode failure becomes [`Error::Protocol`].
    fn send_receive(&mut self, payload: Vec<u8>) -> Result<Response> {
        write_frame(&mut self.stream, &payload)?;
        self.stream.flush()?;
        self.buf = read_frame(&mut self.stream)?;
        let resp = flatbuffers::root::<wire::Response>(&self.buf)
            .map_err(|e| Error::Protocol(format!("invalid response buffer: {e}")))?;
        match resp.status() {
            Status::OK => decode_body(resp),
            status => {
                let msg = resp.message().unwrap_or_default().to_string();
                Err(Error::Server(status, msg))
            }
        }
    }
}

/// Map a decoded `Status::OK` response's body union to an owned [`Response`].
fn decode_body(resp: wire::Response) -> Result<Response> {
    match resp.body_type() {
        wire::ResponseBody::NONE => Err(Error::Protocol("OK response with no body".into())),
        wire::ResponseBody::InsertRes => {
            let b = resp.body_as_insert_res().expect("InsertRes");
            let ids = b
                .ids()
                .map(|v| v.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            Ok(Response::Insert(ids))
        }
        wire::ResponseBody::FindRes => {
            let b = resp.body_as_find_res().expect("FindRes");
            let docs = b
                .docs()
                .map(|v| v.iter().map(value_from_wire).collect())
                .unwrap_or_default();
            Ok(Response::Find(docs))
        }
        wire::ResponseBody::CountRes => {
            let b = resp.body_as_count_res().expect("CountRes");
            Ok(Response::Count(b.count()))
        }
        wire::ResponseBody::ExistsRes => {
            let b = resp.body_as_exists_res().expect("ExistsRes");
            Ok(Response::Exists(b.exists()))
        }
        wire::ResponseBody::UpdateRes => {
            let b = resp.body_as_update_res().expect("UpdateRes");
            Ok(Response::Update(b.count()))
        }
        wire::ResponseBody::ReplaceRes => {
            let b = resp.body_as_replace_res().expect("ReplaceRes");
            Ok(Response::Replace(b.count()))
        }
        wire::ResponseBody::DeleteRes => {
            let b = resp.body_as_delete_res().expect("DeleteRes");
            Ok(Response::Delete(b.count()))
        }
        wire::ResponseBody::SearchRes => {
            let b = resp.body_as_search_res().expect("SearchRes");
            let hits = b
                .hits()
                .map(|v| {
                    v.iter()
                        .map(|h| (value_from_wire(h.doc().expect("hit doc")), h.score()))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Response::Search(hits))
        }
        wire::ResponseBody::GroupRes => {
            let b = resp.body_as_group_res().expect("GroupRes");
            let groups = b
                .groups()
                .map(|v| v.iter().map(value_from_wire).collect())
                .unwrap_or_default();
            Ok(Response::Group(groups))
        }
        wire::ResponseBody::StatsRes => {
            let b = resp.body_as_stats_res().expect("StatsRes");
            let per_index = b
                .per_index()
                .map(|v| {
                    v.iter()
                        .map(|s| IndexStat {
                            field: s.field().expect("field").to_string(),
                            entries: s.entries(),
                            distinct: s.distinct(),
                            memory: s.memory(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(Response::Stats(Stats {
                docs: b.docs(),
                docs_memory: b.docs_memory(),
                indexes: b.indexes(),
                total_memory: b.total_memory(),
                per_index,
            }))
        }
        wire::ResponseBody(t) => {
            Err(Error::Protocol(format!("unknown response body variant {t}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Collection: the entry points
// ---------------------------------------------------------------------------

/// A named collection over a live [`Client`] connection.
pub struct Collection<'c> {
    client: &'c mut Client,
    name: String,
}

impl<'c> Collection<'c> {
    /// The collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Insert one document; returns its `_id` (auto-generated if absent).
    pub fn insert(&mut self, doc: &Value) -> Result<String> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let d = value_to_wire(b, doc);
            let docs = b.create_vector(&[d]);
            let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs) });
            (Command::InsertCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Insert(ids) => Ok(ids
                .into_iter()
                .next()
                .ok_or_else(|| Error::Protocol("insert returned no id".into()))?),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for insert"))),
        }
    }

    /// Insert many documents; returns their `_id`s in request order.
    pub fn insert_many(&mut self, docs: &[Value]) -> Result<Vec<String>> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let offs: Vec<_> = docs.iter().map(|d| value_to_wire(b, d)).collect();
            let docs_off = b.create_vector(&offs);
            let cmd = wire::InsertCmd::create(b, &wire::InsertCmdArgs { docs: Some(docs_off) });
            (Command::InsertCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Insert(ids) => Ok(ids),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for insert_many"))),
        }
    }

    /// A lazy query chain: `find(filter)` -> `.sort(...)` -> `.skip(n)` ->
    /// `.limit(m)` -> a terminal (`.to_list()` / `.first()` / `.count()`).
    /// The RPC runs only at the terminal.
    pub fn find(&mut self, filter: impl Into<Value>) -> Query<'_> {
        Query {
            client: &mut *self.client,
            name: &self.name,
            filter: filter.into(),
            sort_field: None,
            sort_desc: false,
            skip: 0,
            limit: 0,
            one: false,
        }
    }

    /// Eager `find_one`: the first match (storage/index order, no pipeline),
    /// or `None`.
    pub fn find_one(&mut self, filter: impl Into<Value>) -> Result<Option<Value>> {
        self.find(filter).find_one()
    }

    /// Eager `count(filter)`: the number of matching documents.
    pub fn count(&mut self, filter: impl Into<Value>) -> Result<u64> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let f = value_to_wire(b, &filter.into());
            let cmd = wire::CountCmd::create(b, &wire::CountCmdArgs { filter: Some(f) });
            (Command::CountCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Count(n) => Ok(n),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for count"))),
        }
    }

    /// Eager `exists(filter)`: whether any document matches.
    pub fn exists(&mut self, filter: impl Into<Value>) -> Result<bool> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let f = value_to_wire(b, &filter.into());
            let cmd = wire::ExistsCmd::create(b, &wire::ExistsCmdArgs { filter: Some(f) });
            (Command::ExistsCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Exists(e) => Ok(e),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for exists"))),
        }
    }

    /// `update_one`: apply a Mongo operator object to the first match. Errors
    /// [`Status::NoMatch`] when nothing matches (the engine's convention).
    pub fn update_one(&mut self, filter: impl Into<Value>, update: impl Into<Value>) -> Result<u64> {
        self.update(filter.into(), update.into(), false)
    }

    /// `update_many`: apply a Mongo operator object to every match (0 valid).
    pub fn update_many(&mut self, filter: impl Into<Value>, update: impl Into<Value>) -> Result<u64> {
        self.update(filter.into(), update.into(), true)
    }

    fn update(&mut self, filter: Value, update: Value, many: bool) -> Result<u64> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let f = value_to_wire(b, &filter);
            let u = value_to_wire(b, &update);
            let cmd = wire::UpdateCmd::create(
                b,
                &wire::UpdateCmdArgs { filter: Some(f), update: Some(u), many },
            );
            (Command::UpdateCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Update(n) => Ok(n),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for update"))),
        }
    }

    /// `replace_one`: wholesale-replace the first match (preserving `_id`).
    /// Errors [`Status::NoMatch`] when nothing matches.
    pub fn replace_one(&mut self, filter: impl Into<Value>, new_doc: impl Into<Value>) -> Result<u64> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let f = value_to_wire(b, &filter.into());
            let nd = value_to_wire(b, &new_doc.into());
            let cmd = wire::ReplaceCmd::create(b, &wire::ReplaceCmdArgs { filter: Some(f), new_doc: Some(nd) });
            (Command::ReplaceCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Replace(n) => Ok(n),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for replace"))),
        }
    }

    /// `delete_one`: remove the first match; true if a doc was removed.
    pub fn delete_one(&mut self, filter: impl Into<Value>) -> Result<bool> {
        Ok(self.delete(filter.into(), false)? == 1)
    }

    /// `delete_many`: remove every match; returns the count removed.
    pub fn delete_many(&mut self, filter: impl Into<Value>) -> Result<u64> {
        self.delete(filter.into(), true)
    }

    fn delete(&mut self, filter: Value, many: bool) -> Result<u64> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let f = value_to_wire(b, &filter);
            let cmd = wire::DeleteCmd::create(b, &wire::DeleteCmdArgs { filter: Some(f), many });
            (Command::DeleteCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Delete(n) => Ok(n),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for delete"))),
        }
    }

    /// Brute-force cosine vector search (requires a vector index on `field`).
    /// Returns the top `limit` hits as `(doc, cosine_score)`, best first.
    /// `limit == 0` means no limit.
    pub fn vector_search(
        &mut self,
        field: &str,
        query: &[f32],
        limit: u64,
    ) -> Result<Vec<(Value, f64)>> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let field_off = b.create_string(field);
            let q = b.create_vector(query);
            let cmd = wire::VectorSearchCmd::create(
                b,
                &wire::VectorSearchCmdArgs { field: Some(field_off), query: Some(q), limit },
            );
            (Command::VectorSearchCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Search(hits) => Ok(hits),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for vector_search"))),
        }
    }

    /// BM25 text search (requires a text index on `field`).
    pub fn text_search(
        &mut self,
        field: &str,
        query: &str,
        limit: u64,
    ) -> Result<Vec<(Value, f64)>> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let field_off = b.create_string(field);
            let q = b.create_string(query);
            let cmd = wire::TextSearchCmd::create(
                b,
                &wire::TextSearchCmdArgs { field: Some(field_off), query: Some(q), limit },
            );
            (Command::TextSearchCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Search(hits) => Ok(hits),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for text_search"))),
        }
    }

    /// RRF hybrid search (requires BOTH a text index on `text_field` and a
    /// vector index on `vec_field`).
    pub fn hybrid_search(
        &mut self,
        text_field: &str,
        vec_field: &str,
        query_text: &str,
        query_vec: &[f32],
        limit: u64,
    ) -> Result<Vec<(Value, f64)>> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let tf = b.create_string(text_field);
            let vf = b.create_string(vec_field);
            let qt = b.create_string(query_text);
            let qv = b.create_vector(query_vec);
            let cmd = wire::HybridSearchCmd::create(
                b,
                &wire::HybridSearchCmdArgs {
                    text_field: Some(tf),
                    vec_field: Some(vf),
                    query_text: Some(qt),
                    query_vec: Some(qv),
                    limit,
                },
            );
            (Command::HybridSearchCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Search(hits) => Ok(hits),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for hybrid_search"))),
        }
    }

    /// Collection statistics (docs, per-index estimates, counts).
    pub fn stats(&mut self) -> Result<Stats> {
        let req_id = self.client.take_id();
        let payload = build_request(&self.name, req_id, |b| {
            let cmd = wire::StatsCmd::create(b, &wire::StatsCmdArgs::default());
            (Command::StatsCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Stats(s) => Ok(s),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for stats"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Query: the lazy chain (filter -> sort -> skip -> limit -> terminal)
// ---------------------------------------------------------------------------

/// A lazy query over a [`Collection`]. The single RPC runs only at a terminal
/// (`.to_list()` / `.first()` / `.count()` / `.find_one()`) or when chained
/// into a [`GroupQuery`].
pub struct Query<'q> {
    client: &'q mut Client,
    name: &'q str,
    filter: Value,
    sort_field: Option<String>,
    sort_desc: bool,
    skip: u64,
    limit: u64,
    one: bool,
}

impl<'q> Query<'q> {
    /// Sort the pipeline by `field` (engine total order); a later `.sort`
    /// replaces the earlier one. `desc` reverses the whole (value, `_id`) order.
    pub fn sort(mut self, field: impl Into<String>, desc: bool) -> Self {
        self.sort_field = Some(field.into());
        self.sort_desc = desc;
        self
    }

    /// Drop the first `n` matched docs of the (sorted) stream.
    pub fn skip(mut self, n: u64) -> Self {
        self.skip = n;
        self
    }

    /// Return at most `m` matched docs. `limit(0)` = no limit.
    pub fn limit(mut self, m: u64) -> Self {
        self.limit = m;
        self
    }

    /// Run the `find` pipeline and return every matched doc, in pipeline order.
    pub fn to_list(self) -> Result<Vec<Value>> {
        match self.find_pipeline()? {
            Response::Find(docs) => Ok(docs),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for find"))),
        }
    }

    /// Run the `find` pipeline and return the first matched doc (in pipeline
    /// order: filter → sort; `skip`/`limit` do not apply to a one-shot), or
    /// `None` when nothing matches.
    pub fn first(self) -> Result<Option<Value>> {
        // Reuse the one-shot pipeline: find_one semantics = one=true, no
        // sort/skip/limit overrides beyond what is already set. We clear skip
        // and force limit 1 via a dedicated one-shot request to match the
        // eager `find_one` contract (first in pipeline order).
        let req_id = self.client.take_id();
        let payload = build_request(self.name, req_id, |b| {
            let f = value_to_wire(b, &self.filter);
            let sort_field = self.sort_field.as_ref().map(|s| b.create_string(s));
            let cmd = wire::FindCmd::create(
                b,
                &wire::FindCmdArgs {
                    filter: Some(f),
                    sort_field,
                    sort_desc: self.sort_desc,
                    skip: 0,
                    limit: 1,
                    one: true,
                },
            );
            (Command::FindCmd, Some(cmd.as_union_value()))
        });
        let resp = self.client.send_receive(payload)?;
        match resp {
            Response::Find(docs) => Ok(docs.into_iter().next()),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for first"))),
        }
    }

    /// Run the `find` pipeline and return the number of matched docs (the
    /// count of the full filtered set — skip/limit do not apply to counts).
    pub fn count(self) -> Result<u64> {
        // The count terminal counts the filtered set (the server's CountCmd
        // takes only a filter; sort/skip/limit are pipeline-only and the
        // engine's `Query::count` counts matches after filter, before limit).
        let req_id = self.client.take_id();
        let payload = build_request(self.name, req_id, |b| {
            let f = value_to_wire(b, &self.filter);
            let cmd = wire::CountCmd::create(b, &wire::CountCmdArgs { filter: Some(f) });
            (Command::CountCmd, Some(cmd.as_union_value()))
        });
        match self.client.send_receive(payload)? {
            Response::Count(n) => Ok(n),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for count"))),
        }
    }

    /// Run the `find` pipeline and return the first matched doc, or `None`.
    pub fn find_one(self) -> Result<Option<Value>> {
        self.first()
    }

    /// Chain into an aggregation: group the (filtered, pipelined) stream by
    /// `field`. The query's sort/skip/limit become the *pre-group* pipeline.
    pub fn group(self, field: impl Into<String>) -> GroupQuery<'q> {
        GroupQuery {
            client: self.client,
            name: self.name,
            filter: self.filter,
            q_sort_field: self.sort_field,
            q_sort_desc: self.sort_desc,
            q_skip: self.skip,
            q_limit: self.limit,
            group_field: field.into(),
            g_sort_field: None,
            g_sort_desc: false,
            g_limit: 0,
        }
    }

    /// Shared FindCmd runner for the list terminal. Consumes the query.
    fn find_pipeline(self) -> Result<Response> {
        let req_id = self.client.take_id();
        let payload = build_request(self.name, req_id, |b| {
            let f = value_to_wire(b, &self.filter);
            let sort_field = self.sort_field.as_ref().map(|s| b.create_string(s));
            let cmd = wire::FindCmd::create(
                b,
                &wire::FindCmdArgs {
                    filter: Some(f),
                    sort_field,
                    sort_desc: self.sort_desc,
                    skip: self.skip,
                    limit: self.limit,
                    one: self.one,
                },
            );
            (Command::FindCmd, Some(cmd.as_union_value()))
        });
        self.client.send_receive(payload)
    }
}

// ---------------------------------------------------------------------------
// GroupQuery: the aggregation chain
// ---------------------------------------------------------------------------

/// A lazy aggregation over a [`Collection`]. Built from
/// `Collection::group(filter)` or `Query::group(field)`. The terminal is
/// `.agg(fn, field)`.
pub struct GroupQuery<'c> {
    client: &'c mut Client,
    name: &'c str,
    filter: Value,
    // pre-group (query-level) pipeline
    q_sort_field: Option<String>,
    q_sort_desc: bool,
    q_skip: u64,
    q_limit: u64,
    // group-level
    group_field: String,
    g_sort_field: Option<String>,
    g_sort_desc: bool,
    g_limit: u64,
}

impl<'c> GroupQuery<'c> {
    /// Re-sort the *group documents* by `field` (total order, ties by `_id`);
    /// defaults to group-key order. `desc` reverses the order.
    pub fn sort(mut self, field: impl Into<String>, desc: bool) -> Self {
        self.g_sort_field = Some(field.into());
        self.g_sort_desc = desc;
        self
    }

    /// Keep at most `m` group documents. `limit(0)` = no limit.
    pub fn limit(mut self, m: u64) -> Self {
        self.g_limit = m;
        self
    }

    /// Run the aggregation and return one result doc per group:
    /// `{ "_id": <group key>, "<fn-name>": <result> }`.
    pub fn agg(self, fn_: AggFn, field: impl Into<String>) -> Result<Vec<Value>> {
        let req_id = self.client.take_id();
        let payload = build_request(self.name, req_id, |b| {
            let f = value_to_wire(b, &self.filter);
            let qsf = self.q_sort_field.as_ref().map(|s| b.create_string(s));
            let gf = b.create_string(&self.group_field);
            let af = b.create_string(&field.into());
            let gsf = self.g_sort_field.as_ref().map(|s| b.create_string(s));
            let cmd = wire::GroupCmd::create(
                b,
                &wire::GroupCmdArgs {
                    filter: Some(f),
                    sort_field: qsf,
                    sort_desc: self.q_sort_desc,
                    skip: self.q_skip,
                    limit: self.q_limit,
                    group_field: Some(gf),
                    agg_fn: agg_fn_to_wire(fn_),
                    agg_field: Some(af),
                    group_sort_field: gsf,
                    group_sort_desc: self.g_sort_desc,
                    group_limit: self.g_limit,
                },
            );
            (Command::GroupCmd, Some(cmd.as_union_value()))
        });
        let resp = self.client.send_receive(payload)?;
        match resp {
            Response::Group(groups) => Ok(groups),
            other => Err(Error::Protocol(format!("unexpected response {other:?} for agg"))),
        }
    }
}

/// Map the engine [`AggFn`] to its wire discriminant.
fn agg_fn_to_wire(f: AggFn) -> wire::AggFn {
    use AggFn::*;
    match f {
        Count => wire::AggFn::Count,
        Sum => wire::AggFn::Sum,
        Mean => wire::AggFn::Mean,
        Min => wire::AggFn::Min,
        Max => wire::AggFn::Max,
        Collect => wire::AggFn::Collect,
        First => wire::AggFn::First,
        Last => wire::AggFn::Last,
    }
}

// ---------------------------------------------------------------------------
// Unit tests (no network): value round-trip + error helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a nested value tree through the wire encoder/decoder.
    #[test]
    fn value_roundtrips_losslessly() {
        let v = Value::Object(vec![
            ("_id".to_string(), Value::Str("deep".into())),
            ("i".to_string(), Value::I64(-42)),
            ("f".to_string(), Value::F64(-1.5)),
            ("b".to_string(), Value::Bool(true)),
            ("n".to_string(), Value::Null),
            ("s".to_string(), Value::Str("moo \u{1F404}".into())),
            (
                "arr".to_string(),
                Value::Array(vec![Value::I64(1), Value::Str("x".into()), Value::Bool(false)]),
            ),
            (
                "obj".to_string(),
                Value::Object(vec![
                    ("k1".to_string(), Value::I64(7)),
                    ("k2".to_string(), Value::Str("inner".into())),
                ]),
            ),
        ]);
        let mut b = FlatBufferBuilder::new();
        let off = value_to_wire(&mut b, &v);
        b.finish_minimal(off);
        let w = flatbuffers::root::<wire::Value>(b.finished_data()).unwrap();
        assert_eq!(value_from_wire(w), v, "value tree must round-trip");
    }

    /// Object key order is preserved through the wire (parallel keys/vals).
    #[test]
    fn object_key_order_preserved() {
        let v = Value::Object(vec![
            ("z".to_string(), Value::I64(1)),
            ("a".to_string(), Value::I64(2)),
            ("m".to_string(), Value::I64(3)),
        ]);
        let mut b = FlatBufferBuilder::new();
        let off = value_to_wire(&mut b, &v);
        b.finish_minimal(off);
        let w = flatbuffers::root::<wire::Value>(b.finished_data()).unwrap();
        let back = value_from_wire(w);
        let keys: Vec<&str> = match &back {
            Value::Object(p) => p.iter().map(|(k, _)| k.as_str()).collect(),
            _ => panic!("expected object"),
        };
        assert_eq!(keys, vec!["z", "a", "m"]);
    }

    /// `Error::status` exposes the typed status only for server errors.
    #[test]
    fn error_status_helper() {
        assert_eq!(
            Error::Server(Status::NoMatch, "x".into()).status(),
            Some(Status::NoMatch)
        );
        assert_eq!(Error::Protocol("y".into()).status(), None);
        let io = Error::Io(io::Error::new(io::ErrorKind::BrokenPipe, "z"));
        assert_eq!(io.status(), None);
        assert!(io.to_string().starts_with("I/O error"));
        let s = Error::Server(Status::DuplicateId, "dup".into());
        assert_eq!(s.to_string(), "server error DuplicateId: dup");
    }

    /// The `AggFn` mapping covers every engine variant.
    #[test]
    fn agg_fn_maps_every_variant() {
        assert_eq!(agg_fn_to_wire(AggFn::Count).0, wire::AggFn::Count.0);
        assert_eq!(agg_fn_to_wire(AggFn::Sum).0, wire::AggFn::Sum.0);
        assert_eq!(agg_fn_to_wire(AggFn::Mean).0, wire::AggFn::Mean.0);
        assert_eq!(agg_fn_to_wire(AggFn::Min).0, wire::AggFn::Min.0);
        assert_eq!(agg_fn_to_wire(AggFn::Max).0, wire::AggFn::Max.0);
        assert_eq!(agg_fn_to_wire(AggFn::Collect).0, wire::AggFn::Collect.0);
        assert_eq!(agg_fn_to_wire(AggFn::First).0, wire::AggFn::First.0);
        assert_eq!(agg_fn_to_wire(AggFn::Last).0, wire::AggFn::Last.0);
    }
}
