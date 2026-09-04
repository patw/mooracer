# MooRacer

![MooRacer](images/mooracer-banner.png)

An in-memory, network-accessible document data engine in **pure Rust**. No disk,
no persistence, no durable state — think "Redis, but a document store and
entirely in RAM." It provides the full query surface of
[Moofile](https://github.com/patw/moofile) **except** auto-embedding / semantic
search (voyage-4-nano ONNX inference is out of scope). Everything lives in
memory and every operator is optimized for raw throughput and low latency.

MooRacer is a **Rust workspace** with an in-memory engine, a FlatBuffers-over-TCP
server, Rust and Python clients, a benchmark harness, and a shared wire crate:

| crate          | package           | what it is                                                        |
|----------------|-------------------|-------------------------------------------------------------------|
| `engine/`      | `mooracer-engine` | The in-memory document engine (store, indexes, queries, search, aggregation). |
| `wire/`        | `mooracer-wire`   | The FlatBuffers wire schema (`mooracer.fbs`) + codegen (shared by server & clients). |
| `server/`      | `mooracer-server` | TCP server: length-prefixed FlatBuffers frames, thread pool, RwLock store. |
| `client-rust/` | `mooracer-client` | Synchronous Rust client exposing the Mongo-style chain API over the wire. |
| `client-python/` | `mooracer`        | Pure-Python client (generated wire + socket) — the `mooracer` package. |
| `bench/`       | `mooracer-bench`  | Benchmark binary: ops/s + p50/p99 µs across 14 workloads.         |

## Relationship to Moofile

[MooFile](https://github.com/patw/moofile) is Pat Wendorf's embedded,
single-file, **durable** document store — a library that reads/writes a `.bson`
file on disk (and can auto-embed text for on-device semantic search). **MooRacer
is its in-memory, network-accessible sibling.** It inherits Moofile's
Mongo-style query surface and its text/vector/hybrid search model, but flips the
storage story: instead of a file on disk, everything lives in RAM behind a TCP
server — no disk I/O, no durability, no file to open, just raw throughput and
low latency on a live process.

| | Moofile | **MooRacer** |
|---|---|---|
| Storage | single `.bson` file | **RAM only (in-process)** |
| Persistence / durability | ✓ | **none** |
| Auto-embedding (ONNX) | ✓ | no — caller supplies vectors |
| Network server + clients | ✗ | ✓ (FlatBuffers over TCP, Rust + Python) |
| Multi-process safe | ✓ | no — single process, threads |
| Query API | Mongo-style | **same Mongo-style chain** |
| Text / vector / hybrid search | ✓ | ✓ |

Choose **Moofile** when data must survive a restart or be shared between
processes. Choose **MooRacer** when you want a hot, in-RAM document store behind
a network protocol and don't mind losing everything on exit.

## Use cases — why a pure in-memory store

There's no disk, no durability, and no recovery, so MooRacer is **not** a
replacement for a database — but it's an excellent fit for data that's
ephemeral, volatile, or derived, where the cost/benefit of persistence isn't
worth it. Situations where a no-persistence document store genuinely shines:

- **Caching layer.** A hot, queryable cache in front of a slow source of truth
  (DB, HTTP API, computed value), invalidated on a TTL or a "rebuild" trigger.
  Documents let you cache rich aggregates — not just scalar keys.
- **Session / per-request state.** Live sessions, cart contents, feature-flag or
  A/B configuration snapshots — naturally regenerated or extinguished each
  visit.
- **Rate limiting & counters.** A high-throughput counter/flag store (Redis-like)
  but over nested documents and queries, e.g. per-user, per-day counters with a
  `find` + `$inc` pattern.
- **Ephemeral analytics / metrics.** Streaming events, live dashboards, and
  real-time rollups that are recomputed or discarded — aggregate what's
  happening *now*, not years of history.
- **Scratch / staging for batch work.** A fast in-memory workspace to hold,
  filter, and reshape intermediate datasets before the real work, with no temp
  file cleanup.
- **Test fixtures & mocks.** Deterministic, in-memory datasets that start empty
  each run and need no teardown; a document engine with indexes and search is a
  far richer test double than a plain hash map.
- **Demos, prototypes & embedded tooling.** Thin, fast data storage for a demo or
  CLI that doesn't want to ship a database dependency or a `.bson` file.

The trade-off is deliberate: **if a process restart must not lose data, MooRacer
is the wrong tool** — use [Moofile](https://github.com/patw/moofile) (or a real
database) instead.

## Layout & build

The workspace builds with a **maxed-out release profile** (the point of a
"racer" build): `[profile.release]` uses `opt-level = 3`, `lto = "fat"`,
`codegen-units = 1`, and `.cargo/config.toml` sets `-C target-cpu=native` for
every build. Debug (`cargo test`) stays fast; release is what the server and
benchmark ship with.

```sh
cargo build            # debug
cargo build --release  # optimized (LTO)
cargo test             # full suite (all crates; Rust only — see Python below)
```

> **Generated code is vendored.** The FlatBuffers types for the Rust `wire`
> crate are checked in at `wire/src/generated.rs` (and the Python client's
> `wire/` package), so the workspace builds **without** needing a system
> `flatc`. `flatc` is only required when you *edit* `schema/mooracer.fbs` and
> want to regenerate (`flatc --rust -o wire/src schema/mooracer.fbs`), or
> regenerate the Python package. Do **not** hand-edit the generated files.

## Quick start (Rust client)

```sh
# terminal 1 — start the server (defaults: 127.0.0.1:4141, pool 8)
MOORACER_ADDR=127.0.0.1:4141 MOORACER_THREADS=8 \
    cargo run --release -p mooracer-server

# terminal 2 — connect and run the Mongo-style chain API
```

```rust
use mooracer_client::Client;
use mooracer_engine::Value;

let mut client = Client::connect("127.0.0.1:4141").unwrap();
let mut col = client.collection("cows");

// insert
let id = col.insert(Value::object_from(vec![
    ("name".to_string(), Value::str("Bella".to_string())),
    ("age".to_string(),  Value::i64(5)),
])).unwrap();

// lazy query chain: filter → sort → limit, single RPC at the terminal
let filter = Value::object_from(vec![
    ("age".to_string(),
     Value::object_from(vec![("$gte".to_string(), Value::i64(4))])),
]);
let docs: Vec<Value> = col.find(filter)
    .sort("age", false)
    .limit(10)
    .to_list()
    .unwrap();

// eager entry points
let n: u64    = col.count(Value::object()).unwrap();
let first: _  = col.find_one(Value::object()).unwrap();
let any: bool = col.exists(Value::object()).unwrap();

// updates / deletes (operator object; `_id` is immutable)
let upd = Value::object_from(vec![
    ("$set".to_string(), Value::object_from(vec![("region".to_string(), Value::str("north".to_string()))])),
    ("$inc".to_string(), Value::object_from(vec![("age".to_string(),  Value::i64(1))])),
]);
col.update_one(Value::object_from(vec![("_id".to_string(), Value::str(id.clone()))]), upd).unwrap();
```

## Query API

Fluent, lazy, Mongo-style. Filters are `Value` objects in Mongo syntax; `{}`
matches all. Terminal `.to_list()` / `.first()` / `.count()` / `.find_one()`
run the single RPC.

- **Comparison:** implicit `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, ranges
  (`{"age": {"$gte": 25, "$lt": 40}}`).
- **Set:** `$in`, `$nin`.
- **Logical:** `$and`, `$or`, `$not`.
- **Element:** `$exists`.
- **Array:** `$elemMatch` (equality, comparison, sub-document).
- **Pipeline:** `.sort(field, desc)`, `.skip(n)`, `.limit(n)` (limit 0 = none).

An **index is used automatically** when the driving top-level field is indexed;
otherwise the engine does a full scan. `_id` is always indexed. Index-driven and
full-scan results are always identical (every index candidate is re-verified).

## Write API

- `insert(doc)` / `insert_many(docs)` — auto-assigns a 24-char hex `_id` when
  missing; duplicates raise an error; `insert_many` is atomic (staged).
- `update_one` / `update_many(filter, {"$set":…,"$inc":…,"$unset":…})` —
  `update_one` errors on no-match (`NoMatch`).
- `replace_one(filter, new_doc)` — wholesale replace, `_id` preserved.
- `delete_one(filter) -> bool`, `delete_many(filter) -> count`.
- **Atomic batch:** `collection.begin() -> Transaction` buffers writes; reads
  inside see the pre-batch state; `commit()` applies all + all indexes
  atomically; drop/`rollback()` discards; rollback on error.

All writes keep every index (value / vector / text) in lockstep.

## Search & aggregation

- `vector_search(field, query_vec, limit)` — brute-force **cosine** over the
  configured-dim `Vec<f32>` index (vectors are unit-normalized once at write
  time; scores are a single SIMD dot product).
- `text_search(field, query, limit)` — **BM25** + **Porter stemming** over an
  inverted index.
- `hybrid_search(text_field, vec_field, query_text, query_vec, limit)` —
  **Reciprocal Rank Fusion** (RRF, `K = 60`) of the two rankings.
- `find(filter).group(field).agg(fn, field)` — group by a field with
  `count`/`sum`/`mean`/`min`/`max`/`collect`/`first`/`last`, optional
  `.sort`/`.limit` on the grouped result.

All search/aggregation results are returned **best-first / by group key**, full
document clones, with the `limit(0) = no limit` convention.

## Wire protocol (FlatBuffers)

One `.fbs` schema (`schema/mooracer.fbs`, code-generated into `mooracer-wire`)
defines the request/response envelope. Frames are **4-byte little-endian `u32`
length-prefixed** FlatBuffers buffers over a binary TCP stream (256 MiB cap).
`file_identifier "MOOR"`, envelope `version 1`.

- **Request**: `version` + `collection` + a **command union** (one payload per
  engine surface: `Insert`/`Find`/`Count`/`Exists`/`Update`/`Replace`/`Delete`/
  `VectorSearch`/`TextSearch`/`HybridSearch`/`Group`/`Stats`/`Index`). `one`/
  `many` flags collapse `*_one`/`*_many` onto one payload each; `limit == 0` =
  no limit; `filter {}` = all.
- **Response**: `version` + `status` + `message` + a **response union** body
  (`InsertRes{ids}`, `FindRes{docs}`, `CountRes`, `ExistsRes`, `UpdateRes`,
  `ReplaceRes`, `DeleteRes{count}`, `SearchRes{hits:[{doc,score}]}`,
  `GroupRes{groups}`, `StatsRes{…}`, `IndexRes`). `body` is present only on
  `OK`.
- **Values** are a single `Value` table discriminated by `kind`
  (`Null|Bool|I64|F64|Str|Array|Object`); an object is parallel `keys`/`vals`
  vectors (insertion order preserved).
- **Status codes** are `OK` + the nine engine `StoreError` variants verbatim
  (`NotAnObject`, `IdMustBeString`, `DuplicateId`, `IdMismatch`, `NoIndex`,
  `PrimaryIndex`, `NoMatch`, `InvalidUpdate`, `VectorDimMismatch`) + four
  transport codes (`MalformedRequest`, `UnknownCommand`, `UnsupportedVersion`,
  `InternalError`). Clients surface these as typed errors.

The numeric discriminants of every enum/union **are** the wire format and are
pinned by `wire/tests/schema.rs` (a protocol break fails the suite loudly).

### Server model

`mooracer-server` owns one shared store: `RwLock<HashMap<String, Collection>>`.
Read commands take the shared lock (concurrent reads); write commands take the
exclusive lock (serialized writes). A small, **configurable thread pool**
(`MOORACER_THREADS`) shares one `mpsc` receiver; each worker pulls one
connection and serves it to EOF in a request/response loop. A missing collection
is not an error for reads (empty/zero result) — searches return `NoIndex`; writes
create the collection on first use. **Indexes are managed over the wire** with an
`IndexCmd` (create/drop value, vector, and text indexes) — see
[Index management](#index-management) below — so a client can enable search at
runtime without any server-side setup.

```sh
MOORACER_ADDR=127.0.0.1:4141 MOORACER_THREADS=8 cargo run --release -p mooracer-server
```

### Index management

Create/drop index types on a collection over the wire. This is what enables
vector, text, and hybrid search — and value-indexed range/equality fast paths —
from any client:

```rust
col.create_index("region")?;          // value index (equality + ranges)
col.create_vector_index("emb", 64)?;  // vector index (cosine search)
col.create_text_index("body")?;       // BM25 text index
// col.drop_index("region")?            // etc. (`_id` cannot be dropped)
```

```python
col.create_index("region")
col.create_vector_index("emb", 64)
col.create_text_index("body")
```

Dropping the primary `_id` index and dropping a nonexistent index are typed
errors (`PrimaryIndex` / `NoIndex`).

## Python client

Pure Python (no Rust FFI) — it is a network client. The package
`client-python/mooracer/` has a hand-written `client.py` plus a checked-in
`wire/` subpackage generated by `flatc --python` from the same schema. It is a
proper PyPI package (`[project]` in `client-python/pyproject.toml`) — install
with `pip install ./client-python` (or `pip install -e ./client-python`).

```python
from mooracer import Client

c = Client.connect("127.0.0.1:4141")
col = c.collection("cows")

col.insert({"name": "Bella", "age": 5})
col.find({"age": {"$gte": 4}}).sort("age", False).limit(10).to_list()  # [dict, …]
col.find_one({})       # dict | None
col.count({})          # int
col.exists({})         # bool
col.update_one({"_id": "…"}, {"$set": {"region": "north"}, "$inc": {"age": 1}})
col.delete_one({"_id": "…"})   # bool
col.vector_search("vec", [0.0]*64, 10)   # [(dict, float), …]
col.find({"region": "north"}).group("region").agg("count", None)  # [dict, …]
```

Native mapping: `dict`→Object (order preserved), `list`/`tuple`→Array,
`str`→Str, `int`→I64, `float`→F64, `bool`→Bool (checked before `int`),
`None`→Null. Errors: `MooracerError` base; `MooracerIOError`, `ProtocolError`,
`ServerError(status, message, name)`.

The Python **tests** drive a dev server (`server/src/bin/mooracer-devserver.rs`)
that can pre-create indexes from env (`MOORACER_VECTOR_INDEX=coll:field:dim;…`,
`MOORACER_TEXT_INDEX=coll:field;…`); index creation over the wire is also
covered directly (see [Index management](#index-management)). Run the Python
suite with:

```sh
python3 -m pytest client-python/tests -q
```

## Benchmarking

`bench` builds a deterministic in-memory corpus and runs **14 workloads**
(insert, insert_many, indexed-equality, range-find, scan, sort-limit,
update_one, delete, `$and`/`$or`, `$elemMatch`, vector-search, text-search,
hybrid-search, group-agg), reporting **ops/s + p50/p99 µs** per workload.

```sh
cargo run --release --bin bench                    # 20k docs
cargo run --release --bin bench -- --size 250000   # larger
cargo run --release --bin bench -- --quick         # small, fast
cargo run --release --bin bench -- --write-md      # regenerate BENCH.md
```

See [`BENCH.md`](BENCH.md) for the current report and methodology.

## Examples

The [`examples/`](examples/) directory has runnable scripts (Python client)
that drive a live server end-to-end:

- [`examples/simple_crud.py`](examples/simple_crud.py) — insert / find / update
  / delete / count / exists, error handling, and stats. The "hello world" for
  the query chain.
- [`examples/complex_usage.py`](examples/complex_usage.py) — vector search, BM25
  text search, hybrid (RRF) search, group-by aggregation, and stats over a
  small product corpus.

See [`examples/README.md`](examples/README.md) for how to start the server (and
the dev server for the search indexes) and run them.

## Verification

- **spec** — `spec.md` (behavior + API contract).
- **code** — matches the spec's purpose and API.
- **tests** — a real `cargo test` suite covers every operator, search mode,
  aggregation, atomic batch, network round-trip, and the Rust client's happy +
  error paths; the Python client has its own pytest suite.
- **docs** — this README + `spec.md` + `BENCH.md` + `CHANGELOG.md` + `examples/`.

## License

MIT — see [`LICENSE`](LICENSE).

## Notes / non-goals

No persistence / disk I/O / durability / recovery. No auto-embedding, ONNX, or
semantic search. No cross-process file locking (single process, in-memory). No
MongoDB/BSON compatibility layer — documents are in-memory native value trees.
