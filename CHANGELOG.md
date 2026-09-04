# Changelog

All notable changes to MooRacer are documented here. This is the initial public
release; entries follow a Keep-a-Changelog-style format.

## [0.1.0] - 2026-09-04

Initial public release of **MooRacer** — an in-memory, network-accessible
document data engine in pure Rust. No disk, no persistence, no durable state:
everything lives in RAM and every operator is tuned for raw throughput and low
latency. It is the high-performance, in-memory sibling of
[Moofile](https://github.com/patw/moofile), sharing Moofile's Mongo-style query
surface but shipping as a Rust workspace with a FlatBuffers-over-TCP server and
both Rust and Python clients.

### Added

- **In-memory document engine** (`engine/`) — a `Collection` of nested
  JSON-like `Value` trees with automatic `_id` assignment, a primary index, and
  per-field value indexes.
- **Mongo-style lazy query chain** — `find(filter).sort().skip().limit()`
  with `$eq/$ne/$gt/$gte/$lt/$lte`, ranges, `$in/$nin`, `$and/$or/$not`,
  `$exists`, and `$elemMatch`, terminating in `to_list()/first()/count()/
  find_one()`. Index-driven and full-scan results are always identical.
- **Write API** — `insert`/`insert_many` (atomic, staged), `update_one`/
  `update_many` (`$set`/`$inc`/`$unset`), `replace_one` (preserves `_id`),
  `delete_one`/`delete_many`, and an atomic batch `Transaction`
  (`begin()`/`commit()`/`rollback()`). All indexes are kept in lockstep.
- **Vector search** — brute-force cosine over a configured-dim `Vec<f32>`
  index (unit-normalized once at write time; SIMD dot products).
- **Text search** — BM25 (K1 = 1.2, b = 0.75) with Porter stemming over an
  inverted index.
- **Hybrid search** — Reciprocal Rank Fusion (`K = 60`) of the BM25 and vector
  rankings.
- **Aggregation** — `find(filter).group(field).agg(...)` with
  `count/sum/mean/min/max/collect/first/last`, plus group-level `sort`/`limit`.
- **FlatBuffers wire protocol** (`wire/`) — a single `.fbs` schema
  (`schema/mooracer.fbs`, `file_identifier "MOOR"`, envelope v1) defining a
  command union and a typed result union, length-prefixed frames over TCP.
  Numeric discriminants are pinned by the schema test suite.
- **TCP server** (`server/`) — threaded listener with a configurable thread
  pool, shared `RwLock<HashMap<String, Collection>>` store, concurrent reads /
  serialized writes, and typed error status codes on the wire.
- **Rust client** (`client-rust/`) — a synchronous, reborrowing client exposing
  the same chain API over the wire with typed errors
  (`Io`/`Protocol`/`Server(status, message)`).
- **Python client** (`client-python/`) — a pure-Python (no FFI) network client
  with a generated `wire/` subpackage, native `dict`/`list`/`str`/`int`/
  `float`/`bool`/`None` mapping, and typed `MooracerError` hierarchy.
- **Benchmark harness** (`bench/`) — 14 representative workloads reporting
  ops/s plus p50/p99 latency (µs), driven by a deterministic in-memory corpus,
  with a `--release` CPU-native profile. See `BENCH.md` for the report.
- **Documentation** — `README.md`, `spec.md` (behavior + API contract),
  `BENCH.md` (benchmark report), `examples/`, and this changelog.

### Notes

- **No persistence / durability / recovery.** State is process-bound; a restart
  loses everything. This is by design (see the README "Use cases").
- **No auto-embedding / ONNX / semantic search.** Vector search expects caller
  supplied embeddings (a configured-dim `Vec<f32>`).
- **No cross-process locking.** Thread-safe within one process (concurrent
  reads, serialized writes); single-process by design.
- **Prerequisite:** the `wire` crate runs the FlatBuffers compiler (`flatc`) in
  its `build.rs` — it must be on `PATH` (or set `FLATC=/path/to/flatc`).
