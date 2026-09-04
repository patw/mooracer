# MooRacer benchmark

Raw throughput + latency report for the in-memory document engine.

## Methodology

- **Build**: `--release` (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `-C target-cpu=native`). These are the shipped numbers.
- **Dataset**: 20000 deterministic documents, each with scalars (`age`, `score`, `region`, `name`), a text `body`, an `items` array of sub-documents, and a 64-dim `vec`. Value indexes on `age`/`region`, a vector index on `vec`, and a BM25 text index on `body` are created once.
- **Read workloads** sample a bounded number of operations (corpus size scales the per-op cost, the op count stays fixed) so any run is a real, reproducible amount of work; **mutating workloads** run on bounded scratch collections so they never corrupt the shared read fixture.
- **Measurement**: each operation is timed individually; `p50`/`p99` are percentiles of the per-op latency distribution (µs), and `ops/s` is `ops / wall_time` over the whole workload.

## Results — dataset size = 20000 docs

| workload | ops | ops/s | p50 (µs) | p99 (µs) |
|---|---:|---:|---:|---:|
| insert | 20000 | 702154 | 1.38 | 1.44 |
| insert_many | 20 | 316 | 3074.23 | 3149.87 |
| indexed-equality | 1500 | 25352 | 40.54 | 46.45 |
| range-find | 1500 | 1745 | 570.07 | 601.36 |
| scan | 1500 | 1355 | 710.74 | 1189.11 |
| sort-limit | 1500 | 37830 | 26.12 | 30.08 |
| update_one | 1500 | 188798 | 5.22 | 7.13 |
| delete | 4000 | 209578 | 4.69 | 6.73 |
| logical-and-or | 1500 | 288 | 3457.18 | 3805.77 |
| elemMatch | 1500 | 162 | 6151.21 | 6306.49 |
| vector-search | 1500 | 753 | 1322.87 | 1398.81 |
| text-search | 1500 | 4453 | 228.11 | 249.53 |
| hybrid-search | 1500 | 207 | 4834.91 | 4965.89 |
| group-agg | 1500 | 356 | 2172.69 | 4145.89 |

## Reading the numbers

- **Fast (µs, index-/key-driven):** `indexed-equality`, `sort-limit`, `update_one`, `delete` — index-driven point/range lookups and primary-key writes.
- **Mid (hundreds of µs):** `range-find`, `text-search` (posting-list walk), `scan` (full scan). `insert` is ~1.4 µs/doc amortized.
- **Slow (ms):** `logical-and-or`, `elemMatch`, `group-agg` (full scans), `vector-search` (brute-force over the whole corpus), and `hybrid-search` (two full rankings per op) — the natural targets for further optimization.

## Regenerating

```sh
cargo run --release --bin bench -- --write-md   # 20k docs
cargo run --release --bin bench -- --size 250000 --write-md
cargo run --release --bin bench -- --quick   # small, fast
```

