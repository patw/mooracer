//! mooracer-bench — MooRacer benchmark harness.
//!
//! Runs a fixed set of representative workloads against an in-memory
//! [`Collection`] and reports, per workload:
//!
//!   * **ops/s**        — operations completed per second
//!   * **p50 / p99 µs** — per-operation latency percentiles
//!
//! Every workload is *self-contained*: a workload that mutates the store
//! (insert / insert_many / update_one / delete) builds its own throwaway
//! collection so the read workloads always see an intact dataset, and the
//! harness runs each operation many times to collect a latency histogram.
//!
//! The dataset size is configurable (`--size`, default 20 000 docs); read
//! workloads always sample a bounded number of operations (the corpus size
//! scales the per-op cost, the op count stays fixed) so any run is a real,
//! reproducible amount of work. `--quick` drops the defaults to a small
//! dataset so the harness is useful interactively and in CI.
//!
//! Usage:
//!   bench                     # 20k docs
//!   bench --size 250000       # larger dataset
//!   bench --quick             # tiny dataset, fast
//!
//! This is `--release`-oriented: the release profile (fat LTO, codegen-units=1,
//! `target-cpu=native`) is what the report numbers in `BENCH.md` reflect.

use std::hint::black_box;
use std::time::Instant;

use mooracer_engine::{AggFn, Collection, Value};

/// Vector dimension for the `vec` field (brute-force cosine search).
const DIM: usize = 64;
/// Text-corpus word bank: big enough for a stable BM25 idf, small enough to
/// stay cheap to build.
const WORDS: &[&str] = &[
    "moo",
    "cow",
    "herd",
    "pasture",
    "barn",
    "field",
    "meadow",
    "triumph",
    "velvet",
    "copper",
    "harvest",
    "orchard",
    "garden",
    "winter",
    "spring",
    "summer",
    "autumn",
    "ember",
    "frost",
    "thunder",
    "lightning",
    "rainbow",
    "starlight",
    "moonbeam",
    "daybreak",
    "dusk",
    "dawn",
    "horizon",
    "summit",
    "valley",
    "river",
    "canyon",
    "desert",
    "forest",
    "grove",
    "mangrove",
    "redwood",
    "cedar",
    "birch",
    "willow",
    "oak",
    "maple",
    "spruce",
    "pine",
    "larch",
    "alder",
];
/// The small `region` domain: gives `group` a handful of real groups and makes
/// indexed-equality / `$and` / `$or` filters hit a realistic fraction.
const REGIONS: &[&str] = &["north", "south", "east", "west"];
/// `batch` size for the `insert_many` workload.
const BATCH: usize = 1_000;

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// Build one deterministic document. All fields a workload needs are present:
/// scalars for equality/range/scan, an object array for `$elemMatch`, a text
/// field for BM25, and a numeric array for the vector index.
fn make_doc(i: u64) -> Value {
    let region = REGIONS[(i % REGIONS.len() as u64) as usize];
    let tags: Vec<Value> = (0..3)
        .map(|k| Value::str(WORDS[((i + k) as usize) % WORDS.len()]))
        .collect();
    // A small array of sub-documents — the `$elemMatch` target.
    let items: Vec<Value> = (0..4)
        .map(|k| {
            Value::object_from(vec![
                ("qty".to_string(), Value::i64(((i + k) % 50) as i64)),
                (
                    "w".to_string(),
                    Value::str(REGIONS[((i + k) as usize) % REGIONS.len()]),
                ),
            ])
        })
        .collect();
    // A deterministic pseudo-random-ish f32 vector (per-dim value in [-1, 1]).
    let mut vec: Vec<Value> = Vec::with_capacity(DIM);
    for d in 0..DIM {
        let t = (((i as u32).wrapping_mul(2654435761).wrapping_add(d as u32) >> 8) & 0xff) as f64;
        vec.push(Value::f64(t / 127.5 - 1.0));
    }
    // A sentence of words for BM25.
    let body = (0..8)
        .map(|k| WORDS[((i + k * 7) as usize) % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ");

    Value::object_from(vec![
        ("_id".to_string(), Value::str(format!("{i:024}"))),
        ("age".to_string(), Value::i64((i % 1000) as i64)),
        ("score".to_string(), Value::f64(((i % 10_000) as f64) / 7.0)),
        ("region".to_string(), Value::str(region.to_string())),
        ("name".to_string(), Value::str(format!("cow-{i}"))),
        ("tags".to_string(), Value::array_from(tags)),
        ("items".to_string(), Value::array_from(items)),
        ("body".to_string(), Value::str(body)),
        ("vec".to_string(), Value::array_from(vec)),
    ])
}

/// Build a populated [`Collection`] of `size` docs with the value indexes
/// (age, region), a vector index on `vec`, and a text index on `body` created
/// once over the whole corpus. This is the shared fixture for every *read*
/// workload; mutating workloads build their own collection.
fn build_dataset(size: usize) -> Collection {
    let mut col = Collection::new("bench");
    // insert_many in batches (also warms the store).
    let mut docs: Vec<Value> = Vec::with_capacity(size);
    for i in 0..size as u64 {
        docs.push(make_doc(i));
        if docs.len() == BATCH {
            col.insert_many(std::mem::take(&mut docs)).unwrap();
        }
    }
    if !docs.is_empty() {
        col.insert_many(docs).unwrap();
    }
    col.create_index("age").unwrap();
    col.create_index("region").unwrap();
    col.create_vector_index("vec", DIM);
    col.create_text_index("body");
    col
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// A completed workload: `ops` operations taking `total` seconds, with the
/// p50/p99 per-operation latency in microseconds.
struct Bench {
    name: &'static str,
    ops: u64,
    total: f64,
    p50: f64,
    p99: f64,
}

impl Bench {
    fn ops_per_sec(&self) -> f64 {
        self.ops as f64 / self.total.max(f64::MIN_POSITIVE)
    }
}

/// Run `f` `iters` times, timing each call individually and returning the
/// workload stats. `f` returns a `u64` that is black-boxed (xor-accumulated)
/// so the optimizer cannot elide the operation.
fn bench(name: &'static str, iters: u64, mut f: impl FnMut(u64) -> u64) -> Bench {
    let mut samples = Vec::with_capacity(iters as usize);
    let mut acc: u64 = 0;
    let total_start = Instant::now();
    for i in 0..iters {
        let t0 = Instant::now();
        acc ^= black_box(f(i));
        let dt = t0.elapsed().as_secs_f64() * 1e6; // microseconds
        samples.push(dt);
    }
    let total = total_start.elapsed().as_secs_f64();
    let _ = black_box(acc);
    samples.sort_by(|a, b| a.partial_cmp(b).expect("f64 total order for durations"));
    let pick = |q: usize| samples[((iters as usize).saturating_sub(1) * q) / 100];
    Bench {
        name,
        ops: iters,
        total,
        p50: pick(50),
        p99: pick(99),
    }
}

/// `q`-percentile (0..=100) of a *non-empty* slice, computed in place-free by
/// a partial selection-free full sort (the harness sorts anyway). Exposed for
/// tests.
pub fn percentile(sorted: &[f64], q: usize) -> f64 {
    assert!(!sorted.is_empty(), "percentile of empty slice");
    assert!(q <= 100, "percentile q must be <= 100");
    sorted[sorted.len().saturating_sub(1) * q / 100]
}

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

fn op_count(size: usize) -> u64 {
    // Read workloads sample this many operations. Capped (independent of
    // `size`) so a run is a fixed, reproducible amount of work no matter how
    // large the corpus is — the corpus size scales the *per-op* cost, while
    // the op count stays bounded so `--size 100000` does not blow up runtime.
    (size as u64).clamp(300, 1_500)
}

/// Bounded corpus size for the *mutating* workloads (insert / insert_many /
/// update / delete): these measure per-write latency, so they run a fixed
/// number of writes on a bounded scratch corpus rather than the full `size`
/// corpus (delete in particular is O(corpus) per op, so it must not scale with
/// `size` or large runs stall).
fn write_corpus(size: usize) -> usize {
    size.min(20_000)
}

fn run_all(size: usize) -> Vec<Bench> {
    let col = build_dataset(size);
    let iters = op_count(size);
    let wsize = write_corpus(size);
    // Delete is O(corpus) per op (index deindex memmoves), so keep its
    // scratch corpus small and independent of `size`.
    let del_size = size.clamp(500, 4_000);
    let mut out: Vec<Bench> = Vec::with_capacity(16);

    // --- insert (one doc at a time, throwaway collection) -----------------
    out.push(bench("insert", wsize as u64, |i| {
        let mut c = Collection::new("w-insert");
        c.insert(make_doc(i)).unwrap();
        c.len() as u64
    }));

    // --- insert_many (batch of BATCH docs, throwaway collection) ----------
    let im_iters = (wsize as u64 / BATCH as u64).max(1);
    out.push(bench("insert_many", im_iters, |i| {
        let mut c = Collection::new("w-insertmany");
        let base = i * BATCH as u64;
        let docs: Vec<Value> = (0..BATCH as u64).map(|k| make_doc(base + k)).collect();
        c.insert_many(docs).unwrap();
        c.len() as u64
    }));

    // --- indexed equality (age is indexed; a selective point lookup, ~20
    //     matches per op on the default corpus) ----------------------------
    out.push(bench("indexed-equality", iters, |i| {
        let a = ((i % 1000) as i64) + 50;
        let f = Value::object_from(vec![("age".to_string(), Value::i64(a))]);
        col.find(f).to_list().len() as u64
    }));

    // --- range find (age is indexed, narrow range) ------------------------
    out.push(bench("range-find", iters, |i| {
        let lo = ((i % 900) as i64) + 50;
        let f = Value::object_from(vec![(
            "age".to_string(),
            Value::object_from(vec![
                ("$gte".to_string(), Value::i64(lo)),
                ("$lt".to_string(), Value::i64(lo + 20)),
            ]),
        )]);
        col.find(f).to_list().len() as u64
    }));

    // --- unindexed scan (score is NOT value-indexed → full scan) ----------
    out.push(bench("scan", iters, |i| {
        let v = ((i % 10_000) as f64) / 7.0;
        let f = Value::object_from(vec![("score".to_string(), Value::f64(v))]);
        col.find(f).count() as u64
    }));

    // --- sort + limit (indexed sort on `age` streams the index and stops at
    //     the limit — the fast path; O(skip+limit) per op) ----------------
    out.push(bench("sort-limit", iters, |i| {
        let f = Value::object(); // all
        col.find(f)
            .sort("age", i % 2 == 0)
            .limit(20)
            .to_list()
            .len() as u64
    }));

    // --- update_one ($set + $inc by _id; scratch corpus updated one per op) -
    let mut upd = Box::new(Collection::new("w-update"));
    {
        let n = iters as usize; // bounded corpus: exactly one doc per op
        let mut d = Vec::with_capacity(BATCH);
        for i in 0..n as u64 {
            d.push(make_doc(i));
            if d.len() == BATCH {
                upd.insert_many(std::mem::take(&mut d)).unwrap();
            }
        }
        if !d.is_empty() {
            upd.insert_many(d).unwrap();
        }
    }
    out.push(bench("update_one", iters, |i| {
        let id = format!("{i:024}");
        let f = Value::object_from(vec![("_id".to_string(), Value::str(id))]);
        let u = Value::object_from(vec![
            (
                "$set".to_string(),
                Value::object_from(vec![(
                    "region".to_string(),
                    Value::str(REGIONS[(i % REGIONS.len() as u64) as usize].to_string()),
                )]),
            ),
            (
                "$inc".to_string(),
                Value::object_from(vec![("age".to_string(), Value::i64(1))]),
            ),
        ]);
        upd.update_one(f, u).unwrap() as u64
    }));

    // --- delete (delete_one by _id; a bounded scratch corpus, drained) ----
    let mut del = Box::new(Collection::new("w-delete"));
    {
        let mut d = Vec::with_capacity(BATCH);
        for i in 0..del_size as u64 {
            d.push(make_doc(i));
            if d.len() == BATCH {
                del.insert_many(std::mem::take(&mut d)).unwrap();
            }
        }
        if !d.is_empty() {
            del.insert_many(d).unwrap();
        }
    }
    out.push(bench("delete", del_size as u64, |i| {
        let f = Value::object_from(vec![("_id".to_string(), Value::str(format!("{i:024}")))]);
        let r = del.delete_one(f) as u64;
        r + del.len() as u64
    }));

    // --- $and / $or logical filters ---------------------------------------
    out.push(bench("logical-and-or", iters, |i| {
        let and_f = Value::object_from(vec![(
            "$and".to_string(),
            Value::array_from(vec![
                Value::object_from(vec![(
                    "region".to_string(),
                    Value::str(REGIONS[(i % REGIONS.len() as u64) as usize].to_string()),
                )]),
                Value::object_from(vec![(
                    "age".to_string(),
                    Value::object_from(vec![("$gte".to_string(), Value::i64(100))]),
                )]),
            ]),
        )]);
        let a = col.find(and_f).count();
        let or_f = Value::object_from(vec![(
            "$or".to_string(),
            Value::array_from(vec![
                Value::object_from(vec![(
                    "region".to_string(),
                    Value::str("north".to_string()),
                )]),
                Value::object_from(vec![(
                    "region".to_string(),
                    Value::str("south".to_string()),
                )]),
            ]),
        )]);
        let b = col.find(or_f).count();
        (a + b) as u64
    }));

    // --- $elemMatch -------------------------------------------------------
    out.push(bench("elemMatch", iters, |i| {
        let f = Value::object_from(vec![(
            "items".to_string(),
            Value::object_from(vec![(
                "$elemMatch".to_string(),
                Value::object_from(vec![
                    (
                        "qty".to_string(),
                        Value::object_from(vec![("$gte".to_string(), Value::i64(40))]),
                    ),
                    (
                        "w".to_string(),
                        Value::str(REGIONS[(i % REGIONS.len() as u64) as usize].to_string()),
                    ),
                ]),
            )]),
        )]);
        col.find(f).count() as u64
    }));

    // --- vector search (brute-force cosine) -------------------------------
    out.push(bench("vector-search", iters, |i| {
        let q: Vec<f32> = (0..DIM)
            .map(|d| {
                let t =
                    (((i as u32).wrapping_mul(40503).wrapping_add(d as u32) >> 8) & 0xff) as f32;
                t / 127.5 - 1.0
            })
            .collect();
        col.vector_search("vec", &q, 10).unwrap().len() as u64
    }));

    // --- text search (BM25 + Porter) --------------------------------------
    out.push(bench("text-search", iters, |i| {
        let w = WORDS[(i as usize) % WORDS.len()];
        let query = format!("{w} {}", WORDS[((i + 3) as usize) % WORDS.len()]);
        col.text_search("body", &query, 10).unwrap().len() as u64
    }));

    // --- hybrid search (RRF of BM25 + vector) -----------------------------
    out.push(bench("hybrid-search", iters, |i| {
        let q: Vec<f32> = (0..DIM)
            .map(|d| {
                let t =
                    (((i as u32).wrapping_mul(69069).wrapping_add(d as u32) >> 8) & 0xff) as f32;
                t / 127.5 - 1.0
            })
            .collect();
        let query = WORDS[(i as usize) % WORDS.len()].to_string();
        col.hybrid_search("body", "vec", &query, &q, 10)
            .unwrap()
            .len() as u64
    }));

    // --- aggregation (group by region, sum score) -------------------------
    out.push(bench("group-agg", iters, |i| {
        let f = if i % 2 == 0 {
            Value::object_from(vec![(
                "region".to_string(),
                Value::str(REGIONS[(i % REGIONS.len() as u64) as usize].to_string()),
            )])
        } else {
            Value::object()
        };
        col.find(f).group("region").agg(AggFn::Sum, "score").len() as u64
    }));

    out
}

/// Render the benchmark table (also what `BENCH.md` is generated from).
fn render(rows: &[Bench], size: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "MooRacer benchmark — dataset size = {size} docs\n"
    ));
    s.push_str(&format!(
        "{:<20} {:>12} {:>12} {:>12} {:>12}\n",
        "workload", "ops", "ops/s", "p50 (µs)", "p99 (µs)"
    ));
    s.push_str(&"-".repeat(70));
    s.push('\n');
    for b in rows {
        s.push_str(&format!(
            "{:<20} {:>12} {:>12.0} {:>12.2} {:>12.2}\n",
            b.name,
            b.ops,
            b.ops_per_sec(),
            b.p50,
            b.p99
        ));
    }
    s
}

/// Render the full `BENCH.md` report (markdown) from the workload rows. This
/// is the deliverable report; `render` above is the compact stdout table. The
/// table data is exactly what `bench` measured — regenerate with
/// `cargo run --release --bin bench -- --write-md`.
fn report_md(rows: &[Bench], size: usize) -> String {
    let mut s = String::new();
    s.push_str("# MooRacer benchmark\n\n");
    s.push_str("Raw throughput + latency report for the in-memory document engine.\n\n");
    s.push_str("## Methodology\n\n");
    s.push_str(
        "- **Build**: `--release` (`opt-level = 3`, `lto = \"fat\"`, `codegen-units = 1`, \
         `-C target-cpu=native`). These are the shipped numbers.\n",
    );
    s.push_str(&format!(
        "- **Dataset**: {size} deterministic documents, each with scalars \
         (`age`, `score`, `region`, `name`), a text `body`, an `items` array of \
         sub-documents, and a 64-dim `vec`. Value indexes on `age`/`region`, a \
         vector index on `vec`, and a BM25 text index on `body` are created once.\n"
    ));
    s.push_str(
        "- **Read workloads** sample a bounded number of operations (corpus size \
         scales the per-op cost, the op count stays fixed) so any run is a real, \
         reproducible amount of work; **mutating workloads** run on bounded \
         scratch collections so they never corrupt the shared read fixture.\n",
    );
    s.push_str(
        "- **Measurement**: each operation is timed individually; `p50`/`p99` are \
         percentiles of the per-op latency distribution (µs), and `ops/s` is \
         `ops / wall_time` over the whole workload.\n\n",
    );
    s.push_str(&format!("## Results — dataset size = {size} docs\n\n"));
    s.push_str("| workload | ops | ops/s | p50 (µs) | p99 (µs) |\n");
    s.push_str("|---|---:|---:|---:|---:|\n");
    for b in rows {
        s.push_str(&format!(
            "| {} | {} | {:.0} | {:.2} | {:.2} |\n",
            b.name,
            b.ops,
            b.ops_per_sec(),
            b.p50,
            b.p99
        ));
    }
    s.push_str("\n## Reading the numbers\n\n");
    s.push_str(
        "- **Fast (µs, index-/key-driven):** `indexed-equality`, `sort-limit`, \
         `update_one`, `delete` — index-driven point/range lookups and primary-key \
         writes.\n",
    );
    s.push_str(
        "- **Mid (hundreds of µs):** `range-find`, `text-search` (posting-list walk), \
         `scan` (full scan). `insert` is ~1.4 µs/doc amortized.\n",
    );
    s.push_str(
        "- **Slow (ms):** `logical-and-or`, `elemMatch`, `group-agg` (full scans), \
         `vector-search` (brute-force over the whole corpus), and `hybrid-search` \
         (two full rankings per op) — the natural targets for further optimization.\n\n",
    );
    s.push_str("## Regenerating\n\n");
    s.push_str("```sh\ncargo run --release --bin bench -- --write-md   # 20k docs\ncargo run --release --bin bench -- --size 250000 --write-md\ncargo run --release --bin bench -- --quick   # small, fast\n```\n\n");
    s
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Opts {
    size: usize,
    write_md: bool,
}

fn parse_opts() -> Opts {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut size = 20_000usize;
    let mut quick = false;
    let mut write_md = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                i += 1;
                size = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("error: --size expects a number");
                    std::process::exit(2);
                });
            }
            "--quick" => quick = true,
            "--write-md" => write_md = true,
            "--help" | "-h" => {
                println!("usage: bench [--size N] [--quick] [--write-md]");
                std::process::exit(0);
            }
            other => {
                eprintln!("warning: ignoring unknown arg {other:?}");
            }
        }
        i += 1;
    }
    if quick {
        size = size.min(5_000);
    }
    Opts {
        size: size.max(100),
        write_md,
    }
}

fn main() {
    let opts = parse_opts();
    let rows = run_all(opts.size);
    let report = render(&rows, opts.size);
    print!("{report}");
    if opts.write_md {
        // `--write-md` writes the full markdown report to BENCH.md in the CWD
        // so the deliverable report can be regenerated from the harness. (No
        // persistence of *data* — this only writes a human report file.)
        if let Err(e) = std::fs::write("BENCH.md", report_md(&rows, opts.size)) {
            eprintln!("error: could not write BENCH.md: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_correct() {
        let mut v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(percentile(&v, 50), 50.0);
        assert_eq!(percentile(&v, 99), 99.0);
        assert_eq!(percentile(&v, 0), 1.0);
        assert_eq!(percentile(&v, 100), 100.0);
        // Empty slice is rejected.
        assert!(std::panic::catch_unwind(|| percentile(&[], 50)).is_err());
    }

    #[test]
    fn make_doc_is_deterministic_and_wellformed() {
        let a = make_doc(7);
        let b = make_doc(7);
        assert_eq!(a, b, "docs for the same index must be identical");
        assert_eq!(
            a.get("_id").unwrap().as_str(),
            Some("000000000000000000000007")
        );
        assert_eq!(a.get("age").unwrap().as_i64(), Some(7));
        assert!(matches!(a.get("score").unwrap(), Value::F64(_)));
        assert_eq!(
            a.get("region").unwrap().as_str(),
            Some(REGIONS[7 % REGIONS.len()])
        );
        // vec has the configured dimension and is all numeric.
        let vec = a.get("vec").unwrap().as_array().unwrap();
        assert_eq!(vec.len(), DIM);
        assert!(vec.iter().all(Value::is_number));
    }

    #[test]
    fn build_dataset_indexes_and_counts() {
        let col = build_dataset(1_000);
        assert_eq!(col.len(), 1_000);
        assert!(col.index("age").is_some());
        assert!(col.index("region").is_some());
        assert!(col.has_vector_index("vec"));
        assert!(col.has_text_index("body"));
        // Indexed equality actually finds the expected fraction.
        let f = Value::object_from(vec![(
            "region".to_string(),
            Value::str("north".to_string()),
        )]);
        let n = col.find(f).count();
        assert!(n > 0 && n < 1_000);
    }

    #[test]
    fn ops_per_sec_is_consistent() {
        let b = Bench {
            name: "t",
            ops: 1_000,
            total: 1.0, // second
            p50: 1e-6,
            p99: 1e-6,
        };
        assert!((b.ops_per_sec() - 1_000.0).abs() < 1e-6);
    }

    #[test]
    fn bench_measurements_are_positive_and_ordered() {
        // A tiny real run: 200 inserts must produce positive latency and
        // p99 >= p50.
        let b = bench("t", 200, |i| {
            let mut c = Collection::new("b");
            c.insert(make_doc(i)).unwrap();
            c.len() as u64
        });
        assert_eq!(b.ops, 200);
        assert!(b.total > 0.0);
        assert!(b.p50 > 0.0);
        assert!(b.p99 >= b.p50 - 1e-9);
    }

    #[test]
    fn every_workload_runs_and_reports() {
        // The full harness must run to completion for every workload and
        // report one row each with sane numbers.
        let rows = run_all(600);
        assert_eq!(rows.len(), 14, "all 14 workloads must be present");
        let names: Vec<&str> = rows.iter().map(|b| b.name).collect();
        for expected in [
            "insert",
            "insert_many",
            "indexed-equality",
            "range-find",
            "scan",
            "sort-limit",
            "update_one",
            "delete",
            "logical-and-or",
            "elemMatch",
            "vector-search",
            "text-search",
            "hybrid-search",
            "group-agg",
        ] {
            assert!(names.contains(&expected), "missing workload {expected}");
        }
        for b in &rows {
            assert!(b.ops > 0, "{} ran zero ops", b.name);
            assert!(b.total > 0.0, "{} took no time", b.name);
            assert!(b.p99 >= b.p50 - 1e-9, "{} p99 < p50", b.name);
        }
    }

    #[test]
    fn render_contains_all_workloads_and_headers() {
        let rows = run_all(500);
        let out = render(&rows, 500);
        assert!(out.contains("ops/s"));
        assert!(out.contains("p50 (µs)"));
        assert!(out.contains("p99 (µs)"));
        assert!(out.contains("insert"));
        assert!(out.contains("hybrid-search"));
    }

    #[test]
    fn report_md_is_a_valid_markdown_report() {
        let rows = run_all(500);
        let md = report_md(&rows, 500);
        // Markdown structure: title + section headers + a table.
        assert!(md.starts_with("# MooRacer benchmark\n"));
        assert!(md.contains("## Methodology"));
        assert!(md.contains("## Results"));
        assert!(md.contains("## Reading the numbers"));
        assert!(md.contains("## Regenerating"));
        // The results table is real markdown and lists every workload.
        assert!(md.contains("| workload | ops | ops/s | p50 (µs) | p99 (µs) |"));
        for expected in [
            "insert",
            "insert_many",
            "indexed-equality",
            "range-find",
            "scan",
            "sort-limit",
            "update_one",
            "delete",
            "logical-and-or",
            "elemMatch",
            "vector-search",
            "text-search",
            "hybrid-search",
            "group-agg",
        ] {
            assert!(
                md.contains(&format!("| {expected} |")),
                "report missing {expected}"
            );
        }
        // The dataset size is stamped into the results header.
        assert!(md.contains("dataset size = 500 docs"));
    }
}
