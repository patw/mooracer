# MooRacer Roadmap

Future work, sharp edges, and ideas, roughly prioritized. This is a living
document for thinking about where MooRacer goes next; it complements
`spec.md` (the behavior/API contract) and `CHANGELOG.md` (what shipped). Items
are marked **[planned]**, **[in progress]**, or **[idea]**.

> Guiding principle: *perf is already there* ("it's fast enough"), so this list
> is about **usability, operator experience, and removing sharp edges** — not
> squeezing more throughput.

---

## [done] Wire-level index management

Indexes were **server-side only**: the wire v1 protocol had no index-management
command, so a client could not enable vector / text / hybrid search at runtime
without reaching into the store. The only pre-existing way to use search was
the `mooracer-devserver` env-var hack
(`MOORACER_VECTOR_INDEX=coll:field:dim`, `MOORACER_TEXT_INDEX=coll:field`),
which was awkward and Python-test-only.

**Status:** implemented and tested (not yet committed/dev-released). `IndexCmd`
(create/drop value, vector, text) + `IndexRes` were added as **additive**
union variants (existing on-wire discriminants unchanged, so it is
backward-compatible). The new methods are exposed in both the Rust and Python
clients; the Python `complex_usage` example creates its own indexes. Covered by
server TCP, Rust client, Python, and wire contract tests.

**Justification:** search is a headline feature but it was unusable from a real
client. Now a client can enable search at runtime — not an MVP without it.

**Follow-ups to consider:** changing the envelope `version` and, once the Python
client is pip-installable (below), converting the test suite off the dev-server
index seeding in favor of the wire command.

---

## [planned] Snapshot / resume on clean shutdown

Serialize the in-memory store on a clean shutdown signal and optionally load it
back on startup, so scheduled maintenance runs can resume where they left off.

> ⚠️ **Philosophy tension:** `spec.md` explicitly lists **"no persistence, no
> disk I/O, no durability, no recovery"** as non-goals. This is arguably a form
> of all four. It is a deliberate steer, not an accident — so if we build it, we
> must frame it as an **opt-in snapshot for resumed scheduled work**, clearly
> documented as *not* a durability/consistency guarantee, not a silent
> "MooRacer persists now."

**Justification:** lets long-running maintenance/batch jobs checkpoint and
resume without a live process, at low cost. Biggest gotchas: capturing a
consistent point under the `RwLock` while writes are in flight, serializing the
full value tree **and** the vector/text indexes (vs. rebuilding them on load),
atomic write (temp + rename), and versioning the snapshot format.

---

## [done] Pip-installable Python client + packaging

`client-python/` was a `PYTHONPATH=client-python` folder with only a
`requirements.txt` and a hand-checked-in generated `wire/` subpackage.

**Status:** implemented and tested (not yet committed/dev-released). Added a
`pyproject.toml` (setuptools), a package `README`, and a `LICENSE`; the
generated `wire/` subpackage ships as package-data and the `flatbuffers` runtime
dependency is declared. `pip install ./client-python` (and `pip install -e
./client-python`) now work; a `py3-none-any` wheel + sdist build cleanly and the
installed wheel was smoke-tested against a live server.

**Follow-ups to consider:** the `!Sync` client (one request in flight) is
documented as one-client-per-thread; a small thread-pool helper remains an
option. The test suite still seeds indexes via the dev server env vars — once
this packaging lands, consider converting it to the wire `IndexCmd`.

---

## [done] CI + release scaffolding

No CI on a public repo was itself a sharp edge.

**Status:** implemented (not yet committed). `.github/workflows/ci.yml` runs
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`,
a release build, and the Python `pytest` suite (spinning up the devserver).
`.github/workflows/release.yml` builds the pure-Python sdist/wheel on a `v*`
tag, verifies it's a real `py3-none-any` artifact, publishes to PyPI via trusted
publishing (OIDC), and cuts a GitHub Release. The Rust codebase was made
fmt- and clippy-clean so the gates are honest.

**Follow-ups to consider:** crates.io publishing for the 5 Rust crates (per-crate
tokens + publish order) and a `rustfmt.toml`/clippy config are not wired yet.

---

## [planned] More Python examples

Beyond `simple_crud`/`complex_usage`: a real-ish **app** example (contacts
manager), a **search + aggregation** showcase, an **error-handling / index**
example, and a **concurrency** example (one client per thread) that doubles as
the `!Sync` documentation.

**Justification:** examples are the fastest path to understanding an API; the
few current ones leave search, aggregation, and concurrency under-explored.
Cheap and directly answers "what can I do with this?"

---

## [idea] Network-level benchmark harness

The current `bench` is **in-process**, so the published numbers are not what you
actually get over the wire. A Redis-`redis-benchmark`-style *wire* benchmark
would give credible, defensible numbers for the "racer" claim.

**Justification:** the project is fast enough, but the published numbers are
misleading without an on-the-wire measurement. This is about proof/credibility,
not performance.

---

## [idea] Server ops: logging, config, and collection management

- Replace env-var-only server config with a small config file / flags.
- Structured, optional logging.
- Add `list_collections` / `drop_collection` (today a collection is only ever
  created implicitly on first write — you can't see or remove them over the
  wire).

**Justification:** makes MooRacer operable rather than just runnable.

---

## Not in scope (by design)

- Durability / persistence / recovery as a *guarantee* (see the snapshot note).
- Auto-embedding / ONNX / semantic search — caller supplies vectors.
- Cross-process locking.
