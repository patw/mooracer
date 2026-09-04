# MooRacer examples

Runnable end-to-end examples for the **Python client**. Each connects to a
running MooRacer server, so you'll need to start one first (see below).

- [`simple_crud.py`](simple_crud.py) — insert, find, update, delete, count,
  exists, error handling, stats. The "hello world" for the query chain.
- [`complex_usage.py`](complex_usage.py) — vector search, BM25 text search,
  hybrid (RRF) search, group-by aggregation, and stats over a product corpus.

## Prerequisites

1. **Build the server.**
   ```sh
   cargo build --release -p mooracer-server
   ```
   (`flatc` must be on `PATH` — see the README.)

2. **Install the Python client.**
   ```sh
   pip install -e client-python   # or: PYTHONPATH=client-python
   ```

## Running the examples

### simple_crud.py — a plain server, no indexes needed

```sh
# terminal 1
MOORACER_ADDR=127.0.0.1:4141 cargo run --release -p mooracer-server

# terminal 2
MOORACER_ADDR=127.0.0.1:4141 python3 examples/simple_crud.py
```

### complex_usage.py — a plain server (indexes created over the wire)

The example creates its own value/vector/text indexes with the client
(`create_index` / `create_vector_index` / `create_text_index`), so it also runs
against the **plain** `mooracer-server` — no index env vars, no dev server:

```sh
# terminal 1
MOORACER_ADDR=127.0.0.1:4141 cargo run --release -p mooracer-server

# terminal 2
MOORACER_ADDR=127.0.0.1:4141 python3 examples/complex_usage.py
```

> The `mooracer-devserver` binary is still used by the *test* suite to
> pre-create indexes (and remains useful for that), but is no longer needed to
> run this example.
