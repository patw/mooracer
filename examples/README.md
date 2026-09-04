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

### complex_usage.py — needs a server with vector/text indexes

The wire v1 protocol has **no index-management command**, so indexes must be
created server-side. Use the *dev* server, which pre-creates them from
environment variables before serving:

```sh
# terminal 1 — pre-create a vector index on `embedding` (dim 8) and a
# text index on `description` for the `products` collection
MOORACER_ADDR=127.0.0.1:4141 \
MOORACER_VECTOR_INDEX=products:embedding:8 \
MOORACER_TEXT_INDEX=products:description \
    cargo run --release -p mooracer-server --bin mooracer-devserver

# terminal 2
MOORACER_ADDR=127.0.0.1:4141 python3 examples/complex_usage.py
```
