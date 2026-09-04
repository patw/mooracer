# mooracer — Python client

The pure-Python network client for
[MooRacer](https://github.com/patw/mooracer), the in-memory document data engine.
No Rust FFI — it speaks the same FlatBuffers-over-TCP wire protocol as any MooRacer
server, exposes a Mongo-style chain API, and returns native Python `dict`/`list`/
`str`/`int`/`float`/`bool`/`None`.

```python
from mooracer import Client

c = Client.connect("127.0.0.1:4141")       # "host:port"
col = c.collection("cows")

col.insert({"name": "Bella", "age": 5})     # -> _id (str)
col.find({"age": {"$gte": 4}}).sort("age").limit(10).to_list()
col.find_one({})                            # dict | None
col.count({})                               # int
col.exists({})                              # bool
col.update_one({"_id": "…"}, {"$set": {"region": "north"}, "$inc": {"age": 1}})
col.delete_one({"_id": "…"})                # bool

# Indexes are managed over the wire:
col.create_index("region")
col.create_vector_index("emb", 64)
col.create_text_index("body")

# Search + aggregation:
col.vector_search("emb", [0.0] * 64, 10)    # [(dict, float), …]
col.text_search("body", "moo", 10)
col.find({"region": "north"}).group("region").agg("count", None)
```

Install:

```sh
pip install mooracer          # from PyPI
# or, from the checkout:
pip install ./client-python
```

Requires `flatbuffers>=23.5.26`. The generated `wire/` subpackage is checked in
(regenerate with `flatc --python -o mooracer/ ../schema/mooracer.fbs`, then move
`mooracer/mooracer/wire` → `mooracer/wire`).

`Client` owns one TCP connection and is strictly request/response (one in-flight
request at a time) — use **one client per thread** for concurrency.

See the repo [`examples/`](../examples) for runnable demos.
