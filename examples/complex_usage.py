"""Complex MooRacer usage — text, vector and hybrid search + aggregation.

Demonstrates the search-and-aggregation surface over a small product corpus:
    * `text_search`    — BM25 + Porter stemming
    * `vector_search`  — brute-force cosine (`embedding`)
    * `hybrid_search`  — Reciprocal Rank Fusion of the two
    * group-by aggregation over the same query chain

Indexes are created **over the wire** with the client (`create_vector_index`,
`create_text_index`), so this runs against the plain server:

    MOORACER_ADDR=127.0.0.1:4141 cargo run --release -p mooracer-server

Then, in another terminal:
    MOORACER_ADDR=127.0.0.1:4141 python3 examples/complex_usage.py
"""

import os

from mooracer import Client

ADDR = os.environ.get("MOORACER_ADDR", "127.0.0.1:4141")
DIM = 8


def v(*xs: float) -> list:
    """A DIM-length embedding vector (the vector index was built at DIM=8)."""
    return list(xs)


# Small, deliberately separable corpus. Descriptions drive BM25;
# `embedding` is a crude 8-d flavour vector for cosine search.
PRODUCTS = [
    {"name": "Chocolate Stout",  "category": "beer",   "price": 9.0,
     "description": "rich chocolate stout with coffee grounds and dark cacao",
     "embedding": v(0.9, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Pale Ale",         "category": "beer",   "price": 7.5,
     "description": "crisp pale ale with grassy hops and citrus peel",
     "embedding": v(0.1, 0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Rosé",             "category": "wine",   "price": 14.0,
     "description": "light rosé with strawberry and stone fruit notes",
     "embedding": v(0.1, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Cabernet",         "category": "wine",   "price": 18.0,
     "description": "bold cabernet with blackberry and oak tannins",
     "embedding": v(0.1, 0.1, 0.3, 0.9, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Cold Brew",        "category": "coffee", "price": 5.0,
     "description": "smooth cold brew coffee with chocolate finish",
     "embedding": v(0.8, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Espresso",         "category": "coffee", "price": 4.5,
     "description": "intense espresso with chocolate and caramel notes",
     "embedding": v(0.7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)},
    {"name": "Dry Cider",        "category": "cider",  "price": 6.0,
     "description": "dry apple cider with crisp tart finish",
     "embedding": v(0.1, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0)},
]


def main() -> None:
    print(f"=== MooRacer — search & aggregation @ {ADDR} ===\n")
    c = Client.connect(ADDR)
    try:
        col = c.collection("products")
        col.delete_many({})
        ids = col.insert_many(PRODUCTS)
        print(f"inserted {len(ids)} products")

        # Create the indexes over the wire (backfills from existing docs).
        # Value index -> range/equality fast paths; vector -> cosine search;
        # text -> BM25. The `_id` index is implicit.
        col.create_index("category")
        col.create_vector_index("embedding", DIM)
        col.create_text_index("description")
        print("created value index on 'category', vector index on 'embedding', "
              f"and text index on 'description' (dim={DIM})")

        # ------------------------------------------------------------------
        # TEXT SEARCH — BM25 + Porter stemming
        # ------------------------------------------------------------------
        print("\n--- text_search('description', 'coffee') ---")
        for doc, score in col.text_search("description", "coffee", limit=3):
            print(f"  {score:6.3f}  {doc['name']}")

        # ------------------------------------------------------------------
        # VECTOR SEARCH — brute-force cosine (a "coffee" flavour vector)
        # ------------------------------------------------------------------
        print("\n--- vector_search('embedding', coffee-vector, limit=3) ---")
        coffee_vec = v(0.9, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)
        for doc, score in col.vector_search("embedding", coffee_vec, limit=3):
            print(f"  {score:6.3f}  {doc['name']}")

        # ------------------------------------------------------------------
        # HYBRID SEARCH — RRF fusion of BM25 + cosine
        # ------------------------------------------------------------------
        print("\n--- hybrid_search(text='coffee chocolate', vec=coffee-vector) ---")
        for doc, score in col.hybrid_search(
            "description", "embedding", "coffee chocolate", coffee_vec, limit=5
        ):
            print(f"  {score:6.3f}  {doc['name']}")

        # ------------------------------------------------------------------
        # AGGREGATION — group by category
        # ------------------------------------------------------------------
        print("\n--- group by category: count / mean(price) ---")
        counts = col.find({}).group("category").agg("count", None)
        print("  counts:", {g["_id"]: g["count"] for g in counts})
        means = col.find({}).group("category").agg("mean", "price")
        print("  mean price:", {g["_id"]: round(g["mean"], 2) for g in means})

        # A filtered aggregation: count only coffee items
        coffee_counts = col.find({"category": "coffee"}).group("category").agg("count", None)
        print("  coffee count:", coffee_counts[0]["count"])

        # ------------------------------------------------------------------
        # STATS
        # ------------------------------------------------------------------
        s = col.stats()
        print(f"\nstats: docs={s['docs']}, indexes={s['indexes']}, "
              f"total_memory={s['total_memory']} bytes")

        col.delete_many({})
    finally:
        c.close()


if __name__ == "__main__":
    main()
