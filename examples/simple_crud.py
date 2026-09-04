"""Simple CRUD with MooRacer — insert, find, update, delete, count, exists.

Run against a running server (see examples/README.md):
    MOORACER_ADDR=127.0.0.1:4141 python3 examples/simple_crud.py
"""

import os

from mooracer import Client, ServerError

ADDR = os.environ.get("MOORACER_ADDR", "127.0.0.1:4141")


def main() -> None:
    print(f"=== MooRacer — simple CRUD @ {ADDR} ===\n")
    c = Client.connect(ADDR)
    try:
        col = c.collection("contacts")
        col.delete_many({})  # clean any previous run

        # ------------------------------------------------------------------
        # INSERT
        # ------------------------------------------------------------------
        alice_id = col.insert(
            {"name": "Alice", "email": "alice@example.com", "age": 30, "status": "active"}
        )
        print(f"inserted Alice -> _id={alice_id}")

        ids = col.insert_many(
            [
                {"name": "Bob",   "email": "bob@example.com",   "age": 25, "status": "trial"},
                {"name": "Carol", "email": "carol@example.com", "age": 40, "status": "active"},
            ]
        )
        print(f"insert_many -> {len(ids)} ids")
        print(f"total docs: {col.count({})}")

        # ------------------------------------------------------------------
        # FIND
        # ------------------------------------------------------------------
        active = col.find({"status": "active"}).to_list()
        print(f"active: {[d['name'] for d in active]}")

        young = col.find({"age": {"$lt": 30}}).sort("age").to_list()
        print(f"under 30 (sorted by age): {[(d['name'], d['age']) for d in young]}")

        one = col.find_one({"email": "alice@example.com"})
        print(f"find_one: {one['name']}, age={one['age']}")
        print(f"exists? alice={col.exists({'email': 'alice@example.com'})} "
              f"zed={col.exists({'email': 'zed@example.com'})}")

        # ------------------------------------------------------------------
        # UPDATE
        # ------------------------------------------------------------------
        col.update_one({"email": "alice@example.com"}, {"$set": {"age": 31}})
        col.update_one({"email": "bob@example.com"}, {"$inc": {"logins": 1}})
        print(f"alice age now: {col.find_one({'email': 'alice@example.com'})['age']}")

        promoted = col.update_many({"status": "trial"}, {"$set": {"status": "active"}})
        print(f"promoted {promoted} trial user(s)")

        col.replace_one(
            {"name": "Carol"},
            {"name": "Carol H.", "email": "carol@example.com", "age": 41, "status": "active"},
        )
        print(f"after replace: {col.find_one({'email': 'carol@example.com'})}")

        # ------------------------------------------------------------------
        # DELETE
        # ------------------------------------------------------------------
        col.insert({"_id": "ghost", "name": "Temp"})
        print(f"deleted ghost: {col.delete_one({'_id': 'ghost'})}")
        print(f"deleted {col.delete_many({'status': 'active'})} active")
        print(f"remaining: {col.count({})}")

        # ------------------------------------------------------------------
        # ERROR HANDLING (typed errors with a `.name`)
        # ------------------------------------------------------------------
        print("\n--- error handling ---")
        col.insert({"_id": "fixed", "name": "Fixed"})
        try:
            col.insert({"_id": "fixed", "name": "Duplicate"})
        except ServerError as e:
            print(f"caught {e.name}: {e}")

        try:
            col.update_one({"email": "nobody@example.com"}, {"$set": {"age": 99}})
        except ServerError as e:
            print(f"caught {e.name}: {e}")

        # ------------------------------------------------------------------
        # STATS
        # ------------------------------------------------------------------
        s = col.stats()
        print(f"stats: docs={s['docs']}, indexes={s['indexes']}, "
              f"total_memory={s['total_memory']} bytes")

        col.delete_many({})
    finally:
        c.close()


if __name__ == "__main__":
    main()
