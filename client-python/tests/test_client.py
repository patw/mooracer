"""End-to-end tests for the Python client over a real TCP devserver.

Each test starts a FRESH server (isolated in-memory store), so tests never
share state. Index-backed workloads pass the index config to the server
(`server(vector=..., text=...)`); docs are seeded over the wire and the
indexes are maintained on insert by the engine.
"""

import pytest

from mooracer import Client, ServerError

COWS = [
    {"_id": "bess", "age": 3, "region": "north", "emb": [1.0, 0.0],
     "text": "brown cow by the barn"},
    {"_id": "daisy", "age": 9, "region": "south", "emb": [0.0, 1.0],
     "text": "the white daisy likes hay"},
    {"_id": "moo", "age": 9, "region": "north", "emb": [1.0, 1.0],
     "text": "moo moo brown field"},
    {"_id": "pete", "age": 5, "region": "south", "emb": [0.0, 0.0],
     "text": "quiet pete in the shade"},
]


def connect(server, name="cows", *, vector=None, text=None, docs=COWS):
    addr = server(vector=vector, text=text)
    c = Client.connect(addr)
    herd = c.collection(name)
    if docs:
        herd.insert_many(docs)
    return c, herd


def ids(docs):
    return [d["_id"] for d in docs]


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def test_insert_explicit_and_auto_id(server):
    c = Client.connect(server())
    herd = c.collection("c")
    assert herd.insert({"_id": "a", "age": 3}) == "a"
    auto = herd.insert({"age": 5})
    assert len(auto) == 24
    assert all(ch in "0123456789abcdef" for ch in auto)


def test_insert_many_returns_ids_in_order(server):
    c, herd = connect(server, docs=None)
    got = herd.insert_many([
        {"_id": "a", "n": 1}, {"_id": "b", "n": 2}, {"_id": "c", "n": 3},
    ])
    assert got == ["a", "b", "c"]
    assert herd.count({}) == 3


def test_duplicate_id_is_typed_error(server):
    c, herd = connect(server, docs=[{"_id": "a"}, {"_id": "b"}])
    with pytest.raises(ServerError) as ei:
        herd.insert({"_id": "a"})
    assert ei.value.status == 3  # Status.DuplicateId
    assert ei.value.name == "DuplicateId"


def test_replace_on_empty_collection_is_typed_no_match(server):
    c = Client.connect(server())
    herd = c.collection("c")
    with pytest.raises(ServerError) as ei:
        herd.replace_one({"x": 1}, {"x": [1, 2]})
    assert ei.value.status == 7  # Status.NoMatch
    assert ei.value.name == "NoMatch"


# ---------------------------------------------------------------------------
# Find / query pipeline
# ---------------------------------------------------------------------------

def test_find_all_and_equality(server):
    c, herd = connect(server)
    assert len(herd.find({}).to_list()) == 4
    got = herd.find({"_id": "daisy"}).to_list()
    assert ids(got) == ["daisy"]
    assert got[0]["age"] == 9
    assert got[0]["emb"] == [0.0, 1.0]  # float elements stay floats


def test_find_operators(server):
    c, herd = connect(server)
    # Unsorted finds return storage order (a hash table), so compare sets
    # unless a sort is applied (the _id point lookup below is index-driven).
    assert set(ids(herd.find({"age": {"$gte": 9}}).to_list())) == {"daisy", "moo"}
    assert set(ids(herd.find({"region": "north"}).to_list())) == {"bess", "moo"}
    assert set(ids(herd.find({"region": {"$in": ["south", "north"]}}).to_list())) \
        == {"bess", "daisy", "moo", "pete"}
    assert set(ids(herd.find({"age": {"$ne": 9}}).to_list())) == {"bess", "pete"}
    assert ids(herd.find({"_id": {"$in": ["moo", "bess"]}}).sort("_id").to_list()) \
        == ["bess", "moo"]


def test_sort_limit_skip(server):
    c, herd = connect(server)
    assert ids(herd.find({}).sort("age", False).to_list()) == ["bess", "pete", "daisy", "moo"]
    # Equal-value ties reverse in desc (9: daisy, moo -> moo, daisy by _id desc).
    assert ids(herd.find({}).sort("age", True).to_list()) == ["moo", "daisy", "pete", "bess"]
    assert ids(herd.find({}).sort("age", False).skip(1).limit(2).to_list()) == ["pete", "daisy"]
    assert ids(herd.find({}).sort("age", True).limit(1).to_list()) == ["moo"]
    # limit(0) = no limit.
    assert len(herd.find({}).sort("age").limit(0).to_list()) == 4


def test_find_one_first_count_exists(server):
    c, herd = connect(server)
    doc = herd.find_one({"region": "south", "age": 5})
    assert doc["_id"] == "pete"
    assert herd.find_one({"age": 99}) is None
    assert herd.find({"age": 5}).first()["_id"] == "pete"
    assert herd.find({"age": 5}).find_one()["_id"] == "pete"
    assert herd.find({"age": 9}).first()["_id"] in ("daisy", "moo")
    assert herd.count({"region": "north"}) == 2
    assert herd.count({}) == 4
    assert herd.exists({"_id": "bess"}) is True
    assert herd.exists({"_id": "nope"}) is False
    assert herd.find({"age": {"$lt": 3}}).count() == 0


# ---------------------------------------------------------------------------
# Update / replace / delete
# ---------------------------------------------------------------------------

def test_update_set_inc_unset_refresh_index(server):
    c, herd = connect(server)
    assert herd.update_one({"_id": "bess"}, {"$set": {"age": 4}}) == 1
    assert herd.count({"age": 4}) == 1          # index refreshed
    assert herd.count({"age": 3}) == 0
    assert herd.update_many({"region": "north"}, {"$inc": {"age": 1}}) == 2
    # bess: 3 ->($set) 4 ->($inc) 5 ; moo: 9 -> 10.
    assert sorted(d["age"] for d in herd.find({"region": "north"}).to_list()) == [5, 10]
    assert herd.update_one({"_id": "pete"}, {"$unset": ["region"]}) == 1
    assert herd.count({"region": {"$exists": False}}) == 1


def test_update_one_no_match_is_typed_error(server):
    c, herd = connect(server)
    with pytest.raises(ServerError) as ei:
        herd.update_one({"_id": "ghost"}, {"$set": {"age": 1}})
    assert ei.value.status == 7  # Status.NoMatch
    assert ei.value.name == "NoMatch"
    # no-match on update_many is a valid zero, not an error
    assert herd.update_many({"_id": "ghost"}, {"$set": {"age": 1}}) == 0


def test_replace_one_preserves_id_and_is_wholesale(server):
    c, herd = connect(server)
    assert herd.replace_one({"_id": "moo"}, {"age": 1, "region": "east"}) == 1
    doc = herd.find_one({"_id": "moo"})
    assert doc == {"_id": "moo", "age": 1, "region": "east"}  # emb/text dropped
    assert herd.count({"region": "east"}) == 1                # index refreshed
    with pytest.raises(ServerError) as ei:
        herd.replace_one({"_id": "ghost"}, {"age": 1})
    assert ei.value.status == 7


def test_delete_one_and_many(server):
    c, herd = connect(server)
    assert herd.delete_one({"_id": "bess"}) is True
    assert herd.delete_one({"_id": "bess"}) is False  # already gone
    assert herd.count({}) == 3
    assert herd.delete_many({"age": 9}) == 2
    assert herd.count({}) == 1
    assert herd.delete_many({}) == 1
    assert herd.count({}) == 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_vector_search_ranking_and_scores(server):
    c, herd = connect(server, vector=["cows:emb:2"])
    hits = herd.vector_search("emb", [1.0, 0.0], 0)
    assert len(hits) == 4
    assert hits[0][0]["_id"] == "bess"
    assert abs(hits[0][1] - 1.0) < 1e-6          # exact alignment
    by_id = {d["_id"]: s for d, s in hits}
    # (0,1) orthogonal -> 0 ; (1,1) 45deg -> ~0.707 ; zero vec -> 0.
    assert abs(by_id["daisy"] - 0.0) < 1e-6
    assert abs(by_id["moo"] - 0.70710678) < 1e-3
    assert abs(by_id["pete"] - 0.0) < 1e-6
    assert all(0.0 <= s <= 1.0 + 1e-9 for _, s in hits)


def test_vector_search_limit_and_no_index(server):
    c, herd = connect(server, vector=["cows:emb:2"])
    assert len(herd.vector_search("emb", [1.0, 0.0], 2)) == 2
    c2, h2 = connect(server, docs=None)
    with pytest.raises(ServerError) as ei:
        h2.vector_search("emb", [1.0, 0.0], 2)
    assert ei.value.status == 5  # Status.NoIndex


def test_text_search_bm25(server):
    c, herd = connect(server, text=["cows:text"])
    hits = herd.text_search("text", "brown", 0)
    assert len(hits) == 2                    # bess + moo contain "brown"
    assert hits[0][0]["_id"] in ("bess", "moo")
    assert all(s > 0.0 for _, s in hits)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(herd.text_search("text", "brown", 1)) == 1
    assert herd.text_search("text", "zzzz", 0) == []  # rare term, no hits


def test_text_search_no_index(server):
    c, herd = connect(server, docs=None)
    with pytest.raises(ServerError) as ei:
        herd.text_search("text", "moo", 0)
    assert ei.value.status == 5


def test_hybrid_search_rrf(server):
    c, herd = connect(server, vector=["cows:emb:2"], text=["cows:text"])
    hits = herd.hybrid_search("text", "emb", "brown", [1.0, 0.0], 0)
    assert len(hits) == 4                    # union over both signals
    # bess: rank 1 in vector (cos 1.0) and rank 1/2 in text -> best fused.
    assert hits[0][0]["_id"] == "bess"
    assert all(s > 0.0 for _, s in hits)
    # missing one index -> NoIndex
    c2, h2 = connect(server, vector=["cows:emb:2"], docs=None)
    with pytest.raises(ServerError) as ei:
        h2.hybrid_search("text", "emb", "brown", [1.0, 0.0], 0)
    assert ei.value.status == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_group_count_and_sum(server):
    c, herd = connect(server)
    groups = herd.find({}).group("region").agg("count")
    assert {g["_id"]: g["count"] for g in groups} == {"north": 2, "south": 2}

    groups = herd.find({}).group("region").agg("sum", "age")
    assert {g["_id"]: g["sum"] for g in groups} == {"north": 12, "south": 14}


def test_group_mean_min_max_first_last(server):
    c, herd = connect(server)
    g = {x["_id"]: x for x in herd.find({}).group("region").agg("mean", "age")}
    assert abs(g["north"]["mean"] - 6.0) < 1e-9
    assert abs(g["south"]["mean"] - 7.0) < 1e-9
    g = {x["_id"]: x for x in herd.find({}).group("region").agg("min", "age")}
    assert g["north"]["min"] == 3
    g = {x["_id"]: x for x in herd.find({}).group("region").agg("max", "age")}
    assert g["south"]["max"] == 9
    # first/last follow the query stream: sort by _id within region.
    g = {x["_id"]: x
         for x in herd.find({"region": "north"}).sort("_id").group("region")
         .agg("first", "_id")}
    assert g["north"]["first"] == "bess"
    g = {x["_id"]: x
         for x in herd.find({"region": "north"}).sort("_id").group("region")
         .agg("last", "_id")}
    assert g["north"]["last"] == "moo"


def test_group_sort_limit_and_filter(server):
    c, herd = connect(server)
    groups = herd.find({}).group("region").sort("count", True).limit(1).agg("count")
    assert len(groups) == 1
    groups = herd.find({"age": {"$gte": 5}}).group("region").agg("count")
    assert {g["_id"]: g["count"] for g in groups} == {"north": 1, "south": 2}
    # AggFn given as the enum number works too (0 = Count).
    groups = herd.find({}).group("region").agg(0)
    assert {g["_id"]: g["count"] for g in groups} == {"north": 2, "south": 2}


# ---------------------------------------------------------------------------
# Stats / protocol
# ---------------------------------------------------------------------------

def test_stats_shape(server):
    c, herd = connect(server)
    s = herd.stats()
    assert s["docs"] == 4
    assert s["indexes"] == 1                       # only the primary `_id`
    assert s["total_memory"] == s["docs_memory"] + sum(
        i["memory"] for i in s["per_index"])
    assert [i["field"] for i in s["per_index"]] == ["_id"]
    assert s["per_index"][0]["entries"] == 4
    assert s["per_index"][0]["distinct"] == 4


def test_missing_collection_read_is_empty(server):
    c = Client.connect(server())
    ghost = c.collection("ghost")
    assert ghost.find({}).to_list() == []
    assert ghost.count({}) == 0
    assert ghost.exists({}) is False
    assert ghost.stats()["docs"] == 0


def test_req_id_echo_and_reuse(server):
    c, herd = connect(server, docs=None)
    herd.insert({"_id": "a"})
    assert c.last_req_id == 1
    herd.insert({"_id": "b"})
    assert c.last_req_id == 2
    herd.count({})
    assert c.last_req_id == 3


def test_nested_docs_roundtrip(server):
    c, herd = connect(server, docs=None)
    doc = {"_id": "n", "meta": {"tags": ["x", "y"], "score": 1.5},
           "path": {"a": [1, None, {"b": True}]}}
    herd.insert(doc)
    got = herd.find_one({"_id": "n"})
    assert got == doc
    assert list(got["meta"].keys()) == ["tags", "score"]
