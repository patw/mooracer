"""Pure-Python MooRacer client: FlatBuffers over length-prefixed TCP frames.

Mirrors the Rust client's Mongo-style chain API (see
`client-rust/src/lib.rs` — the spec's "Rust client decisions" block is the
reference for this client):

    from mooracer import Client
    c = Client.connect("127.0.0.1:4141")
    herd = c.collection("cows")
    herd.insert({"name": "daisy", "age": 3})            # -> "_id" (str)
    herd.find({"age": {"$gte": 2}}).sort("age", True).limit(2).to_list()
    herd.find({"region": "x"}).group("region").agg("count")

`Client` owns one TCP connection and is strictly request/response (one
in-flight request at a time) — use one client per thread. Values are plain
Python types: `dict` / `list` / `str` / `int` / `float` / `bool` / `None`.

The wire format (v1) is defined in `schema/mooracer.fbs`; the generated
`mooracer.wire` package is checked in (regenerate with
`flatc --python -o mooracer/ ../schema/mooracer.fbs` and move
`mooracer/mooracer/wire -> mooracer/wire`).
"""

from __future__ import annotations

import socket
import struct

import flatbuffers

# NOTE: each generated module shadows its class name (module `wire/Value.py`
# contains class `Value`), so import the classes from their modules.
from .wire.AggFn import AggFn
from .wire.Command import Command
from .wire.CountCmd import CountCmd
from .wire.CountRes import CountRes
from .wire.DeleteCmd import DeleteCmd
from .wire.DeleteRes import DeleteRes
from .wire.ExistsCmd import ExistsCmd
from .wire.ExistsRes import ExistsRes
from .wire.FindCmd import FindCmd
from .wire.FindRes import FindRes
from .wire.GroupCmd import GroupCmd
from .wire.GroupRes import GroupRes
from .wire.HybridSearchCmd import HybridSearchCmd
from .wire.IndexCmd import IndexCmd
from .wire.IndexKind import IndexKind
from .wire.IndexRes import IndexRes
from .wire.InsertCmd import InsertCmd
from .wire.InsertRes import InsertRes
from .wire.ReplaceCmd import ReplaceCmd
from .wire.ReplaceRes import ReplaceRes
from .wire.Request import Request
from .wire.Response import Response
from .wire.ResponseBody import ResponseBody
from .wire.SearchRes import SearchRes
from .wire.StatsCmd import StatsCmd
from .wire.StatsRes import StatsRes
from .wire.Status import Status
from .wire.TextSearchCmd import TextSearchCmd
from .wire.UpdateCmd import UpdateCmd
from .wire.UpdateRes import UpdateRes
from .wire.Value import Value
from .wire.ValueKind import ValueKind
from .wire.VectorSearchCmd import VectorSearchCmd
from .wire.CountCmd import CountCmdAddFilter, CountCmdEnd, CountCmdStart
from .wire.DeleteCmd import DeleteCmdAddFilter, DeleteCmdAddMany, DeleteCmdEnd, DeleteCmdStart
from .wire.ExistsCmd import ExistsCmdAddFilter, ExistsCmdEnd, ExistsCmdStart
from .wire.FindCmd import (
    FindCmdAddFilter,
    FindCmdAddLimit,
    FindCmdAddOne,
    FindCmdAddSkip,
    FindCmdAddSortDesc,
    FindCmdAddSortField,
    FindCmdEnd,
    FindCmdStart,
)
from .wire.GroupCmd import (
    GroupCmdAddAggField,
    GroupCmdAddAggFn,
    GroupCmdAddFilter,
    GroupCmdAddGroupField,
    GroupCmdAddGroupLimit,
    GroupCmdAddGroupSortDesc,
    GroupCmdAddGroupSortField,
    GroupCmdAddLimit,
    GroupCmdAddSkip,
    GroupCmdAddSortDesc,
    GroupCmdAddSortField,
    GroupCmdEnd,
    GroupCmdStart,
)
from .wire.HybridSearchCmd import (
    HybridSearchCmdAddLimit,
    HybridSearchCmdAddQueryText,
    HybridSearchCmdAddQueryVec,
    HybridSearchCmdAddTextField,
    HybridSearchCmdAddVecField,
    HybridSearchCmdEnd,
    HybridSearchCmdStart,
    HybridSearchCmdStartQueryVecVector,
)
from .wire.IndexCmd import (
    IndexCmdAddDim,
    IndexCmdAddField,
    IndexCmdAddKind,
    IndexCmdEnd,
    IndexCmdStart,
)
from .wire.InsertCmd import (
    InsertCmdAddDocs,
    InsertCmdEnd,
    InsertCmdStart,
    InsertCmdStartDocsVector,
)
from .wire.ReplaceCmd import (
    ReplaceCmdAddFilter,
    ReplaceCmdAddNewDoc,
    ReplaceCmdEnd,
    ReplaceCmdStart,
)
from .wire.Request import (
    RequestAddCollection,
    RequestAddCommand,
    RequestAddCommandType,
    RequestAddReqId,
    RequestAddVersion,
    RequestEnd,
    RequestStart,
)
from .wire.StatsCmd import StatsCmdEnd, StatsCmdStart
from .wire.TextSearchCmd import (
    TextSearchCmdAddField,
    TextSearchCmdAddLimit,
    TextSearchCmdAddQuery,
    TextSearchCmdEnd,
    TextSearchCmdStart,
)
from .wire.UpdateCmd import (
    UpdateCmdAddFilter,
    UpdateCmdAddMany,
    UpdateCmdAddUpdate,
    UpdateCmdEnd,
    UpdateCmdStart,
)
from .wire.VectorSearchCmd import (
    VectorSearchCmdAddField,
    VectorSearchCmdAddLimit,
    VectorSearchCmdAddQuery,
    VectorSearchCmdEnd,
    VectorSearchCmdStart,
    VectorSearchCmdStartQueryVector,
)
from .wire.Value import (
    ValueAddArr,
    ValueAddB,
    ValueAddF,
    ValueAddI,
    ValueAddKeys,
    ValueAddKind,
    ValueAddS,
    ValueAddVals,
    ValueEnd,
    ValueStart,
    ValueStartArrVector,
    ValueStartKeysVector,
    ValueStartValsVector,
)

WIRE_VERSION = 1
MAX_FRAME = 256 * 1024 * 1024  # must match the server cap
FILE_IDENTIFIER = b"MOOR"  # must match the schema's file_identifier
_CHUNK = 1 << 20

# Response body union type -> generated result table.
_BODY = {
    ResponseBody.InsertRes: InsertRes,
    ResponseBody.FindRes: FindRes,
    ResponseBody.CountRes: CountRes,
    ResponseBody.ExistsRes: ExistsRes,
    ResponseBody.UpdateRes: UpdateRes,
    ResponseBody.ReplaceRes: ReplaceRes,
    ResponseBody.DeleteRes: DeleteRes,
    ResponseBody.SearchRes: SearchRes,
    ResponseBody.GroupRes: GroupRes,
    ResponseBody.StatsRes: StatsRes,
    ResponseBody.IndexRes: IndexRes,
}

_STATUS_NAMES = {
    Status.OK: "OK",
    Status.NotAnObject: "NotAnObject",
    Status.IdMustBeString: "IdMustBeString",
    Status.DuplicateId: "DuplicateId",
    Status.IdMismatch: "IdMismatch",
    Status.NoIndex: "NoIndex",
    Status.PrimaryIndex: "PrimaryIndex",
    Status.NoMatch: "NoMatch",
    Status.InvalidUpdate: "InvalidUpdate",
    Status.VectorDimMismatch: "VectorDimMismatch",
    Status.MalformedRequest: "MalformedRequest",
    Status.UnknownCommand: "UnknownCommand",
    Status.UnsupportedVersion: "UnsupportedVersion",
    Status.InternalError: "InternalError",
}

_AGG_NAMES = {
    "count": AggFn.Count,
    "sum": AggFn.Sum,
    "mean": AggFn.Mean,
    "min": AggFn.Min,
    "max": AggFn.Max,
    "collect": AggFn.Collect,
    "first": AggFn.First,
    "last": AggFn.Last,
}


# ---------------------------------------------------------------------------
# Errors (mirror client::Error = Io | Protocol | Server(Status, message))
# ---------------------------------------------------------------------------

class MooracerError(Exception):
    """Base class for all client errors."""


class MooracerIOError(MooracerError, OSError):
    """TCP / socket failure (client::Error::Io)."""


class ProtocolError(MooracerError):
    """Framing or decode failure — the connection is poisoned; close it."""


class ServerError(MooracerError):
    """A typed server-side error (client::Error::Server).

    `status` is the wire `Status` code (the 9 engine `StoreError` codes +
    the 4 transport codes) and `message` is the server's human detail.
    """

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"{self.name}: {message}")

    @property
    def name(self) -> str:
        return _STATUS_NAMES.get(self.status, f"Status({self.status})")


# ---------------------------------------------------------------------------
# Value <-> native Python conversion
# ---------------------------------------------------------------------------

def _text(b) -> str:
    if isinstance(b, bytes):
        return b.decode("utf-8")
    return b


def decode_value(v: Value):
    """Wire `Value` table -> native Python value (dict/list/str/int/float/bool/None)."""
    kind = v.Kind()
    if kind == ValueKind.Null:
        return None
    if kind == ValueKind.Bool:
        return v.B()
    if kind == ValueKind.I64:
        return v.I()
    if kind == ValueKind.F64:
        return v.F()
    if kind == ValueKind.Str:
        return _text(v.S())
    if kind == ValueKind.Array:
        n = v.ArrLength()
        return [decode_value(v.Arr(j)) for j in range(n)]
    if kind == ValueKind.Object:
        n = v.KeysLength()
        out = {}
        for j in range(n):
            out[_text(v.Keys(j))] = decode_value(v.Vals(j))
        return out
    raise ProtocolError(f"unknown ValueKind {kind}")


def encode_value(b: flatbuffers.Builder, x) -> int:
    """Native Python value -> wire `Value` table offset on `b`.

    Note the `bool` check precedes `int` (bool is an int subclass); the
    engine's `I64`/`F64` split maps to Python `int`/`float`.

    FlatBuffers rule: every nested object (string, vector, child table) must
    be created BEFORE the enclosing table's `Start` (which asserts and sets
    the builder's `nested` flag), so all content is built first.
    """
    if x is None:
        kind = ValueKind.Null
    elif isinstance(x, bool):
        kind = ValueKind.Bool
    elif isinstance(x, int):
        kind = ValueKind.I64
    elif isinstance(x, float):
        kind = ValueKind.F64
    elif isinstance(x, str):
        kind = ValueKind.Str
    elif isinstance(x, (list, tuple)):
        kind = ValueKind.Array
    elif isinstance(x, dict):
        kind = ValueKind.Object
    else:
        raise TypeError(f"cannot encode {type(x).__name__} as a MooRacer value")

    s_off = arr_off = keys_off = vals_off = 0
    if kind == ValueKind.Str:
        s_off = b.CreateString(x.encode("utf-8"))
    elif kind == ValueKind.Array:
        offs = [encode_value(b, e) for e in x]
        ValueStartArrVector(b, len(offs))
        for o in reversed(offs):
            b.PrependUOffsetTRelative(o)
        arr_off = b.EndVector()  # count tracked by StartVector
    elif kind == ValueKind.Object:
        items = list(x.items())
        for k, _ in items:
            if not isinstance(k, str):
                raise TypeError(f"object keys must be str, got {type(k).__name__}")
        vals = [encode_value(b, v) for _, v in items]
        ValueStartValsVector(b, len(vals))
        for o in reversed(vals):
            b.PrependUOffsetTRelative(o)
        vals_off = b.EndVector()
        keys = [b.CreateString(k.encode("utf-8")) for k, _ in items]
        ValueStartKeysVector(b, len(keys))
        for o in reversed(keys):
            b.PrependUOffsetTRelative(o)
        keys_off = b.EndVector()

    ValueStart(b)
    ValueAddKind(b, kind)
    if kind == ValueKind.Bool:
        ValueAddB(b, x)
    elif kind == ValueKind.I64:
        ValueAddI(b, x)
    elif kind == ValueKind.F64:
        ValueAddF(b, x)
    elif kind == ValueKind.Str:
        ValueAddS(b, s_off)
    elif kind == ValueKind.Array:
        ValueAddArr(b, arr_off)
    elif kind == ValueKind.Object:
        ValueAddKeys(b, keys_off)
        ValueAddVals(b, vals_off)
    return ValueEnd(b)


def encode_float_vec(b: flatbuffers.Builder, start, x) -> int:
    """[f] vector offset for a search query (all values coerced to float)."""
    vals = list(x)
    start(b, len(vals))
    for f in reversed(vals):
        b.PrependFloat32(float(f))
    return b.EndVector()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Client:
    """One TCP connection; one in-flight request at a time (not thread-safe)."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._recvbuf = bytearray()  # reused across calls
        self._req_id = 0
        self.last_req_id = 0  # echoes the last completed request's req_id

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def connect(cls, addr: str, *, timeout: float | None = None) -> "Client":
        """`addr` = "host:port" (or a bare host, defaulting to port 4141)."""
        host, _, port = addr.rpartition(":")
        sock = socket.create_connection(
            (host or "127.0.0.1", int(port or 4141)), timeout=timeout
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # low latency
        return cls(sock)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def collection(self, name: str) -> "Collection":
        return Collection(self, name)

    # -- transport ----------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        buf = self._recvbuf
        while len(buf) < n:
            chunk = self._sock.recv(min(_CHUNK, n - len(buf)))
            if not chunk:
                raise ProtocolError("connection closed by server")
            buf.extend(chunk)
        out = bytes(buf[:n])
        del buf[:n]
        return out

    def _rpc(self, name: str, cmd_type: int, build) -> Response:
        """Build the command on a fresh builder, wrap it in a Request, round-trip.

        `build(b)` returns the command table offset; `b` is fresh per call
        (a FlatBufferBuilder cannot be reused after Finish).
        """
        b = flatbuffers.Builder(0)
        cmd = build(b)
        col_off = b.CreateString(name.encode("utf-8"))  # before RequestStart
        RequestStart(b)
        RequestAddVersion(b, WIRE_VERSION)
        self._req_id = (self._req_id + 1) & 0xFFFFFFFFFFFFFFFF
        RequestAddReqId(b, self._req_id)
        RequestAddCollection(b, col_off)
        # The Python codegen splits a union field into type + offset adds.
        RequestAddCommandType(b, cmd_type)
        RequestAddCommand(b, cmd)
        req = RequestEnd(b)
        # The server checks the "MOOR" file identifier on every request.
        b.Finish(req, FILE_IDENTIFIER)
        payload = b.Output()

        try:
            self._sock.sendall(struct.pack("<I", len(payload)) + payload)
            length = int.from_bytes(self._recv_exact(4), "little")
            if length > MAX_FRAME:
                raise ProtocolError(f"frame length {length} exceeds MAX_FRAME")
            body = self._recv_exact(length)
        except OSError as e:
            raise MooracerIOError(str(e)) from e

        resp = Response.GetRootAs(body, 0)
        self.last_req_id = resp.ReqId()
        status = resp.Status()
        if status != Status.OK:
            raise ServerError(status, _text(resp.Message()) or "")
        body_type = resp.BodyType()
        table_cls = _BODY.get(body_type)
        if table_cls is None:
            raise ProtocolError(f"unknown response body type {body_type}")
        # `Body()` returns a bare `Table` positioned at the union payload
        # (the Python codegen has no `BodyAs<Variant>` accessors); re-Init
        # the typed table on the same (Bytes, Pos).
        union = resp.Body()
        table = table_cls()
        table.Init(union.Bytes, union.Pos)
        self._body = table
        return resp

    def _search(self, name: str, cmd_type: int, build) -> list:
        self._rpc(name, cmd_type, build)
        res = self._body
        n = res.HitsLength()
        return [(decode_value(res.Hits(i).Doc()), float(res.Hits(i).Score()))
                for i in range(n)]


# ---------------------------------------------------------------------------
# Collection (all commands live here; mirrors the Rust client)
# ---------------------------------------------------------------------------

class Collection:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    # -- insert ------------------------------------------------------------

    def insert(self, doc: dict) -> str:
        """Insert one doc; returns its `_id` (explicit, or auto-generated)."""
        self.client._rpc(self.name, Command.InsertCmd,
                         lambda b: _build_insert(b, [doc]))
        return _text(self.client._body.Ids(0))

    def insert_many(self, docs) -> list:
        """Insert several docs (order preserved); returns the ids in order."""
        docs = list(docs)
        self.client._rpc(self.name, Command.InsertCmd,
                         lambda b: _build_insert(b, docs))
        ids = self.client._body
        n = ids.IdsLength()
        return [_text(ids.Ids(i)) for i in range(n)]

    # -- query ---------------------------------------------------------------

    def find(self, filter: dict | None = None) -> "Query":
        """Lazy query; the RPC runs only at a terminal."""
        return Query(self, dict(filter) if filter is not None else {})

    def find_one(self, filter: dict | None = None):
        """First match as a native dict, or None."""
        self.client._rpc(self.name, Command.FindCmd,
                         lambda b: _build_find(b, dict(filter or {}), one=True))
        docs = self.client._body
        return decode_value(docs.Docs(0)) if docs.DocsLength() else None

    def count(self, filter: dict | None = None) -> int:
        self.client._rpc(self.name, Command.CountCmd,
                         lambda b: _build_count(b, dict(filter or {})))
        return int(self.client._body.Count())

    def exists(self, filter: dict | None = None) -> bool:
        self.client._rpc(self.name, Command.ExistsCmd,
                         lambda b: _build_exists(b, dict(filter or {})))
        return bool(self.client._body.Exists())

    # -- update / replace / delete ------------------------------------------

    def update_one(self, filter: dict, update: dict) -> int:
        """Raises `ServerError(status=NoMatch)` when no document matches."""
        self.client._rpc(self.name, Command.UpdateCmd,
                         lambda b: _build_update(b, filter, update, many=False))
        return int(self.client._body.Count())

    def update_many(self, filter: dict, update: dict) -> int:
        self.client._rpc(self.name, Command.UpdateCmd,
                         lambda b: _build_update(b, filter, update, many=True))
        return int(self.client._body.Count())

    def replace_one(self, filter: dict, new_doc: dict) -> int:
        """Raises `ServerError(status=NoMatch)` when no document matches."""
        self.client._rpc(self.name, Command.ReplaceCmd,
                         lambda b: _build_replace(b, filter, new_doc))
        return int(self.client._body.Count())

    def delete_one(self, filter: dict | None = None) -> bool:
        self.client._rpc(self.name, Command.DeleteCmd,
                         lambda b: _build_delete(b, filter, many=False))
        return int(self.client._body.Count()) == 1

    def delete_many(self, filter: dict | None = None) -> int:
        self.client._rpc(self.name, Command.DeleteCmd,
                         lambda b: _build_delete(b, filter, many=True))
        return int(self.client._body.Count())

    # -- search --------------------------------------------------------------

    def vector_search(self, field: str, query, limit: int = 10) -> list:
        """-> list of (doc dict, cosine score float), best-first, post-limit."""
        def build(b):
            field_off = b.CreateString(field.encode("utf-8"))
            q = encode_float_vec(b, VectorSearchCmdStartQueryVector, list(query))
            VectorSearchCmdStart(b)
            VectorSearchCmdAddField(b, field_off)
            VectorSearchCmdAddQuery(b, q)
            VectorSearchCmdAddLimit(b, int(limit))
            return VectorSearchCmdEnd(b)
        return self.client._search(self.name, Command.VectorSearchCmd, build)

    def text_search(self, field: str, query: str, limit: int = 10) -> list:
        """BM25 hits -> list of (doc dict, score float)."""
        def build(b):
            field_off = b.CreateString(field.encode("utf-8"))
            query_off = b.CreateString(query.encode("utf-8"))
            TextSearchCmdStart(b)
            TextSearchCmdAddField(b, field_off)
            TextSearchCmdAddQuery(b, query_off)
            TextSearchCmdAddLimit(b, int(limit))
            return TextSearchCmdEnd(b)
        return self.client._search(self.name, Command.TextSearchCmd, build)

    def hybrid_search(self, text_field: str, vec_field: str, query_text: str,
                      query_vec, limit: int = 10) -> list:
        """RRF-fused BM25 + vector hits -> list of (doc dict, score float)."""
        def build(b):
            tf_off = b.CreateString(text_field.encode("utf-8"))
            vf_off = b.CreateString(vec_field.encode("utf-8"))
            qt_off = b.CreateString(query_text.encode("utf-8"))
            q = encode_float_vec(b, HybridSearchCmdStartQueryVecVector,
                                 list(query_vec))
            HybridSearchCmdStart(b)
            HybridSearchCmdAddTextField(b, tf_off)
            HybridSearchCmdAddVecField(b, vf_off)
            HybridSearchCmdAddQueryText(b, qt_off)
            HybridSearchCmdAddQueryVec(b, q)
            HybridSearchCmdAddLimit(b, int(limit))
            return HybridSearchCmdEnd(b)
        return self.client._search(self.name, Command.HybridSearchCmd, build)

    # -- aggregation -----------------------------------------------------------

    def stats(self) -> dict:
        self.client._rpc(self.name, Command.StatsCmd, lambda b: _build_stats(b))
        s = self.client._body
        per = []
        for i in range(s.PerIndexLength()):
            ix = s.PerIndex(i)
            per.append({
                "field": _text(ix.Field()),
                "entries": int(ix.Entries()),
                "distinct": int(ix.Distinct()),
                "memory": int(ix.Memory()),
            })
        return {
            "docs": int(s.Docs()),
            "docs_memory": int(s.DocsMemory()),
            "indexes": int(s.Indexes()),
            "total_memory": int(s.TotalMemory()),
            "per_index": per,
        }

    # -- index management (wire-level; enables search at runtime) ------------

    def create_index(self, field: str) -> None:
        """Create a value field index on `field` (enables equality/range scans)."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.CreateValue, field, 0))

    def drop_index(self, field: str) -> None:
        """Drop a value field index on `field`. `_id` cannot be dropped."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.DropValue, field, 0))

    def create_vector_index(self, field: str, dim: int) -> None:
        """Create a vector index on `field` with dimension `dim`."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.CreateVector, field, dim))

    def drop_vector_index(self, field: str) -> None:
        """Drop a vector index on `field`."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.DropVector, field, 0))

    def create_text_index(self, field: str) -> None:
        """Create a BM25 text index on `field`."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.CreateText, field, 0))

    def drop_text_index(self, field: str) -> None:
        """Drop a text index on `field`."""
        self.client._rpc(self.name, Command.IndexCmd,
                         lambda b: _build_index(b, IndexKind.DropText, field, 0))


# ---------------------------------------------------------------------------
# Lazy query chain: find -> sort/skip/limit -> terminal
# ---------------------------------------------------------------------------

class Query:
    def __init__(self, col: Collection, filter: dict):
        self.col = col
        self.filter = filter
        self._sort_field = None
        self._sort_desc = False
        self._skip = 0
        self._limit = 0

    def sort(self, field: str, descending: bool = False) -> "Query":
        self._sort_field = field
        self._sort_desc = bool(descending)
        return self

    def skip(self, n: int) -> "Query":
        self._skip = int(n)
        return self

    def limit(self, n: int) -> "Query":
        self._limit = int(n)  # 0 = no limit (Mongo cursor convention)
        return self

    def group(self, field: str) -> "GroupQuery":
        """Carry this query's pipeline as the *pre-group* stage."""
        return GroupQuery(self.col, self.filter, field,
                          self._sort_field, self._sort_desc, self._skip,
                          self._limit)

    # -- terminals (each runs the single RPC) --------------------------------

    def to_list(self) -> list:
        self.col.client._rpc(self.col.name, Command.FindCmd,
                             lambda b: _build_find(b, self.filter,
                                                   self._sort_field,
                                                   self._sort_desc,
                                                   self._skip, self._limit,
                                                   one=False))
        docs = self.col.client._body
        return [decode_value(docs.Docs(i)) for i in range(docs.DocsLength())]

    def first(self):
        self.col.client._rpc(self.col.name, Command.FindCmd,
                             lambda b: _build_find(b, self.filter,
                                                   self._sort_field,
                                                   self._sort_desc,
                                                   self._skip, self._limit,
                                                   one=True))
        docs = self.col.client._body
        return decode_value(docs.Docs(0)) if docs.DocsLength() else None

    find_one = first

    def count(self) -> int:
        # Same filter, same pipeline semantics as the eager count; the server
        # counts filtered docs (sort/skip/limit are irrelevant to a count).
        self.col.client._rpc(self.col.name, Command.CountCmd,
                             lambda b: _build_count(b, self.filter))
        return int(self.col.client._body.Count())


class GroupQuery:
    def __init__(self, col, filter, group_field,
                 sort_field=None, sort_desc=False, skip=0, limit=0):
        self.col = col
        self.filter = filter
        self.group_field = group_field
        self.q_sort_field = sort_field
        self.q_sort_desc = sort_desc
        self.q_skip = skip
        self.q_limit = limit
        self._g_sort_field = None
        self._g_sort_desc = False
        self._g_limit = 0

    def sort(self, field: str, descending: bool = False) -> "GroupQuery":
        """Sort the *group documents* (post-group stage)."""
        self._g_sort_field = field
        self._g_sort_desc = bool(descending)
        return self

    def limit(self, n: int) -> "GroupQuery":
        """Limit the group documents (post-group); 0 = no limit."""
        self._g_limit = int(n)
        return self

    def agg(self, fn, field: str | None = None) -> list:
        """Terminal: one result doc per group
        `{ "_id": <key>, "<fn>": <value> }` as native dicts."""
        if isinstance(fn, str):
            fn = _AGG_NAMES.get(fn.lower())
            if fn is None:
                raise ValueError(f"unknown AggFn {fn!r} "
                                 f"(expected one of {sorted(_AGG_NAMES)})")
        if field is None:
            field = ""

        def build(b):
            f = encode_value(b, self.filter)
            group_off = b.CreateString(self.group_field.encode("utf-8"))
            agg_off = b.CreateString(field.encode("utf-8"))
            q_sort_off = (b.CreateString(self.q_sort_field.encode("utf-8"))
                          if self.q_sort_field is not None else 0)
            g_sort_off = (b.CreateString(self._g_sort_field.encode("utf-8"))
                          if self._g_sort_field is not None else 0)
            GroupCmdStart(b)
            GroupCmdAddFilter(b, f)
            if self.q_sort_field is not None:
                GroupCmdAddSortField(b, q_sort_off)
                GroupCmdAddSortDesc(b, self.q_sort_desc)
            if self.q_skip:
                GroupCmdAddSkip(b, self.q_skip)
            if self.q_limit:
                GroupCmdAddLimit(b, self.q_limit)
            GroupCmdAddGroupField(b, group_off)
            GroupCmdAddAggFn(b, fn)
            GroupCmdAddAggField(b, agg_off)
            if self._g_sort_field is not None:
                GroupCmdAddGroupSortField(b, g_sort_off)
                GroupCmdAddGroupSortDesc(b, self._g_sort_desc)
            if self._g_limit:
                GroupCmdAddGroupLimit(b, self._g_limit)
            return GroupCmdEnd(b)

        self.col.client._rpc(self.col.name, Command.GroupCmd, build)
        groups = self.col.client._body
        return [decode_value(groups.Groups(i))
                for i in range(groups.GroupsLength())]


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_insert(b, docs) -> int:
    docs = list(docs)
    offs = [encode_value(b, d) for d in docs]
    InsertCmdStartDocsVector(b, len(offs))
    for o in reversed(offs):
        b.PrependUOffsetTRelative(o)
    docs_off = b.EndVector()
    InsertCmdStart(b)
    InsertCmdAddDocs(b, docs_off)
    return InsertCmdEnd(b)


def _build_find(b, filter, sort_field=None, sort_desc=False,
                skip=0, limit=0, one=False) -> int:
    f = encode_value(b, filter)
    sort_off = (b.CreateString(sort_field.encode("utf-8"))
                if sort_field is not None else 0)
    FindCmdStart(b)
    FindCmdAddFilter(b, f)
    if sort_field is not None:
        FindCmdAddSortField(b, sort_off)
        FindCmdAddSortDesc(b, bool(sort_desc))
    if skip:
        FindCmdAddSkip(b, int(skip))
    if limit:
        FindCmdAddLimit(b, int(limit))
    if one:
        FindCmdAddOne(b, True)
    return FindCmdEnd(b)


def _build_count(b, filter) -> int:
    f = encode_value(b, filter)
    CountCmdStart(b)
    CountCmdAddFilter(b, f)
    return CountCmdEnd(b)


def _build_exists(b, filter) -> int:
    f = encode_value(b, filter)
    ExistsCmdStart(b)
    ExistsCmdAddFilter(b, f)
    return ExistsCmdEnd(b)


def _build_update(b, filter, update, many) -> int:
    f = encode_value(b, filter)
    u = encode_value(b, update)
    UpdateCmdStart(b)
    UpdateCmdAddFilter(b, f)
    UpdateCmdAddUpdate(b, u)
    if many:
        UpdateCmdAddMany(b, True)
    return UpdateCmdEnd(b)


def _build_replace(b, filter, new_doc) -> int:
    f = encode_value(b, filter)
    d = encode_value(b, new_doc)
    ReplaceCmdStart(b)
    ReplaceCmdAddFilter(b, f)
    ReplaceCmdAddNewDoc(b, d)
    return ReplaceCmdEnd(b)


def _build_delete(b, filter, many) -> int:
    f = encode_value(b, dict(filter) if filter is not None else {})
    DeleteCmdStart(b)
    DeleteCmdAddFilter(b, f)
    if many:
        DeleteCmdAddMany(b, True)
    return DeleteCmdEnd(b)


def _build_stats(b) -> int:
    StatsCmdStart(b)
    return StatsCmdEnd(b)


def _build_index(b, kind, field, dim) -> int:
    field_off = b.CreateString(field.encode("utf-8"))
    IndexCmdStart(b)
    IndexCmdAddKind(b, kind)
    IndexCmdAddField(b, field_off)
    IndexCmdAddDim(b, int(dim))
    return IndexCmdEnd(b)
