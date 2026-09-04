"""Pure-protocol tests for the generated mooracer.wire module + the client's
value encoder/decoder. No server required: every buffer is built and parsed
with the Python flatbuffers runtime, and the enum discriminants are pinned to
the wire contract (same numbers as wire/tests/schema.rs on the Rust side).
"""

import flatbuffers
import pytest

from mooracer import decode_value, encode_value
from mooracer.wire.AggFn import AggFn
from mooracer.wire.Command import Command
from mooracer.wire.FindCmd import FindCmd
from mooracer.wire.Request import Request
from mooracer.wire.ResponseBody import ResponseBody
from mooracer.wire.Status import Status
from mooracer.wire.Value import Value
from mooracer.wire.ValueKind import ValueKind
from mooracer.client import _build_find


def _roundtrip(x):
    """Encode x as a root Value, decode it back, and compare."""
    b = flatbuffers.Builder(0)
    off = encode_value(b, x)
    b.Finish(off)
    v = Value.GetRootAs(b.Output(), 0)
    return decode_value(v)


# ---------------------------------------------------------------------------
# Value round-trips (all 7 kinds, nested, order-preserving)
# ---------------------------------------------------------------------------

def test_roundtrip_all_kinds():
    assert _roundtrip(None) is None
    assert _roundtrip(True) is True
    assert _roundtrip(False) is False
    assert _roundtrip(42) == 42
    assert _roundtrip(-7) == -7
    assert _roundtrip(2**62) == 2**62
    assert _roundtrip(3.5) == 3.5
    assert _roundtrip(-0.25) == -0.25
    assert _roundtrip("moo") == "moo"
    assert _roundtrip("") == ""
    assert _roundtrip("héllo 🐄") == "héllo 🐄"


def test_roundtrip_int_vs_float_vs_bool_are_distinct_kinds():
    # bool is an int subclass in Python — it must encode as Bool, not I64.
    for x, kind in [(True, ValueKind.Bool), (1, ValueKind.I64), (1.0, ValueKind.F64)]:
        b = flatbuffers.Builder(0)
        off = encode_value(b, x)
        b.Finish(off)
        assert Value.GetRootAs(b.Output(), 0).Kind() == kind


def test_roundtrip_arrays_preserve_order_and_dups():
    x = [1, "two", 3.0, None, True, [1, 1], {"a": [1, 2]}]
    assert _roundtrip(x) == x


def test_roundtrip_objects_preserve_key_order_and_nulls():
    x = {"z": 1, "a": None, "m": {"n": [True]}, "_id": "daisy"}
    out = _roundtrip(x)
    assert out == x
    assert list(out.keys()) == ["z", "a", "m", "_id"]


def test_roundtrip_deep_nesting():
    x = {"l1": [{"l2": {"l3": ["a", "b", {"l4": 1.5}]}}]}
    assert _roundtrip(x) == x


def test_encode_rejects_unknown_types():
    with pytest.raises(TypeError):
        encode_value(flatbuffers.Builder(0), object())
    with pytest.raises(TypeError):
        encode_value(flatbuffers.Builder(0), b"bytes")
    with pytest.raises(TypeError):
        encode_value(flatbuffers.Builder(0), {1: 2})  # non-str key


# ---------------------------------------------------------------------------
# Request envelope: build a Find request, parse it back, check every field
# ---------------------------------------------------------------------------

def test_request_envelope_decodes():
    b = flatbuffers.Builder(0)
    cmd = _build_find(b, {"age": {"$gte": 3}, "tags": ["x", "y"]},
                      sort_field="age", sort_desc=True, skip=2, limit=10)
    col_off = b.CreateString(b"cows")
    from mooracer.wire.Request import (
        RequestAddCollection,
        RequestAddCommand,
        RequestAddCommandType,
        RequestAddReqId,
        RequestAddVersion,
        RequestEnd,
        RequestStart,
    )

    RequestStart(b)
    RequestAddVersion(b, 1)
    RequestAddReqId(b, 7)
    RequestAddCollection(b, col_off)
    RequestAddCommandType(b, Command.FindCmd)
    RequestAddCommand(b, cmd)
    req_off = RequestEnd(b)
    b.Finish(req_off, b"MOOR")
    buf = b.Output()

    req = Request.GetRootAs(buf, 0)
    assert req.Version() == 1
    assert req.ReqId() == 7
    assert req.Collection() == b"cows"
    assert req.CommandType() == Command.FindCmd

    union = req.Command()
    f = FindCmd()
    f.Init(union.Bytes, union.Pos)
    assert f.SortField() == b"age"
    assert f.SortDesc() is True
    assert f.Skip() == 2
    assert f.Limit() == 10
    assert f.One() is False
    assert decode_value(f.Filter()) == {"age": {"$gte": 3}, "tags": ["x", "y"]}


def test_file_identifier_present():
    b = flatbuffers.Builder(0)
    cmd = _build_find(b, {})
    col_off = b.CreateString(b"c")
    from mooracer.wire.Request import (
        RequestAddCollection,
        RequestAddCommand,
        RequestAddCommandType,
        RequestEnd,
        RequestStart,
    )

    RequestStart(b)
    RequestAddCollection(b, col_off)
    RequestAddCommandType(b, Command.FindCmd)
    RequestAddCommand(b, cmd)
    b.Finish(RequestEnd(b), b"MOOR")
    assert Request.RequestBufferHasIdentifier(b.Output(), 0)


# ---------------------------------------------------------------------------
# Pinned wire discriminants (contract: identical to wire/tests/schema.rs)
# ---------------------------------------------------------------------------

def test_enum_discriminants_are_the_wire_contract():
    assert [ValueKind.Null, ValueKind.Bool, ValueKind.I64, ValueKind.F64,
            ValueKind.Str, ValueKind.Array, ValueKind.Object] == list(range(7))

    assert [AggFn.Count, AggFn.Sum, AggFn.Mean, AggFn.Min, AggFn.Max,
            AggFn.Collect, AggFn.First, AggFn.Last] == list(range(8))

    assert [Command.NONE, Command.InsertCmd, Command.FindCmd, Command.CountCmd,
            Command.ExistsCmd, Command.UpdateCmd, Command.ReplaceCmd,
            Command.DeleteCmd, Command.VectorSearchCmd, Command.TextSearchCmd,
            Command.HybridSearchCmd, Command.GroupCmd, Command.StatsCmd] \
        == list(range(13))

    assert [ResponseBody.NONE, ResponseBody.InsertRes, ResponseBody.FindRes,
            ResponseBody.CountRes, ResponseBody.ExistsRes, ResponseBody.UpdateRes,
            ResponseBody.ReplaceRes, ResponseBody.DeleteRes, ResponseBody.SearchRes,
            ResponseBody.GroupRes, ResponseBody.StatsRes] == list(range(11))

    assert [Status.OK, Status.NotAnObject, Status.IdMustBeString,
            Status.DuplicateId, Status.IdMismatch, Status.NoIndex,
            Status.PrimaryIndex, Status.NoMatch, Status.InvalidUpdate,
            Status.VectorDimMismatch, Status.MalformedRequest,
            Status.UnknownCommand, Status.UnsupportedVersion,
            Status.InternalError] == list(range(14))
