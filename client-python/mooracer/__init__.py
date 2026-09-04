"""mooracer — pure-Python client for the MooRacer in-memory document engine.

Speaks the v1 FlatBuffers wire protocol (schema/mooracer.fbs) over
length-prefixed TCP frames. See `mooracer.client` for the API.
"""

from .client import (
    Client,
    Collection,
    GroupQuery,
    Query,
    MooracerError,
    MooracerIOError,
    ProtocolError,
    ServerError,
    decode_value,
    encode_value,
    WIRE_VERSION,
    MAX_FRAME,
)
from . import wire  # generated FlatBuffers tables (flatc --python)

__version__ = "0.1.0"

__all__ = [
    "Client",
    "Collection",
    "GroupQuery",
    "Query",
    "MooracerError",
    "MooracerIOError",
    "ProtocolError",
    "ServerError",
    "decode_value",
    "encode_value",
    "wire",
    "WIRE_VERSION",
    "MAX_FRAME",
]
