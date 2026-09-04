"""Shared fixtures: a fresh mooracer-devserver per test.

The devserver (server/src/bin/mooracer-devserver.rs) is protocol-identical to
mooracer-server but can pre-create indexes from env (the v1 wire protocol has
no index-management command). Docs are seeded over the wire by the tests
themselves; the indexes are maintained on insert by the engine's normal
maintenance path.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLIENT_DIR = Path(__file__).resolve().parents[1]  # client-python/
WORKSPACE = CLIENT_DIR.parent

sys.path.insert(0, str(CLIENT_DIR))

DEVSERVER_BIN = WORKSPACE / "target" / "release" / "mooracer-devserver"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.1)
            return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"devserver did not open port {port} within {timeout}s")


@pytest.fixture(scope="session")
def devserver_bin():
    """The release devserver binary, built once if missing."""
    if not DEVSERVER_BIN.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "mooracer-devserver"],
            cwd=WORKSPACE,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    assert DEVSERVER_BIN.exists()
    return DEVSERVER_BIN


@pytest.fixture
def server(devserver_bin):
    """Yields `start(*, vector=None, text=None) -> addr`.

    Each call starts a FRESH server (isolated store) with optional
    pre-created indexes: `vector=["coll:field:dim", ...]`,
    `text=["coll:field", ...]`. Servers are terminated at test teardown.
    """
    procs = []

    def start(*, vector=None, text=None) -> str:
        port = _free_port()
        env = dict(os.environ, MOORACER_ADDR=f"127.0.0.1:{port}")
        if vector:
            env["MOORACER_VECTOR_INDEX"] = ";".join(vector)
        if text:
            env["MOORACER_TEXT_INDEX"] = ";".join(text)
        p = subprocess.Popen(
            [str(devserver_bin)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
        _wait_port(port)
        return f"127.0.0.1:{port}"

    yield start
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=5)
