from __future__ import annotations

import socket
import subprocess
import sys

import pytest

from agent_bench.serving.sglang import pd_supervisor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_assert_port_available_rejects_bound_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = int(sock.getsockname()[1])
        with pytest.raises(RuntimeError, match="already in use"):
            pd_supervisor._assert_port_available("127.0.0.1", port, "prefill")


def test_wait_for_port_fails_if_child_exits_before_listening() -> None:
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    proc.wait(timeout=5)

    with pytest.raises(RuntimeError, match="exited before listening"):
        pd_supervisor._wait_for_port("127.0.0.1", port, "prefill", timeout=5, proc=proc)
