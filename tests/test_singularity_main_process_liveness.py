from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace

from agent_bench.deployments.containers.singularity_liveness import (
    poll_tracked_main_process,
)


def test_poll_tracked_main_process_reports_child_exit():
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    holder = SimpleNamespace(_main_pids={"gateway": proc.pid})

    deadline = time.monotonic() + 5
    result = None
    while time.monotonic() < deadline and result is None:
        result = poll_tracked_main_process(holder, "gateway")
        if result is None:
            time.sleep(0.01)

    assert result == 7
    assert "gateway" not in holder._main_pids


def test_poll_tracked_main_process_keeps_live_child_running():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    holder = SimpleNamespace(_main_pids={"gateway": proc.pid})
    try:
        assert poll_tracked_main_process(holder, "gateway") is None
        assert holder._main_pids["gateway"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)
