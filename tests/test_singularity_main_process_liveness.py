from __future__ import annotations

import subprocess
import sys
import time
import os
import re

import pytest
from agent_bench.deployments.containers.singularity import SingularityContainer
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


def _holder(log="", running=True, tracked=True):
    return SimpleNamespace(
        _main_pids={"gateway": 123} if tracked else {},
        logs=lambda h: log, is_running=lambda h: running,
    ), SimpleNamespace(name="gateway")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_readiness_rejects_dead_command_even_with_matching_log(monkeypatch, exit_code):
    holder, handle = _holder("READY\nprocess exited")
    monkeypatch.setattr(os, "waitpid", lambda pid, options: (pid, exit_code << 8))
    with pytest.raises(RuntimeError, match=f"rc={exit_code}") as error:
        SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 300)
    assert "process exited" in str(error.value)
    assert "gateway" not in holder._main_pids


def test_readiness_preserves_live_command(monkeypatch):
    holder, handle = _holder("READY")
    monkeypatch.setattr(os, "waitpid", lambda pid, options: (0, 0))
    SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 300)
    assert holder._main_pids["gateway"] == 123


def test_readiness_without_main_command_uses_instance_and_log():
    holder, handle = _holder("READY", tracked=False)
    SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 300)


def test_readiness_reports_dead_instance(monkeypatch):
    holder, handle = _holder("initializing", running=False)
    monkeypatch.setattr(os, "waitpid", lambda pid, options: (0, 0))
    with pytest.raises(RuntimeError, match="instance gateway exited"):
        SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 300)


def test_readiness_timeout_uses_monotonic_time(monkeypatch):
    holder, handle = _holder("initializing", tracked=False)
    ticks = iter([0, 0.2, 2])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    def wall_clock():
        raise AssertionError("Readiness must not depend on wall-clock adjustments")
    monkeypatch.setattr(time, "time", wall_clock)
    with pytest.raises(RuntimeError, match="timed out waiting"):
        SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 1)


def test_readiness_rejects_already_reaped_main_process(monkeypatch):
    holder, handle = _holder("READY")
    def reaped(pid, options):
        raise ChildProcessError
    monkeypatch.setattr(os, "waitpid", reaped)
    with pytest.raises(RuntimeError, match="rc=unknown"):
        SingularityContainer.wait_log_pattern(holder, handle, re.compile("READY"), 300)
