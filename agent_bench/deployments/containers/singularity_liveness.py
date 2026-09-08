"""Singularity main-process liveness checks for readiness polling."""
from __future__ import annotations

import os
import time


def poll_tracked_main_process(container, name: str):
    """Return a terminal exit code for the tracked bg exec, else ``None``.

    A Singularity instance itself is kept alive by ``sleep infinity``.  The
    OpenClaw gateway is a child ``singularity exec`` process whose PID is
    stored in ``_main_pids``.  Reap that child with ``waitpid(WNOHANG)`` so a
    dead gateway cannot masquerade as a live instance during readiness.
    """
    pid = getattr(container, "_main_pids", {}).get(name)
    if not pid:
        return None
    try:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # The tracked child has already been reaped elsewhere.  A PID retained
        # in _main_pids after that point is no longer a live gateway process.
        getattr(container, "_main_pids", {}).pop(name, None)
        return "unknown"
    except OSError:
        return None

    if waited_pid == 0:
        return None

    getattr(container, "_main_pids", {}).pop(name, None)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return "unknown"


def _wait_log_pattern_with_main_liveness(
    self,
    h,
    pattern,
    timeout_s: float,
    *,
    poll_s: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_tail = ""
    while time.monotonic() < deadline:
        log_text = self.logs(h)
        last_tail = log_text[-1500:]

        exit_code = poll_tracked_main_process(self, h.name)
        if exit_code is not None:
            raise RuntimeError(
                f"singularity main process for {h.name} exited before readiness "
                f"(rc={exit_code}). last log:\n{last_tail}"
            )

        if pattern.search(log_text):
            return

        if not self.is_running(h):
            raise RuntimeError(
                f"singularity instance {h.name} exited before log pattern matched. "
                f"last log:\n{last_tail}"
            )
        time.sleep(poll_s)

    raise RuntimeError(
        f"timed out waiting {timeout_s}s for log pattern in singularity "
        f"instance {h.name}. last log:\n{last_tail}"
    )


def install_singularity_main_process_liveness() -> None:
    """Make Singularity readiness fail fast when its bg command exits."""
    from .singularity import SingularityContainer

    current = SingularityContainer.wait_log_pattern
    if getattr(current, "_agent_bench_main_process_liveness", False):
        return
    setattr(
        _wait_log_pattern_with_main_liveness,
        "_agent_bench_main_process_liveness",
        True,
    )
    SingularityContainer.wait_log_pattern = _wait_log_pattern_with_main_liveness
