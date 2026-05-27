"""OpenHands dispatcher — shares the OC remote-exec bridge protocol.

When the agent runtime is the OpenHands SDK (Python) rather than OpenClaw
(node), tool calls (bash, file edits, …) still need to land in the per-run
SWE-bench sandbox container — same problem the OC bridge already solves.
Rather than build a parallel mechanism, we reuse:

  - host-side socket server: `agent_bench/runtime/exec_bridge/server.py` (unchanged)
  - in-container CLI client: `agent_bench/runtime/exec_bridge/bridge.py` (unchanged)
  - HMAC-signed JSONL wire protocol                         (unchanged)

What differs from `OpenClawRemoteDispatcher` is only how the bridge gets
WIRED into the harness container:

  - In-container binary path is `/usr/local/bin/oh-exec-bridge`
    (kept distinct from oc-exec-bridge so a hybrid host running both
    harnesses doesn't accidentally cross-talk).
  - Env vars are `OPENHANDS_EXEC_BRIDGE` / `_SOCKET` / `_TOKEN` so the
    OpenHands driver (`agent_bench/harnesses/openhands/driver.py`) picks them up
    without conflating with OpenClaw's namespace.
  - The state_dir is mounted at `/etc/agent-bench/state:ro` in OH
    (no `~/.openclaw` analog needed; OH doesn't read state from $HOME).

See `agent_bench/harnesses/openhands/driver.py` for the patch point that turns SDK
BashTool / FileEditorTool / TerminalTool executors into bridge invocations
(no JS patch like OC needs — the SDK is Python, we monkey-patch tool
executors in-process).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agent_bench.deployments.containers.base import Mount
from agent_bench.harnesses.openclaw.dispatcher import (
    OpenClawRemoteDispatcher,
)


# Mount targets inside the OpenHands harness container. The harness bind-mounts
# state_dir at `_STATE_DIR_IN_OH`, so the socket file (created by server.py
# under state_dir/exec-bridge.sock on the host) appears at the path below.
_STATE_DIR_IN_OH = "/etc/agent-bench/state"
_BRIDGE_SOCKET_IN_OH = f"{_STATE_DIR_IN_OH}/exec-bridge.sock"
_BRIDGE_PATH_IN_OH = "/usr/local/bin/oh-exec-bridge"


class OpenHandsDispatcher(OpenClawRemoteDispatcher):
    """Same lifecycle as OpenClawRemoteDispatcher, OH-flavoured mounts/env."""

    backend_id = "openhands"

    def agent_mounts_and_env(self) -> tuple[list[Mount], dict[str, str]]:
        if self.token is None:
            raise RuntimeError(
                "OpenHandsDispatcher.agent_mounts_and_env() called before start()"
            )
        mounts = [
            Mount(host=Path(self.bridge_script), container=_BRIDGE_PATH_IN_OH, mode="ro"),
        ]
        env = {
            "OPENHANDS_EXEC_BRIDGE": _BRIDGE_PATH_IN_OH,
            "OPENHANDS_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET_IN_OH,
            "OPENHANDS_EXEC_BRIDGE_TOKEN": self.token,
            # bridge.py itself reads the historical OPENCLAW_* names.
            "OPENCLAW_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET_IN_OH,
            "OPENCLAW_EXEC_BRIDGE_TOKEN": self.token,
        }
        return mounts, env

    def agent_container_args(self) -> List[str]:
        mounts, env = self.agent_mounts_and_env()
        args: List[str] = []
        for mount in mounts:
            args += ["-v", f"{mount.host}:{mount.container}:{mount.mode}"]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        return args

    def env_info(self) -> Dict[str, Any]:
        info = super().env_info()
        info["dispatcher"] = self.backend_id
        return info
