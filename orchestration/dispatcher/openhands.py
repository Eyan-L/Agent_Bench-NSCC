"""OpenHands dispatcher — shares the OC remote-exec bridge protocol.

When the agent runtime is the OpenHands SDK (Python) rather than OpenClaw
(node), tool calls (bash, file edits, …) still need to land in the per-run
SWE-bench sandbox container — same problem the OC bridge already solves.
Rather than build a parallel mechanism, we reuse:

  - host-side socket server: `openclaw_remote/server.py`   (unchanged)
  - in-container CLI client: `openclaw_remote/bridge.py`   (unchanged)
  - HMAC-signed JSONL wire protocol                         (unchanged)

What differs from `OpenClawRemoteDispatcher` is only how the bridge gets
WIRED into the harness container:

  - In-container binary path is `/usr/local/bin/oh-exec-bridge`
    (kept distinct from oc-exec-bridge so a hybrid host running both
    harnesses doesn't accidentally cross-talk).
  - Env vars are `OPENHANDS_EXEC_BRIDGE` / `_SOCKET` / `_TOKEN` so the
    OpenHands driver (`harnesses/openhands/driver.py`) picks them up
    without conflating with OpenClaw's namespace.
  - The state_dir is mounted at `/etc/agent-bench/state:ro` in OH
    (no `~/.openclaw` analog needed; OH doesn't read state from $HOME).

See `harnesses/openhands/driver.py` for the patch point that turns SDK
BashTool / FileEditorTool / TerminalTool executors into bridge invocations
(no JS patch like OC needs — the SDK is Python, we monkey-patch tool
executors in-process).
"""
from __future__ import annotations

from typing import Any, Dict, List

from orchestration.dispatcher.openclaw_remote import (
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

    def agent_container_args(self) -> List[str]:
        if self.token is None:
            raise RuntimeError(
                "OpenHandsDispatcher.agent_container_args() called before start()"
            )
        return [
            # Bridge CLI binary the OH driver invokes when its tool executors
            # are wrapped (see harnesses/openhands/driver.py).
            "-v", f"{self.bridge_script}:{_BRIDGE_PATH_IN_OH}:ro",
            # OPENHANDS_-prefixed env so the driver knows it's running in
            # bridged mode without colliding with OC's namespace.
            "-e", f"OPENHANDS_EXEC_BRIDGE={_BRIDGE_PATH_IN_OH}",
            "-e", f"OPENHANDS_EXEC_BRIDGE_SOCKET={_BRIDGE_SOCKET_IN_OH}",
            "-e", f"OPENHANDS_EXEC_BRIDGE_TOKEN={self.token}",
            # bridge.py itself reads token + socket from OPENCLAW_EXEC_BRIDGE_*
            # env (it was originally written for OC's JS patch). Export the
            # canonical names too so the binary just works when invoked.
            "-e", f"OPENCLAW_EXEC_BRIDGE_SOCKET={_BRIDGE_SOCKET_IN_OH}",
            "-e", f"OPENCLAW_EXEC_BRIDGE_TOKEN={self.token}",
        ]

    def env_info(self) -> Dict[str, Any]:
        info = super().env_info()
        info["dispatcher"] = self.backend_id
        return info
