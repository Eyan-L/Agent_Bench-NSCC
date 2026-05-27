"""Hermes dispatcher — shares the Agent_Bench/OpenClaw exec-bridge protocol.

The Hermes patch keeps `TERMINAL_ENV=docker` and hijacks DockerEnvironment's
exec path when HERMES_EXEC_BRIDGE is set. The mounted bridge talks to the
host-side socket server and routes commands into the pre-started SWE-bench
task container.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agent_bench.deployments.containers.base import Mount
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher


_STATE_DIR_IN_HERMES = "/opt/data"
_BRIDGE_SOCKET_IN_HERMES = f"{_STATE_DIR_IN_HERMES}/exec-bridge.sock"
_BRIDGE_PATH_IN_HERMES = "/usr/local/bin/hm-exec-bridge"


class HermesDispatcher(OpenClawRemoteDispatcher):
    """Same lifecycle as OpenClawRemoteDispatcher, Hermes-flavoured env."""

    backend_id = "hermes"

    def agent_mounts_and_env(self) -> tuple[list[Mount], dict[str, str]]:
        if self.token is None:
            raise RuntimeError(
                "HermesDispatcher.agent_mounts_and_env() called before start()"
            )
        mounts = [
            Mount(host=Path(self.bridge_script), container=_BRIDGE_PATH_IN_HERMES, mode="ro"),
        ]
        env = {
            "HERMES_EXEC_BRIDGE": _BRIDGE_PATH_IN_HERMES,
            "HERMES_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET_IN_HERMES,
            "HERMES_EXEC_BRIDGE_TOKEN": self.token,
            # bridge.py itself reads the historical OPENCLAW_* names.
            "OPENCLAW_EXEC_BRIDGE": _BRIDGE_PATH_IN_HERMES,
            "OPENCLAW_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET_IN_HERMES,
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
