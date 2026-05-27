"""OpenClaw remote-exec dispatcher.

Bundles the host-side socket server (`server.py`) and the in-OC-container
client (`bridge.py`) with a `Dispatcher`-conforming class that owns their
per-run lifecycle (token, socket, server subprocess, agent-container
injection, cleanup).

See `doc/exec-bridge.md` for the underlying architecture.
"""
from __future__ import annotations

import logging
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_bench.deployments.containers.base import Mount
from agent_bench.runtime.dispatcher import Dispatcher
from agent_bench.runtime import exec_bridge

logger = logging.getLogger(__name__)

# Path the harness mounts state_dir at inside the OC container (see
# agent_bench/benchmarks/swebench/app.py: `-v {state_dir}:/root/.openclaw:rw`).
# The socket file therefore appears at this fixed path inside the container.
_BRIDGE_SOCKET_IN_OC = "/root/.openclaw/exec-bridge.sock"
_BRIDGE_PATH_IN_OC = "/usr/local/bin/oc-exec-bridge"

_EXEC_BRIDGE_DIR = Path(exec_bridge.__file__).resolve().parent
_SERVER_SCRIPT = _EXEC_BRIDGE_DIR / "server.py"
_BRIDGE_SCRIPT = _EXEC_BRIDGE_DIR / "bridge.py"


class OpenClawRemoteDispatcher(Dispatcher):
    """OpenClaw exec-bridge dispatcher (host-side server + in-container bridge).

    Assumes the harness already bind-mounts `<state_dir>:/root/.openclaw:rw`
    on the OC container, so the unix socket created in state_dir is visible
    inside the container at `_BRIDGE_SOCKET_IN_OC`.
    """

    backend_id = "openclaw_remote"

    _STARTUP_TIMEOUT_SECONDS = 10

    def __init__(self) -> None:
        self.server_script = _SERVER_SCRIPT
        self.bridge_script = _BRIDGE_SCRIPT

        self._proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.socket_path: Optional[Path] = None
        self.token: Optional[str] = None
        self.log_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, *, state_dir: Path, backend: str, target: str) -> None:
        state_dir = Path(state_dir)
        if not self.server_script.is_file():
            raise RuntimeError(
                f"exec-bridge server.py not found at {self.server_script}"
            )
        if not self.bridge_script.is_file():
            raise RuntimeError(
                f"exec-bridge bridge.py not found at {self.bridge_script}"
            )

        token = secrets.token_hex(24)
        token_file = state_dir / "exec-bridge.token"
        token_file.write_text(token)
        token_file.chmod(0o600)

        socket_path = state_dir / "exec-bridge.sock"
        log_path = state_dir / "exec-bridge.log"

        proc = subprocess.Popen(
            [
                "python3", str(self.server_script),
                "--socket", str(socket_path),
                "--token-file", str(token_file),
                "--backend", backend,
                "--target", target,
            ],
            stdout=log_path.open("ab"),
            stderr=subprocess.STDOUT,
        )

        deadline = time.time() + self._STARTUP_TIMEOUT_SECONDS
        while not socket_path.exists():
            if time.time() > deadline:
                proc.terminate()
                raise RuntimeError(
                    f"exec-bridge server did not create socket at "
                    f"{socket_path}; see {log_path}"
                )
            if proc.poll() is not None:
                raise RuntimeError(
                    f"exec-bridge server died on startup "
                    f"(rc={proc.returncode}); see {log_path}"
                )
            time.sleep(0.1)

        self._proc = proc
        self.pid = proc.pid
        self.socket_path = socket_path
        self.token = token
        self.log_path = log_path

    def stop(self) -> None:
        if self.pid is None or self.socket_path is None:
            return
        self.stop_external(pid=self.pid, socket_path=str(self.socket_path))
        self._proc = None
        self.pid = None
        self.socket_path = None
        self.token = None

    @classmethod
    def stop_external(
        cls,
        *,
        pid: Optional[int] = None,
        socket_path: Optional[str] = None,
        **_extra: Any,
    ) -> None:
        # server.py installs a SIGTERM handler that os._exit(0)s after
        # unlinking the socket; SIGTERM is enough — no double-flush worries.
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if socket_path and os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Agent runtime wiring
    # ------------------------------------------------------------------

    def agent_mounts_and_env(self) -> tuple[list[Mount], dict[str, str]]:
        """Runtime-agnostic bridge wiring for the OC container.

        Returns the mounts (just bridge.py, read-only) and env vars
        (`OPENCLAW_EXEC_BRIDGE*`) the harness must inject into the OC
        container's `ContainerSpec`. The actual translation to
        `docker run -v/-e` or `singularity --bind/--env` happens inside
        the `Container` subclass.
        """
        if self.token is None:
            raise RuntimeError(
                "OpenClawRemoteDispatcher.agent_mounts_and_env() called "
                "before start()"
            )
        mounts = [
            # bridge.py is read-only — it talks to the socket, doesn't
            # write to itself. OC's patched spawnSpec spawns this file
            # when it sees OPENCLAW_EXEC_BRIDGE in env.
            Mount(host=Path(self.bridge_script), container=_BRIDGE_PATH_IN_OC, mode="ro"),
        ]
        env = {
            # Trigger the patched exec runtime; OC's spawnSpec returns the
            # bridge argv instead of the built-in `["docker", "exec", ...]`.
            "OPENCLAW_EXEC_BRIDGE": _BRIDGE_PATH_IN_OC,
            "OPENCLAW_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET_IN_OC,
            "OPENCLAW_EXEC_BRIDGE_TOKEN": self.token,
        }
        return mounts, env

    def env_info(self) -> Dict[str, Any]:
        if self.pid is None or self.socket_path is None:
            raise RuntimeError(
                "OpenClawRemoteDispatcher.env_info() called before start()"
            )
        return {
            "bridge_pid": self.pid,
            "bridge_socket": str(self.socket_path),
        }
