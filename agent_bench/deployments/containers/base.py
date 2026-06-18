"""Container runtime interface.

`Container` is the seam between application/harness code and the underlying
container runtime (docker, singularity). The harness builds a `ContainerSpec`
once; whichever runtime is injected at driver time decides how to realize it.

Runtime-specific concepts that don't exist on the other side (docker
network, singularity instance log file path) are exposed as no-ops or
optional fields so neither caller needs to branch.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Mount:
    """One bind-mount entry. `mode` is "rw" or "ro"."""
    host: Path
    container: str
    mode: str = "rw"


@dataclass(frozen=True)
class ContainerSpec:
    """Everything `Container.start()` needs to bring up one container.

    docker maps fields directly to `docker run` flags; singularity uses what
    applies (mounts, env) and ignores the rest (`network`, `extra_hosts`,
    `user`, `labels` — singularity has no equivalents and silently no-ops).
    """
    name: str
    image: str
    mounts: tuple[Mount, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ()                 # args after the image
    entrypoint: Optional[str] = None              # override image entrypoint
    workdir: Optional[str] = None
    user: Optional[str] = None                    # docker --user (singularity: no-op)
    network: Optional[str] = None                 # docker --network (singularity: no-op)
    extra_hosts: dict[str, str] = field(default_factory=dict)  # docker --add-host
    labels: dict[str, str] = field(default_factory=dict)       # docker --label
    gpu: bool = False                             # docker --gpus / singularity --nv
    pull_timeout_s: float = 600.0
    cpus: Optional[float] = None                  # docker --cpus (singularity: no-op)


@dataclass
class ContainerHandle:
    """Reference to a running container.

    docker: `id` is the long container id; `name` is the human name we set.
    singularity: `id` and `name` are both the instance name (instances are
    addressed by name). `log_path` is set when the runtime captures bg-exec
    output to a host file (singularity uses it for wait_log_pattern).
    """
    id: str
    name: str
    backend: str
    log_path: Optional[Path] = None


@runtime_checkable
class Container(Protocol):
    """Runtime-agnostic container operations."""

    backend_id: str  # "docker" | "singularity"

    @property
    def host_address(self) -> str:
        """Hostname/IP the *containerised* OC uses to reach the host.

        docker: "vllm-host" (synthetic, mapped via --add-host host-gateway).
        singularity: "127.0.0.1" (singularity shares the host's net namespace
        by default so localhost works).

        Consumed by the harness when rewriting OC's per-run openclaw.json
        endpoints (vllm baseUrl, OTel collector).
        """
        ...

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def check_available(self) -> None: ...
    def check_image(self, image: str) -> None: ...

    # ------------------------------------------------------------------
    # Per-run docker network. Singularity no-ops both.
    # ------------------------------------------------------------------

    def create_network(self, name: str) -> None: ...
    def remove_network(self, name: str) -> None: ...

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def start(self, spec: ContainerSpec) -> ContainerHandle: ...
    def stop(self, h: ContainerHandle) -> None: ...

    # ------------------------------------------------------------------
    # Exec / IO
    # ------------------------------------------------------------------

    def exec(
        self,
        h: ContainerHandle,
        argv: list[str],
        *,
        check: bool = True,
        timeout: Optional[float] = None,
        input_text: Optional[str] = None,
        workdir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess: ...

    def copy_into(self, h: ContainerHandle, src: Path, dst: str) -> None: ...

    def copy_out(
        self,
        h: ContainerHandle,
        src_in_container: str,
        dst_host: Path,
    ) -> None:
        """Copy a file or directory from a running container out to the host.

        Counterpart of `copy_into`. Used by conditions to archive
        in-container artifacts (OC session JSONL, raw LLM responses, …)
        under the per-run dir for post-hoc analysis. Implementations should
        not raise on missing source — callers wrap with try/except and
        only debug-log a miss, since the artifact may legitimately not
        exist (e.g. agent crashed before writing).
        """
        ...

    def copy_from_image(
        self,
        image: str,
        src_in_image: str,
        dst_host: Path,
        *,
        timeout: float = 600.0,
    ) -> None:
        """Extract a directory tree from a not-running image to a host dir."""
        ...

    # ------------------------------------------------------------------
    # Liveness / log waiting
    # ------------------------------------------------------------------

    def is_running(self, h: ContainerHandle) -> bool: ...
    def logs(self, h: ContainerHandle) -> str: ...
    def wait_log_pattern(
        self,
        h: ContainerHandle,
        pattern: re.Pattern,
        timeout_s: float,
        *,
        poll_s: float = 2.0,
    ) -> None:
        """Poll logs until pattern matches. Raise on container death or timeout."""
        ...
