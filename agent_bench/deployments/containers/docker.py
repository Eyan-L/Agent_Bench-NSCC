"""Docker implementation of the `Container` protocol.

Lifts the `subprocess.run(["docker", ...])` call sites that used to live
directly in `agent_bench/benchmarks/swebench/openclaw/app.py:SWEBenchHarness`. The
argv shapes are preserved verbatim so docker behavior is unchanged after
the refactor.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from agent_bench.deployments.containers.base import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)
from agent_bench.deployments.containers.limits import docker_cpus_args
from agent_bench.telemetry.harness_spans import (
    preview_json,
    preview_text,
    write_harness_span,
)

logger = logging.getLogger(__name__)


class DockerContainer(Container):
    backend_id = "docker"
    _trace_local = threading.local()

    # OC inside the container talks to the host via the synthetic
    # `host-gateway` hostname added by `--add-host vllm-host:host-gateway`.
    # The harness uses this when rewriting per-run openclaw.json endpoints.
    _HOST_ADDRESS = "vllm-host"

    @property
    def host_address(self) -> str:
        return self._HOST_ADDRESS

    def set_trace_path(self, path: Optional[Path]) -> None:
        self._trace_local.path = Path(path) if path is not None else None

    def _trace_path(self) -> Optional[Path]:
        return getattr(self._trace_local, "path", None)

    def _run_traced(
        self,
        span_name: str,
        cmd: list[str],
        *,
        attrs: Optional[dict] = None,
        timeout: Optional[float] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        start_ns = time.time_ns()
        cmd_json, cmd_chars, cmd_truncated = preview_json(cmd, limit=8000)
        span_attrs = {
            "agent_bench.channel": "harness",
            "agent_bench.command_json": cmd_json,
            "agent_bench.command_chars": cmd_chars,
            "agent_bench.command_truncated": cmd_truncated,
        }
        if attrs:
            span_attrs.update(attrs)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input_text,
            )
            stdout, stdout_chars, stdout_truncated = preview_text(
                result.stdout, limit=4000,
            )
            stderr, stderr_chars, stderr_truncated = preview_text(
                result.stderr, limit=4000,
            )
            span_attrs.update({
                "agent_bench.returncode": result.returncode,
                "agent_bench.stdout_preview": stdout,
                "agent_bench.stdout_chars": stdout_chars,
                "agent_bench.stdout_truncated": stdout_truncated,
                "agent_bench.stderr_preview": stderr,
                "agent_bench.stderr_chars": stderr_chars,
                "agent_bench.stderr_truncated": stderr_truncated,
            })
            return result
        except subprocess.TimeoutExpired as e:
            stdout, stdout_chars, stdout_truncated = preview_text(
                e.stdout, limit=4000,
            )
            stderr, stderr_chars, stderr_truncated = preview_text(
                e.stderr, limit=4000,
            )
            span_attrs.update({
                "agent_bench.timeout": True,
                "agent_bench.timeout_seconds": timeout,
                "agent_bench.stdout_preview": stdout,
                "agent_bench.stdout_chars": stdout_chars,
                "agent_bench.stdout_truncated": stdout_truncated,
                "agent_bench.stderr_preview": stderr,
                "agent_bench.stderr_chars": stderr_chars,
                "agent_bench.stderr_truncated": stderr_truncated,
            })
            raise
        finally:
            write_harness_span(
                self._trace_path(),
                name=span_name,
                attrs=span_attrs,
                start_ns=start_ns,
            )

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def check_available(self) -> None:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")

    def check_image(self, image: str) -> None:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"docker image not found: {image}\n{r.stderr.strip()}"
            )

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def create_network(self, name: str) -> None:
        r = self._run_traced(
            "agent_bench.container.network.create",
            ["docker", "network", "create", name],
            attrs={"agent_bench.container.network": name},
        )
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

    def remove_network(self, name: str) -> None:
        # Sweep stragglers attached to the network, then drop it.
        r = self._run_traced(
            "agent_bench.container.network.list",
            ["docker", "ps", "-aq", "--filter", f"network={name}"],
            attrs={"agent_bench.container.network": name},
        )
        for cid in (r.stdout or "").split():
            self._run_traced(
                "agent_bench.container.stop",
                ["docker", "rm", "-f", "-v", cid],
                attrs={
                    "agent_bench.container.id": cid,
                    "agent_bench.container.network": name,
                },
            )
        self._run_traced(
            "agent_bench.container.network.remove",
            ["docker", "network", "rm", name],
            attrs={"agent_bench.container.network": name},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, spec: ContainerSpec) -> ContainerHandle:
        cmd: list[str] = ["docker", "run", "-d", "--name", spec.name]
        if spec.network:
            cmd += ["--network", spec.network]
        for host, ip in spec.extra_hosts.items():
            cmd += ["--add-host", f"{host}:{ip}"]
        if spec.user is not None:
            cmd += ["--user", spec.user]
        if spec.gpu:
            cmd += ["--gpus", "all"]
        cmd += docker_cpus_args(spec.cpus)
        for k, v in spec.labels.items():
            cmd += ["--label", f"{k}={v}" if v else k]
        for m in spec.mounts:
            cmd += ["-v", _bind(m)]
        if spec.workdir:
            cmd += ["-w", spec.workdir]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        if spec.entrypoint:
            cmd += ["--entrypoint", spec.entrypoint]
        cmd += [spec.image]
        cmd += list(spec.command)

        r = self._run_traced(
            "agent_bench.container.start",
            cmd,
            timeout=spec.pull_timeout_s,
            attrs={
                "agent_bench.container.name": spec.name,
                "agent_bench.container.image": spec.image,
                "agent_bench.container.network": spec.network,
                "agent_bench.container.workdir": spec.workdir,
                "agent_bench.container.entrypoint": spec.entrypoint,
            },
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {spec.name}: {r.stderr.strip()}"
            )
        container_id = r.stdout.strip()
        return ContainerHandle(id=container_id, name=spec.name, backend=self.backend_id)

    def stop(self, h: ContainerHandle) -> None:
        # `docker rm -f -v` does both: SIGKILL + remove + drop anon vols.
        target = h.id or h.name
        self._run_traced(
            "agent_bench.container.stop",
            ["docker", "rm", "-f", "-v", target],
            attrs={
                "agent_bench.container.id": h.id,
                "agent_bench.container.name": h.name,
            },
        )

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
    ) -> subprocess.CompletedProcess:
        cmd: list[str] = ["docker", "exec"]
        if workdir:
            cmd += ["-w", workdir]
        if env:
            for k, v in env.items():
                cmd += ["-e", f"{k}={v}"]
        cmd += [h.id, *argv]
        r = self._run_traced(
            "agent_bench.container.exec",
            cmd,
            timeout=timeout,
            input_text=input_text,
            attrs={
                "agent_bench.container.id": h.id,
                "agent_bench.container.name": h.name,
                "agent_bench.container.workdir": workdir,
                "agent_bench.exec.argv_json": preview_json(argv, limit=8000)[0],
            },
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({argv[0] if argv else ''}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r

    def copy_into(self, h: ContainerHandle, src: Path, dst: str) -> None:
        r = self._run_traced(
            "agent_bench.container.copy_into",
            ["docker", "cp", str(src), f"{h.id}:{dst}"],
            attrs={
                "agent_bench.container.id": h.id,
                "agent_bench.container.name": h.name,
                "agent_bench.copy.src": str(src),
                "agent_bench.copy.dst": dst,
            },
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker cp failed: {r.stderr.strip()}")

    def copy_out(
        self,
        h: ContainerHandle,
        src_in_container: str,
        dst_host: Path,
    ) -> None:
        dst_host.parent.mkdir(parents=True, exist_ok=True)
        r = self._run_traced(
            "agent_bench.container.copy_out",
            ["docker", "cp", f"{h.id}:{src_in_container}", str(dst_host)],
            attrs={
                "agent_bench.container.id": h.id,
                "agent_bench.container.name": h.name,
                "agent_bench.copy.src": src_in_container,
                "agent_bench.copy.dst": str(dst_host),
            },
        )
        if r.returncode != 0:
            # Caller decides how loud to be (conditions just debug-log).
            raise RuntimeError(f"docker cp out failed: {r.stderr.strip()}")

    def copy_from_image(
        self,
        image: str,
        src_in_image: str,
        dst_host: Path,
        *,
        timeout: float = 600.0,
    ) -> None:
        scratch_name = f"oc-eval-prime-{dst_host.name[:12]}-{os.urandom(2).hex()}"
        r = self._run_traced(
            "agent_bench.container.image.create_scratch",
            ["docker", "create", "--name", scratch_name,
             "--entrypoint", "true", image],
            timeout=timeout,
            attrs={
                "agent_bench.container.name": scratch_name,
                "agent_bench.container.image": image,
            },
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker create scratch failed: {r.stderr.strip()}")
        try:
            # `:/testbed/.` → flatten contents into dst (without the wrapper dir).
            r = self._run_traced(
                "agent_bench.container.copy_from_image",
                ["docker", "cp", f"{scratch_name}:{src_in_image.rstrip('/')}/.",
                 str(dst_host).rstrip("/") + "/"],
                timeout=timeout,
                attrs={
                    "agent_bench.container.name": scratch_name,
                    "agent_bench.container.image": image,
                    "agent_bench.copy.src": src_in_image,
                    "agent_bench.copy.dst": str(dst_host),
                },
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"docker cp from {image}:{src_in_image} failed: {r.stderr.strip()}"
                )
        finally:
            self._run_traced(
                "agent_bench.container.stop",
                ["docker", "rm", "-f", scratch_name],
                attrs={
                    "agent_bench.container.name": scratch_name,
                    "agent_bench.container.image": image,
                },
            )

    # ------------------------------------------------------------------
    # Liveness / log waiting
    # ------------------------------------------------------------------

    def is_running(self, h: ContainerHandle) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", h.id],
            capture_output=True, text=True,
        )
        return (r.stdout or "").strip() == "true"

    def logs(self, h: ContainerHandle) -> str:
        r = subprocess.run(
            ["docker", "logs", h.id],
            capture_output=True, text=True, timeout=30,
        )
        return (r.stdout or "") + (r.stderr or "")

    def wait_log_pattern(
        self,
        h: ContainerHandle,
        pattern: re.Pattern,
        timeout_s: float,
        *,
        poll_s: float = 2.0,
    ) -> None:
        deadline = time.time() + timeout_s
        last_tail = ""
        while time.time() < deadline:
            log_text = self.logs(h)
            if pattern.search(log_text):
                return
            last_tail = log_text[-1500:]
            if not self.is_running(h):
                raise RuntimeError(
                    f"container {h.name} exited before log pattern matched. "
                    f"last log:\n{last_tail}"
                )
            time.sleep(poll_s)
        raise RuntimeError(
            f"container {h.name} did not match log pattern within "
            f"{timeout_s}s. last log:\n{last_tail}"
        )


def _bind(m: Mount) -> str:
    return f"{m.host}:{m.container}:{m.mode}"
