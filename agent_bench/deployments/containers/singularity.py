"""Singularity implementation of the `Container` protocol.

Maps the same interface DockerContainer satisfies onto
`singularity instance start/stop` + `singularity exec instance://<name>`.

Two singularity-specific quirks the docker side doesn't have to think about:

1.  **Keepalive startscript.** `singularity instance` keeps the instance
    alive only as long as its startscript is running. Most sifs converted
    from docker images run the image's CMD as startscript — which often
    exits immediately (e.g. `bash`) or runs an unwanted long process
    (e.g. starts a server we wanted to start ourselves). We bind a tiny
    `exec sleep infinity` script over `/.singularity.d/startscript` to
    guarantee the instance stays up regardless of what the sif ships.
    Our actual command (the OC gateway, etc.) runs as a separate
    `singularity exec` invocation captured to a host log file.

2.  **No docker network / extra_hosts / --user.** Singularity shares the
    host's network namespace by default, so `localhost` inside the
    container reaches the host. `host_address` is `127.0.0.1`. `--user`,
    `--network`, `--add-host`, and `--label` have no equivalents and are
    silently ignored — the harness builds one `ContainerSpec` and these
    fields just don't fire here.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from agent_bench.deployments.containers.base import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)

logger = logging.getLogger(__name__)

# The startscript shim. Runs as the instance's main process; the instance
# stays alive while sleep is running.
_KEEPALIVE_STARTSCRIPT = "#!/bin/sh\nexec sleep infinity\n"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _set_singularity_env(out: dict[str, str], k: str, v: str) -> None:
    """Inject one env var through SINGULARITYENV_*.

    `singularity exec --env` parses commas as assignment separators on some
    NSCC installs; OTEL_RESOURCE_ATTRIBUTES and similar values then become
    broken shell fragments inside /.inject-singularity-env.sh. The parent
    environment form preserves the value verbatim.
    """
    if k == "HOME":
        return
    if not _ENV_NAME_RE.match(k):
        raise ValueError(f"invalid environment variable name for singularity: {k!r}")
    out[f"SINGULARITYENV_{k}"] = str(v)


def _subprocess_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Environment for singularity subprocesses.

    Keep most caller-provided environment intact, but never let
    SINGULARITYENV_HOME leak in from a parent shell. Singularity refuses that
    variable and prints a warning on every exec; HOME is handled explicitly by
    the instance `--home` flag instead.
    """
    env = os.environ.copy()
    env.pop("SINGULARITYENV_HOME", None)
    if extra:
        env.update(extra)
        env.pop("SINGULARITYENV_HOME", None)
    return env


class SingularityContainer(Container):
    backend_id = "singularity"

    @property
    def host_address(self) -> str:
        # Singularity shares the host net namespace; OC inside the
        # container reaches the host via plain localhost.
        return "127.0.0.1"

    def __init__(self, *, log_dir: Optional[Path] = None,
                 keepalive_dir: Optional[Path] = None) -> None:
        """
        log_dir:        where to capture per-instance bg-exec stdout/stderr
                        (used by `wait_log_pattern`). Default:
                        $SCRATCH/.singularity/logs/ or /tmp/.
        keepalive_dir:  where to drop the startscript-override shim. The
                        shim is bound :ro into the instance.
        """
        scratch = os.environ.get("SCRATCH") or os.environ.get("HOME") or "/tmp"
        self._log_dir = Path(log_dir or f"{scratch}/.singularity/logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._keepalive_dir = Path(keepalive_dir or f"{scratch}/.singularity/keepalive")
        self._keepalive_dir.mkdir(parents=True, exist_ok=True)
        # Per-instance bg-exec PIDs, so `stop` can clean them up.
        self._main_pids: dict[str, int] = {}
        # Per-handle start-time env, replayed on every exec() so the
        # singularity backend matches docker semantics: env set at container
        # creation is visible to subsequent execs. `singularity exec
        # instance://<name>` does *not* inherit the instance's start-time
        # env on its own, so we stash and re-pass via SINGULARITYENV_*. (Distinct
        # from `_instance_env()` below, which produces SINGULARITYENV_*
        # for the initial instance start.)
        self._handle_env: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def check_available(self) -> None:
        if shutil.which("singularity") is None:
            raise RuntimeError(
                "singularity not on PATH; on NSCC compute nodes run "
                "`module load singularity` before invoking the driver."
            )

    def check_image(self, image: str) -> None:
        # In singularity mode `image` is an absolute path to a .sif file.
        p = Path(image)
        if not p.is_file():
            raise RuntimeError(
                f"singularity image not found: {image}\n"
                "For agent runtime SIFs: build with "
                "`build_oc_sif.sh`, `build_oh_sif.sh`, or `build_hm_sif.sh` "
                "under `agent_bench/deployments/pbs/`, or enable STAGE_OC_CHECK in "
                "`run_benchmark.pbs`.\n"
                "For SWE-bench task sifs: pull with `agent_bench/deployments/pbs/pull_images.sh`."
            )

    # ------------------------------------------------------------------
    # Network (no-op)
    # ------------------------------------------------------------------

    def create_network(self, name: str) -> None:
        return None

    def remove_network(self, name: str) -> None:
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _home_path(self, name: str) -> Path:
        """Per-instance scratch dir bound over the container's HOME via --home.

        Lives next to the keepalive shim. Empty by default; OC mounts its own
        state_dir at <HOME>/.openclaw via spec.mounts, layered on top.
        """
        p = self._keepalive_dir / f"{name}.home"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _keepalive_path(self, name: str) -> Path:
        p = self._keepalive_dir / f"{name}.startscript"
        if not p.exists():
            p.write_text(_KEEPALIVE_STARTSCRIPT)
            p.chmod(0o755)
        return p

    def _log_path(self, name: str) -> Path:
        return self._log_dir / f"{name}.log"

    def _instance_env(self, env: dict[str, str]) -> dict[str, str]:
        """Translate spec.env into SINGULARITYENV_* on the parent so they
        propagate into the instance's startscript.

        Skip HOME — singularity prints "Overriding HOME … not permitted" and
        ignores it; we set HOME via --home <hostdir>:<containerpath> in start()
        instead.
        """
        out = _subprocess_env()
        for k, v in env.items():
            _set_singularity_env(out, k, v)
        return out

    def start(self, spec: ContainerSpec) -> ContainerHandle:
        self.check_image(spec.image)

        cmd: list[str] = ["singularity", "instance", "start"]
        if spec.gpu:
            cmd += ["--nv"]
        # NSCC compute nodes disable unprivileged user namespaces — without
        # them, `singularity instance start`'s default `mount -t proc proc /proc`
        # fails with EPERM. `--no-mount proc` skips that mount; then we
        # bind-mount host's /proc into the container so reads like
        # /proc/<pid>/statm (Node's process.memoryUsage(), OC's diagnostic
        # memory sampler) keep working. Bind doesn't need user ns.
        cmd += ["--no-mount", "proc", "--bind", "/proc:/proc"]
        # Bind keepalive over the runscript so the instance stays alive
        # regardless of what `spec.command` (or the sif's default CMD) does.
        ka = self._keepalive_path(spec.name)
        cmd += ["--bind", f"{ka}:/.singularity.d/startscript:ro"]
        # singularity refuses SINGULARITYENV_HOME ("Overriding HOME environment
        # variable with SINGULARITYENV_HOME is not permitted"). If spec.env
        # asks for a specific HOME (e.g. OC wants HOME=/root so it reads
        # /root/.openclaw/openclaw.json — same path docker mode uses),
        # bind-mount a scratch host dir over that container path via --home.
        # This both creates the dir inside the container and exports HOME to it.
        target_home = spec.env.get("HOME")
        if target_home:
            host_home = self._home_path(spec.name)
            cmd += ["--home", f"{host_home}:{target_home}"]
        for m in spec.mounts:
            cmd += ["--bind", _bind(m)]
        cmd += [spec.image, spec.name]

        r = subprocess.run(
            cmd,
            env=self._instance_env(spec.env),
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"singularity instance start failed for {spec.name}: "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )

        # Stash for exec() replay (see __init__ comment). Filtering HOME
        # happens on read in exec(), so the rule stays in one place.
        self._handle_env[spec.name] = dict(spec.env)

        log_path = self._log_path(spec.name)
        log_path.write_text("")

        # If the spec carries a main command (e.g. `openclaw gateway`), run
        # it in the background as a separate `singularity exec`. Output is
        # captured for wait_log_pattern.
        if spec.command:
            exec_cmd: list[str] = ["singularity", "exec"]
            if spec.workdir:
                exec_cmd += ["--pwd", spec.workdir]
            exec_env = _subprocess_env()
            for k, v in spec.env.items():
                _set_singularity_env(exec_env, k, v)
            exec_cmd += [f"instance://{spec.name}"]
            if spec.entrypoint:
                exec_cmd += [spec.entrypoint]
            exec_cmd += list(spec.command)

            with open(log_path, "ab") as logf:
                proc = subprocess.Popen(
                    exec_cmd,
                    stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                    start_new_session=True,  # own PGID so we can SIGTERM the tree
                    env=exec_env,
                )
            self._main_pids[spec.name] = proc.pid

        return ContainerHandle(
            id=spec.name, name=spec.name, backend=self.backend_id,
            log_path=log_path,
        )

    def stop(self, h: ContainerHandle) -> None:
        # Kill any bg-exec we tracked (e.g. the OC gateway exec).
        pid = self._main_pids.pop(h.name, None)
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        subprocess.run(
            ["singularity", "instance", "stop", h.name],
            capture_output=True, check=False, timeout=60,
            env=_subprocess_env(),
        )
        self._handle_env.pop(h.name, None)

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
        cmd: list[str] = ["singularity", "exec"]
        if workdir:
            cmd += ["--pwd", workdir]
        # Replay the instance's start-time env (docker-like semantics) and
        # let any caller-supplied env override on conflict.
        merged: dict[str, str] = {}
        stashed = self._handle_env.get(h.name)
        if stashed:
            merged.update(stashed)
        if env:
            merged.update(env)
        exec_env = _subprocess_env()
        for k, v in merged.items():
            _set_singularity_env(exec_env, k, v)
        cmd += [f"instance://{h.name}", *argv]
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, input=input_text,
            env=exec_env,
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"singularity exec failed ({argv[0] if argv else ''}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r

    def copy_into(self, h: ContainerHandle, src: Path, dst: str) -> None:
        """Stream src into the instance via stdin → `cat > dst`.

        Avoids assuming the destination path lives on a bound host dir.
        For binary safety we pipe raw bytes and use base64 inside the
        container to round-trip.
        """
        import base64
        data = Path(src).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        # `printf %s` survives shells that mangle echo (dash, ash).
        # Write to dst.tmp then mv to dst — atomic-ish overwrite.
        tmp_dst = f"{dst}.tmp.{os.getpid()}"
        script = (
            f"printf %s '{b64}' | base64 -d > '{tmp_dst}' && "
            f"mv '{tmp_dst}' '{dst}'"
        )
        self.exec(h, ["sh", "-c", script])

    def copy_out(
        self,
        h: ContainerHandle,
        src_in_container: str,
        dst_host: Path,
    ) -> None:
        """Copy a file or directory tree out of a running instance to host.

        Uses `tar -c | tar -x` over a pipe so we don't have to know up front
        whether `src_in_container` is a file or a directory. The basename of
        src is preserved under dst_host's parent (mirrors `docker cp`'s
        behavior).
        """
        dst_host = Path(dst_host)
        # Tar in the instance from the parent dir, naming just the basename;
        # extract into dst_host's parent. This puts <dst_host_parent>/<basename>
        # on disk, then we rename to dst_host if names differ.
        src = src_in_container.rstrip("/")
        src_parent = os.path.dirname(src) or "/"
        src_base = os.path.basename(src)
        if not src_base:
            raise RuntimeError(f"copy_out: refusing to copy root path {src_in_container!r}")

        dst_host.parent.mkdir(parents=True, exist_ok=True)
        # Extract into a temp dir so we can rename atomically into dst_host
        # regardless of whether src is a file or directory.
        tmp_extract = Path(tempfile.mkdtemp(prefix="oc-copyout-", dir=str(dst_host.parent)))
        try:
            producer = subprocess.Popen(
                ["singularity", "exec", f"instance://{h.name}",
                 "tar", "-C", src_parent, "-cf", "-", src_base],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_subprocess_env(),
            )
            consumer = subprocess.Popen(
                ["tar", "-C", str(tmp_extract), "-xf", "-"],
                stdin=producer.stdout, stderr=subprocess.PIPE,
            )
            if producer.stdout is not None:
                producer.stdout.close()
            consumer_err = consumer.communicate(timeout=300)[1]
            producer_err = producer.communicate(timeout=10)[1]
            if producer.returncode != 0:
                raise RuntimeError(
                    f"copy_out: in-container tar -c failed for {src!r}: "
                    f"{(producer_err or b'').decode(errors='replace')[-500:]}"
                )
            if consumer.returncode != 0:
                raise RuntimeError(
                    f"copy_out: host tar -x failed: "
                    f"{(consumer_err or b'').decode(errors='replace')[-500:]}"
                )

            extracted = tmp_extract / src_base
            if not extracted.exists():
                raise RuntimeError(f"copy_out: extraction missing {extracted}")

            # Move into place. If dst_host already exists, replace it (mirrors
            # docker cp -a behavior on overwrite).
            if dst_host.exists():
                if dst_host.is_dir():
                    shutil.rmtree(dst_host)
                else:
                    dst_host.unlink()
            shutil.move(str(extracted), str(dst_host))
        finally:
            if tmp_extract.exists():
                shutil.rmtree(tmp_extract, ignore_errors=True)

    def copy_from_image(
        self,
        image: str,
        src_in_image: str,
        dst_host: Path,
        *,
        timeout: float = 600.0,
    ) -> None:
        """Pipe `tar -cf -` from inside the sif into a host `tar -xf -`."""
        self.check_image(image)
        dst_host.mkdir(parents=True, exist_ok=True)
        src = src_in_image.rstrip("/")
        producer = subprocess.Popen(
            ["singularity", "exec", image,
             "tar", "-C", src, "-cf", "-", "."],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_subprocess_env(),
        )
        consumer = subprocess.Popen(
            ["tar", "-C", str(dst_host), "-xf", "-"],
            stdin=producer.stdout, stderr=subprocess.PIPE,
        )
        if producer.stdout is not None:
            # let consumer own the read end
            producer.stdout.close()
        try:
            consumer_err = consumer.communicate(timeout=timeout)[1]
            producer_err = producer.communicate(timeout=10)[1]
        except subprocess.TimeoutExpired:
            consumer.kill()
            producer.kill()
            raise RuntimeError(
                f"copy_from_image timed out after {timeout}s "
                f"({image}:{src_in_image} → {dst_host})"
            )
        if consumer.returncode != 0:
            raise RuntimeError(
                f"copy_from_image: host tar -x failed: {(consumer_err or b'').decode(errors='replace')[-500:]}"
            )
        if producer.returncode != 0:
            raise RuntimeError(
                f"copy_from_image: in-sif tar -c failed: {(producer_err or b'').decode(errors='replace')[-500:]}"
            )

    # ------------------------------------------------------------------
    # Liveness / log waiting
    # ------------------------------------------------------------------

    def is_running(self, h: ContainerHandle) -> bool:
        r = subprocess.run(
            ["singularity", "instance", "list", "--json", h.name],
            capture_output=True, text=True,
            env=_subprocess_env(),
        )
        if r.returncode != 0:
            return False
        try:
            data = json.loads(r.stdout or "{}")
        except json.JSONDecodeError:
            return False
        return bool(data.get("instances"))

    def logs(self, h: ContainerHandle) -> str:
        if h.log_path is None or not h.log_path.exists():
            return ""
        return h.log_path.read_text(errors="replace")

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
                    f"singularity instance {h.name} exited before log pattern matched. "
                    f"last log:\n{last_tail}"
                )
            time.sleep(poll_s)
        raise RuntimeError(
            f"singularity instance {h.name} did not match log pattern within "
            f"{timeout_s}s. last log:\n{last_tail}"
        )


def _bind(m: Mount) -> str:
    return f"{m.host}:{m.container}:{m.mode}"
