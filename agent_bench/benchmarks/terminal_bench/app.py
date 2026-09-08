"""Terminal-Bench harnesses (Docker + Singularity).

Two backends, same surface as ``agent_bench/benchmarks/swebench/app.py``:

  - ``TerminalBenchOpenClawHarness``     — container-agnostic OpenClaw path.
                                           It starts the OC runtime and task
                                           runtime through an injected Container,
                                           with the exec bridge targeting task.
  - ``TerminalBenchDockerHarness`` and ``TerminalBenchSingularityHarness``
                                           — legacy task-only paths retained for
                                           the ``tb_react`` fallback.

OpenClaw architecture (Docker or Singularity):

    host  ──── only vLLM runs here ──── http://localhost:8000/v1
                                  ▲
                                  │
    ┌─────────────────────────────────────┐
    │ oc-<rid>   agent-bench/openclaw:<v> │  (optional; only when the
    │ - openclaw gateway                  │   harness config has an
    │ - exec-bridge client                │   `openclaw` block — i.e. the
    └────────┬────────────────────────────┘   condition is openclaw_agent)
             │
             ↓
    ┌─────────────────────────────────────┐
    │ openclaw-sandbox-<sess>             │
    │ image: <task.toml docker_image>     │
    │ - task WORKDIR (rw, workspace)      │
    │ - /logs/verifier (rw, reward.txt)   │
    └─────────────────────────────────────┘

For the ``tb_react`` condition (no OC), the harness still brings up the task
container but skips the OC container + bridge; ``env_info["task_container_id"]``
is what the condition's docker-exec calls use.

Scoring isolation:
    /tests is not mounted while the agent is solving the task. run_tests copies
    the vendored tests into the live task container immediately before running
    the verifier. /logs/verifier is bind-mounted rw so the verifier can write
    /logs/verifier/reward.txt as its last step.
"""
from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent_bench.benchmarks.conditions.gateway_helpers import (
    pick_free_gateway_port as _pick_free_gateway_port,
    release_gateway_port as _release_gateway_port,
)
from agent_bench.benchmarks.swebench.app import (
    DEFAULT_OPENCLAW_RUNTIME_IMAGE,
    _force_rmtree,
    _oc_sandbox_name,
)
from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher
from agent_bench.core.config import runtime_endpoint_for_container, load_active_model_and_serving
from agent_bench.core.profile_io import load_profile
from agent_bench.deployments.containers.limits import (
    configured_task_container_cpus,
    docker_cpus_args,
)
from agent_bench.deployments.containers import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)

logger = logging.getLogger(__name__)


# Serialize only the OpenClaw gateway warmup window. This mirrors the
# proven SWE-bench NSCC path: plugin installation is serialized, while
# agent execution remains concurrent after the gateway becomes ready.
_OPENCLAW_WARMUP_SEM = threading.BoundedSemaphore(
    int(os.environ.get("AB_HARNESS_WARMUP_CONCURRENCY", "1"))
)


# ----------------------------------------------------------------------
# Workspace helpers — Terminal-Bench tasks normally use /app, but a few task
# images set a different WORKDIR. Keep every harness on the task/image workdir
# so priming, binds, cwd, and scoring all see the same filesystem root.
# ----------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"


def _terminal_bench_profile_section() -> Dict:
    """Return adapter-specific settings from the active harness profile."""
    harness_path = os.environ.get("AB_HARNESS_CONFIG")
    if not harness_path or not os.path.isfile(harness_path):
        return {}
    return dict(load_profile(harness_path).get("terminal_bench") or {})


def _terminal_bench_prompt_filename(field: str, default: str) -> str:
    """Resolve one prompt asset selected by the active harness profile."""
    filename = str(_terminal_bench_profile_section().get(field) or default)
    if Path(filename).name != filename:
        raise ValueError(
            f"terminal_bench.{field} must be a filename, got {filename!r}"
        )
    return filename


def _terminal_bench_profile_dir(
    field: str,
    *,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve an optional repo-relative prompt or skill directory."""
    raw = _terminal_bench_profile_section().get(field)
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        root = Path(
            repo_root
            or os.environ.get("AB_REPO_ROOT")
            or _THIS_DIR.parents[2]
        )
        path = root / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"terminal_bench.{field} directory missing: {path}")
    return path


def _terminal_bench_agent_assets_dir(
    *, repo_root: Optional[Path] = None
) -> Path:
    return _terminal_bench_profile_dir(
        "agent_assets_dir", repo_root=repo_root
    ) or (_THIS_DIR / "openclaw" / "agents")


def _normalize_container_path(raw: Optional[str], default: str = "/app") -> str:
    path = (raw or "").strip() or default
    if not path.startswith("/"):
        path = f"/{path}"
    normalized = posixpath.normpath(path)
    if normalized in {"", "."}:
        return default
    return normalized


def _docker_image_workdir(docker_image: str) -> Optional[str]:
    try:
        r = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                docker_image,
                "--format",
                "{{.Config.WorkingDir}}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("docker image workdir inspect failed for %s: %s", docker_image, exc)
        return None
    if r.returncode != 0:
        logger.debug(
            "docker image workdir inspect failed for %s: %s",
            docker_image,
            r.stderr.strip(),
        )
        return None
    workdir = (r.stdout or "").strip()
    if not workdir or workdir == "<no value>":
        return None
    return _normalize_container_path(workdir)


def _resolve_task_workspace(
    instance_data: Optional[Dict],
    image_workdir: Optional[str] = None,
) -> str:
    """Resolve the task container workspace for both new and old datasets.

    New materialized Terminal-Bench rows carry `workspace_source`. Old rows may
    have the loader's previous `/app` default even when the image WORKDIR is
    `/workspace`, so image metadata is allowed to override only that legacy
    default shape.
    """
    instance_data = instance_data or {}
    workspace = instance_data.get("workspace")
    source = str(instance_data.get("workspace_source") or "")
    image_workspace = (
        _normalize_container_path(image_workdir)
        if image_workdir and str(image_workdir).strip() not in {"/", "."}
        else None
    )

    if source == "task.toml" and workspace:
        return _normalize_container_path(str(workspace))
    if image_workspace:
        return image_workspace
    if workspace and str(workspace).strip() != "/app":
        return _normalize_container_path(str(workspace))
    if workspace:
        return _normalize_container_path(str(workspace))
    return "/app"


def _require_tests_host_path(instance_data: Optional[Dict]) -> Path:
    if not instance_data:
        raise RuntimeError(
            "TerminalBench scoring requires instance_data with tests_host_path."
        )
    raw = instance_data.get("tests_host_path")
    if not raw:
        raise RuntimeError(
            "TerminalBench scoring requires instance_data['tests_host_path']."
        )
    tests_host = Path(raw)
    if not tests_host.is_dir():
        raise FileNotFoundError(f"TerminalBench tests directory missing: {tests_host}")
    return tests_host


def _clear_reward_file(host_logs_dir: Path) -> Path:
    host_logs_dir.mkdir(parents=True, exist_ok=True)
    host_reward = host_logs_dir / "reward.txt"
    host_reward.unlink(missing_ok=True)
    return host_reward


def _terminal_bench_reward(raw: str, *, timed_out: bool) -> tuple[str, Optional[float]]:
    """Interpret binary Terminal-Bench and fractional RoadmapBench rewards."""
    if timed_out:
        return ("TIMEOUT", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return ("ERROR", None)
    if not 0.0 <= value <= 1.0:
        return ("ERROR", None)
    return ("PASS" if value >= 1.0 else "FAIL", value)


def _as_text(value) -> str:
    """Normalize subprocess output for JSON-safe verifier metadata."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _render_workspace_doc_text(text: str, task_workspace: str) -> str:
    workspace = _normalize_container_path(task_workspace)
    if workspace == "/app":
        return text
    return text.replace("/app", workspace)


def _seed_workspace_helpers(
    app_dir: Path,
    instance_data: Dict,
    task_workspace: str = "/app",
) -> None:
    """Materialize configured prompt assets in the task workspace bind.

    Terminal-Bench keeps its historical BOOTSTRAP-only behavior. Benchmark
    profiles can opt into additional workspace files with
    ``terminal_bench.workspace_asset_files``. This lets RoadmapBench replace
    OpenClaw's generated AGENTS/IDENTITY/TOOLS context without changing the
    natural Terminal-Bench prompt.

    ``app_dir`` is the host side of the resolved task workspace bind, so files
    must be written directly below it even when the container path is not
    ``/app``. Document contents are rendered with that container path.
    """
    try:
        section = _terminal_bench_profile_section()
        configured = section.get("workspace_asset_files")
        if configured is None:
            source_names = [
                _terminal_bench_prompt_filename(
                    "workspace_bootstrap_file", "BOOTSTRAP.md"
                )
            ]
            destination_names = ["BOOTSTRAP.md"]
        else:
            if not isinstance(configured, list) or not configured:
                raise ValueError(
                    "terminal_bench.workspace_asset_files must be a non-empty list"
                )
            source_names = [str(name) for name in configured]
            destination_names = list(source_names)

        assets_dir = _terminal_bench_agent_assets_dir()
        rendered = []
        for source_name, destination_name in zip(source_names, destination_names):
            if (
                not source_name
                or Path(source_name).name != source_name
                or Path(destination_name).name != destination_name
            ):
                raise ValueError(
                    "workspace asset names must be plain filenames: "
                    f"{source_name!r}"
                )
            source = assets_dir / source_name
            if not source.is_file():
                raise FileNotFoundError(f"workspace asset not found: {source}")
            rendered.append((
                destination_name,
                _render_workspace_doc_text(source.read_text(), task_workspace),
            ))

        app_dir.mkdir(parents=True, exist_ok=True)
        for destination_name, text in rendered:
            (app_dir / destination_name).write_text(text)
    except OSError as e:
        logger.warning("workspace prompt asset materialization failed: %s", e)


# ----------------------------------------------------------------------
# Docker harness
# ----------------------------------------------------------------------


class TerminalBenchDockerHarness:
    """Two-container docker harness for Terminal-Bench tasks.

    Mirrors ``SWEBenchHarness`` end-to-end: per-run network, pre-started task
    container with the OC-predicted name, exec-bridge dispatcher on the host,
    optional OpenClaw container with state_dir + gateway. The only TB-specific
    pieces are the solve-time bind layout (task/image workdir +
    ``/logs/verifier`` instead of ``/testbed``), scoring-time test injection
    into ``/tests``, and the scoring path (``bash /tests/test.sh`` →
    ``reward.txt`` instead of SWE-bench's ``run_evaluation``).

    OC is **optional**: the harness only starts it when the active harness
    config carries an ``openclaw`` block (which signals an OC-driven condition
    like ``openclaw_agent``). The ``tb_react`` condition harness config has no
    such block, so the OC container + bridge are skipped and only the task
    container is brought up.
    """

    backend_id = "docker"

    _GATEWAY_READY_RE = re.compile(
        r"\[gateway\][^\n]{0,80}(ready|listening on )", re.IGNORECASE,
    )
    _GATEWAY_READY_TIMEOUT_SECONDS = 180
    _DEFAULT_VERIFIER_TIMEOUT_SECONDS = 900

    def __init__(self, benchmark_data_root: Optional[Path] = None,
                 oc_image: Optional[str] = None,
                 workspace_base: Optional[Path] = None):
        # benchmark_data_root accepted for driver-API parity; unused by TB.
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.oc_image = oc_image or DEFAULT_OPENCLAW_RUNTIME_IMAGE

        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")

        # OC image is only required when the harness config asks for it; we
        # check lazily in prepare_environment so a pure tb_react run on a
        # machine without the OC image still works.
        self.task_container_cpus = configured_task_container_cpus()

        self.workspace_base = Path(
            workspace_base
            or os.environ.get("AB_HARNESS_WORKSPACE_BASE",
                              os.path.expanduser("~/.cache/oc-eval/workspaces"))
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _has_openclaw_block(harness_cfg_path: Optional[str]) -> bool:
        if not harness_cfg_path or not os.path.isfile(harness_cfg_path):
            return False
        try:
            return bool(load_profile(harness_cfg_path).get("openclaw"))
        except Exception:
            return False

    @staticmethod
    def _agent_id_from_harness_cfg() -> Optional[str]:
        """Pull the first registered agent's id out of the active harness
        config's `openclaw.agents.list[]`. Returns None when the config has
        no openclaw block or no agents."""
        p = os.environ.get("AB_HARNESS_CONFIG")
        if not p or not os.path.isfile(p):
            return None
        try:
            cfg = load_profile(p)
        except Exception:
            return None
        agents = ((cfg.get("openclaw") or {}).get("agents") or {}).get("list") or []
        for a in agents:
            if isinstance(a, dict) and a.get("id"):
                return str(a["id"])
        return None

    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,                           # ignored — TB
        base_commit: str,                       # ignored — TB
        container_image: Optional[str] = None,  # ignored — read from task.toml
        workspace_root: Optional[str] = None,   # ignored — TB uses task/image workdir
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                "TerminalBenchDockerHarness.prepare_environment requires "
                "instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchDockerHarness only handles benchmark=terminal_bench "
                f"(got benchmark={instance_data.get('benchmark')!r})"
            )

        docker_image = instance_data["docker_image"]
        image_workdir = _docker_image_workdir(docker_image)
        task_workspace = _resolve_task_workspace(instance_data, image_workdir)

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"

        # Predict OC's sandbox container name — same algorithm as SWE-bench so
        # an OC-driven TB condition (openclaw_agent) finds the task container
        # as a "hot container" instead of spawning a new one.
        #
        # Agent id MUST match three places: (1) the harness config's
        # `openclaw.agents.list[0].id`, (2) the dirname under state_dir/agents/
        # where prompts are dropped, (3) what the condition passes to
        # `openclaw agent --agent <id>` (TerminalBenchOpenClawCondition uses
        # "terminal_bench"). Pull from OC config when present so a custom
        # harness can override.
        sandbox_scope = "agent"
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        logger.info(
            "preflight cleanup for TB OpenClaw run resources: oc=%s task=%s net=%s",
            oc_name, task_container_name, net_name,
        )
        self._best_effort_cleanup(net_name, oc_name, task_container_name)

        # Per-run host dirs: task workspace + /logs/verifier (where reward.txt
        # lands) bind targets. The host directory remains named "app" for
        # backward-compatible env_info/cleanup paths; its container target is
        # task_workspace.
        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        app_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Whether to bring up the OC container + bridge — driven by whether
        # the active harness config has an `openclaw` block.
        harness_cfg_path = os.environ.get("AB_HARNESS_CONFIG")
        use_openclaw = self._has_openclaw_block(harness_cfg_path)
        if use_openclaw:
            r = subprocess.run(
                ["docker", "image", "inspect", self.oc_image],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"openclaw image not found: {self.oc_image}\n"
                    "Build it first: bash agent_bench/harnesses/openclaw/docker/build.sh"
                )

        logger.info(
            "Preparing TB run %s (task image=%s, task_workspace=%s, oc=%s, use_openclaw=%s)",
            run_id, docker_image, task_workspace,
            self.oc_image if use_openclaw else "(skipped)", use_openclaw,
        )

        # 1. Per-run docker network.
        r = subprocess.run(
            ["docker", "network", "create", net_name],
            capture_output=True, text=True,
        )
        if r.returncode != 0 and "already exists" not in (r.stderr or ""):
            raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

        # 2. (Optional) prime the task workspace from the task image so
        #    starter files at WORKDIR are not masked by the empty bind.
        try:
            self._prime_workspace_from_image(docker_image, task_workspace, app_dir)
        except Exception as e:
            logger.warning("TB workspace prime skipped: %s", e)

        # 2b. Seed BOOTSTRAP.md into the task workspace bind. BOOTSTRAP.md
        #     silences OC's "[Bootstrap pending] please read BOOTSTRAP.md"
        #     injection (system-prompt-BDZ2g8e3.js:133+) and replaces it
        #     with explicit TB path rules.
        _seed_workspace_helpers(app_dir, instance_data, task_workspace)

        # 3. (Optional) per-run state_dir for OC — only when OC is in play.
        state_dir: Optional[Path] = None
        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        if use_openclaw:
            state_dir = self._build_state_dir(
                run_id=run_id,
                workspace_root=task_workspace,
                sandbox_image=docker_image,
                sandbox_binds=[
                    f"{app_dir}:{task_workspace}:rw",
                    f"{logs_dir}:/logs/verifier:rw",
                ],
                container_prefix=sandbox_prefix,
                scope=sandbox_scope,
            )

        # 4. Exec-bridge dispatcher (only meaningful when OC is up; we still
        #    start it so condition-side code that consults env_info["bridge_*"]
        #    has consistent shape).
        if use_openclaw:
            dispatcher = OpenClawRemoteDispatcher()
            try:
                dispatcher.start(
                    state_dir=state_dir,
                    backend=self.backend_id,
                    target=task_container_name,
                )
            except BaseException:
                self._best_effort_cleanup(net_name, oc_name, task_container_name)
                raise

        # 5. Start the TB task container with the OC-predicted name and the
        #    TB bind layout. Use `--user 0` so pip installs in test.sh land
        #    in the system site-packages (kv-store-grpc tests rely on this).
        task_run_cmd = [
            "docker", "run", "-d",
            "--name", task_container_name,
            "--network", net_name,
            "--user", "0",
            "--label", "openclaw.sandbox=1",
            "--label", f"openclaw.sessionKey={scope_key}",
            "--label", f"oc-eval.run_id={run_id}",
            "-v", f"{app_dir}:{task_workspace}:rw",
            "-v", f"{logs_dir}:/logs/verifier:rw",
            "-w", task_workspace,
            *docker_cpus_args(self.task_container_cpus),
            "--entrypoint", "sleep",
            docker_image,
            "infinity",
        ]
        try:
            r = subprocess.run(task_run_cmd, capture_output=True, text=True, timeout=600)
        except BaseException:
            if dispatcher:
                dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise
        if r.returncode != 0:
            if dispatcher:
                dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise RuntimeError(
                f"task env container start failed: {r.stderr.strip()}"
            )
        task_container_id = r.stdout.strip()

        env_info: Dict = {
            # `container_id` is the field the driver passes positionally into
            # condition.run(). For tb_react this is the TB task container so
            # the existing handlers in driver.py just work; for openclaw_agent
            # we overwrite it below with the OC container id (same as SWE-bench).
            "container_id": task_container_id,
            "container_name": task_container_name,
            "backend": "docker",
            "task_container_id": task_container_id,
            "task_container_name": task_container_name,
            "network_name": net_name,
            "workspace_root": task_workspace,
            "task_workspace_root": task_workspace,
            "workspace_host_path": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "sandbox_image": docker_image,
            # TB-specific echo (so conditions don't re-parse task.toml)
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get(
                "verifier_entrypoint", "/tests/test.sh"
            ),
            "tb_reward_file": instance_data.get(
                "reward_file", "/logs/verifier/reward.txt"
            ),
            # For react.py: tells it which engine to use for exec.
            "container_engine": "docker",
            "instance_name": task_container_name,
        }

        if not use_openclaw:
            logger.info(
                "  TB task container up: %s (%s) — OC skipped (no `openclaw` "
                "block in harness config)",
                task_container_name, task_container_id[:12],
            )
            return env_info

        # 6. Start the OC container (mirrors SWEBenchHarness).
        gateway_port = _pick_free_gateway_port()
        gateway_token = secrets.token_hex(24)
        assert dispatcher is not None and state_dir is not None  # for type-checkers

        # Wire the bridge explicitly, matching the SWE-bench harness pattern.
        # Without these env vars, OpenClaw falls back to its built-in Docker
        # tools and bridge-side tool-call logging never runs.
        bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
        bridge_args: List[str] = []
        for mount in bridge_mounts:
            bridge_args += ["-v", f"{mount.host}:{mount.container}:{mount.mode}"]
        for key, value in bridge_env.items():
            bridge_args += ["-e", f"{key}={value}"]

        cmd = [
            "docker", "run", "-d",
            "--name", oc_name,
            "--network", net_name,
            "--add-host", "vllm-host:host-gateway",
            "--user", "0",
            "-v", f"{state_dir}:/root/.openclaw:rw",
            # NB: don't bind {app_dir}:/app on the OC container — the OC image
            # ships `/usr/local/bin/openclaw -> /app/openclaw.mjs`, so a bind
            # over /app masks the binary and the entrypoint fails. The agent's
            # workspace lives in the task container (sandbox); OC reaches it
            # via the bridge, not via a host bind.
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-w", "/root",
            "-e", f"OPENCLAW_GATEWAY_TOKEN={gateway_token}",
            "-e", f"OPENCLAW_GATEWAY_OTEL_ENDPOINT={os.environ.get('AB_TELEMETRY_ENDPOINT', 'http://vllm-host:4318')}",
            "-e", f"OPENCLAW_RUN_ID={run_id}",
            "-e", "OPENCLAW_RAW_RESPONSE_LOG=/root/.openclaw/logs/llm-raw-responses.jsonl",
            "-e", f"OTEL_RESOURCE_ATTRIBUTES=oc.run_id={run_id}",
            "-e", "DOCKER_API_VERSION=1.44",
            *bridge_args,
            "--entrypoint", "/usr/local/bin/openclaw",
            self.oc_image,
            "gateway", "--allow-unconfigured",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except BaseException:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise
        if r.returncode != 0:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise RuntimeError(f"oc container start failed: {r.stderr.strip()}")
        oc_container_id = r.stdout.strip()

        try:
            self._wait_for_gateway_ready(oc_container_id)
        except BaseException:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

        gw_env = (
            f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789\n"
            f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
            f"export OPENCLAW_RUN_ID={run_id}\n"
        )
        try:
            subprocess.run(
                ["docker", "exec", oc_container_id, "bash", "-c",
                 f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF"],
                capture_output=True, text=True,
            )
        except BaseException:
            _release_gateway_port(gateway_port)
            dispatcher.stop()
            self._best_effort_cleanup(net_name, oc_name, task_container_name)
            raise

        logger.info(
            "  ready: oc=%s task=%s bridge_pid=%s net=%s",
            oc_container_id[:12], task_container_id[:12],
            dispatcher.pid, net_name,
        )
        env_info.update({
            # OC takes the `container_id` slot so condition.run() (which uses
            # it for openclaw_agent's docker-exec) targets the OC container.
            "container_id": oc_container_id,
            "container_name": oc_name,
            "oc_container_id": oc_container_id,
            "state_dir": str(state_dir),
            "gateway_port": gateway_port,
            "gateway_token": gateway_token,
            "gateway_url": "ws://127.0.0.1:18789",
            **dispatcher.env_info(),
        })
        return env_info

    # ------------------------------------------------------------------
    # State-dir compose. Container lifecycle remains Terminal-Bench based,
    # while active harness profiles may select benchmark-specific assets.
    # ------------------------------------------------------------------

    def _build_state_dir(
        self,
        *,
        run_id: str,
        workspace_root: str,
        sandbox_image: str,
        sandbox_binds: List[str],
        container_prefix: str,
        scope: str = "agent",
    ) -> Path:
        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        # Agent dirname must match the id the condition will invoke with
        # `openclaw agent --agent <id>` AND the openclaw.json `agents.list[].id`.
        # TerminalBenchOpenClawCondition uses "terminal_bench"; pull from OC
        # config when present so a custom harness can override.
        agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        os.makedirs(state_dir / "agents" / agent_id, exist_ok=True)

        override = os.environ.get("AB_HARNESS_STATE_TEMPLATE")
        if override:
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"AB_HARNESS_STATE_TEMPLATE points to {override} but is not a directory."
                )
            shutil.rmtree(state_dir)
            shutil.copytree(override, state_dir)
        else:
            harness_path = os.environ.get(
                "AB_HARNESS_CONFIG",
                os.path.join(repo_root, "profiles/harnesses/openclaw_terminal_bench.yaml"),
            )
            if not os.path.isfile(harness_path):
                raise FileNotFoundError(
                    f"harness config not found at {harness_path} "
                    "(set AB_HARNESS_CONFIG or AB_HARNESS_STATE_TEMPLATE)."
                )
            harness_cfg = load_profile(harness_path)
            oc_cfg = dict(harness_cfg.get("openclaw") or {})
            if not oc_cfg:
                raise ValueError(
                    f"harness config {harness_path} has no `openclaw` block "
                    "(required when use_openclaw=True)."
                )
            model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
            providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
            agent_model_p = providers.setdefault("agent_model", {})
            agent_model_p["api"] = _OPENAI_CHAT_COMPLETIONS_API
            agent_model_p.setdefault("apiKey", model_cfg.api_key or "sk-dummy")
            agent_model_p["models"] = [{
                "id": model_cfg.name,
                "name": model_cfg.name,
                "api": _OPENAI_CHAT_COMPLETIONS_API,
                "contextWindow": model_cfg.context_window,
                "maxTokens": model_cfg.max_tokens,
            }]
            agents_d = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
            agents_d.setdefault("model", {})["primary"] = f"agent_model/{model_cfg.name}"
            with open(state_dir / "openclaw.json", "w") as fh:
                json.dump(oc_cfg, fh, indent=2)

            tb_agents = str(
                _terminal_bench_agent_assets_dir(repo_root=Path(repo_root))
            )
            tb_skills = os.path.join(repo_root, "agent_bench/benchmarks/terminal_bench/openclaw/skills")
            swe_skills = os.path.join(repo_root, "agent_bench/benchmarks/swebench/openclaw/skills")

            # Prompt and skill sources can be benchmark-specific while the
            # Terminal-Bench container/runtime adapter remains shared.
            src_agents = tb_agents
            configured_skills = _terminal_bench_profile_dir(
                "skills_dir", repo_root=Path(repo_root)
            )
            src_skills = str(configured_skills) if configured_skills else (
                tb_skills if (os.path.isdir(tb_skills) and os.listdir(tb_skills))
                else swe_skills
            )
            if os.path.isdir(src_agents):
                agent_prompt_file = _terminal_bench_prompt_filename(
                    "agent_prompt_file", "AGENTS.md"
                )
                agent_prompt_path = Path(src_agents) / agent_prompt_file
                if not agent_prompt_path.is_file():
                    raise FileNotFoundError(
                        "Terminal-Bench agent prompt selected by harness is "
                        f"missing: {agent_prompt_path}"
                    )
                # BOOTSTRAP.md is workspace-scoped (lives in task workspace at runtime,
                # see _seed_workspace_helpers). The file in agents/ is just
                # the source-of-truth — don't also copy it into
                # state_dir/agents/<id>/, where it'd be agent-persona context.
                for f in os.listdir(src_agents):
                    if f == "BOOTSTRAP.md" or f.startswith("BOOTSTRAP.md."):
                        continue
                    if f == "AGENTS.md" or f.startswith("AGENTS.md."):
                        if f != agent_prompt_file:
                            continue
                        output_name = "AGENTS.md"
                    else:
                        output_name = f
                    s = os.path.join(src_agents, f)
                    if os.path.isfile(s):
                        dst = state_dir / "agents" / agent_id / output_name
                        if f.endswith((".md", ".txt")):
                            text = Path(s).read_text()
                            dst.write_text(
                                _render_workspace_doc_text(text, workspace_root)
                            )
                        else:
                            shutil.copyfile(s, dst)
            if os.path.isdir(src_skills):
                shutil.copytree(src_skills, state_dir / "skills")

        cfg_path = state_dir / "openclaw.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        agents_defaults["workspace"] = workspace_root
        agents_defaults["sandbox"] = {
            "mode": "all",
            "scope": scope,
            "backend": "docker",
            "workspaceAccess": "rw",
            "docker": {
                "image": sandbox_image,
                "containerPrefix": container_prefix,
                "workdir": workspace_root,
                "binds": list(sandbox_binds),
                "dangerouslyAllowExternalBindSources": True,
            },
        }
        cfg.setdefault("tools", {}).setdefault("exec", {})["host"] = "sandbox"
        if (state_dir / "skills").is_dir():
            cfg.setdefault("skills", {}).setdefault("load", {})["extraDirs"] = [
                "/root/.openclaw/skills"
            ]
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        serving_endpoint = runtime_endpoint_for_container(
            model_cfg.endpoint, "vllm-host"
        )
        providers = cfg.setdefault("models", {}).setdefault("providers", {})
        agent_model_provider = providers.setdefault("agent_model", {})
        agent_model_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API
        agent_model_provider["baseUrl"] = serving_endpoint
        for model in agent_model_provider.get("models") or []:
            if isinstance(model, dict):
                model["api"] = _OPENAI_CHAT_COMPLETIONS_API
        otel_cfg = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel_cfg.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "AB_TELEMETRY_ENDPOINT", "http://vllm-host:4318",
            )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

    # ------------------------------------------------------------------

    def _prime_workspace_from_image(
        self,
        docker_image: str,
        task_workspace: str,
        app_dir: Path,
    ) -> None:
        """Best-effort copy of the image WORKDIR into the host bind dir."""
        scratch_name = f"oc-eval-tb-prime-{app_dir.name[:12]}-{os.urandom(2).hex()}"
        r = subprocess.run(
            ["docker", "create", "--name", scratch_name,
             "--entrypoint", "true", docker_image],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker create scratch failed: {r.stderr.strip()}")
        try:
            r = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{scratch_name}:{task_workspace.rstrip('/')}/.",
                    str(app_dir) + "/",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                # docker cp from an empty/missing source can fail on some
                # daemons; treat it as "nothing to prime".
                logger.debug(
                    "TB prime: docker cp returned %d for %s (likely empty): %s",
                    r.returncode, task_workspace, r.stderr.strip(),
                )
        finally:
            subprocess.run(["docker", "rm", "-f", scratch_name],
                           capture_output=True, check=False)

    # ------------------------------------------------------------------

    def _wait_for_gateway_ready(self, oc_container_id: str) -> None:
        deadline = time.time() + self._GATEWAY_READY_TIMEOUT_SECONDS
        last_log_tail = ""
        while time.time() < deadline:
            r = subprocess.run(
                ["docker", "logs", oc_container_id],
                capture_output=True, text=True, timeout=30,
            )
            log_text = (r.stdout or "") + (r.stderr or "")
            if self._GATEWAY_READY_RE.search(log_text):
                return
            last_log_tail = log_text[-1500:]
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", oc_container_id],
                capture_output=True, text=True,
            )
            if (r.stdout or "").strip() != "true":
                raise RuntimeError(
                    f"openclaw container exited before gateway ready. "
                    f"last log:\n{last_log_tail}"
                )
            time.sleep(2)
        raise RuntimeError(
            f"gateway did not signal ready within "
            f"{self._GATEWAY_READY_TIMEOUT_SECONDS}s. last log:\n{last_log_tail}"
        )

    # ------------------------------------------------------------------
    # exec_in_container — callable for conditions that want to drive the task
    # container directly (tb_react). Matches the singularity harness's shape.
    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        container: str,                  # docker id or name
        argv: List[str],
        timeout_seconds: int = 300,
        stdin_text: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> Dict:
        full = ["docker", "exec", "-i"]
        if workdir:
            full.extend(["-w", workdir])
        full.extend([container, *argv])
        try:
            r = subprocess.run(
                full,
                input=stdin_text,
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            def _as_str(x):
                if x is None:
                    return ""
                return (x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray)) else x)
            return {
                "rc": -1,
                "stdout": _as_str(e.stdout),
                "stderr": (_as_str(e.stderr)
                           + f"\n[harness] timeout after {timeout_seconds}s"),
                "timed_out": True,
            }
        return {
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": False,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _copy_tests_into_docker_container(
        container_name: str,
        tests_host: Path,
    ) -> None:
        for cmd, timeout in (
            (["docker", "exec", container_name, "rm", "-rf", "/tests"], 60),
            (["docker", "cp", str(tests_host), f"{container_name}:/tests"], 300),
        ):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0:
                raise RuntimeError(
                    "failed to inject TerminalBench tests into docker "
                    f"container {container_name}: {r.stderr.strip()}"
                )

    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,            # ignored — TB has no patch
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None,
        **_kw,
    ) -> Dict:
        """Inject tests, run the verifier, then read reward.txt from host logs."""
        # Target the task container by name (deterministic from run_id + the
        # agent id used at prepare_environment time).
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        sandbox_agent_id = self._agent_id_from_harness_cfg() or "terminal_bench"
        scope_key = f"agent:{sandbox_agent_id}"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = _oc_sandbox_name(sandbox_prefix, scope_key)

        host_logs_dir = self.workspace_base / run_id / "logs" / "verifier"
        image_workdir = (
            _docker_image_workdir(instance_data["docker_image"])
            if instance_data and instance_data.get("docker_image")
            else None
        )
        task_workspace = _resolve_task_workspace(instance_data, image_workdir)

        timeout = timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        if instance_data:
            timeout = int(instance_data.get("verifier_timeout_sec", timeout))

        verifier = (instance_data or {}).get(
            "verifier_entrypoint", "/tests/test.sh"
        )
        tests_host = _require_tests_host_path(instance_data)
        logger.info(
            "scoring TB container %s (workdir=%s, verifier=%s, tests=%s, timeout=%ds)",
            task_container_name, task_workspace, verifier, tests_host, timeout,
        )
        self._copy_tests_into_docker_container(task_container_name, tests_host)
        host_reward = _clear_reward_file(host_logs_dir)
        result = self.exec_in_container(
            task_container_name,
            ["bash", verifier],
            timeout_seconds=timeout,
            workdir=task_workspace,
        )

        reward_text = ""
        try:
            reward_text = host_reward.read_text().strip()
        except FileNotFoundError:
            pass

        status, reward_value = _terminal_bench_reward(
            reward_text, timed_out=result["timed_out"]
        )

        return {
            "status": status,
            "test_pass_rate": reward_value,
            "passed": 1 if status == "PASS" else 0,
            "failed": 0 if status == "PASS" else 1,
            "verifier_rc": result["rc"],
            "verifier_stdout_tail": result["stdout"][-2000:],
            "verifier_stderr_tail": result["stderr"][-2000:],
            "reward_raw": reward_text or None,
            "reward_path": str(host_reward),
            "evaluation_dir": str(host_logs_dir),
            "error_message": (
                None if status == "PASS"
                else f"verifier exit {result['rc']}, reward={reward_text!r}"
            ),
        }

    # ------------------------------------------------------------------

    def cleanup(
        self,
        container_id: Optional[str] = None,
        oc_container_id: Optional[str] = None,
        task_container_id: Optional[str] = None,
        task_container_name: Optional[str] = None,
        container_name: Optional[str] = None,
        network_name: Optional[str] = None,
        gateway_port: Optional[int] = None,
        workspace_host_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        bridge_pid: Optional[int] = None,
        bridge_socket: Optional[str] = None,
        host_logs_dir: Optional[str] = None,
        **_extra,
    ):
        if gateway_port is not None:
            _release_gateway_port(gateway_port)
        OpenClawRemoteDispatcher.stop_external(
            pid=bridge_pid, socket_path=bridge_socket,
        )

        uid = os.getuid()
        gid = os.getgid()
        for cid in {oc_container_id, container_id}:
            if not cid:
                continue
            subprocess.run(
                ["docker", "exec", cid, "chown", "-R",
                 f"{uid}:{gid}", "/root/.openclaw"],
                capture_output=True, check=False, timeout=30,
            )

        targets = [t for t in (oc_container_id, container_id,
                               task_container_id, task_container_name,
                               container_name)
                   if t]
        for cid in dict.fromkeys(targets):
            subprocess.run(
                ["docker", "rm", "-f", "-v", cid],
                capture_output=True, check=False,
            )
        if network_name:
            r = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"network={network_name}"],
                capture_output=True, text=True,
            )
            for cid in (r.stdout or "").split():
                subprocess.run(["docker", "rm", "-f", "-v", cid],
                               capture_output=True, check=False)
            subprocess.run(["docker", "network", "rm", network_name],
                           capture_output=True, check=False)

        keep = os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") == "1"
        if not keep and workspace_host_path and os.path.isdir(workspace_host_path):
            # workspace_host_path points at the host bind dir; also clean the
            # logs sibling and parent run dir so per-run dirs don't pile up.
            run_root = Path(workspace_host_path).parent
            _force_rmtree(run_root, ignore_errors=True)
        if not keep and state_dir and os.path.isdir(state_dir):
            _force_rmtree(Path(state_dir), ignore_errors=True)

    # ------------------------------------------------------------------

    def _best_effort_cleanup(
        self,
        net: str,
        oc: str,
        task: Optional[str] = None,
    ) -> None:
        for name in (oc, task):
            if name:
                subprocess.run(["docker", "rm", "-f", "-v", name],
                               capture_output=True, check=False)
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"network={net}"],
            capture_output=True, text=True,
        )
        for cid in (r.stdout or "").split():
            subprocess.run(["docker", "rm", "-f", "-v", cid],
                           capture_output=True, check=False)
        subprocess.run(["docker", "network", "rm", net],
                       capture_output=True, check=False)


# ----------------------------------------------------------------------
# Container-agnostic OpenClaw harness
# ----------------------------------------------------------------------


class TerminalBenchOpenClawHarness(TerminalBenchDockerHarness):
    """OpenClaw Terminal-Bench lifecycle implemented through ``Container``.

    The legacy Docker harness remains the compatibility path for ``tb_react``.
    This class owns both the task runtime and the OpenClaw runtime, so its
    ``container_id`` is always the OpenClaw handle that conditions must invoke.
    """

    backend_id = "container-agnostic"
    _GATEWAY_READY_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        benchmark_data_root: Optional[Path] = None,
        *,
        container: Container,
        oc_image: Optional[str] = None,
        workspace_base: Optional[Path] = None,
    ) -> None:
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.container = container
        if self.container.backend_id == "singularity":
            self.oc_image = os.environ.get("AB_AGENT_RUNTIME_SIF", "")
            if not self.oc_image:
                raise RuntimeError(
                    "AB_AGENT_RUNTIME_SIF is required for Terminal-Bench "
                    "OpenClaw runs with the singularity backend."
                )
        else:
            self.oc_image = oc_image or DEFAULT_OPENCLAW_RUNTIME_IMAGE
        self.container.check_available()
        self.container.check_image(self.oc_image)
        self.task_container_cpus = configured_task_container_cpus()
        self.workspace_base = Path(
            workspace_base
            or os.environ.get("AB_HARNESS_WORKSPACE_BASE", "~/.cache/oc-eval/workspaces")
        ).expanduser()
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)
        self._task_specs: Dict[str, ContainerSpec] = {}
        self._task_specs_lock = threading.Lock()

    def _resolve_task_image(self, docker_image: str) -> str:
        if self.container.backend_id == "docker":
            return docker_image
        sif_dir = os.environ.get("AB_DEPLOYMENT_TASK_IMAGE_DIR")
        if not sif_dir:
            raise RuntimeError(
                "AB_DEPLOYMENT_TASK_IMAGE_DIR is required for Terminal-Bench "
                "with the singularity backend."
            )
        sif_name = docker_image.replace("/", "__").replace(":", "__") + ".sif"
        sif_path = Path(sif_dir) / sif_name
        if not sif_path.is_file():
            raise FileNotFoundError(
                f"prepared Terminal-Bench task SIF missing: {sif_path}; "
                "PBS prepare_images must create it before the run."
            )
        return str(sif_path)

    def _build_state_dir(self, **kwargs) -> Path:  # type: ignore[override]
        state_dir = super()._build_state_dir(**kwargs)
        # The inherited TB renderer predates the runtime abstraction and uses
        # Docker's synthetic hostname. Rewrite only the runtime-specific URLs.
        cfg_path = state_dir / "openclaw.json"
        cfg = json.loads(cfg_path.read_text())
        model_cfg, _serving_cfg = load_active_model_and_serving(
            repo_root=os.environ.get("AB_REPO_ROOT", os.getcwd())
        )
        endpoint = os.environ.get("AB_SERVING_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        ) or model_cfg.endpoint
        cfg.setdefault("models", {}).setdefault("providers", {}).setdefault(
            "agent_model", {}
        )["baseUrl"] = runtime_endpoint_for_container(
            endpoint, self.container.host_address
        )
        otel = (cfg.get("diagnostics") or {}).get("otel") or {}
        if otel.get("enabled"):
            cfg["diagnostics"]["otel"]["endpoint"] = os.environ.get(
                "AB_TELEMETRY_ENDPOINT",
                f"http://{self.container.host_address}:4318",
            )

        # /app inside the OpenClaw runtime belongs to the runtime image itself
        # and is read-only under Singularity. Terminal-Bench tasks commonly
        # use /app as their task workspace, so expose that host workspace to
        # OpenClaw as /workspace instead. This matches the proven Atlas path.
        workspace_root = _normalize_container_path(
            str(kwargs.get("workspace_root") or "/app")
        )
        if workspace_root == "/app":
            cfg.setdefault("agents", {}).setdefault(
                "defaults", {}
            )["workspace"] = "/workspace"

        cfg_path.write_text(json.dumps(cfg, indent=2))
        return state_dir

    @staticmethod
    def _task_container_name(run_id: str) -> str:
        agent_id = TerminalBenchDockerHarness._agent_id_from_harness_cfg() or "terminal_bench"
        return _oc_sandbox_name(f"oc-sbx-{run_id[:12]}-", f"agent:{agent_id}")

    def _runtime_workspace_mount(self, app_dir: Path, task_workspace: str) -> Optional[Mount]:
        # The OpenClaw runtime image uses /app for its own launcher/package.
        # Never mask it. For Terminal-Bench tasks whose workspace is /app,
        # expose the same host workspace to the runtime at /workspace instead.
        # The task container still sees it at its canonical /app path.
        if task_workspace == "/app":
            return Mount(app_dir, "/workspace", "rw")
        return Mount(app_dir, task_workspace, "rw")

    def _best_effort_stop(self, *handles: Optional[ContainerHandle]) -> None:
        for handle in handles:
            if handle is None:
                continue
            try:
                self.container.stop(handle)
            except Exception:
                pass

    def prepare_environment(
        self, instance_id: str, repo_id: str, base_commit: str,
        container_image: Optional[str] = None, workspace_root: Optional[str] = None,
        run_id: Optional[str] = None, instance_data: Optional[Dict] = None,
    ) -> Dict:
        if not instance_data or instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError("TerminalBenchOpenClawHarness requires terminal_bench instance_data")
        docker_image = instance_data["docker_image"]
        if self.container.backend_id == "docker":
            image_workdir = _docker_image_workdir(docker_image)
            task_workspace = _resolve_task_workspace(instance_data, image_workdir)
        else:
            task_workspace = _resolve_task_workspace(instance_data)
        task_image = self._resolve_task_image(docker_image)
        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"
        agent_id = (
            TerminalBenchDockerHarness._agent_id_from_harness_cfg()
            or "terminal_bench"
        )
        scope_key = f"agent:{agent_id}"
        task_name = self._task_container_name(run_id)

        # run_id is deterministic, so an interrupted prior run may have left
        # instances with exactly these names behind. Clean those stale
        # resources before retrying.
        self._best_effort_stop(
            ContainerHandle(oc_name, oc_name, self.container.backend_id),
            ContainerHandle(task_name, task_name, self.container.backend_id),
        )
        self.container.remove_network(net_name)
        self.container.create_network(net_name)

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        tests_dir = run_root / "tests"
        app_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)

        # Docker can create /tests later with docker cp. A Singularity SIF has
        # an immutable root filesystem, so give it an empty writable /tests
        # bind from the start. No verifier files are placed here until after
        # the agent has finished solving the task.
        task_mounts = [
            Mount(app_dir, task_workspace),
            Mount(logs_dir, "/logs/verifier"),
        ]
        if self.container.backend_id == "singularity":
            task_mounts.append(Mount(tests_dir, "/tests"))
        workspace_prepare_start_ns = time.time_ns()
        try:
            self.container.copy_from_image(task_image, task_workspace, app_dir)
        except Exception as exc:
            _force_rmtree(run_root, ignore_errors=True)
            self.container.remove_network(net_name)
            raise RuntimeError(
                "TerminalBench workspace prime failed; refusing to start the "
                f"agent on a partial workspace ({docker_image}:{task_workspace})"
            ) from exc
        _seed_workspace_helpers(app_dir, instance_data, task_workspace)
        workspace_prepare_end_ns = time.time_ns()

        state_dir = self._build_state_dir(
            run_id=run_id,
            workspace_root=task_workspace,
            sandbox_image=docker_image,
            sandbox_binds=[f"{app_dir}:{task_workspace}:rw", f"{logs_dir}:/logs/verifier:rw"],
            container_prefix=f"oc-sbx-{rid_short}-",
        )
        task_handle: Optional[ContainerHandle] = None
        oc_handle: Optional[ContainerHandle] = None
        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        gateway_port: Optional[int] = None
        try:
            task_spec = ContainerSpec(
                name=task_name, image=task_image, network=net_name, user="0",
                labels={
                    "openclaw.sandbox": "1",
                    "openclaw.sessionKey": scope_key,
                    "oc-eval.run_id": run_id,
                },
                mounts=tuple(task_mounts),
                env={"HOME": "/root"}, workdir=task_workspace,
                entrypoint="sleep", command=("infinity",), cpus=self.task_container_cpus,
            )
            with self._task_specs_lock:
                self._task_specs[task_name] = task_spec
            task_handle = self.container.start(task_spec)
            dispatcher = OpenClawRemoteDispatcher()
            dispatcher.start(state_dir=state_dir, backend=self.container.backend_id, target=task_name)
            gateway_port = _pick_free_gateway_port()
            gateway_token = secrets.token_hex(24)
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            mounts = [Mount(state_dir, "/root/.openclaw"), *bridge_mounts]
            workspace_mount = self._runtime_workspace_mount(app_dir, task_workspace)
            if workspace_mount:
                mounts.append(workspace_mount)
            if self.container.backend_id == "docker":
                mounts.append(Mount(Path("/var/run/docker.sock"), "/var/run/docker.sock"))
            runtime_env = {
                "HOME": "/root",
                "OPENCLAW_GATEWAY_PORT": str(gateway_port),
                "OPENCLAW_GATEWAY_TOKEN": gateway_token,
                "OPENCLAW_GATEWAY_OTEL_ENDPOINT": os.environ.get(
                    "AB_TELEMETRY_ENDPOINT", f"http://{self.container.host_address}:4318"
                ),
                "OPENCLAW_RUN_ID": run_id,
                "OPENCLAW_RAW_RESPONSE_LOG": "/root/.openclaw/logs/llm-raw-responses.jsonl",
                "OTEL_RESOURCE_ATTRIBUTES": f"oc.run_id={run_id}",
                "OPENCLAW_DISABLE_BONJOUR": "1",
                "NPM_CONFIG_REGISTRY": os.environ.get(
                    "AB_NPM_REGISTRY", "https://registry.npmmirror.com"
                ),
                "npm_config_registry": os.environ.get(
                    "AB_NPM_REGISTRY", "https://registry.npmmirror.com"
                ),
                "NPM_CONFIG_FETCH_RETRIES": "1",
                "NPM_CONFIG_FETCH_TIMEOUT": "15000",
                "NODE_OPTIONS": "--dns-result-order=ipv4first",
                **bridge_env,
            }
            if self.container.backend_id == "singularity":
                runtime_env["OPENCLAW_BYPASS_ENSURE_SANDBOX"] = "1"
                runtime_env["OPENCLAW_PLUGIN_STAGE_DIR"] = (
                    f"/tmp/oc-plugin-deps/{run_id}"
                )
            else:
                runtime_env["DOCKER_API_VERSION"] = "1.44"
            oc_spec = ContainerSpec(
                name=oc_name, image=self.oc_image, network=net_name, user="0",
                mounts=tuple(mounts), env=runtime_env, workdir="/root",
                extra_hosts={"vllm-host": "host-gateway"},
                entrypoint="/usr/local/bin/openclaw",
                command=("gateway", "--allow-unconfigured"),
            )
            gateway_warmup_wait_start_ns = time.time_ns()
            logger.info(
                "  [warmup] waiting for gateway-warmup slot (run=%s)", run_id
            )
            with _OPENCLAW_WARMUP_SEM:
                gateway_warmup_start_ns = time.time_ns()
                logger.info(
                    "  [warmup] slot acquired; starting gateway (run=%s)",
                    run_id,
                )
                oc_handle = self.container.start(oc_spec)
                self.container.wait_log_pattern(
                    oc_handle,
                    self._GATEWAY_READY_RE,
                    self._GATEWAY_READY_TIMEOUT_SECONDS,
                )
                gateway_ready_ns = time.time_ns()
            logger.info(
                "  [warmup] gateway ready; slot released (run=%s)", run_id
            )
            gateway_url = f"ws://127.0.0.1:{gateway_port}"
            cli_stage_export = ""
            if self.container.backend_id == "singularity":
                cli_stage_export = (
                    "export OPENCLAW_PLUGIN_STAGE_DIR="
                    f"/tmp/oc-plugin-deps/{run_id}-cli\n"
                )
            self.container.exec(oc_handle, ["bash", "-c", (
                "cat > /tmp/oc_gateway_env <<'EOF'\n"
                f"export OPENCLAW_GATEWAY_URL={gateway_url}\n"
                f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
                f"export OPENCLAW_RUN_ID={run_id}\n"
                f"{cli_stage_export}"
                "EOF"
            )])
            session_exporter_pid = self._start_roadmap_session_exporter(
                oc_handle=oc_handle,
                state_dir=state_dir,
                agent_id=agent_id,
            )
            return {
                "prepare_phase_timing": {
                    "workspace_prepare_start_ns": workspace_prepare_start_ns,
                    "workspace_prepare_end_ns": workspace_prepare_end_ns,
                    "workspace_prepare_seconds": (workspace_prepare_end_ns - workspace_prepare_start_ns) / 1e9,
                    "gateway_warmup_wait_start_ns": gateway_warmup_wait_start_ns,
                    "gateway_warmup_start_ns": gateway_warmup_start_ns,
                    "gateway_ready_ns": gateway_ready_ns,
                    "gateway_warmup_queue_seconds": max(0.0, (gateway_warmup_start_ns - gateway_warmup_wait_start_ns) / 1e9),
                    "gateway_warmup_seconds": max(0.0, (gateway_ready_ns - gateway_warmup_start_ns) / 1e9),
                },
                "container_id": oc_handle.id,
                "container_name": oc_handle.name,
                "oc_container_id": oc_handle.id,
                "task_container_id": task_handle.id,
                "task_container_name": task_handle.name,
                "backend": self.container.backend_id,
                "network_name": net_name,
                "workspace_root": task_workspace,
                "task_workspace_root": task_workspace,
                "workspace_host_path": str(app_dir),
                "host_logs_dir": str(logs_dir),
                "state_dir": str(state_dir),
                "gateway_port": gateway_port,
                "gateway_token": gateway_token,
                "gateway_url": gateway_url,
                "session_exporter_pid": session_exporter_pid,
                "gateway_log_path": (
                    str(oc_handle.log_path) if oc_handle.log_path else None
                ),
                "sandbox_image": docker_image,
                "tb_instruction": instance_data["instruction"],
                "tb_verifier_entrypoint": instance_data.get("verifier_entrypoint", "/tests/test.sh"),
                "tb_reward_file": instance_data.get("reward_file", "/logs/verifier/reward.txt"),
                **dispatcher.env_info(),
            }
        except BaseException:
            with self._task_specs_lock:
                self._task_specs.pop(task_name, None)
            if gateway_port is not None:
                _release_gateway_port(gateway_port)
            if dispatcher:
                dispatcher.stop()
            self._best_effort_stop(oc_handle, task_handle)
            self.container.remove_network(net_name)
            raise

    def _start_roadmap_session_exporter(
        self,
        *,
        oc_handle: ContainerHandle,
        state_dir: Path,
        agent_id: str,
    ) -> Optional[int]:
        """Project OpenClaw v9's STRICT SQLite store for host-side watchers."""
        version = str(os.environ.get("AB_AGENT_RUNTIME_VERSION") or "").lstrip("v")
        if agent_id != "roadmap_bench" or version != "2026.9.2":
            return None

        source = (
            Path(__file__).resolve().parents[1]
            / "roadmap_bench"
            / "openclaw"
            / "sqlite_session_exporter.mjs"
        )
        target = state_dir / "agent-bench" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        ready_path = target.parent / "sqlite-session-exporter.ready.json"
        ready_path.unlink(missing_ok=True)
        pid = self.container.exec_background(
            oc_handle,
            [
                "bash",
                "-c",
                "mkdir -p /root/.openclaw/logs; "
                "exec node /root/.openclaw/agent-bench/sqlite_session_exporter.mjs "
                f"/root/.openclaw/agents/{agent_id}/agent/openclaw-agent.sqlite "
                f"/root/.openclaw/agents/{agent_id}/sessions "
                "/root/.openclaw/agent-bench/sqlite-session-exporter.ready.json "
                ">/root/.openclaw/logs/sqlite-session-exporter.log 2>&1 "
                "</dev/null",
            ],
        )
        deadline = time.monotonic() + 30
        while not ready_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.25)
        if not ready_path.is_file():
            raise RuntimeError(
                "OpenClaw SQLite session exporter did not become ready within 30s"
            )
        logger.info("  started Roadmap SQLite session exporter pid=%s", pid)
        return pid

    @staticmethod
    def _wait_for_process_exit(pid: Optional[int], timeout: float = 10.0) -> None:
        if not pid:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                waited, _status = os.waitpid(int(pid), os.WNOHANG)
                if waited == int(pid):
                    return
            except ChildProcessError:
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"exec bridge process {pid} did not stop")

    def quiesce_orphan_children(self, env_info: Dict) -> Dict:
        """Stop all agent writers and restart only the task runtime.

        Stopping the task runtime is deliberate: terminating the gateway and
        bridge prevents new writes, while restarting the task from its
        original spec guarantees an already-dispatched exec cannot overlap the
        verifier. The host-bound workspace survives the restart unchanged.
        """
        task_name = str(env_info.get("task_container_name") or "")
        task_id = str(env_info.get("task_container_id") or task_name)
        oc_id = str(env_info.get("oc_container_id") or env_info.get("container_id") or "")
        if not task_name or not task_id or not oc_id:
            raise RuntimeError("missing runtime handles for orphan quiescence")
        with self._task_specs_lock:
            task_spec = self._task_specs.get(task_name)
        if task_spec is None:
            raise RuntimeError(f"missing saved task ContainerSpec for {task_name}")

        bridge_pid = env_info.get("bridge_pid")
        OpenClawRemoteDispatcher.stop_external(
            pid=bridge_pid,
            socket_path=env_info.get("bridge_socket"),
        )
        self._wait_for_process_exit(bridge_pid)
        env_info["bridge_pid"] = None
        env_info["bridge_socket"] = None

        oc_handle = ContainerHandle(oc_id, oc_id, self.container.backend_id)
        self.container.stop(oc_handle)
        if self.container.is_running(oc_handle):
            raise RuntimeError(f"OpenClaw runtime remained active after stop: {oc_id}")

        task_handle = ContainerHandle(task_id, task_name, self.container.backend_id)
        self.container.stop(task_handle)
        if self.container.is_running(task_handle):
            raise RuntimeError(f"task runtime remained active after stop: {task_name}")

        restarted = self.container.start(task_spec)
        if not self.container.is_running(restarted):
            raise RuntimeError(f"restarted task runtime is not active: {task_name}")
        env_info["task_container_id"] = restarted.id
        env_info["task_container_name"] = restarted.name
        env_info["agent_runtime_quiesced"] = True
        return {
            "workspaceQuiesced": True,
            "quiescenceMethod": "stop-agent-and-restart-task-runtime",
            "openClawRuntimeStopped": True,
            "execBridgeStopped": True,
            "taskRuntimeRestarted": True,
        }

    def run_tests(self, container_id: Optional[str], instance_id: str, workspace_root: str,
                  timeout_seconds: int, patch: Optional[str] = None, run_id: Optional[str] = None,
                  instance_data: Optional[Dict] = None, **_kw) -> Dict:
        run_id = run_id or instance_id
        task_name = self._task_container_name(run_id)
        task = ContainerHandle(task_name, task_name, self.container.backend_id)
        task_workspace = _resolve_task_workspace(instance_data)
        timeout = int((instance_data or {}).get(
            "verifier_timeout_sec", timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        ))
        tests_host = _require_tests_host_path(instance_data)
        if self.container.backend_id == "singularity":
            # /tests is a persistent bind mount in Singularity mode, so do not
            # remove the mount point itself. Remove only its contents before
            # injecting the authoritative verifier tree.
            self.container.exec(
                task,
                [
                    "sh",
                    "-c",
                    "find /tests -mindepth 1 -maxdepth 1 "
                    "-exec rm -rf -- {} +",
                ],
                timeout=60,
            )
        else:
            self.container.exec(task, ["rm", "-rf", "/tests"], timeout=60)

        self.container.copy_into(task, tests_host, "/tests")
        host_logs = self.workspace_base / run_id / "logs" / "verifier"
        reward = _clear_reward_file(host_logs)
        try:
            result = self.container.exec(
                task, ["bash", (instance_data or {}).get("verifier_entrypoint", "/tests/test.sh")],
                check=False, timeout=timeout, workdir=task_workspace,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            if stderr:
                stderr += "\n"
            stderr += f"[harness] verifier timeout after {timeout}s"
            result = subprocess.CompletedProcess([], -1, stdout, stderr)
            timed_out = True
        reward_text = reward.read_text().strip() if reward.exists() else ""
        status, reward_value = _terminal_bench_reward(
            reward_text, timed_out=timed_out
        )
        return {"status": status, "verifier_status": status, "reward": reward_value,
                "test_pass_rate": reward_value, "passed": int(status == "PASS"),
                "failed": int(status != "PASS"), "verifier_rc": result.returncode,
                "verifier_timed_out": timed_out, "verifier_timeout_seconds": timeout,
                "verifier_stdout_tail": _as_text(result.stdout)[-2000:],
                "verifier_stderr_tail": _as_text(result.stderr)[-2000:], "reward_raw": reward_text or None,
                "reward_path": str(reward), "evaluation_dir": str(host_logs),
                "error_message": None if status == "PASS" else f"verifier exit {result.returncode}, reward={reward_text!r}"}

    def cleanup(self, container_id: Optional[str] = None, oc_container_id: Optional[str] = None,
                task_container_id: Optional[str] = None, task_container_name: Optional[str] = None,
                network_name: Optional[str] = None, gateway_port: Optional[int] = None,
                workspace_host_path: Optional[str] = None, state_dir: Optional[str] = None,
                bridge_pid: Optional[int] = None, bridge_socket: Optional[str] = None, **_kw) -> None:
        if task_container_name:
            with self._task_specs_lock:
                self._task_specs.pop(task_container_name, None)
        if gateway_port is not None:
            _release_gateway_port(gateway_port)
        OpenClawRemoteDispatcher.stop_external(pid=bridge_pid, socket_path=bridge_socket)
        for ident in dict.fromkeys(x for x in (oc_container_id, container_id, task_container_id, task_container_name) if x):
            self._best_effort_stop(ContainerHandle(ident, ident, self.container.backend_id))
        if network_name:
            self.container.remove_network(network_name)
        if os.environ.get("AB_HARNESS_KEEP_WORKSPACE", "0") != "1":
            if workspace_host_path:
                _force_rmtree(Path(workspace_host_path).parent, ignore_errors=True)
            if state_dir:
                _force_rmtree(Path(state_dir), ignore_errors=True)


# ----------------------------------------------------------------------
# Singularity harness (HPC fallback) — unchanged from the original
# TerminalBenchHarness, just renamed.
# ----------------------------------------------------------------------


class TerminalBenchSingularityHarness:
    """Singularity-instance harness for TB tasks (HPC nodes with no docker)."""

    backend_id = "singularity"

    _PULL_TIMEOUT_SECONDS = 1800
    _INSTANCE_START_TIMEOUT_SECONDS = 60
    _DEFAULT_VERIFIER_TIMEOUT_SECONDS = 900

    def __init__(self, benchmark_data_root: Optional[Path] = None,
                 workspace_base: Optional[Path] = None):
        self.benchmark_data_root = Path(benchmark_data_root) if benchmark_data_root else None
        self.workspace_base = Path(
            workspace_base or os.environ.get(
                "AB_HARNESS_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)

        self.sif_dir = Path(os.environ.get(
            "AB_DEPLOYMENT_TASK_IMAGE_DIR",
            str(self.workspace_base.parent / "tb-sif"),
        ))
        self.sif_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("singularity"):
            raise RuntimeError("singularity not on PATH")

    @staticmethod
    def _sif_name_for(docker_image: str) -> str:
        return docker_image.replace("/", "__").replace(":", "__") + ".sif"

    def _ensure_sif(self, docker_image: str) -> Path:
        sif_path = self.sif_dir / self._sif_name_for(docker_image)
        if sif_path.exists() and sif_path.stat().st_size > 0:
            return sif_path
        logger.info("pulling %s -> %s", docker_image, sif_path)
        tmp = sif_path.with_suffix(".sif.partial")
        if tmp.exists():
            tmp.unlink()
        r = subprocess.run(
            ["singularity", "pull", "--name", tmp.name,
             f"docker://{docker_image}"],
            cwd=str(sif_path.parent),
            capture_output=True, text=True,
            timeout=self._PULL_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"singularity pull docker://{docker_image} failed "
                f"(rc={r.returncode}): {r.stderr[-600:]}"
            )
        tmp.rename(sif_path)
        return sif_path

    @staticmethod
    def _prime_workspace_from_sif(
        sif_path: Path,
        task_workspace: str,
        app_dir: Path,
    ) -> None:
        app_dir.mkdir(parents=True, exist_ok=True)
        pack = subprocess.Popen(
            [
                "singularity",
                "exec",
                str(sif_path),
                "tar",
                "-C",
                task_workspace,
                "-cf",
                "-",
                ".",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            extract = subprocess.run(
                ["tar", "-C", str(app_dir), "-xf", "-"],
                stdin=pack.stdout,
                capture_output=True,
                timeout=600,
            )
        finally:
            if pack.stdout:
                pack.stdout.close()

        pack_stderr = b""
        if pack.stderr:
            pack_stderr = pack.stderr.read()
        try:
            pack_rc = pack.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pack.kill()
            pack_rc = -1
            pack_stderr += b"\n[harness] tar pack timeout"

        def _decode(value: bytes) -> str:
            return value.decode("utf-8", errors="replace")

        if pack_rc != 0 or extract.returncode != 0:
            stderr = _decode(pack_stderr) + _decode(extract.stderr or b"")
            raise RuntimeError(
                "failed to prime TerminalBench workspace from sif "
                f"{sif_path}:{task_workspace}: {stderr.strip()}"
            )

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,
        base_commit: str,
        container_image: Optional[str] = None,
        workspace_root: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        if instance_data is None:
            raise RuntimeError(
                "TerminalBenchSingularityHarness.prepare_environment requires "
                "instance_data; the driver should pass it through."
            )
        if instance_data.get("benchmark") != "terminal_bench":
            raise RuntimeError(
                "TerminalBenchSingularityHarness only handles "
                "benchmark=terminal_bench instances "
                f"(got benchmark={instance_data.get('benchmark')!r})"
            )

        docker_image = instance_data["docker_image"]
        task_workspace = _resolve_task_workspace(instance_data)

        sif_path = self._ensure_sif(docker_image)

        run_id = run_id or instance_id
        run_root = self.workspace_base / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        app_dir = run_root / "app"
        logs_dir = run_root / "logs" / "verifier"
        app_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._prime_workspace_from_sif(sif_path, task_workspace, app_dir)
        except Exception as exc:  # noqa: BLE001 - empty/missing workdir is non-fatal
            logger.debug("TB singularity workspace prime skipped: %s", exc)

        _seed_workspace_helpers(app_dir, instance_data, task_workspace)

        instance_name = self._instance_name_for(run_id)
        self._try_stop_instance(instance_name)

        argv = [
            "singularity", "instance", "start",
            "--writable-tmpfs",
            "--bind", f"{app_dir}:{task_workspace}:rw",
            "--bind", f"{logs_dir}:/logs/verifier:rw",
            str(sif_path),
            instance_name,
        ]
        logger.info("starting TB instance %s from %s", instance_name, sif_path.name)
        r = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=self._INSTANCE_START_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"singularity instance start failed (rc={r.returncode}): "
                f"{r.stderr[-600:]}"
            )

        return {
            "container_id": None,
            "container_name": instance_name,
            "backend": self.backend_id,
            "workspace_root": task_workspace,
            "task_workspace_root": task_workspace,
            "host_workspace_root": str(app_dir),
            "workspace_host_path": str(app_dir),
            "host_logs_dir": str(logs_dir),
            "task_sif": str(sif_path),
            "instance_name": instance_name,
            "container_engine": "singularity",
            "tb_instruction": instance_data["instruction"],
            "tb_verifier_entrypoint": instance_data.get(
                "verifier_entrypoint", "/tests/test.sh"
            ),
            "tb_reward_file": instance_data.get(
                "reward_file", "/logs/verifier/reward.txt"
            ),
        }

    def exec_in_container(
        self,
        instance_name: str,
        argv: List[str],
        timeout_seconds: int = 300,
        stdin_text: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> Dict:
        full = ["singularity", "exec"]
        if workdir:
            full.extend(["--pwd", workdir])
        full.extend([f"instance://{instance_name}", *argv])
        try:
            r = subprocess.run(
                full,
                input=stdin_text,
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "rc": -1,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + f"\n[harness] timeout after {timeout_seconds}s",
                "timed_out": True,
            }
        return {
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": False,
        }

    @staticmethod
    def _copy_tests_into_singularity_instance(
        instance_name: str,
        tests_host: Path,
    ) -> None:
        reset = subprocess.run(
            [
                "singularity",
                "exec",
                f"instance://{instance_name}",
                "sh",
                "-lc",
                "rm -rf /tests && mkdir -p /tests",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if reset.returncode != 0:
            raise RuntimeError(
                "failed to prepare /tests in singularity instance "
                f"{instance_name}: {reset.stderr.strip()}"
            )

        pack = subprocess.Popen(
            ["tar", "-C", str(tests_host), "-cf", "-", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            extract = subprocess.run(
                [
                    "singularity",
                    "exec",
                    f"instance://{instance_name}",
                    "tar",
                    "-C",
                    "/tests",
                    "-xf",
                    "-",
                ],
                stdin=pack.stdout,
                capture_output=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            if pack.stdout:
                pack.stdout.close()
            pack.kill()
            pack.wait(timeout=30)
            raise RuntimeError(
                "timed out injecting TerminalBench tests into singularity "
                f"instance {instance_name}"
            ) from exc
        finally:
            if pack.stdout:
                pack.stdout.close()

        pack_stderr = b""
        if pack.stderr:
            pack_stderr = pack.stderr.read()
        try:
            pack_rc = pack.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pack.kill()
            pack_rc = -1
            pack_stderr += b"\n[harness] tar pack timeout"

        def _decode(value: bytes) -> str:
            return value.decode("utf-8", errors="replace")

        if pack_rc != 0 or extract.returncode != 0:
            stderr = _decode(pack_stderr) + _decode(extract.stderr or b"")
            raise RuntimeError(
                "failed to inject TerminalBench tests into singularity "
                f"instance {instance_name}: {stderr.strip()}"
            )

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
        run_dir: Optional[Path] = None,
        **_kw,
    ) -> Dict:
        instance_name = self._instance_name_for(run_id or instance_id)
        verifier = (instance_data or {}).get("verifier_entrypoint", "/tests/test.sh")
        host_logs_dir = (
            self.workspace_base / (run_id or instance_id) / "logs" / "verifier"
        )
        task_workspace = _resolve_task_workspace(instance_data)

        timeout = timeout_seconds or self._DEFAULT_VERIFIER_TIMEOUT_SECONDS
        if instance_data:
            timeout = int(instance_data.get("verifier_timeout_sec", timeout))

        tests_host = _require_tests_host_path(instance_data)
        logger.info(
            "scoring TB instance %s (workdir=%s, verifier=%s, tests=%s, timeout=%ds)",
            instance_name, task_workspace, verifier, tests_host, timeout,
        )
        self._copy_tests_into_singularity_instance(instance_name, tests_host)
        host_reward = _clear_reward_file(host_logs_dir)
        result = self.exec_in_container(
            instance_name,
            ["bash", verifier],
            timeout_seconds=timeout,
            workdir=task_workspace,
        )

        reward_text = ""
        try:
            reward_text = host_reward.read_text().strip()
        except FileNotFoundError:
            pass

        status, reward_value = _terminal_bench_reward(
            reward_text, timed_out=result["timed_out"]
        )

        return {
            "status": status,
            "test_pass_rate": reward_value,
            "passed": 1 if status == "PASS" else 0,
            "failed": 0 if status == "PASS" else 1,
            "verifier_rc": result["rc"],
            "verifier_stdout_tail": result["stdout"][-2000:],
            "verifier_stderr_tail": result["stderr"][-2000:],
            "reward_raw": reward_text or None,
            "reward_path": str(host_reward),
            "evaluation_dir": str(host_logs_dir),
            "error_message": (
                None if status == "PASS"
                else f"verifier exit {result['rc']}, reward={reward_text!r}"
            ),
        }

    def cleanup(self, container_id: Optional[str] = None, **kwargs) -> None:
        instance_name = kwargs.get("container_name") or kwargs.get("instance_name")
        if instance_name:
            self._try_stop_instance(instance_name)

    @staticmethod
    def _instance_name_for(run_id: str) -> str:
        safe = "".join(c if c.isalnum() or c == "-" else "-" for c in run_id)
        return f"tb-{safe}"[:32]

    @staticmethod
    def _try_stop_instance(name: str) -> None:
        try:
            subprocess.run(
                ["singularity", "instance", "stop", name],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            logger.debug("instance stop %s: ignored exception", name, exc_info=True)


# Default export used by the driver when no AB_DEPLOYMENT_CONTAINER_BACKEND override is set.
# Picks docker on a host with docker, singularity otherwise.
def _default_harness_cls():
    if shutil.which("docker"):
        return TerminalBenchDockerHarness
    if shutil.which("singularity"):
        return TerminalBenchSingularityHarness
    raise RuntimeError(
        "neither docker nor singularity is on PATH; "
        "cannot construct a TerminalBench harness"
    )


# Back-compat alias for `from agent_bench.benchmarks.terminal_bench.app import TerminalBenchHarness`.
# Driver selects via AB_DEPLOYMENT_CONTAINER_BACKEND now (see agent_bench/runtime/driver.py).
TerminalBenchHarness = TerminalBenchDockerHarness
