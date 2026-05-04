"""SWE-bench harness integration — container-based, B1-a model.

Model:
    - Per-run: `docker run -d` a SWE-bench task image as a persistent container.
    - Host Node.js + host OpenClaw package mounted read-only.
    - Conditions interact via `docker cp` and `docker exec`.
    - Evaluation delegates to SWE-bench's run_evaluation (which handles its own
      scoring containers internally).

Mounts per run container:
    /usr/local/bin/node            : host Node runtime (ro)
    /opt/openclaw                  : host OpenClaw package incl. node_modules (ro)
    (workspace stays inside the container at /testbed — SWE-bench image has it)

Networking: --network host. vLLM is reachable at http://localhost:8000 inside
the container.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Host paths — override via env if your OpenClaw lives elsewhere.
# ----------------------------------------------------------------------
HOST_NODE_BIN = os.environ.get("OC_HOST_NODE", "/usr/local/bin/node")


def _autodetect_openclaw_pkg() -> str:
    """Find the openclaw package dir. Prefers a user-level npm install whose
    `dist/extensions/*/node_modules` are already populated — required because
    we mount the package read-only, so OpenClaw cannot npm-install plugin
    runtime deps at startup. Falls back to the system path with a clear error
    surface in the constructor below.
    """
    env = os.environ.get("OC_HOST_OPENCLAW")
    if env:
        return env
    # Resolve `openclaw` on PATH → its package dir is two levels up from
    # bin/openclaw (typically <prefix>/bin/openclaw → <prefix>/lib/node_modules/openclaw).
    bin_path = shutil.which("openclaw")
    if bin_path:
        real = os.path.realpath(bin_path)
        # `openclaw.mjs` lives at <pkg>/openclaw.mjs; bin link points to it.
        if os.path.basename(real) in ("openclaw.mjs", "openclaw"):
            cand = os.path.dirname(real)
            if os.path.exists(os.path.join(cand, "dist", "extensions")):
                return cand
        # npm-style: <prefix>/bin/openclaw → <prefix>/lib/node_modules/openclaw
        prefix = os.path.dirname(os.path.dirname(bin_path))
        cand = os.path.join(prefix, "lib", "node_modules", "openclaw")
        if os.path.isdir(cand):
            return cand
    return "/usr/local/lib/node_modules/openclaw"


HOST_OPENCLAW_PKG = _autodetect_openclaw_pkg()

# Per-container Gateway port range. Avoids 18789 (host's other gateway) and
# leaves headroom for sidecars (Gateway picks port+2 for the browser plugin
# control surface). Stride of 4 → up to ~25 concurrent runs.
GATEWAY_PORT_MIN = int(os.environ.get("OC_GATEWAY_PORT_MIN", "18800"))
GATEWAY_PORT_MAX = int(os.environ.get("OC_GATEWAY_PORT_MAX", "18899"))
GATEWAY_PORT_STRIDE = 4


# Process-global reservation — the bind probe releases immediately, so two
# concurrent threads can race on the same port. Hold a logical reservation
# from `_pick_free_gateway_port` until `_release_gateway_port`.
_GATEWAY_PORT_LOCK = threading.Lock()
_GATEWAY_PORTS_INUSE: Set[int] = set()


def _pick_free_gateway_port(attempts: int = 25) -> int:
    """Reserve a free port in [GATEWAY_PORT_MIN, GATEWAY_PORT_MAX].

    Range stride accounts for Gateway opening a sidecar at port+2 (browser
    plugin control surface). Reservation is process-global so concurrent
    callers don't collide.
    """
    import random
    candidates = list(range(GATEWAY_PORT_MIN, GATEWAY_PORT_MAX + 1, GATEWAY_PORT_STRIDE))
    random.shuffle(candidates)
    with _GATEWAY_PORT_LOCK:
        for port in candidates[:attempts]:
            if port in _GATEWAY_PORTS_INUSE or (port + 2) in _GATEWAY_PORTS_INUSE:
                continue
            try:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port + 2))
            except OSError:
                continue
            _GATEWAY_PORTS_INUSE.add(port)
            _GATEWAY_PORTS_INUSE.add(port + 2)
            return port
    raise RuntimeError(
        f"no free Gateway port in [{GATEWAY_PORT_MIN},{GATEWAY_PORT_MAX}]"
    )


def wrap_singularity_argv(
    argv: List[str], env: Dict, binds: List, sif: str, overlay: Optional[str] = None,
) -> tuple:
    """Prefix `argv` with `singularity exec [--overlay <ov>:ro] --bind ... <sif>`.

    Caller-supplied `sif` and `overlay` (no env-var sniffing) so call sites
    that already know they're on the singularity backend can wrap explicitly.

    `binds` is a list of (src, dst) path pairs. Use src==dst when possible so
    absolute paths in env vars (OPENCLAW_STATE_DIR etc) resolve identically
    inside the container.

    OPENCLAW_*, OTEL_*, OC_HOST_* env vars are forwarded into the container
    via the SINGULARITYENV_ prefix.
    """
    bind_args: List[str] = []
    for src, dst in binds:
        if src and dst:
            bind_args += ["--bind", f"{src}:{dst}"]

    overlay_args: List[str] = []
    if overlay:
        # `:ro` keeps the patched layer immutable across concurrent runs.
        overlay_args = ["--overlay", f"{overlay}:ro"]

    new_argv = (
        ["singularity", "exec"] + overlay_args + bind_args + [sif] + argv
    )
    new_env = dict(env)
    for k in list(new_env.keys()):
        if k.startswith(("OPENCLAW_", "OTEL_", "OC_HOST_")):
            new_env[f"SINGULARITYENV_{k}"] = new_env[k]
    return new_argv, new_env


def _kill_pgrp(pid: int) -> None:
    """SIGTERM then SIGKILL the process group rooted at pid."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        time.sleep(0.1)
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _release_gateway_port(port: int) -> None:
    with _GATEWAY_PORT_LOCK:
        _GATEWAY_PORTS_INUSE.discard(port)
        _GATEWAY_PORTS_INUSE.discard(port + 2)


class SWEBenchHarness:
    """Per-run container harness for SWE-bench tasks.

    prepare_environment:  docker run -d  → returns container_id
    apply_patch:          docker cp + docker exec git apply
    run_tests:            docker exec git diff to collect patch,
                          then swebench.harness.run_evaluation to score
    cleanup:              docker stop && docker rm
    """

    def __init__(self, swebench_root: Path, workspace_base: Optional[Path] = None):
        self.swebench_root = Path(swebench_root)

        try:
            import swebench  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"swebench Python package not importable from {self.swebench_root}. "
                "Run `pip install -e .` inside the SWE-bench clone."
            ) from e

        for path, label in [
            (HOST_NODE_BIN, "Node.js binary (set OC_HOST_NODE to override)"),
            (HOST_OPENCLAW_PKG, "OpenClaw package dir (set OC_HOST_OPENCLAW to override)"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} not found at {path}")

        # The package is mounted read-only, so OpenClaw cannot install
        # plugin runtime deps at startup. Require them to be pre-populated
        # in the host package — fail fast with a clear remedy instead of
        # letting the agent crash inside the container with cryptic ENOENT.
        ext_root = Path(HOST_OPENCLAW_PKG) / "dist" / "extensions"
        sentinel = ext_root / "diagnostics-otel" / "node_modules"
        if not sentinel.exists():
            raise RuntimeError(
                f"OpenClaw plugin runtime deps not installed at {ext_root}.\n"
                "The package is mounted read-only into eval containers, so "
                "OpenClaw cannot npm-install plugins at startup. Use a host "
                "openclaw whose extensions/*/node_modules are already populated "
                "(e.g. a user-level `npm i -g openclaw` install), and set "
                "OC_HOST_OPENCLAW to point at it."
            )

        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"docker not available: {r.stderr.strip()}")

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,
        base_commit: str,
        container_image: str,
        workspace_root: str = "/testbed",
        run_id: Optional[str] = None,
    ) -> Dict:
        logger.info("Preparing container for %s (image=%s)", instance_id, container_image)

        container_name = f"oc-eval-{instance_id}-{os.urandom(3).hex()}"

    
        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        oc_config_file = f"{kit_root}/openclaw_state/openclaw.json"
        oc_agents_dir = f"{kit_root}/openclaw_state/agents"

        for path, label in [
            (oc_config_file, "openclaw.json"),
            (oc_agents_dir, "agents dir"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"OpenClaw {label} not found at {path}. "
                    f"Set KIT_ROOT or prepare openclaw_state/ directory."
                )

        # Plugin cache dir — lazily populated by the first container start.
        # Location is a per-host dir outside the project (stale cache is easy
        # to blow away: rm -rf $OC_PLUGIN_CACHE).
        oc_plugin_cache = os.environ.get(
            "OC_PLUGIN_CACHE",
            os.path.expanduser("~/.cache/oc-eval/openclaw-plugin-deps"),
        )
        os.makedirs(oc_plugin_cache, exist_ok=True)

        # Mount the whole npm package dir so the `require('../lib/cli.js')`
        # inside /usr/local/lib/node_modules/npm/bin/npm-cli.js resolves.
        # Do NOT mount /usr/local/bin/npm itself — on many hosts it's a
        # symlink and Docker doesn't follow it. Instead we create a container-
        # side symlink after startup (see the exec below).
        host_npm_pkg = os.environ.get(
            "OC_HOST_NPM_PKG", "/usr/local/lib/node_modules/npm"
        )
        if not os.path.exists(host_npm_pkg):
            raise FileNotFoundError(
                f"npm package dir not found at {host_npm_pkg}. "
                "Set OC_HOST_NPM_PKG if npm is installed elsewhere."
            )

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",
            "-v", f"{HOST_NODE_BIN}:/usr/local/bin/node:ro",
            "-v", f"{host_npm_pkg}:/usr/local/lib/node_modules/npm:ro",
            "-v", f"{HOST_OPENCLAW_PKG}:/opt/openclaw:ro",
            "-v", f"{oc_config_file}:/root/.openclaw/openclaw.json:ro",
            # agents/ cannot be mounted wholesale: OpenClaw mkdirs sessions/,
            # memory/ etc. under the agent's dir at runtime. Mount only the
            # static prompt files, leaving the parent dir writable.
            *sum((
                ["-v", f"{oc_agents_dir}/swebench/{f}:/root/.openclaw/agents/swebench/{f}:ro"]
                for f in ("AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md", "HEARTBEAT.md")
            ), []),
            # Plugin runtime deps cache — shared across runs to avoid re-downloading
            # hundreds of MB of Node modules on every container start. First container
            # to start populates it; subsequent containers reuse. rw because OpenClaw
            # may touch lockfiles etc. even on reuse.
            "-v", f"{oc_plugin_cache}:/root/.openclaw/plugin-runtime-deps:rw",
            "-w", workspace_root,
            "--entrypoint", "sleep",
            container_image,
            "infinity",
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {instance_id}: {r.stderr.strip()}"
            )
        container_id = r.stdout.strip()

        head = self._exec(
            container_id, ["git", "-C", workspace_root, "rev-parse", "HEAD"],
            check=False,
        )
        if head.returncode == 0:
            got = head.stdout.strip()
            if got != base_commit:
                logger.warning(
                    "Container HEAD %s != base_commit %s — checking out.",
                    got[:12], base_commit[:12],
                )
                self._exec(
                    container_id,
                    ["git", "-C", workspace_root, "checkout", "-q", base_commit],
                )

        # Create PATH entries for our mounted tooling.
        # chmod may fail (ro fs) but ln -sf and the symlinks work anyway.
        self._exec(
            container_id,
            ["ln", "-sf", "/opt/openclaw/openclaw.mjs", "/usr/local/bin/openclaw"],
            check=False,
        )
        self._exec(
            container_id,
            ["ln", "-sf", "/usr/local/lib/node_modules/npm/bin/npm-cli.js", "/usr/local/bin/npm"],
            check=False,
        )

        # Start the Gateway inside the container. openclaw_agent uses gateway
        # mode (no --local) so tool calls don't fall back and OTel spans are
        # emitted by the patched server-side dispatcher. AR baselines don't
        # need the gateway but starting it unconditionally is cheap once the
        # plugin cache is warm and keeps prepare_environment per-condition
        # uniform.
        gateway_port = _pick_free_gateway_port()
        gateway_token = secrets.token_hex(24)
        gateway_url = f"ws://127.0.0.1:{gateway_port}"

        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        gw_up = f"{kit_root}/eval/frontend/openclaw/gateway_up.sh"
        if not os.path.exists(gw_up):
            raise FileNotFoundError(f"missing helper: {gw_up}")
        gw_up_argv = ["bash", gw_up, container_id, str(gateway_port), gateway_token]
        if run_id:
            gw_up_argv.append(run_id)
        r = subprocess.run(
            gw_up_argv,
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            logger.error("gateway-up failed: %s", r.stderr.strip())
            raise RuntimeError(
                f"openclaw_gateway_up.sh failed (rc={r.returncode}): {r.stderr.strip()}"
            )
        logger.info(
            "  gateway ready at %s in container %s", gateway_url, container_name
        )

        logger.info("  container %s ready at %s", container_name, workspace_root)
        return {
            "container_id": container_id,
            "container_name": container_name,
            "backend": "docker",
            "workspace_root": workspace_root,
            "gateway_port": gateway_port,
            "gateway_token": gateway_token,
            "gateway_url": gateway_url,
        }

    def _exec(
        self,
        container_id: str,
        argv: List[str],
        check: bool = True,
        timeout: Optional[int] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        cmd = ["docker", "exec", container_id] + argv
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, input=input_text
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({argv[0]}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r

    # ------------------------------------------------------------------
    # Patch application (conditions that produce .patch files use this)
    # ------------------------------------------------------------------

    def apply_patch(
        self,
        container_id: str,
        patch_path: Path,
        workspace_root: str,
    ) -> bool:
        logger.info("Applying patch %s in %s", patch_path, container_id[:12])

        target = f"{workspace_root}/oc_candidate.patch"
        r = subprocess.run(
            ["docker", "cp", str(patch_path), f"{container_id}:{target}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error("docker cp failed: %s", r.stderr)
            return False

        dry = self._exec(
            container_id,
            ["git", "-C", workspace_root, "apply", "--check", target],
            check=False,
        )
        if dry.returncode != 0:
            logger.warning("Patch does not apply: %s", dry.stderr[-300:])
            return False

        self._exec(container_id, ["git", "-C", workspace_root, "apply", target])
        return True

    # ------------------------------------------------------------------
    # Patch collection (conditions that edit files in place use this)
    # ------------------------------------------------------------------

    # Files OpenClaw writes into any workspace on first run. These must be
    # removed before diffing so they don't leak into the SWE-bench patch.
    _OC_SCAFFOLD_FILES = (
        "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
        "BOOT.md MEMORY.md fix.patch"
    )
    _OC_SCAFFOLD_DIRS = (".openclaw memory sessions")

    def collect_patch(self, container_id: str, workspace_root: str) -> str:
        # Strip OpenClaw scaffolding that was written into the workspace at
        # agent startup. These are not part of the SWE-bench task and would
        # cause run_evaluation to reject the patch.
        ws = workspace_root
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OC_SCAFFOLD_FILES} && "
             f"rm -rf {self._OC_SCAFFOLD_DIRS}"],
            check=False,
        )
        # Stage + diff.
        self._exec(container_id, ["git", "-C", ws, "add", "-A"])
        r = self._exec(container_id, ["git", "-C", ws, "diff", "--cached"])
        return r.stdout

    # ------------------------------------------------------------------
    # Scoring via SWE-bench's run_evaluation
    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: str,
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        **_kw,
    ) -> Dict:
        logger.info("Scoring %s", instance_id)

        if patch is None:
            patch = self.collect_patch(container_id, workspace_root)

        if not patch.strip():
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 1,
                "error_message": "Empty patch (agent made no changes)",
                "evaluation_dir": None,
            }

        if run_id is None:
            run_id = f"oc-eval-{instance_id}-{os.urandom(3).hex()}"

        with tempfile.TemporaryDirectory(prefix="oc_pred_") as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "predictions.json"
            pred_path.write_text(
                json.dumps(
                    [{
                        "instance_id": instance_id,
                        "model_name_or_path": "oc-eval",
                        "model_patch": patch,
                    }]
                )
            )

            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Verified",
                "--predictions_path", str(pred_path),
                "--instance_ids", instance_id,
                "--max_workers", "1",
                "--run_id", run_id,
                "--cache_level", "env",
                "--timeout", str(timeout_seconds),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.swebench_root),
                    capture_output=True, text=True,
                    timeout=timeout_seconds + 300,
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "TIMEOUT",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": "run_evaluation wall-clock timeout",
                    "evaluation_dir": None,
                }

            if proc.returncode != 0:
                logger.error("run_evaluation stderr:\n%s", proc.stderr[-2000:])
                return {
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": f"run_evaluation exit {proc.returncode}: {proc.stderr[-500:]}",
                    "evaluation_dir": None,
                }

            return self._find_and_parse_verdict(run_id, instance_id)

    def _find_and_parse_verdict(self, run_id: str, instance_id: str) -> Dict:
        """Locate the report SWE-bench wrote and parse it.

        Layout SWE-bench actually uses (as of the version we're running):
            <swebench_root>/logs/run_evaluation/<run_id>/<model>/<instance>/report.json
        Our predictions.json passes "model_name_or_path": "oc-eval", so the model
        dir is literally 'oc-eval'. We also rglob as a fallback in case the
        layout shifts across versions.
        """
        # Primary path (current SWE-bench layout).
        primary = (
            self.swebench_root
            / "logs" / "run_evaluation" / run_id / "oc-eval" / instance_id
            / "report.json"
        )
        if primary.exists():
            return self._parse_verdict(primary, instance_id)

        # Secondary: older "evaluation_results/" layout, kept for forward
        # compatibility if we ever run against an older install.
        legacy = (
            self.swebench_root
            / "evaluation_results" / run_id / instance_id / "report.json"
        )
        if legacy.exists():
            return self._parse_verdict(legacy, instance_id)

        # Fallback: search under the likely roots for any matching report.
        for root in (
            self.swebench_root / "logs" / "run_evaluation" / run_id,
            self.swebench_root / "evaluation_results" / run_id,
        ):
            if root.exists():
                matches = list(root.rglob("report.json"))
                if matches:
                    return self._parse_verdict(matches[0], instance_id)

        return {
            "status": "ERROR",
            "test_pass_rate": 0.0,
            "passed": 0,
            "failed": 0,
            "error_message": f"report.json not found for {instance_id} (run_id={run_id})",
            "evaluation_dir": str(primary.parent),
        }

    def _parse_verdict(self, report_path: Path, instance_id: str) -> Dict:
        try:
            report = json.loads(report_path.read_text())
        except Exception as e:
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0,
                "failed": 0,
                "error_message": f"Failed to parse {report_path}: {e}",
                "evaluation_dir": str(report_path.parent),
            }

        inst = report.get(instance_id, report)
        resolved = inst.get("resolved", False)

        ts = inst.get("tests_status", {})
        f2p = ts.get("FAIL_TO_PASS", {"success": [], "failure": []})
        p2p = ts.get("PASS_TO_PASS", {"success": [], "failure": []})

        passed = len(f2p.get("success", [])) + len(p2p.get("success", []))
        failed = len(f2p.get("failure", [])) + len(p2p.get("failure", []))
        total = passed + failed

        return {
            "status": "PASS" if resolved else "FAIL",
            "test_pass_rate": passed / total if total else 0.0,
            "passed": passed,
            "failed": failed,
            "evaluation_dir": str(report_path.parent),
            "raw_report_path": str(report_path),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self,
        container_id: Optional[str] = None,
        workspace_root: Optional[str] = None,
        gateway_port: Optional[int] = None,
        **_extra,  # ignore host-mode kwargs (gateway_pid, host_state_dir)
    ):
        if gateway_port is not None:
            _release_gateway_port(gateway_port)
        if not container_id:
            return
        logger.info("Cleaning up container %s", container_id[:12])
        subprocess.run(["docker", "stop", "-t", "5", container_id],
                       capture_output=True, check=False)
        subprocess.run(["docker", "rm", "-f", container_id],
                       capture_output=True, check=False)


# ----------------------------------------------------------------------
# HostHarness — no-docker variant.
#
# Runs the agent directly on the host. The repo is cloned into a temporary
# workspace (no SWE-bench task image needed). Scoring is intentionally
# deferred — emits a `predictions.json` plus a verdict with status=DEFERRED
# so a docker-equipped machine can later run `swebench.harness.run_evaluation`
# against the predictions and produce the real verdict.
#
# Use case: cloud nodes without docker. Trade-off: no isolation, no per-image
# Python env, no scoring locally.
# ----------------------------------------------------------------------


class HostHarness:
    """Host-side harness. Same surface as SWEBenchHarness but no docker.

    Per run we:
      1. Clone the repo into a host workspace dir.
      2. Stand up a per-run OPENCLAW_STATE_DIR (copied from openclaw_state/
         in the repo, with the swebench agent's `workspace` rewritten to
         point at the cloned dir).
      3. Start `openclaw gateway run` as a background process bound to a
         loopback port so the agent can route via gateway (matches docker
         mode — needed for OTel span emission, since OpenClaw --local does
         not emit OTel).

    Spans go to OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318
    from openclaw.json). If no collector is reachable they're silently
    dropped — agent still runs, just no OTel artifacts. Run `otelcol-contrib`
    natively on the host (no docker required) if you want spans.
    """

    # Match "[gateway] ready (...)" tolerant of ANSI color codes inserted
    # by openclaw between the bracket and the word "ready".
    _GATEWAY_READY_RE = re.compile(r"\[gateway\][^\n]{0,40}ready", re.IGNORECASE)
    _GATEWAY_READY_TIMEOUT_SECONDS = 180

    # Identifies the backend in the env_info dict returned by prepare_environment.
    # Conditions branch on this instead of sniffing OC_OC_SIF env vars.
    backend_id = "host"

    def __init__(self, swebench_root: Path, workspace_base: Optional[Path] = None):
        # swebench_root retained for API parity (driver passes it). It's not
        # required for host-mode agent execution; it is needed only if you
        # later run scoring on a docker-equipped machine.
        self.swebench_root = Path(swebench_root) if swebench_root else None
        self.workspace_base = Path(
            workspace_base or os.environ.get(
                "OC_HOST_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        # Per-run state dirs live under the workspace base so they're easy to
        # clean en masse. The directory itself is small (kBs).
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)

        if not shutil.which("openclaw"):
            raise RuntimeError(
                "openclaw not on PATH. Install (`npm i -g openclaw`) or set OC_BIN."
            )
        if not shutil.which("git"):
            raise RuntimeError("git not on PATH.")

    # ------------------------------------------------------------------

    def prepare_environment(
        self,
        instance_id: str,
        repo_id: str,
        base_commit: str,
        container_image: Optional[str] = None,  # ignored
        workspace_root: Optional[str] = None,    # ignored — we pick our own
        run_id: Optional[str] = None,
    ) -> Dict:
        ws = self.workspace_base / (run_id or instance_id)
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)

        logger.info("Cloning %s @ %s into %s", repo_id, base_commit[:12], ws)
        # Local repo cache to avoid re-cloning for every run of the same repo.
        cache = self.workspace_base.parent / "repo-cache" / repo_id.replace("/", "__")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            r = subprocess.run(
                ["git", "clone", "--bare", f"https://github.com/{repo_id}.git", str(cache)],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(f"git clone failed for {repo_id}: {r.stderr[-400:]}")

        # Working tree at base_commit. Use clone --reference to share objects.
        r = subprocess.run(
            ["git", "clone", "--reference", str(cache),
             f"https://github.com/{repo_id}.git", str(ws)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone (working tree) failed: {r.stderr[-400:]}")
        r = subprocess.run(
            ["git", "-C", str(ws), "checkout", "-q", base_commit],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git checkout {base_commit} failed: {r.stderr[-400:]}")

        logger.info("  host workspace ready at %s", ws)

        # Per-run state dir: copy openclaw_state and rewrite the swebench
        # agent's workspace to our cloned dir.
        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        src_state = (
            os.environ.get("OC_HOST_STATE_TEMPLATE")
            or os.path.join(kit_root, "openclaw_state")
        )
        if not os.path.isdir(src_state):
            raise FileNotFoundError(
                f"openclaw_state template not found at {src_state} "
                "(set OC_HOST_STATE_TEMPLATE or place an openclaw_state/ dir at KIT_ROOT)."
            )
        state_dir = self.state_base / (run_id or instance_id)
        if state_dir.exists():
            shutil.rmtree(state_dir)
        shutil.copytree(src_state, state_dir)
        cfg_path = state_dir / "openclaw.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        # New OC schema: workspace + tools live at top level (or under
        # agents.defaults), not on each agent.list item.
        cfg.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = str(ws)
        # tools.exec.host=node would route execs through a paired node host;
        # gateway-routed exec keeps tool.call spans flowing without needing
        # a separate node process.
        tools = cfg.setdefault("tools", {})
        exec_cfg = tools.setdefault("exec", {})
        exec_cfg["host"] = "gateway"
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)

        # Start a host gateway so the agent emits OTel + uses the patched
        # gateway-side dispatcher (matches docker mode behavior).
        gw = self._start_host_gateway(state_dir, run_id)

        return {
            # Sentinel: container_id=None signals non-docker mode to conditions.
            "container_id": None,
            "container_name": f"{self.backend_id}-{instance_id}",
            "backend": self.backend_id,
            "workspace_root": str(ws),
            "host_state_dir": str(state_dir),
            "gateway_port": gw["port"],
            "gateway_token": gw["token"],
            "gateway_url": f"ws://127.0.0.1:{gw['port']}",
            "gateway_pid": gw["pid"],
            "gateway_log": gw["log_path"],
            "node_pid": gw.get("node_pid"),
            "node_log": gw.get("node_log_path"),
            **self._extra_env_info(str(ws), str(state_dir)),
        }

    # ------------------------------------------------------------------
    # Hooks for backend variants (overridden by SingularityHarness).

    def _wrap_gateway_argv(
        self, argv: List[str], env: Dict, state_dir: Path,
    ) -> tuple:
        """Default host-mode: no wrapping. Overridden in SingularityHarness."""
        return argv, env

    def _extra_env_info(self, workspace_root: str, state_dir: str) -> Dict:
        """Backend-specific keys to merge into env_info. Default: none."""
        return {}

    # ------------------------------------------------------------------

    def _start_host_gateway(
        self, state_dir: Path, run_id: Optional[str],
    ) -> Dict:
        port = _pick_free_gateway_port()
        token = secrets.token_hex(24)
        log_path = state_dir / "openclaw_gateway.log"

        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(state_dir)
        env["OPENCLAW_GATEWAY_TOKEN"] = token
        env["OPENCLAW_GATEWAY_PORT"] = str(port)
        if run_id:
            env["OTEL_RESOURCE_ATTRIBUTES"] = f"oc.run_id={run_id}"

        log_fh = open(log_path, "w")
        argv = ["openclaw", "gateway", "run",
                "--port", str(port),
                "--auth", "token",
                "--bind", "loopback"]
        argv, env = self._wrap_gateway_argv(argv, env, state_dir)
        proc = subprocess.Popen(
            argv,
            stdout=log_fh, stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if os.name == "posix" else None,
        )

        # Poll log for readiness marker.
        deadline = time.time() + self._GATEWAY_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            if proc.poll() is not None:
                _release_gateway_port(port)
                tail = log_path.read_text()[-1000:]
                raise RuntimeError(
                    f"{self.backend_id} gateway exited (rc={proc.returncode}) before ready. "
                    f"Last log:\n{tail}"
                )
            try:
                if self._GATEWAY_READY_RE.search(log_path.read_text()):
                    logger.info(
                        "  %s gateway ready on port %d (pid=%d)",
                        self.backend_id, port, proc.pid,
                    )
                    return {"pid": proc.pid, "port": port, "token": token,
                            "log_path": str(log_path),
                            "node_pid": None, "node_log_path": None}
            except OSError:
                pass
            time.sleep(1)

        # Timed out — kill and report.
        _kill_pgrp(proc.pid)
        _release_gateway_port(port)
        tail = log_path.read_text()[-1500:]
        raise RuntimeError(
            f"{self.backend_id} gateway did not become ready in "
            f"{self._GATEWAY_READY_TIMEOUT_SECONDS}s.\nLast log:\n{tail}"
        )

    def _start_host_node(
        self, state_dir: Path, gateway_port: int, gateway_token: str,
    ) -> Dict:
        """Spawn `openclaw node run` so tools.exec.host=node has a paired node.

        Without a paired node host the agent's exec calls fail with
        "requires a paired node (none available)". The node host registers
        with the gateway over WebSocket and auto-pairs (gateway logs
        `device pairing auto-approved`). We poll for the pair line and
        return the pid so cleanup can stop it.
        """
        log_path = state_dir / "openclaw_node.log"
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(state_dir)
        env["OPENCLAW_GATEWAY_TOKEN"] = gateway_token

        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            ["openclaw", "node", "run",
             "--host", "127.0.0.1",
             "--port", str(gateway_port)],
            stdout=log_fh, stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if os.name == "posix" else None,
        )

        # Poll the *gateway* log for the auto-approve line — that's our signal
        # the node has paired and exec.host=node will resolve.
        gw_log = state_dir / "openclaw_gateway.log"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = log_path.read_text()[-800:]
                raise RuntimeError(
                    f"{self.backend_id} node exited (rc={proc.returncode}) before pairing. "
                    f"Last log:\n{tail}"
                )
            try:
                if "device pairing auto-approved" in gw_log.read_text():
                    logger.info("  %s node paired (pid=%d)", self.backend_id, proc.pid)
                    return {"pid": proc.pid, "log_path": str(log_path)}
            except OSError:
                pass
            time.sleep(0.5)

        _kill_pgrp(proc.pid)
        tail = log_path.read_text()[-1000:]
        raise RuntimeError(
            f"{self.backend_id} node did not pair in 60s. Last log:\n{tail}"
        )

    # ------------------------------------------------------------------

    def run_tests(
        self,
        container_id: Optional[str],
        instance_id: str,
        workspace_root: str,
        timeout_seconds: int,
        patch: Optional[str] = None,
        run_id: Optional[str] = None,
        instance_data: Optional[Dict] = None,
    ) -> Dict:
        """Score the patch. Three back-ends, picked by env (in order):

        1. `OC_SCORING_SIF_DIR` set → run pytest inside a per-instance
           Singularity .sif (`<dir>/<instance_id>.sif`). No docker, no
           server. Best fit for HPC clusters where docker is unavailable.
        2. `OC_SCORING_URL` set → POST to a self-hosted scoring server
           (eval/scoring_server/main.py on a docker host).
        3. Otherwise defer: drop predictions.json next to the workspace.
        """
        if patch is None:
            patch = ""

        if not patch.strip():
            return {
                "status": "FAIL",
                "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": "empty patch — nothing to score",
                "evaluation_dir": None,
            }

        sif_dir = os.environ.get("OC_SCORING_SIF_DIR")
        if sif_dir:
            return self._score_via_singularity(
                Path(sif_dir), instance_id, instance_data,
                workspace_root, timeout_seconds, run_id,
            )

        scoring_url = os.environ.get("OC_SCORING_URL")
        if scoring_url:
            return self._score_via_server(
                scoring_url, instance_id, patch, timeout_seconds,
            )

        # No scoring backend → defer.
        pred_path = Path(workspace_root).parent / f"predictions-{instance_id}.json"
        pred_path.write_text(json.dumps([{
            "instance_id": instance_id,
            "model_name_or_path": "oc-eval",
            "model_patch": patch,
        }], indent=2))
        return {
            "status": "DEFERRED",
            "test_pass_rate": 0.0,
            "passed": 0, "failed": 0,
            "error_message": (
                f"host mode: scoring deferred. predictions written to {pred_path}. "
                "Set OC_SCORING_URL to a scoring server, or score manually with "
                "`python -m swebench.harness.run_evaluation`."
            ),
            "evaluation_dir": None,
            "predictions_path": str(pred_path),
        }

    def _score_via_singularity(
        self,
        sif_dir: Path,
        instance_id: str,
        instance_data: Optional[Dict],
        workspace_root: str,
        timeout_seconds: int,
        run_id: Optional[str],
    ) -> Dict:
        """Score by `singularity exec`-ing the per-instance SWE-bench image.

        Reuses swebench's official scoring path: builds the same eval_script
        the docker harness would run, executes it inside the .sif with the
        host workspace bound at /testbed, captures the test log, and parses
        it through `swebench.harness.grading.get_eval_report` so the verdict
        shape is identical to docker mode.
        """
        if instance_data is None:
            return {
                "status": "ERROR", "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": "singularity scoring needs instance_data dict",
                "evaluation_dir": None,
            }
        sif_path = sif_dir / f"{instance_id}.sif"
        if not sif_path.exists():
            return {
                "status": "ERROR", "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": f"sif missing: {sif_path}. Pull with pbs/pull_images.sh",
                "evaluation_dir": None,
            }
        if not shutil.which("singularity"):
            return {
                "status": "ERROR", "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": "singularity not on PATH",
                "evaluation_dir": None,
            }

        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
            from swebench.harness.grading import get_eval_report
        except ImportError as e:
            return {
                "status": "ERROR", "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": f"swebench not importable: {e}",
                "evaluation_dir": None,
            }

        spec = make_test_spec(instance_data)
        eval_dir = Path(workspace_root).parent / f"eval-{instance_id}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        script_path = eval_dir / "eval.sh"
        script_path.write_text(spec.eval_script)
        log_path = eval_dir / "test_output.txt"

        # Bind host workspace at /testbed (matches the path the eval_script
        # was generated for). The .sif image already has /testbed populated
        # at base_commit; binding our workspace shadows it with our edits.
        cmd = [
            "singularity", "exec",
            "--bind", f"{workspace_root}:/testbed",
            "--bind", f"{script_path}:/oc_eval.sh:ro",
            str(sif_path),
            "bash", "/oc_eval.sh",
        ]
        logger.info("singularity scoring %s via %s", instance_id, sif_path.name)
        with open(log_path, "wb") as fh:
            try:
                proc = subprocess.run(
                    cmd, stdout=fh, stderr=subprocess.STDOUT,
                    timeout=timeout_seconds + 600,
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "TIMEOUT", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"singularity exec timed out for {instance_id}",
                    "evaluation_dir": str(eval_dir),
                }

        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "oc-eval",
            "model_patch": "<applied-in-workspace>",
        }
        try:
            report_map = get_eval_report(
                spec, prediction, str(log_path), include_tests_status=True,
            )
        except Exception as e:
            return {
                "status": "ERROR", "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": f"get_eval_report failed: {e} (rc={proc.returncode})",
                "evaluation_dir": str(eval_dir),
            }
        inst = report_map.get(instance_id, {})
        resolved = inst.get("resolved", False)
        ts = inst.get("tests_status", {}) or {}
        f2p = ts.get("FAIL_TO_PASS", {"success": [], "failure": []})
        p2p = ts.get("PASS_TO_PASS", {"success": [], "failure": []})
        passed = len(f2p.get("success", [])) + len(p2p.get("success", []))
        failed = len(f2p.get("failure", [])) + len(p2p.get("failure", []))
        total = passed + failed
        return {
            "status": "PASS" if resolved else "FAIL",
            "test_pass_rate": passed / total if total else 0.0,
            "passed": passed,
            "failed": failed,
            "evaluation_dir": str(eval_dir),
            "raw_report_path": str(log_path),
        }

    @staticmethod
    def _score_via_server(
        base_url: str, instance_id: str, patch: str, timeout_seconds: int,
    ) -> Dict:
        """POST to scoring server, poll for verdict."""
        import urllib.request
        import urllib.error

        base_url = base_url.rstrip("/")
        body = json.dumps({
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": "oc-eval",
            "timeout_seconds": timeout_seconds,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/submit",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                state = json.loads(r.read())
        except urllib.error.URLError as e:
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0, "passed": 0, "failed": 0,
                "error_message": f"scoring submit failed: {e}",
                "evaluation_dir": None,
            }
        sid = state["submission_id"]
        logger.info("scoring submitted %s -> sid=%s", instance_id, sid)

        # Poll. Server-side timeout already enforced via run_evaluation;
        # client-side wait gives some slack.
        deadline = time.time() + timeout_seconds + 900
        while time.time() < deadline:
            time.sleep(5)
            try:
                with urllib.request.urlopen(
                    f"{base_url}/v1/submission/{sid}", timeout=30,
                ) as r:
                    state = json.loads(r.read())
            except urllib.error.URLError as e:
                logger.warning("poll failed (%s); retrying", e)
                continue
            status = state.get("status")
            if status == "done":
                return state.get("verdict") or {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": "server returned done with no verdict",
                    "evaluation_dir": None,
                }
            if status == "failed":
                return {
                    "status": "ERROR", "test_pass_rate": 0.0,
                    "passed": 0, "failed": 0,
                    "error_message": f"scoring server failed: {state.get('error')}",
                    "evaluation_dir": None,
                }
        return {
            "status": "TIMEOUT", "test_pass_rate": 0.0,
            "passed": 0, "failed": 0,
            "error_message": f"scoring server poll timed out (sid={sid})",
            "evaluation_dir": None,
        }

    # ------------------------------------------------------------------

    def cleanup(
        self,
        container_id: Optional[str] = None,
        workspace_root: Optional[str] = None,
        gateway_port: Optional[int] = None,
        gateway_pid: Optional[int] = None,
        host_state_dir: Optional[str] = None,
        node_pid: Optional[int] = None,
        **_extra,
    ):
        # Stop node host before gateway; otherwise gateway shutdown leaves it
        # spinning trying to reconnect.
        if node_pid:
            logger.info("Stopping %s node pid=%d", self.backend_id, node_pid)
            _kill_pgrp(node_pid)
        # Stop the gateway so it can flush OTel before we tear state.
        if gateway_pid:
            logger.info("Stopping %s gateway pid=%d", self.backend_id, gateway_pid)
            _kill_pgrp(gateway_pid)
        if gateway_port is not None:
            _release_gateway_port(gateway_port)

        # Persist gateway/node logs out of the per-run state dir so they
        # survive cleanup. Lives under $OC_LOG_DIR/runs/<state-dir-name>/;
        # default $OC_LOG_DIR is $KIT_ROOT/logs (set by the PBS driver).
        if host_state_dir and Path(host_state_dir).exists():
            try:
                log_root = Path(
                    os.environ.get("OC_LOG_DIR")
                    or os.path.join(os.environ.get("KIT_ROOT", os.getcwd()), "logs")
                ) / "runs" / Path(host_state_dir).name
                log_root.mkdir(parents=True, exist_ok=True)
                for name in ("openclaw_gateway.log", "openclaw_node.log"):
                    src = Path(host_state_dir) / name
                    if src.exists():
                        shutil.copy2(src, log_root / name)
            except OSError as e:
                logger.debug("log archive skipped: %s", e)

        keep = os.environ.get("OC_HOST_KEEP_WORKSPACE", "0") == "1"
        if keep:
            logger.info("Keeping host workspace + state (OC_HOST_KEEP_WORKSPACE=1)")
            return
        if host_state_dir and Path(host_state_dir).exists():
            shutil.rmtree(host_state_dir, ignore_errors=True)
        if workspace_root and Path(workspace_root).exists():
            shutil.rmtree(workspace_root, ignore_errors=True)


# ----------------------------------------------------------------------
# SingularityHarness — runs OpenClaw inside an OC runtime .sif.
#
# The host has no openclaw installed; openclaw lives inside an alpine/openclaw
# Singularity image (OC_OC_SIF). Patches (eval/frontend/openclaw/patch/*) live
# in a read-only overlay (OC_OC_OVERLAY). Every `openclaw` invocation is
# wrapped in `singularity exec`. Per-run workspace + state still live on the
# host (so git, scoring sif, etc. operate on plain files).
#
# Use case: HPC clusters where docker is unavailable but singularity is.
# ----------------------------------------------------------------------


class SingularityHarness(HostHarness):
    """Singularity-backed variant of HostHarness.

    Differences from HostHarness:
      - Does not require host `openclaw` on PATH; instead requires a runtime
        sif at OC_OC_SIF (and optional patch overlay at OC_OC_OVERLAY).
      - Gateway/agent invocations run via `singularity exec` with the sif and
        overlay supplied by this harness — not via env-var sniffing.
      - env_info advertises backend="singularity" and the sif/overlay paths so
        conditions can wrap their own `openclaw` calls (e.g. `agent run`).
    """

    backend_id = "singularity"

    def __init__(self, swebench_root: Path, workspace_base: Optional[Path] = None):
        sif = os.environ.get("OC_OC_SIF")
        if not sif:
            raise RuntimeError(
                "SingularityHarness requires OC_OC_SIF "
                "(path to OpenClaw runtime .sif)."
            )
        if not Path(sif).exists():
            raise FileNotFoundError(f"OC_OC_SIF does not exist: {sif}")
        overlay = os.environ.get("OC_OC_OVERLAY") or None
        if overlay and not Path(overlay).exists():
            raise FileNotFoundError(f"OC_OC_OVERLAY does not exist: {overlay}")

        if not shutil.which("singularity"):
            raise RuntimeError("singularity not on PATH")
        if not shutil.which("git"):
            raise RuntimeError("git not on PATH.")

        # Don't call HostHarness.__init__ — it requires host `openclaw`. Mirror
        # only the bits we need (workspace + state base dirs, swebench_root).
        self.swebench_root = Path(swebench_root) if swebench_root else None
        self.workspace_base = Path(
            workspace_base or os.environ.get(
                "OC_HOST_WORKSPACE_BASE",
                os.path.expanduser("~/.cache/oc-eval/workspaces"),
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.state_base = self.workspace_base.parent / "states"
        self.state_base.mkdir(parents=True, exist_ok=True)

        self._sif = sif
        self._overlay = overlay

        # Heads-up if the plugin cache looks empty — first-run npm install
        # against npmjs.org will hang offline (typical for HPC compute nodes).
        cache = Path(self._plugin_cache_dir())
        warm = any(cache.rglob("package.json"))
        if not warm:
            logger.warning(
                "OpenClaw plugin cache at %s is empty. "
                "On HPC compute nodes the gateway will hang on npm install. "
                "Run pbs/warm_oc_plugins.sh on a network-equipped node first.",
                cache,
            )

    def _plugin_cache_dir(self) -> str:
        """Shared plugin runtime deps cache.

        OpenClaw's gateway npm-installs plugin runtime deps (acpx, browser, ...)
        into OPENCLAW_STATE_DIR/plugin-runtime-deps on first start. With per-run
        state dirs that cache would be cold every time *and* HPC compute nodes
        typically have no internet — startup hangs on ECONNRESET. Bind a single
        shared host dir over the per-run path so the cache survives across runs
        and can be pre-populated once on a network-equipped node.
        """
        cache = os.environ.get(
            "OC_PLUGIN_CACHE",
            os.path.expanduser("~/.cache/oc-eval/openclaw-plugin-deps"),
        )
        os.makedirs(cache, exist_ok=True)
        return cache

    def _bind_plugin_cache(self, state_dir: Path) -> tuple:
        """Mkdir the bind target inside state_dir; return (host_cache, target)."""
        target = state_dir / "plugin-runtime-deps"
        target.mkdir(parents=True, exist_ok=True)
        return self._plugin_cache_dir(), str(target)

    def _wrap_gateway_argv(
        self, argv: List[str], env: Dict, state_dir: Path,
    ) -> tuple:
        # Bind state_dir so OPENCLAW_STATE_DIR resolves inside the container,
        # workspace_base (parent of all per-run workspaces) so
        # agents.defaults.workspace in openclaw.json references a real path,
        # and a shared plugin cache so the gateway doesn't try to npm-install
        # plugin runtime deps on every run (and so it works offline once warmed).
        cache_src, cache_dst = self._bind_plugin_cache(state_dir)
        binds = [
            (str(state_dir), str(state_dir)),
            (str(self.workspace_base), str(self.workspace_base)),
            (cache_src, cache_dst),
        ]
        return wrap_singularity_argv(argv, env, binds, self._sif, self._overlay)

    def _extra_env_info(self, workspace_root: str, state_dir: str) -> Dict:
        return {
            "singularity_sif": self._sif,
            "singularity_overlay": self._overlay,
            "plugin_cache_dir": self._plugin_cache_dir(),
        }


# ----------------------------------------------------------------------
# Fairness validator — unchanged from Claude's original.
# ----------------------------------------------------------------------

def validate_fairness_constraints(run_configs: List):
    instance_sets: Dict[str, set] = {}
    model_families: Dict[str, set] = {}
    serving_endpoints: Dict[str, set] = {}

    # Per-instance state: same instance must look identical across all
    # conditions (same repo+commit, same container image, same wall budget).
    # Multi-instance sweeps are fine — different instances naturally span
    # multiple repos.
    repo_commits_per_instance: Dict[str, set] = {}
    container_images_per_instance: Dict[str, set] = {}
    wall_budgets_per_instance: Dict[str, set] = {}

    for cfg in run_configs:
        sweep_key = cfg.condition.condition_id
        instance_sets.setdefault(sweep_key, set()).add(cfg.instance_id)
        model_families.setdefault(sweep_key, set()).add(cfg.condition.model.name)
        serving_endpoints.setdefault(sweep_key, set()).add(cfg.condition.model.endpoint)
        repo_commits_per_instance.setdefault(cfg.instance_id, set()).add(
            (cfg.repo_id, cfg.base_commit)
        )
        container_images_per_instance.setdefault(cfg.instance_id, set()).add(
            cfg.container_image
        )
        wall_budgets_per_instance.setdefault(cfg.instance_id, set()).add(
            cfg.wall_clock_budget_seconds
        )

    all_instance_sets = list(instance_sets.values())
    if len(set(frozenset(s) for s in all_instance_sets)) > 1:
        raise ValueError(f"FAIRNESS VIOLATION: instance sets: {instance_sets}")

    for k, v in model_families.items():
        if len(v) > 1:
            logger.warning("Multiple model families in %s: %s", k, v)
    for k, v in serving_endpoints.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: endpoints in {k}: {v}")
    for inst, v in repo_commits_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: commits for {inst}: {v}")
    for inst, v in container_images_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: images for {inst}: {v}")
    for inst, v in wall_budgets_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: budgets for {inst}: {v}")

    logger.info("Fairness constraints validated across all runs.")