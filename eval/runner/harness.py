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
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Host paths — override via env if your OpenClaw lives elsewhere.
# ----------------------------------------------------------------------
HOST_NODE_BIN = os.environ.get("OC_HOST_NODE", "/usr/local/bin/node")
HOST_OPENCLAW_PKG = os.environ.get(
    "OC_HOST_OPENCLAW", "/usr/local/lib/node_modules/openclaw"
)

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
# Fairness validator — unchanged from Claude's original.
# ----------------------------------------------------------------------

def validate_fairness_constraints(run_configs: List):
    instance_sets: Dict[str, set] = {}
    model_families: Dict[str, set] = {}
    serving_endpoints: Dict[str, set] = {}
    repo_commits: Dict[str, set] = {}
    container_images: Dict[str, set] = {}
    wall_budgets: Dict[str, set] = {}

    for cfg in run_configs:
        sweep_key = cfg.condition.condition_id
        instance_sets.setdefault(sweep_key, set()).add(cfg.instance_id)
        model_families.setdefault(sweep_key, set()).add(cfg.condition.model.name)
        serving_endpoints.setdefault(sweep_key, set()).add(cfg.condition.model.endpoint)
        repo_commits.setdefault(sweep_key, set()).add((cfg.repo_id, cfg.base_commit))
        container_images.setdefault(sweep_key, set()).add(cfg.container_image)
        wall_budgets.setdefault(sweep_key, set()).add(cfg.wall_clock_budget_seconds)

    all_instance_sets = list(instance_sets.values())
    if len(set(frozenset(s) for s in all_instance_sets)) > 1:
        raise ValueError(f"FAIRNESS VIOLATION: instance sets: {instance_sets}")

    for k, v in model_families.items():
        if len(v) > 1:
            logger.warning("Multiple model families in %s: %s", k, v)
    for k, v in serving_endpoints.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: endpoints in {k}: {v}")
    for k, v in repo_commits.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: commits in {k}: {v}")
    for k, v in container_images.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: images in {k}: {v}")
    for k, v in wall_budgets.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: budgets in {k}: {v}")

    logger.info("Fairness constraints validated across all runs.")