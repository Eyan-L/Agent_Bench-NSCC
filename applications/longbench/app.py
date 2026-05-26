"""LongBench RAG harness (OpenClaw agent + ClawRAG retrieval).

Subclass of :class:`DeepResearchHarness` — reuses its two-container model
(OC gateway + sandbox on a per-run docker network, exec-bridge dispatcher,
gateway-ready wait, cleanup). LongBench-specific overrides:

  - ``prepare_environment`` — verifies the shared ClawRAG service is up, seeds
    ``/workspace/clawrag_search.py`` (collection + url baked in), and gives the
    sandbox container ``--add-host clawrag-host:host-gateway`` so the agent's
    search script can reach ClawRAG on the host.
  - ``_build_state_dir`` — seeds longbench agents/skills + the longbench harness
    config.
  - ``run_tests`` — reads ``predictions.jsonl`` (written by the condition loop)
    and scores it with the subtask's LongBench metric.

instance == one LongBench subtask; the condition loops over its examples reusing
these warm containers (see conditions/openclaw_agent.py).
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen

from applications.conditions.gateway_helpers import (
    pick_free_gateway_port as _pick_free_gateway_port,
    release_gateway_port as _release_gateway_port,
)
from applications.deepresearch_bench.app import DeepResearchHarness
from applications.swebench.app import DEFAULT_OC_IMAGE, _force_rmtree
from orchestration.dispatcher.openclaw_remote import OpenClawRemoteDispatcher
from orchestration.runner.config import endpoint_for_container, load_active_model_and_serving

logger = logging.getLogger(__name__)

# Self-contained task image: ClawRAG (in-process chroma, GPU embeddings) runs
# INSIDE the per-task sandbox container. The agent (and the harness's ingest, via
# `docker exec`) reach it at localhost:8080 inside that container.
DEFAULT_SANDBOX_IMAGE = os.environ.get(
    "LONGBENCH_SANDBOX_IMAGE", "agent-bench/longbench-clawrag-sandbox:1.0")
DEFAULT_AGENT_ID = "longbench_rag"
CLAWRAG_AGENT_URL = "http://localhost:8080"
# GPU device for the task container's embeddings (vLLM is on the other device) and
# a host HF cache mount so the embed model is shared/persisted across runs.
GPU_DEVICE = os.environ.get("LONGBENCH_GPU_DEVICE", "0")
HF_CACHE_HOST = os.environ.get("LONGBENCH_HF_CACHE", "/mnt/vdb/hongrui/longbench/hf_cache")
_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_AGENT_FILES = ("AGENTS.md", "SOUL.md", "USER.md", "BOOTSTRAP.md")

_SEARCH_SCRIPT = '''#!/usr/bin/env python3
"""Seeded by LongBenchHarness. Retrieve top passages from a ClawRAG collection.
Usage: clawrag_search.py "<query>" [k] [collection]  (collection defaults to the
per-run default; the harness passes a per-example collection so examples run in
parallel without colliding)."""
import sys, json, urllib.request
URL = "{url}"
DEFAULT_COLLECTION = "{collection}"
def main():
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    collection = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_COLLECTION
    req = urllib.request.Request(
        URL + "/api/v1/rag/search",
        data=json.dumps({{"query": q, "collections": [collection], "k": k}}).encode(),
        headers={{"Content-Type": "application/json"}}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        docs = json.loads(r.read()).get("documents", [])
    out = [d.get("content", "") for d in docs
           if (d.get("metadata") or {{}}).get("type") != "collection_metadata"]
    print("\\n\\n---\\n\\n".join(out) if out else "(no passages found)")
if __name__ == "__main__":
    main()
'''


class LongBenchHarness(DeepResearchHarness):
    """OpenClaw + ClawRAG harness for LongBench subtasks."""

    def _start_clawrag_in_container(self, task_container_name: str) -> None:
        """Start ClawRAG (clawrag-serve) inside the task container; wait until ready."""
        subprocess.run(["docker", "exec", "-d", task_container_name, "clawrag-serve"],
                       capture_output=True, text=True)
        deadline = time.time() + 180
        while time.time() < deadline:
            r = subprocess.run(
                ["docker", "exec", task_container_name, "curl", "-sf", "-m", "3",
                 "http://localhost:8080/api/v1/rag/collections"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                logger.info("  ClawRAG ready inside %s", task_container_name)
                return
            rr = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", task_container_name],
                capture_output=True, text=True,
            )
            if (rr.stdout or "").strip() != "true":
                raise RuntimeError(f"task container {task_container_name} exited before ClawRAG ready")
            time.sleep(3)
        raise RuntimeError(f"ClawRAG did not come up in {task_container_name} within 180s")

    # ------------------------------------------------------------------
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
        if instance_data is None or instance_data.get("benchmark") != "longbench":
            raise RuntimeError(
                "LongBenchHarness requires instance_data with benchmark=longbench "
                f"(got {None if instance_data is None else instance_data.get('benchmark')!r})"
            )

        # Resolve sandbox image. The app-config / loader placeholder
        # ("longbench/clawrag-sandbox.UNUSED:latest") is not pullable, so fall
        # back to the default reusable sandbox whenever it shows up.
        docker_image = container_image or instance_data.get("docker_image") or DEFAULT_SANDBOX_IMAGE
        if "unused" in docker_image.lower():
            docker_image = DEFAULT_SANDBOX_IMAGE

        run_id = run_id or instance_id
        rid_short = run_id[:12]
        net_name = f"oc-eval-net-{rid_short}"
        oc_name = f"oc-{rid_short}"
        collection = f"lb_{rid_short}"

        # scope="shared": all per-example agents share ONE sandbox container
        # (OC's resolveSandboxScopeKey returns the constant "shared"), so we get
        # N isolated agent sessions but a single sandbox + single ClawRAG.
        # IMPORTANT: for shared scope OC uses the LITERAL "shared" slug for the
        # container name (docker.ts ensureSandboxContainer: scope==="shared"
        # bypasses slugifySessionKey) — i.e. `<prefix>shared`, NOT the hashed
        # `_oc_sandbox_name(...)` form used for session/agent scope. We must
        # pre-create with that exact name so OC discovers & reuses our running
        # ClawRAG container (its hot-container path keeps it on config-hash
        # mismatch) instead of creating a fresh sandbox and rejecting the
        # /workspace bind as a reserved target.
        sandbox_scope = "shared"
        scope_key = "shared"
        sandbox_prefix = f"oc-sbx-{rid_short}-"
        task_container_name = (sandbox_prefix + "shared")[:63]

        run_root = self.workspace_base / run_id
        if run_root.exists():
            _force_rmtree(run_root, ignore_errors=True)
        ws_host = run_root / "workspace"
        ws_host.mkdir(parents=True, exist_ok=True)
        self._seed_workspace(ws_host, instance_data, collection)

        harness_cfg_path = os.environ.get("AB_HARNESS_CONFIG")
        use_openclaw = self._has_openclaw_block(harness_cfg_path)
        if use_openclaw:
            r = subprocess.run(["docker", "image", "inspect", self.oc_image],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"openclaw image not found: {self.oc_image}")
        r = subprocess.run(["docker", "image", "inspect", docker_image],
                           capture_output=True, text=True)
        if r.returncode != 0:
            logger.info("  sandbox image %s not local, pulling…", docker_image)
            r = subprocess.run(["docker", "pull", docker_image],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"sandbox image pull failed: {docker_image}\n{r.stderr.strip()}")

        logger.info("Preparing LongBench run %s (sandbox=%s, collection=%s, use_openclaw=%s)",
                    run_id, docker_image, collection, use_openclaw)

        state_dir: Optional[Path] = None
        dispatcher: Optional[OpenClawRemoteDispatcher] = None
        gateway_port: Optional[int] = None

        def _rollback() -> None:
            if gateway_port is not None:
                _release_gateway_port(gateway_port)
            if dispatcher is not None:
                try:
                    dispatcher.stop()
                except Exception:
                    pass
            self._best_effort_cleanup(net_name, oc_name, task_container_name)

        try:
            r = subprocess.run(["docker", "network", "create", net_name],
                               capture_output=True, text=True)
            if r.returncode != 0 and "already exists" not in (r.stderr or ""):
                raise RuntimeError(f"docker network create failed: {r.stderr.strip()}")

            if use_openclaw:
                state_dir = self._build_state_dir(
                    run_id=run_id,
                    workspace_root="/workspace",
                    sandbox_image=docker_image,
                    sandbox_binds=[f"{ws_host}:/workspace:rw"],
                    container_prefix=sandbox_prefix,
                    scope=sandbox_scope,
                    n_agents=len(instance_data.get("examples", [])) or 1,
                )
                dispatcher = OpenClawRemoteDispatcher()
                dispatcher.start(state_dir=state_dir, backend="docker", target=task_container_name)

            # Sandbox/task container = the self-contained ClawRAG image. ClawRAG
            # runs INSIDE it (started below via clawrag-serve); the agent reaches it
            # at localhost:8080. GPU for embeddings (device GPU_DEVICE; vLLM is on
            # the other device) + a mounted HF cache so the embed model persists
            # and is shared across runs.
            os.makedirs(HF_CACHE_HOST, exist_ok=True)
            task_run_cmd = [
                "docker", "run", "-d",
                "--name", task_container_name,
                "--network", net_name,
                "--gpus", f"device={GPU_DEVICE}",
                "--user", "0",
                "--label", "openclaw.sandbox=1",
                "--label", f"openclaw.sessionKey={scope_key}",
                "--label", f"oc-eval.run_id={run_id}",
                "-v", f"{ws_host}:/workspace:rw",
                "-v", f"{HF_CACHE_HOST}:/opt/clawrag/cache/hf",
                "-w", "/workspace",
                "--entrypoint", "sleep",
                docker_image, "infinity",
            ]
            r = subprocess.run(task_run_cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"sandbox container start failed: {r.stderr.strip()}")
            task_container_id = r.stdout.strip()

            # Bring ClawRAG up inside the task container (in-process chroma).
            self._start_clawrag_in_container(task_container_name)

            env_info: Dict = {
                "container_id": task_container_id,
                "container_name": task_container_name,
                "backend": "docker",
                "task_container_id": task_container_id,
                "task_container_name": task_container_name,
                "network_name": net_name,
                "workspace_root": "/workspace",
                "workspace_host_path": str(ws_host),
                "sandbox_image": docker_image,
                "clawrag_collection": collection,
                "container_engine": "docker",
                "instance_name": task_container_name,
            }

            if not use_openclaw:
                logger.info("  LongBench sandbox up: %s — OC skipped", task_container_name)
                return env_info

            gateway_port = _pick_free_gateway_port()
            gateway_token = secrets.token_hex(24)
            assert dispatcher is not None and state_dir is not None

            # Wire the exec-bridge into the gateway. CRITICAL: use
            # agent_mounts_and_env() (mounts bridge.py + sets OPENCLAW_EXEC_BRIDGE*),
            # NOT the deprecated agent_container_args() which returns []. Without
            # these env vars the OC fs-bridge patch is inactive, so the agent's
            # read/write/edit tools fall back to a node-`fs` boundary check that
            # runs in THIS gateway container against the workspace's real host
            # path (which the gateway can't see) → "Path escapes sandbox mount
            # root". With the bridge wired, those tool ops route through bridge.py
            # → task container (cat/write) and the boundary check is skipped.
            # The socket lives under state_dir (mounted at /root/.openclaw), so it
            # needs no extra mount.
            bridge_mounts, bridge_env = dispatcher.agent_mounts_and_env()
            bridge_args: List[str] = []
            for m in bridge_mounts:
                bridge_args += ["-v", f"{m.host}:{m.container}:{m.mode}"]
            for k, v in bridge_env.items():
                bridge_args += ["-e", f"{k}={v}"]

            cmd = [
                "docker", "run", "-d",
                "--name", oc_name,
                "--network", net_name,
                "--add-host", "vllm-host:host-gateway",
                "--user", "0",
                "-v", f"{state_dir}:/root/.openclaw:rw",
                # Bind the workspace into the GATEWAY too so OpenClaw's
                # workspace-skill scanner (a gateway-side fs scan of
                # <workspace>/skills) can discover /workspace/skills/clawrag_retrieve
                # and surface it in the system prompt. (Agent file reads go through
                # the exec-bridge to the task container, so this mount is for skill
                # discovery, not the read tool.)
                "-v", f"{ws_host}:/workspace:rw",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-w", "/root",
                "-e", f"OPENCLAW_GATEWAY_TOKEN={gateway_token}",
                "-e", f"OPENCLAW_GATEWAY_OTEL_ENDPOINT={os.environ.get('OC_OTEL_ENDPOINT', 'http://vllm-host:4318')}",
                "-e", f"OPENCLAW_RUN_ID={run_id}",
                "-e", f"OTEL_RESOURCE_ATTRIBUTES=oc.run_id={run_id}",
                "-e", "DOCKER_API_VERSION=1.44",
                *bridge_args,
                "--entrypoint", "/usr/local/bin/openclaw",
                self.oc_image,
                "gateway", "--allow-unconfigured",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"oc container start failed: {r.stderr.strip()}")
            oc_container_id = r.stdout.strip()

            self._wait_for_gateway_ready(oc_container_id)

            gw_env = (
                f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789\n"
                f"export OPENCLAW_GATEWAY_TOKEN={gateway_token}\n"
                f"export OPENCLAW_RUN_ID={run_id}\n"
            )
            subprocess.run(
                ["docker", "exec", oc_container_id, "bash", "-c",
                 f"cat > /tmp/oc_gateway_env <<'EOF'\n{gw_env}EOF"],
                capture_output=True, text=True,
            )

            logger.info("  ready: oc=%s sandbox=%s bridge_pid=%s",
                        oc_container_id[:12], task_container_id[:12], dispatcher.pid)
            env_info.update({
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
        except Exception:
            _rollback()
            raise

    # ------------------------------------------------------------------
    @staticmethod
    def _seed_workspace(ws_host: Path, instance_data: Dict, collection: str) -> None:
        (ws_host / "clawrag_search.py").write_text(
            _SEARCH_SCRIPT.format(url=CLAWRAG_AGENT_URL, collection=collection),
            encoding="utf-8",
        )
        src_dir = _THIS_DIR / "openclaw" / "agents"
        for name in _WORKSPACE_AGENT_FILES:
            src = src_dir / name
            if src.is_file():
                try:
                    shutil.copyfile(src, ws_host / name)
                except OSError as e:
                    logger.warning("workspace seed %s failed: %s", name, e)
        # Seed the skills into /workspace too, so the sandboxed agent can read
        # them (its read tool is rooted at /workspace; /root/.openclaw is out of
        # reach). e.g. /workspace/skills/clawrag_retrieve/SKILL.md.
        skills_src = _THIS_DIR / "openclaw" / "skills"
        if skills_src.is_dir():
            try:
                shutil.copytree(skills_src, ws_host / "skills", dirs_exist_ok=True)
            except OSError as e:
                logger.warning("workspace skills seed failed: %s", e)

    # ------------------------------------------------------------------
    def _build_state_dir(
        self, *, run_id: str, workspace_root: str, sandbox_image: str,
        sandbox_binds: List[str], container_prefix: str, scope: str = "shared",
        n_agents: int = 1,
    ) -> Path:
        kit_root = os.environ.get("KIT_ROOT", os.getcwd())
        state_dir = self.state_base / run_id
        if state_dir.exists():
            _force_rmtree(state_dir)
        # One agent per example (longbench_rag_<i>): in this OC build sessions are
        # per-agent, so distinct agents keep concurrent sessions isolated. They all
        # share ONE sandbox via scope="shared".
        agent_ids = [f"{DEFAULT_AGENT_ID}_{i}" for i in range(max(1, n_agents))]
        for aid in agent_ids:
            os.makedirs(state_dir / "agents" / aid, exist_ok=True)

        harness_path = os.environ.get(
            "AB_HARNESS_CONFIG",
            os.path.join(kit_root, "config/harness/openclaw_longbench_clawrag.json"),
        )
        if not os.path.isfile(harness_path):
            raise FileNotFoundError(f"harness config not found at {harness_path}")
        with open(harness_path) as fh:
            harness_cfg = json.load(fh) or {}
        oc_cfg = dict(harness_cfg.get("openclaw") or {})
        if not oc_cfg:
            raise ValueError(f"harness config {harness_path} has no `openclaw` block")

        model_cfg, _serving_cfg = load_active_model_and_serving(kit_root=kit_root)
        providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
        vllm_p = providers.setdefault("vllm", {})
        vllm_p["api"] = _OPENAI_CHAT_COMPLETIONS_API
        vllm_p.setdefault("apiKey", model_cfg.api_key or "sk-dummy")
        vllm_p["models"] = [{
            "id": model_cfg.name, "name": model_cfg.name,
            "api": _OPENAI_CHAT_COMPLETIONS_API,
            "contextWindow": model_cfg.context_window, "maxTokens": model_cfg.max_tokens,
        }]
        oc_cfg.setdefault("agents", {}).setdefault("defaults", {}).setdefault(
            "model", {})["primary"] = f"vllm/{model_cfg.name}"
        # Register one agent per example (each gets its own isolated session).
        # Pin `workspace` on EVERY entry: OC's resolveAgentWorkspaceDir only gives
        # the DEFAULT agent agents.defaults.workspace; non-default agents fall
        # through to a per-agent dir (e.g. /workspace/<agentId>), which would be
        # empty — the seeded files + /workspace/skills (workspace-skill source)
        # live at /workspace. Without this, only longbench_rag_0 sees the skill /
        # BOOTSTRAP.md via the read tool; the rest limp along via `exec`.
        oc_cfg["agents"]["list"] = [
            {"id": aid, "name": aid, "default": (k == 0), "workspace": workspace_root}
            for k, aid in enumerate(agent_ids)
        ]
        with open(state_dir / "openclaw.json", "w") as fh:
            json.dump(oc_cfg, fh, indent=2)

        lb_agents = os.path.join(kit_root, "applications/longbench/openclaw/agents")
        if os.path.isdir(lb_agents):
            for aid in agent_ids:
                for f in os.listdir(lb_agents):
                    if f == "BOOTSTRAP.md":  # workspace-scoped only (seeded into /workspace)
                        continue
                    s = os.path.join(lb_agents, f)
                    if os.path.isfile(s):
                        shutil.copyfile(s, state_dir / "agents" / aid / f)
        # Skills are NOT loaded into OC state (/root/.openclaw/skills): the agent's
        # tools run in the SANDBOX and can't read the OC-container fs (the read
        # tool is rooted at /workspace). The skill is seeded into /workspace/skills
        # instead (see _seed_workspace); AGENTS.md points the agent there.

        cfg_path = state_dir / "openclaw.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        agents_defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
        agents_defaults["workspace"] = workspace_root
        agents_defaults["sandbox"] = {
            "mode": "all", "scope": scope, "backend": "docker", "workspaceAccess": "rw",
            "docker": {
                "image": sandbox_image, "containerPrefix": container_prefix,
                "workdir": workspace_root, "binds": list(sandbox_binds),
                "dangerouslyAllowExternalBindSources": True,
            },
        }
        cfg.setdefault("tools", {}).setdefault("exec", {})["host"] = "sandbox"
        vllm_endpoint = os.environ.get(
            "OC_VLLM_BASE_URL", endpoint_for_container(model_cfg.endpoint, "vllm-host"))
        vp = cfg.setdefault("models", {}).setdefault("providers", {}).setdefault("vllm", {})
        vp["api"] = _OPENAI_CHAT_COMPLETIONS_API
        vp["baseUrl"] = vllm_endpoint
        for m in vp.get("models") or []:
            if isinstance(m, dict):
                m["api"] = _OPENAI_CHAT_COMPLETIONS_API
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return state_dir

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
        from applications.longbench import scoring

        run_id = run_id or instance_id
        dataset = instance_id
        metric = (instance_data or {}).get("metric", "qa_f1_score")
        preds_path = self.workspace_base / run_id / "workspace" / "predictions.jsonl"
        out_dir = self.workspace_base / run_id / "judge"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not preds_path.exists():
            return {
                "status": "ERROR", "test_pass_rate": 0.0, "passed": 0, "failed": 1,
                "error_message": f"predictions.jsonl missing at {preds_path}",
            }
        rows = [json.loads(l) for l in preds_path.read_text().splitlines() if l.strip()]
        per = []
        for r in rows:
            s = scoring.score_one(
                dataset, metric, r.get("prediction", "") or "",
                r.get("answers", []) or [], r.get("all_classes"),
            )
            per.append({
                "_id": r.get("_id"), "score": s,
                "prediction": (r.get("prediction", "") or "")[:300],
                "answers": r.get("answers"),
            })
        mean = sum(p["score"] for p in per) / len(per) if per else 0.0
        (out_dir / "longbench_scores.json").write_text(
            json.dumps({
                "dataset": dataset, "metric": metric, "n": len(per),
                "mean_score": mean, "longbench_score": round(100 * mean, 2),
                "per_example": per,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("LongBench %s: %s = %.2f (n=%d)", dataset, metric, round(100 * mean, 2), len(per))
        return {
            "status": "PASS" if per else "ERROR",
            "test_pass_rate": mean,
            "passed": len(per),
            "failed": 0,
            "metric": metric,
            "n_examples": len(per),
            "longbench_score": round(100 * mean, 2),
            "per_example": per,
            "evaluation_dir": str(out_dir),
            "error_message": None,
        }
