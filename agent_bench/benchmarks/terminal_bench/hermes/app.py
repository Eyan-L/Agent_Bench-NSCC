"""Terminal-Bench application - Hermes harness variant."""
from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Dict

from agent_bench.benchmarks.conditions.agent_otel import otel_env
from agent_bench.benchmarks.terminal_bench.agent_harness_base import TerminalBenchAgentHarnessBase
from agent_bench.core.config import load_active_model_and_serving
from agent_bench.core.profile_io import load_profile
from agent_bench.harnesses.hermes.dispatcher import HermesDispatcher


class TerminalBenchHermesHarness(TerminalBenchAgentHarnessBase):
    harness_name = "hermes"
    short_name = "hm"
    default_runtime_image = "agent-bench/hermes:v2026.5.16"
    dispatcher_cls = HermesDispatcher
    state_dir_in_agent = "/opt/data"
    workspace_in_agent = "/app"
    task_workspace = "/app"

    def _load_harness_config(self) -> Dict:
        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        harness_path = os.environ.get(
            "AB_HARNESS_CONFIG",
            os.path.join(repo_root, "profiles/harnesses/hermes_terminal_bench.yaml"),
        )
        return load_profile(harness_path)

    def _build_state_dir(self, *, run_id: str, docker_image: str, app_dir: Path, logs_dir: Path, instance_data: Dict) -> Path:
        state_dir = self.state_base / run_id
        if state_dir.exists():
            from agent_bench.benchmarks.swebench.app import _force_rmtree
            _force_rmtree(state_dir)
        state_dir.mkdir(parents=True)

        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        harness_cfg = self._load_harness_config()
        hm_cfg = harness_cfg.get("hermes") or {}
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        template_path = self._template_path("hermes", "config.yaml.template")
        values = {
            "AB_MODEL_NAME": os.environ.get("AB_MODEL_NAME") or model_cfg.name,
            "PROVIDER": os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider,
            "BASE_URL": self._runtime_base_url(model_cfg.endpoint),
            "API_KEY": os.environ.get("OPENAI_API_KEY") or model_cfg.api_key,
            "TEMPERATURE": str(model_cfg.temperature),
            "AB_MODEL_MAX_OUTPUT_TOKENS": str(model_cfg.max_tokens),
            "CONTEXT_LENGTH": str(model_cfg.context_window),
            "MAX_TURNS": str((hm_cfg.get("run") or {}).get("max_turns") or 90),
            "TERMINAL_TIMEOUT": str((hm_cfg.get("run") or {}).get("terminal_timeout") or 180),
            "TERMINAL_CWD": self.task_workspace,
            "SANDBOX_IMAGE": docker_image,
        }
        rendered = string.Template(template_path.read_text()).safe_substitute(values)
        (state_dir / "config.yaml").write_text(rendered)
        return state_dir

    def _agent_env(self, *, run_id: str, docker_image: str, instance_data: Dict) -> dict[str, str]:
        model_cfg, _serving_cfg = load_active_model_and_serving()
        model_name = os.environ.get("AB_MODEL_NAME") or model_cfg.name
        provider = os.environ.get("HERMES_INFERENCE_PROVIDER") or model_cfg.provider
        base_url = self._runtime_base_url(model_cfg.endpoint)
        api_key = os.environ.get("OPENAI_API_KEY") or model_cfg.api_key
        env = {
            "HERMES_HOME": self.state_dir_in_agent,
            "HERMES_RUN_ID": run_id,
            "HERMES_INFERENCE_MODEL": model_name,
            "HERMES_INFERENCE_PROVIDER": provider,
            "AB_AGENT_RUNTIME_EVENTS_FILE": f"{self.state_dir_in_agent}/events.jsonl",
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "MINIMAX_API_KEY": api_key,
            "MINIMAX_BASE_URL": base_url,
            "TERMINAL_ENV": "docker",
            "TERMINAL_CWD": self.task_workspace,
            "TERMINAL_DOCKER_IMAGE": docker_image,
        }
        env.update(otel_env(
            service_name="agent-bench-hermes",
            run_id=run_id,
            endpoint=self._otel_endpoint(),
            extra_attrs={"hm.run_id": run_id, "tb.instance_id": instance_data.get("instance_id", "")},
        ))
        return env

    def _condition_env_info(self, state_dir: Path) -> dict[str, str]:
        return {
            "config_path_in_container": f"{self.state_dir_in_agent}/config.yaml",
        }
