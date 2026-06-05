"""Terminal-Bench application - OpenHands harness variant."""
from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Dict

from agent_bench.benchmarks.conditions.agent_otel import otel_config_values, otel_env
from agent_bench.benchmarks.terminal_bench.agent_harness_base import TerminalBenchAgentHarnessBase
from agent_bench.core.config import load_active_model_and_serving
from agent_bench.core.profile_io import load_profile
from agent_bench.harnesses.openhands.dispatcher import OpenHandsDispatcher


class TerminalBenchOpenHandsHarness(TerminalBenchAgentHarnessBase):
    harness_name = "openhands"
    short_name = "oh"
    default_runtime_image = "agent-bench/openhands:1.21.0"
    dispatcher_cls = OpenHandsDispatcher
    state_dir_in_agent = "/etc/agent-bench/state"
    workspace_in_agent = "/app"

    def _build_state_dir(
        self,
        *,
        run_id: str,
        docker_image: str,
        app_dir: Path,
        logs_dir: Path,
        instance_data: Dict,
        task_workspace: str,
    ) -> Path:
        state_dir = self.state_base / run_id
        if state_dir.exists():
            from agent_bench.benchmarks.swebench.app import _force_rmtree
            _force_rmtree(state_dir)
        state_dir.mkdir(parents=True)

        repo_root = os.environ.get("AB_REPO_ROOT", os.getcwd())
        harness_path = os.environ.get(
            "AB_HARNESS_CONFIG",
            os.path.join(repo_root, "profiles/harnesses/openhands_terminal_bench.yaml"),
        )
        harness_cfg = load_profile(harness_path)
        oh_cfg = harness_cfg.get("openhands") or {}
        model_cfg, _serving_cfg = load_active_model_and_serving(repo_root=repo_root)
        template_path = self._template_path("openhands", "config.toml.template")

        model_raw = model_cfg.litellm_model or model_cfg.name
        litellm_prefixes = (
            "openai/", "anthropic/", "vllm/", "hosted_vllm/", "azure/",
            "bedrock/", "gemini/", "groq/", "together_ai/", "huggingface/",
            "mistral/", "cohere/", "ollama/",
        )
        model_resolved = model_raw if model_raw.startswith(litellm_prefixes) else f"openai/{model_raw}"
        values = {
            "AB_MODEL_NAME": model_resolved,
            "BASE_URL": self._runtime_base_url(model_cfg.endpoint),
            "API_KEY": model_cfg.api_key or "EMPTY",
            "TEMPERATURE": str(model_cfg.temperature),
            "AB_MODEL_MAX_OUTPUT_TOKENS": str(model_cfg.max_tokens),
            "CONTEXT_WINDOW": str(model_cfg.context_window),
            "MAX_ITERATIONS": str((oh_cfg.get("run") or {}).get("max_iterations") or 100),
            "AGENT_NAME": (oh_cfg.get("agent") or {}).get("name") or "CodeActAgent",
            "RUNTIME": (oh_cfg.get("run") or {}).get("runtime") or "process",
            "WORKSPACE": self.workspace_in_agent,
            "EVENTS_PATH": f"{self.state_dir_in_agent}/events.jsonl",
            **otel_config_values(
                service_name="agent-bench-openhands",
                run_id=run_id,
                endpoint=self._otel_endpoint(),
                extra_attrs={
                    "oh.run_id": run_id,
                    "tb.instance_id": instance_data.get("instance_id", ""),
                },
                enabled=((harness_cfg.get("telemetry") or {}).get("otel") or {}).get("enabled", True) is not False,
            ),
        }
        rendered = string.Template(template_path.read_text()).safe_substitute(values)
        (state_dir / "config.toml").write_text(rendered)
        (state_dir / "events.jsonl").touch()
        return state_dir

    def _agent_env(
        self,
        *,
        run_id: str,
        docker_image: str,
        instance_data: Dict,
        task_workspace: str,
    ) -> dict[str, str]:
        env = {
            "AB_RUN_ID": run_id,
            "OPENHANDS_RUN_ID": run_id,
            "OPENHANDS_EXEC_BRIDGE_AGENT_WORKSPACE": self.workspace_in_agent,
            "OPENHANDS_EXEC_BRIDGE_TASK_WORKSPACE": task_workspace,
        }
        env.update(otel_env(
            service_name="agent-bench-openhands",
            run_id=run_id,
            endpoint=self._otel_endpoint(),
            extra_attrs={"oh.run_id": run_id, "tb.instance_id": instance_data.get("instance_id", "")},
        ))
        return env

    def _condition_env_info(self, state_dir: Path) -> dict[str, str]:
        return {
            "config_path_in_container": f"{self.state_dir_in_agent}/config.toml",
            "events_path_in_container": f"{self.state_dir_in_agent}/events.jsonl",
            "events_path_on_host": str(state_dir / "events.jsonl"),
        }
