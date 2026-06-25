from __future__ import annotations

from typing import Any, Mapping

from agent_bench.core.config import ModelConfig


_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"


def openclaw_context_window(harness_cfg: Mapping[str, Any], model_context_window: int) -> int:
    """Resolve the OpenClaw model context window.

    The model profile is the default source of truth. Harness or experiment
    configs may lower it for controlled context-retention experiments.
    """
    override = _openclaw_context_override(harness_cfg)
    if override is None:
        return model_context_window
    context_window = _positive_int(override, "openclaw contextWindow")
    return min(model_context_window, context_window)


def openclaw_model_entry(
    model_cfg: ModelConfig,
    harness_cfg: Mapping[str, Any],
    *,
    api: str = _OPENAI_CHAT_COMPLETIONS_API,
    reasoning: bool | None = None,
    compat: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_cfg.name,
        "name": model_cfg.name,
        "api": api,
        "contextWindow": openclaw_context_window(harness_cfg, model_cfg.context_window),
        "maxTokens": model_cfg.max_tokens,
    }
    if reasoning is not None:
        entry["reasoning"] = reasoning
    if compat:
        entry["compat"] = dict(compat)
    return entry


def _openclaw_context_override(harness_cfg: Mapping[str, Any]) -> Any:
    agent_bench_cfg = harness_cfg.get("agent_bench")
    if isinstance(agent_bench_cfg, Mapping):
        for key in ("openclawContextWindow", "openclaw_context_window", "contextWindow", "context_window"):
            if key in agent_bench_cfg:
                return agent_bench_cfg[key]

    openclaw_cfg = harness_cfg.get("openclaw")
    if isinstance(openclaw_cfg, Mapping):
        agent_bench_cfg = openclaw_cfg.get("agent_bench")
        if isinstance(agent_bench_cfg, Mapping):
            for key in ("contextWindow", "context_window"):
                if key in agent_bench_cfg:
                    return agent_bench_cfg[key]
    return None


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be a positive integer")
    return parsed
