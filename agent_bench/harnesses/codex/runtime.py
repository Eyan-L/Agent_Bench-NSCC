"""Benchmark-independent configuration and commands for Codex CLI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PAYLOAD_PATH = "/opt/agent-bench-codex"
STATE_PATH = "/opt/agent-bench-codex-state"


def write_config(state: Path, *, model: str, base_url: str, context_window: int,
                 settings: dict[str, Any]) -> Path:
    """Use a fresh CODEX_HOME and an explicit provider, never personal auth."""
    multi_agent = settings.get("multi_agent", True) is not False
    quote = json.dumps
    lines = [
        f"model = {quote(model)}", 'model_provider = "agent_bench"',
        'approval_policy = "never"', 'sandbox_mode = "danger-full-access"',
        f"model_context_window = {int(context_window)}",
        f"model_auto_compact_token_limit = {int(context_window * 0.8)}",
        'web_search = "disabled"',
        "allow_login_shell = false", 'project_doc_max_bytes = 65536',
        "[analytics]", "enabled = false", "[feedback]", "enabled = false",
        "[features]", f"multi_agent = {str(multi_agent).lower()}",
        "multi_agent_v2 = false", "apps = false", "browser_use = false",
        "image_generation = false", "memories = false",
        "[agents]", f"enabled = {str(multi_agent).lower()}",
        f"max_concurrent_threads_per_session = {int(settings.get('max_subagents', 16))}",
        f"max_depth = {int(settings.get('max_spawn_depth', 1))}",
        "[model_providers.agent_bench]", 'name = "Agent_Bench"',
        f"base_url = {quote(base_url)}", 'wire_api = "responses"',
        'env_key = "AB_CODEX_API_KEY"', "requires_openai_auth = false",
        "supports_websockets = false", "request_max_retries = 2",
        "stream_max_retries = 2", "stream_idle_timeout_ms = 600000",
    ]
    state.mkdir(parents=True, exist_ok=True)
    path = state / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def command(workspace: str, session_id: str | None = None) -> list[str]:
    cli = [f"{PAYLOAD_PATH}/bin/codex", "exec", "--color", "never", "--cd", workspace]
    if session_id:
        cli.append("resume")
    cli += ["--json",
            "--strict-config", "--skip-git-repo-check", "--ignore-rules",
            "--output-last-message", f"{STATE_PATH}/final.txt"]
    if session_id:
        cli.append(session_id)
    cli.append("-")
    # Container.popen_exec deliberately disconnects host stdin. Read the prompt
    # through the per-run state bind without changing other harnesses' exec API.
    prompt_file = "continuation.txt" if session_id else "prompt.txt"
    return ["sh", "-c", f'exec "$@" < {STATE_PATH}/{prompt_file}', "codex-entry", *cli]
