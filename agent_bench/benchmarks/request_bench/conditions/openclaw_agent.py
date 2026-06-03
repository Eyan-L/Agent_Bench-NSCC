"""OpenClaw condition for Request-Bench."""
from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


class RequestBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent that answers a simple natural-language request."""

    agent_id = "request_writer"

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Read the natural-language request in /workspace/PROMPT.md. "
            "Produce a clear, complete answer in Markdown and write it to "
            "/workspace/answer.md. Use the normal file-writing tools so the "
            "benchmark can retrieve the artifact. Reply with a single line "
            "`DONE` after /workspace/answer.md is written.\n\n"
            f"Task id: {instance_data.get('instance_id')}"
        )
