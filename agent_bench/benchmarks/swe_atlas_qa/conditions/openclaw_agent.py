"""OpenClaw condition for SWE-Atlas-QA."""
from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


class SWEAtlasQAOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent driving a SWE-Atlas-QA instance container."""

    agent_id = "swe_atlas_qa"

    def _build_message(self, instance_data: Dict) -> str:
        question = (
            instance_data.get("question")
            or instance_data.get("problem_statement")
            or instance_data.get("prompt")
            or ""
        )

        return (
            "Investigate the following repository question carefully. "
            "Use the available repository, shell, and file tools to inspect "
            "the codebase and gather concrete evidence. Synthesize the evidence "
            "into a complete technical answer, following the answer-submission "
            "requirements in AGENTS.md.\n\n"
            f"Question:\n{question}"
        )

