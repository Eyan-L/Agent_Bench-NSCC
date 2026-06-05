"""Terminal-Bench specialization of the OpenHands agent condition."""
from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.swebench.conditions.openhands_agent import (
    OpenHandsAgentCondition,
)


class TerminalBenchOpenHandsCondition(OpenHandsAgentCondition):
    """OpenHands agent driving a Terminal-Bench task container."""

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Carry out this terminal task in the current container. Install "
            "any required packages, write the required files, and start any "
            "required services so the verifier passes. Declare DONE only when "
            "the task is complete.\n\n"
            f"Task:\n{instance_data['instruction']}"
        )

    def _collect_artifacts(
        self,
        env_info: Dict,
        container_id: str,
        workspace_root: str,
    ) -> str:
        return "# terminal_bench: no patch - agent state in container\n"
