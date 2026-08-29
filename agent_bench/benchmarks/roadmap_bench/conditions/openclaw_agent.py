"""RoadmapBench framing for the shared Terminal-Bench OpenClaw runtime."""
from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


class RoadmapBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw lead engineer implementing a multi-target roadmap."""

    agent_id = "roadmap_bench"

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Lead the implementation of this multi-target software roadmap in "
            "the current repository. Inspect the codebase, identify target "
            "workstreams and dependencies, coordinate parallel work where "
            "appropriate, integrate the complete implementation, and validate "
            "all roadmap requirements. Declare DONE only when the integrated "
            "roadmap is complete.\n\n"
            f"Roadmap:\n{instance_data['instruction']}"
        )
