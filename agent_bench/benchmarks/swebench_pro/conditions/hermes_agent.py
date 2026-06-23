from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition


class SWEBenchProHermesCondition(HermesAgentCondition):
    """Hermes Agent driving a SWE-bench Pro task container."""

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "You are fixing a software bug. The repository is in the current "
            "working directory inside the SWE-bench Pro task image. Read the "
            "issue, modify the source files to fix it, then reply with the "
            "single word DONE.\n\n"
            f"Issue:\n{instance_data['problem_statement']}"
        )

