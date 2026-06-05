"""Terminal-Bench specialization of the OpenClaw agent condition.

Overrides:
  - ``agent_id = "terminal_bench"`` — OC registers the agent under this id
    in the per-run openclaw.json (see ``profiles/harnesses/openclaw_terminal_bench.yaml``).
  - ``_build_message`` — TB-flavoured task framing ("Carry out this terminal
    task ... Declare DONE only when the task is complete") instead of
    SWE-bench's "fixing a bug" framing.

Patch collection is intentionally NOT overridden — TB has no patch artifact
(the agent's effects live inside the task container). The base class writes
an empty ``candidate.patch`` and the driver's ``skip_patch_gate`` at
``agent_bench/runtime/driver.py:187`` allows it through for
``benchmark == "terminal_bench"``.
"""
from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


class TerminalBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent driving a Terminal-Bench task container."""

    agent_id = "terminal_bench"

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Carry out this terminal task in the current container. Install "
            "any required packages, write the required files, and start any "
            "required services so the verifier passes. Declare DONE only when "
            "the task is complete.\n\n"
            f"Task:\n{instance_data['instruction']}"
        )
