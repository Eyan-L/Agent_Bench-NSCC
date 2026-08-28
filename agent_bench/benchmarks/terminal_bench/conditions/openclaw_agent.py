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

import os
from typing import Dict

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition


class TerminalBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent driving a Terminal-Bench task container."""

    agent_id = "terminal_bench"

    def _build_message(self, instance_data: Dict) -> str:
        message = (
            "Carry out this terminal task in the current container. Install "
            "any required packages, write the required files, and start any "
            "required services so the verifier passes. Declare DONE only when "
            "the task is complete.\n\n"
            f"Task:\n{instance_data['instruction']}"
        )

        # Capability-only experiment. Disabled by default so natural
        # Terminal-Bench runs retain their original orchestration policy.
        if os.environ.get("AB_TB_FORCE_ONE_SUBAGENT") == "1":
            message = (
                "CAPABILITY TEST INSTRUCTION:\n"
                "Before solving the task yourself, you MUST call "
                "sessions_spawn exactly once and delegate a meaningful part "
                "of this task to one subagent. Ask the subagent to inspect "
                "the actual task/workspace and use terminal tools as needed; "
                "do not delegate a trivial conversational task. Do not spawn "
                "more than one subagent. After spawning it, allow the child "
                "to do substantive work and use its findings or workspace "
                "changes in the parent trajectory. Use sessions_yield or "
                "sessions_history if needed to wait for or inspect the child "
                "result.\n\n"
                + message
            )

        return message
