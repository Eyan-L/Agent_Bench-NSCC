"""DeepResearch-Bench specialization of the OpenClaw agent condition.

Overrides:
  - ``agent_id = "deep_researcher"`` — must match
    ``config/harness/deepresearch_27b.json`` ``openclaw.agents.list[0].id``
    AND the dirname under state_dir/agents/ where prompts are dropped (see
    ``applications/deepresearch_bench/app.py:_build_state_dir``).
  - ``_build_message`` — research framing ("PROMPT.md on disk, write
    report.md") instead of SWE-bench's bug-fixing framing.

Artifact collection is intentionally NOT overridden — DR has no git patch.
The base class writes an empty ``candidate.patch`` and the driver's
``skip_patch_gate`` at ``orchestration/runner/driver.py`` allows it through
for ``benchmark == "deepresearch_bench"``. The actual artifact (report.md)
is harvested by ``DeepResearchHarness.run_tests`` directly from the sandbox
container.
"""
from __future__ import annotations

from typing import Dict

from applications.conditions.openclaw_agent import OpenClawAgentCondition


class DeepResearchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw agent driving a DeepResearch-Bench task container."""

    agent_id = "deep_researcher"

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Read the research question at /workspace/PROMPT.md, conduct web "
            "research, and write a long-form markdown report with inline "
            "citations to /workspace/report.md. Use the search and page-fetch "
            "tools available to you; cite real public URLs for every "
            "non-trivial factual claim. Reply with a single line `DONE` when "
            "the report is complete.\n\n"
            f"Task id: {instance_data.get('instance_id')}\n"
            f"Language: {instance_data.get('language', 'en')}\n"
            f"Topic:    {instance_data.get('topic', 'unknown')}"
        )
