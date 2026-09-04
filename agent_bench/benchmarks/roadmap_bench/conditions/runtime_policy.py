"""RoadmapBench runtime-validity and terminal-evidence policy.

Formal Roadmap experiments compare gateway-mediated parent/subagent execution.
OpenClaw's CLI can automatically continue in embedded mode after a Gateway
failure, which is a different runtime and therefore an infrastructure-invalid
sample even if it later produces a candidate patch and verifier reward.

Roadmap also records a benchmark-level execution outcome independently of the
upstream OpenClaw terminal-evidence string. When the durable run ends in an
explicit context overflow, preserve that fact in ``agent_lifecycle`` instead of
leaving ``terminalEvidence=unknown``.
"""
from __future__ import annotations

from typing import Callable, Optional


def repair_roadmap_terminal_evidence(result: dict) -> dict:
    """Fill unknown lifecycle evidence from a definitive Roadmap outcome."""
    if not isinstance(result, dict):
        return result
    outcome = str(result.get("agent_execution_outcome") or "").strip().lower()
    if outcome != "context_overflow":
        return result
    lifecycle = result.get("agent_lifecycle")
    if not isinstance(lifecycle, dict):
        return result
    evidence = str(lifecycle.get("terminalEvidence") or "").strip().lower()
    if evidence not in {"", "unknown", "none", "null"}:
        return result

    repaired = dict(result)
    lifecycle = dict(lifecycle)
    lifecycle["terminalEvidence"] = "context_overflow"
    lifecycle["terminalEvidenceSource"] = "agent_execution_outcome"
    repaired["agent_lifecycle"] = lifecycle
    return repaired


def install_roadmap_gateway_only_policy() -> None:
    from agent_bench.runtime import driver

    current: Callable[[dict], Optional[str]] = driver._roadmap_lifecycle_verifier_error
    if not getattr(current, "_roadmap_gateway_only_policy", False):
        def wrapped(result: dict) -> Optional[str]:
            runtime_mode = str(result.get("agent_runtime_mode") or "").strip()
            if runtime_mode and runtime_mode != "gateway":
                return (
                    "RoadmapBench requires OpenClaw gateway runtime; "
                    f"observed agent_runtime_mode={runtime_mode}"
                )
            return current(result)

        setattr(wrapped, "_roadmap_gateway_only_policy", True)
        driver._roadmap_lifecycle_verifier_error = wrapped

    # RoadmapBenchOpenClawCondition.run calls super().run(), so a narrowly
    # guarded base wrapper repairs the result before the generic driver writes
    # the manifest. Other OpenClaw benchmarks pass through unchanged.
    from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

    current_run = OpenClawAgentCondition.run
    if not getattr(current_run, "_roadmap_terminal_evidence_policy", False):
        def run_with_terminal_evidence(self, *args, **kwargs):
            result = current_run(self, *args, **kwargs)
            if getattr(self, "agent_id", "") == "roadmap_bench":
                result = repair_roadmap_terminal_evidence(result)
            return result

        setattr(
            run_with_terminal_evidence,
            "_roadmap_terminal_evidence_policy",
            True,
        )
        OpenClawAgentCondition.run = run_with_terminal_evidence
