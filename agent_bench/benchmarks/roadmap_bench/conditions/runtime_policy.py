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
