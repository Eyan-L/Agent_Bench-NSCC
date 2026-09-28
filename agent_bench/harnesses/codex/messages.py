"""Explicit completion markers for Codex's native turn protocol."""
from __future__ import annotations

import re


def done_marker(text: str | None) -> bool:
    # Same marker used by Roadmap OpenClaw and Hermes durable transcripts.
    return bool(re.search(r"(?im)^\s*DONE(?:\s*$|\b)", str(text or "")))


def blocked_marker(text: str | None) -> bool:
    return bool(re.search(r"(?im)^\s*BLOCKED(?:\s*$|\b)", str(text or "")))


DONE_TURN_INSTRUCTIONS = (
    "This is an unattended tool-using run. Complete the assigned task in this "
    "native turn. Use actual tool calls for remaining implementation, "
    "integration, and validation. Attach any progress preamble to the tool "
    "calls that perform that work, rather than returning a standalone promise "
    "to work later. A Chat Completions stop ends one model response, not the "
    "assigned task. Only finish your assigned task after its requirements "
    "have been implemented and validated. End a successful final answer with "
    "a standalone DONE line. The Roadmap parent must integrate and validate "
    "the entire roadmap and reply with one line, DONE. A subagent must finish "
    "only its assigned scope, report its changes and validation to the parent, "
    "and end that report with DONE. If a concrete blocker prevents completion "
    "and no distinct recovery remains, explain it and end with a standalone "
    "BLOCKED line; do not claim DONE."
)


def assistant_phase(message: dict, *, require_done_marker: bool = False) -> str:
    phase = message.get("phase")
    if phase is not None:
        if phase not in {"commentary", "final_answer"}:
            raise ValueError(f"Unsupported assistant phase: {phase}")
        return phase
    if message.get("tool_calls"):
        return "commentary"
    if require_done_marker and not (
        done_marker(message.get("content")) or blocked_marker(message.get("content"))
    ):
        return "commentary"
    return "final_answer"
