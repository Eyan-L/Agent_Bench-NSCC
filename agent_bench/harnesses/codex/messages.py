"""Recognize explicit working preambles without treating Chat stops as final."""
from __future__ import annotations

import re


def done_marker(text: str | None) -> bool:
    # Same marker used by Roadmap OpenClaw and Hermes durable transcripts.
    return bool(re.search(r"(?im)^\s*DONE(?:\s*$|\b)", str(text or "")))


_ACTION = (
    r"(?:check|inspect|read|look|examine|review|explore|investigate|analy[sz]e|"
    r"plan|implement|modify|edit|write|add|update|fix|integrate|validate|verify|"
    r"test|run|delegate|spawn|wait|continue|finish|complete)\b"
)
_CONTINUATION = re.compile(
    r"(?:^|[.!?]\s+|\n)\s*(?:(?:now|next|first|then)[,:]?\s+)?"
    r"(?:let(?:\s+me|\s+us|['’]s)\s+|"
    r"(?:I|we)(?:\s+will|['’]ll|\s+(?:am|are)\s+going\s+to|"
    r"\s+(?:still\s+)?need\s+to)\s+)" + _ACTION,
    re.IGNORECASE,
)


def continuation_preamble(text: str | None) -> bool:
    """A final paragraph promising further task work is not a closeout.

    Limit the fallback to explicit intent statements. Completed summaries and
    ordinary closing offers such as 'let me know' remain valid non-DONE stops.
    Native Responses phase values do not need this Chat compatibility rule.
    """
    paragraphs = re.split(r"\n\s*\n", str(text or "").strip())
    return bool(_CONTINUATION.search(paragraphs[-1]))


def assistant_phase(message: dict) -> str:
    phase = message.get("phase")
    if phase is not None:
        if phase not in {"commentary", "final_answer"}:
            raise ValueError(f"Unsupported assistant phase: {phase}")
        return phase
    if message.get("tool_calls") or continuation_preamble(message.get("content")):
        return "commentary"
    return "final_answer"
