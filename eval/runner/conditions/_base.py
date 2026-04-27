"""Condition protocol shared by all bench types (code_agent, doc_analysis, websearch, ...)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol


@dataclass
class ConditionResult:
    bench_type: str
    success: bool
    artifacts: dict[str, Path]
    metrics: dict[str, Any]
    error: str | None = None


class ConditionProtocol(Protocol):
    bench_type: ClassVar[str]

    def run(
        self,
        instance: dict[str, Any],
        run_dir: Path,
        seed: int,
        container_id: str | None,
    ) -> ConditionResult: ...
