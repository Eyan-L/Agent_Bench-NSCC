#!/usr/bin/env python3
"""Export Codex Roadmap runs from PBS IDs or run paths, without a paired arm."""
from __future__ import annotations

from typing import Sequence

try:
    from . import analyze_roadmap_runs as analysis
    from . import roadmap_codex_telemetry as telemetry
except ImportError:
    import analyze_roadmap_runs as analysis
    import roadmap_codex_telemetry as telemetry


def main(argv: Sequence[str] | None = None) -> int:
    telemetry.install_analysis(analysis)
    return analysis.main(argv, omit_candidate_workspace=True)


if __name__ == "__main__":
    raise SystemExit(main())
