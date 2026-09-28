#!/usr/bin/env python3
"""Export a single Hermes RoadmapBench arm without requiring a paired run.

This entry point shares Roadmap's selection, metrics, and figures with the
OpenClaw analyzer. It exports Hermes' durable sessions, traces, logs, and
scores, while leaving full candidate workspaces in the original run folders.
"""

from __future__ import annotations

from typing import Sequence

try:
    from . import analyze_roadmap_runs as _analysis
    from . import roadmap_hermes_telemetry as _telemetry
except ImportError:
    import analyze_roadmap_runs as _analysis  # type: ignore
    import roadmap_hermes_telemetry as _telemetry  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    _telemetry.install_analysis(_analysis)
    return _analysis.main(argv, omit_candidate_workspace=True)


if __name__ == "__main__":
    raise SystemExit(main())
