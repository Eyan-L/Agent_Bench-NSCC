#!/usr/bin/env python3
"""Export a single Hermes RoadmapBench arm without requiring a paired run.

This entry point shares Roadmap's selection, metrics, figures, and portable
artifact bundle with the OpenClaw analyzer. The exporter recognizes Hermes'
durable session state, events, and execution logs.
"""

from __future__ import annotations

from typing import Sequence

try:
    from . import analyze_roadmap_runs as _analysis
except ImportError:
    import analyze_roadmap_runs as _analysis  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    return _analysis.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
