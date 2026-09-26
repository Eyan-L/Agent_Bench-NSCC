#!/usr/bin/env python3
"""Export a single Hermes RoadmapBench arm without requiring a paired run.

This entry point shares Roadmap's selection, metrics, figures, and portable
artifact bundle with the OpenClaw analyzer. The exporter recognizes Hermes'
durable session state, events, and execution logs.
"""

from __future__ import annotations

import sys
from typing import Sequence

try:
    from . import analyze_roadmap_runs as _analysis
except ImportError:
    import analyze_roadmap_runs as _analysis  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--runs-root" or arg.startswith("--runs-root=") for arg in args):
        args.extend(("--runs-root", "runs"))
    return _analysis.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
