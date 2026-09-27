#!/usr/bin/env python3
"""Compare paired Hermes RoadmapBench subagent and single-agent PBS arms.

The Roadmap comparison policy is shared with OpenClaw so latency, reward,
selection, and subagent counts retain the same definitions. Hermes artifacts
are exported into the same portable five-part bundle.
"""

from __future__ import annotations

from typing import Sequence

try:
    from . import compare_roadmap_runs as _comparison
except ImportError:
    import compare_roadmap_runs as _comparison  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    return _comparison.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
