#!/usr/bin/env python3
"""Compare paired Hermes RoadmapBench subagent and single-agent PBS arms.

The Roadmap comparison policy is shared with OpenClaw so latency, reward,
selection, and subagent counts retain the same definitions. Hermes candidate
workspaces remain in the original run folders instead of being copied into
the agent-analysis bundle.
"""

from __future__ import annotations

from typing import Sequence

try:
    from . import compare_roadmap_runs as _comparison
    from . import roadmap_hermes_telemetry as _telemetry
except ImportError:
    import compare_roadmap_runs as _comparison  # type: ignore
    import roadmap_hermes_telemetry as _telemetry  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    _telemetry.install_comparison(_comparison)
    return _comparison.main(argv, omit_candidate_workspace=True)


if __name__ == "__main__":
    raise SystemExit(main())
