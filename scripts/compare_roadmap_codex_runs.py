#!/usr/bin/env python3
"""Compare paired Codex Roadmap arms using the shared Roadmap selection policy."""
from __future__ import annotations
from typing import Sequence

try:
    from . import compare_roadmap_runs as comparison
    from . import roadmap_codex_telemetry as telemetry
except ImportError:
    import compare_roadmap_runs as comparison
    import roadmap_codex_telemetry as telemetry


def main(argv: Sequence[str] | None = None) -> int:
    telemetry.install_comparison(comparison)
    return comparison.main(argv, omit_candidate_workspace=True)


if __name__ == "__main__":
    raise SystemExit(main())
