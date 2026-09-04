"""RoadmapBench runtime-validity policy.

Formal Roadmap experiments compare gateway-mediated parent/subagent execution.
OpenClaw's CLI can automatically continue in embedded mode after a Gateway
failure, which is a different runtime and therefore an infrastructure-invalid
sample even if it later produces a candidate patch and verifier reward.

The generic driver already has a pre-verifier Roadmap safety hook. Install a
narrow wrapper around that hook so any *explicitly observed* non-gateway runtime
mode skips verification and is recorded as ERROR. Results without a runtime-mode
field retain the legacy lifecycle-only behavior for backwards compatibility and
unit tests that exercise the helper in isolation.
"""
from __future__ import annotations

from typing import Callable, Optional


def install_roadmap_gateway_only_policy() -> None:
    from agent_bench.runtime import driver

    current: Callable[[dict], Optional[str]] = driver._roadmap_lifecycle_verifier_error
    if getattr(current, "_roadmap_gateway_only_policy", False):
        return

    def wrapped(result: dict) -> Optional[str]:
        runtime_mode = str(result.get("agent_runtime_mode") or "").strip()
        if runtime_mode and runtime_mode != "gateway":
            return (
                "RoadmapBench requires OpenClaw gateway runtime; "
                f"observed agent_runtime_mode={runtime_mode}"
            )
        return current(result)

    setattr(wrapped, "_roadmap_gateway_only_policy", True)
    driver._roadmap_lifecycle_verifier_error = wrapped
