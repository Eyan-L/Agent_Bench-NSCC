"""Placeholder for Terminal-Bench's host-side verification wiring under OpenClaw.

Mirrors applications.swebench.openclaw.verify: re-exports the harness-agnostic
ValidateBridge so a future TB-on-OpenClaw condition can hook self-validation
the same way SWE-bench does. Today's TB ReAct condition does not use the bridge
(it scores once at the end, no mid-run validation), so this module is here for
shape only.
"""
from applications.swebench.openclaw._validate_bridge import ValidateBridge

__all__ = ["ValidateBridge"]
