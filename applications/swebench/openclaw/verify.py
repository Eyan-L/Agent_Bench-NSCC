"""SWE-bench's host-side verification wiring under OpenClaw.

The OpenClaw `verify` skill in `applications/swebench/openclaw/skills/verify/`
instructs the in-container agent to run `bash $WORKSPACE/_oc_validate.sh`,
which round-trips through this bridge to call SWEBenchHarness.run_tests on
the host. The bridge implementation itself lives at
`applications/swebench/openclaw/_validate_bridge.py` because it's harness-agnostic IPC
plumbing — this module just re-exports the entry point so callers can find
it next to the OpenClaw glue for SWE-bench.
"""
from applications.swebench.openclaw._validate_bridge import ValidateBridge

__all__ = ["ValidateBridge"]
