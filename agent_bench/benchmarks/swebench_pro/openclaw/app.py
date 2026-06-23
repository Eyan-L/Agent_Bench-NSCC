from __future__ import annotations

from agent_bench.benchmarks.swebench.openclaw.app import SWEBenchHarness
from agent_bench.benchmarks.swebench_pro.app import SWEBenchProHarnessMixin


class SWEBenchProHarness(SWEBenchProHarnessMixin, SWEBenchHarness):
    """OpenClaw harness for SWE-bench Pro."""

    default_harness_config = "profiles/harnesses/openclaw_swebench_pro.yaml"

