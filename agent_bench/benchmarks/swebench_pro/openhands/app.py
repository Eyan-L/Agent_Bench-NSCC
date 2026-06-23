from __future__ import annotations

from agent_bench.benchmarks.swebench.openhands.app import SWEBenchOpenHandsHarness
from agent_bench.benchmarks.swebench_pro.app import SWEBenchProHarnessMixin


class SWEBenchProOpenHandsHarness(SWEBenchProHarnessMixin, SWEBenchOpenHandsHarness):
    """OpenHands harness for SWE-bench Pro."""

    default_harness_config = "profiles/harnesses/openhands_swebench_pro.yaml"

