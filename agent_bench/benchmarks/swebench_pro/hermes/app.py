from __future__ import annotations

from agent_bench.benchmarks.swebench.hermes.app import SWEBenchHermesHarness
from agent_bench.benchmarks.swebench_pro.app import SWEBenchProHarnessMixin


class SWEBenchProHermesHarness(SWEBenchProHarnessMixin, SWEBenchHermesHarness):
    """Hermes harness for SWE-bench Pro."""

    default_harness_config = "profiles/harnesses/hermes_swebench_pro.yaml"

