import inspect

from agent_bench.benchmarks.deepresearch_bench.app import DeepResearchHarness
from agent_bench.benchmarks.longbench.app import LongBenchHarness
from agent_bench.benchmarks.terminal_bench.agent_harness_base import (
    TerminalBenchAgentHarnessBase,
)
from agent_bench.benchmarks.terminal_bench.app import (
    TerminalBenchDockerHarness,
    TerminalBenchSingularityHarness,
)


def test_non_swe_harness_run_tests_accepts_driver_run_dir():
    harnesses = [
        TerminalBenchDockerHarness,
        TerminalBenchSingularityHarness,
        TerminalBenchAgentHarnessBase,
        DeepResearchHarness,
        LongBenchHarness,
    ]

    for harness in harnesses:
        signature = inspect.signature(harness.run_tests)
        assert "run_dir" in signature.parameters
