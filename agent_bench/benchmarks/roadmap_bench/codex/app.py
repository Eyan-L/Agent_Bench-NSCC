from agent_bench.benchmarks.terminal_bench.codex.app import TerminalBenchCodexHarness


class RoadmapBenchCodexHarness(TerminalBenchCodexHarness):
    def prepare_environment(self, *args, instance_data=None, **kwargs):
        if not instance_data or instance_data.get("benchmark_variant") != "roadmapbench":
            raise ValueError("Roadmap Codex requires RoadmapBench instance_data")
        return super().prepare_environment(*args, instance_data=instance_data, **kwargs)
