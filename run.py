#!/usr/bin/env python3
from agent_bench.core.commands import run_main
from agent_bench.core.strict_benchmark import install_strict_benchmark_failure_policy


if __name__ == "__main__":
    install_strict_benchmark_failure_policy()
    raise SystemExit(run_main())
