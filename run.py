#!/usr/bin/env python3
from agent_bench.core.commands import run_main
from agent_bench.core.strict_benchmark import install_strict_benchmark_failure_policy
from agent_bench.deployments.pbs.openclaw_v9_policy import install_openclaw_v9_pbs_policy


if __name__ == "__main__":
    install_strict_benchmark_failure_policy()
    install_openclaw_v9_pbs_policy()
    raise SystemExit(run_main())
