#!/usr/bin/env python3
import sys

from agent_bench.core.commands import run_main
from agent_bench.core.spec import ConfigError
from agent_bench.core.strict_benchmark import install_strict_benchmark_failure_policy
from agent_bench.deployments.pbs.openclaw_v9_policy import (
    install_openclaw_v9_pbs_policy,
    preflight_openclaw_v9_pbs_candidate,
)


if __name__ == "__main__":
    install_strict_benchmark_failure_policy()
    install_openclaw_v9_pbs_policy()
    try:
        preflight_openclaw_v9_pbs_candidate()
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run_main())
