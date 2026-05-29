from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from agent_bench.core.execution import build_execution_plan, format_execution_plan, run_experiment_stages
from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments.pbs.job_spec import build_pbs_job_spec, dump_spec_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Agent Bench experiment inside a PBS allocation.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the shared execution plan without launching work.")
    parser.add_argument("--print-spec", action="store_true", help="Emit redacted PBS job spec JSON.")
    parser.add_argument("--no-retry-errors", action="store_true", help="Do not retry ERROR runs.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EVAL_MAX_WORKERS", "1")))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text for dry-run.")
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.print_spec:
        try:
            spec = build_pbs_job_spec(resolved, site=os.environ.get("AGENT_BENCH_PBS_SITE", "NSCC"))
        except ConfigError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 2
        print(dump_spec_json(spec))
        return 0

    if args.dry_run:
        try:
            plan = build_execution_plan(resolved)
        except ConfigError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({"experiment": resolved.to_dict(), "execution": plan.to_dict()}, indent=2, sort_keys=True))
        else:
            print(format_execution_plan(plan))
        return 0

    try:
        result = run_experiment_stages(
            resolved,
            retry_errors=not args.no_retry_errors,
            max_workers=args.max_workers,
        )
    except ConfigError as exc:
        print(f"pbs job failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
