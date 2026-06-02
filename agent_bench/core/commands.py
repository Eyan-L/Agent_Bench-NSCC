from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from agent_bench.core.execution import (
    _instance_data_path,
    build_execution_plan,
    format_execution_plan,
    run_experiment_stages,
)
from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments import plan_deployment


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Agent Bench experiment TOML.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--json", action="store_true", help="Emit resolved experiment JSON.")
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(resolved.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"valid: {resolved.spec.name} ({resolved.run_id})")
    return 0


def run_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and run an Agent Bench experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--dry-run", action="store_true", help="Render the execution plan without launching work.")
    parser.add_argument("--render-job", action="store_true", help="Print rendered job script when available.")
    parser.add_argument("--print-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-retry-errors", action="store_true", help="Do not retry ERROR runs.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("AB_RUN_MAX_WORKERS", "1")))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.render_job or args.print_job:
        try:
            deployment_plan = plan_deployment(resolved)
        except ConfigError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({"experiment": resolved.to_dict(), "plan": deployment_plan.to_dict()}, indent=2, sort_keys=True))
            return 0
        if deployment_plan.job_script:
            print(deployment_plan.job_script, end="")
            return 0

    try:
        execution_plan = build_execution_plan(resolved)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.json:
            print(json.dumps({"experiment": resolved.to_dict(), "execution": execution_plan.to_dict()}, indent=2, sort_keys=True))
        else:
            print(format_execution_plan(execution_plan))
        return 0

    try:
        result = run_experiment_stages(
            resolved,
            retry_errors=not args.no_retry_errors,
            max_workers=args.max_workers,
        )
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def analyze_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an Agent Bench run directory.")
    parser.add_argument("--run", required=True, help="Path to a run directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    args = parser.parse_args(argv)

    run_path = Path(args.run).expanduser()
    if not run_path.exists():
        print(f"missing run directory: {run_path}", file=sys.stderr)
        return 2
    if not run_path.is_dir():
        print(f"run path is not a directory: {run_path}", file=sys.stderr)
        return 2

    summary = {
        "run": str(run_path.resolve()),
        "manifest": str((run_path / "manifest.json").resolve()) if (run_path / "manifest.json").exists() else None,
        "result_files": sorted(path.name for path in run_path.glob("*.json")),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"run: {summary['run']}")
        print(f"manifest: {summary['manifest'] or 'not found'}")
        print(f"json files: {len(summary['result_files'])}")
    return 0
