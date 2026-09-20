from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from agent_bench.core.execution import (
    _instance_data_path,
    build_execution_plan,
    format_execution_plan,
    resolve_max_workers,
    run_experiment_stages,
)
from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.core.run_artifacts import create_single_run_bundle
from agent_bench.deployments import plan_deployment


def setup_cadvisor(port: int = 8088):
    """Install/setup cAdvisor if docker is available."""
    if not shutil.which("docker"):
        return

    # Check if cadvisor is already running
    proc = subprocess.run(
        ["docker", "ps", "--filter", "name=cadvisor", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if "cadvisor" in proc.stdout:
        print("cAdvisor is already running.")
        return

    print(f"Starting cAdvisor sidecar on port {port}...")
    # Standard cadvisor run command
    cmd = [
        "docker", "run",
        "--volume=/:/rootfs:ro",
        "--volume=/var/run:/var/run:ro",
        "--volume=/sys:/sys:ro",
        "--volume=/var/lib/docker/:/var/lib/docker:ro",
        "--volume=/dev/disk/:/dev/disk:ro",
        f"--publish={port}:8080",
        "--detach=true",
        "--name=cadvisor",
        "--privileged",
        "--device=/dev/kmsg",
        "gcr.io/cadvisor/cadvisor:latest"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"cAdvisor started on http://localhost:{port}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start cAdvisor: {e}")


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
    parser.add_argument(
        "--run-dir",
        "--ab-run-dir",
        dest="run_dir",
        help="Set AB_RUN_DIR/AB_EXPERIMENT_DIR for this invocation.",
    )
    parser.add_argument(
        "--serving-mode",
        "--ab-serving-mode",
        dest="serving_mode",
        choices=("native", "docker", "singularity"),
        help="Set AB_SERVING_MODE for this invocation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render the execution plan without launching work.")
    parser.add_argument("--render-job", action="store_true", help="Print rendered job script when available.")
    parser.add_argument("--print-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-retry-errors", action="store_true", help="Do not retry ERROR runs.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--skip-cadvisor-setup", action="store_true", help="Skip cAdvisor sidecar setup for container metrics.")
    parser.add_argument("--cadvisor-port", type=int, default=8088, help="Port for cAdvisor API (default 8088).")
    args = parser.parse_args(argv)

    if args.run_dir:
        _apply_cli_run_dir(args.run_dir)
    if args.serving_mode:
        os.environ["AB_SERVING_MODE"] = args.serving_mode
    if not args.skip_cadvisor_setup:
        setup_cadvisor(args.cadvisor_port)
    # Propagate cadvisor port to environment so telemetry can find it
    os.environ["AB_CADVISOR_PORT"] = str(args.cadvisor_port)

    try:
        resolved = load_resolved_experiment(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.serving_mode:
        _apply_cli_serving_mode(resolved.serving.get("backend"), args.serving_mode)

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
        from agent_bench.runtime.driver import _install_signal_handlers

        _install_signal_handlers()
        result = run_experiment_stages(
            resolved,
            retry_errors=not args.no_retry_errors,
            max_workers=resolve_max_workers(resolved, cli_max_workers=args.max_workers),
        )
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _apply_cli_run_dir(value: str) -> None:
    run_dir = str(Path(value).expanduser())
    os.environ["AB_RUN_DIR"] = run_dir
    os.environ["AB_EXPERIMENT_DIR"] = run_dir


def _apply_cli_serving_mode(backend: object, value: str) -> None:
    os.environ["AB_SERVING_MODE"] = value
    if isinstance(backend, str) and backend:
        os.environ[f"{backend.upper()}_MODE"] = value


def analyze_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and export a portable Agent Bench run report."
    )
    parser.add_argument("--run", required=True, help="Path to a run directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument(
        "--output-dir",
        default="reports/analyze",
        help="Parent directory for the portable report bundle.",
    )
    parser.add_argument("--name", help="Optional report bundle name.")
    args = parser.parse_args(argv)

    run_path = Path(args.run).expanduser()
    if not run_path.exists():
        print(f"missing run directory: {run_path}", file=sys.stderr)
        return 2
    if not run_path.is_dir():
        print(f"run path is not a directory: {run_path}", file=sys.stderr)
        return 2

    try:
        bundle_dir, report = create_single_run_bundle(
            run_path,
            Path(args.output_dir),
            name=args.name,
        )
    except (OSError, ValueError) as exc:
        print(f"could not generate analysis bundle: {exc}", file=sys.stderr)
        return 2

    summary = {
        "run": str(run_path.resolve()),
        "manifest": (
            str((run_path / "manifest.json").resolve())
            if (run_path / "manifest.json").exists()
            else None
        ),
        "result_files": sorted(path.name for path in run_path.glob("*.json")),
        "bundle": str(bundle_dir.resolve()),
        "index": str((bundle_dir / "index.html").resolve()),
        **report,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"run: {summary['run']}")
        print(f"manifest: {summary['manifest'] or 'not found'}")
        print(f"json files: {len(summary['result_files'])}")
        print(f"bundle: {summary['bundle']}")
        print(f"viewer: {summary['index']}")
        result = report.get("summary", {}).get("result", {})
        print(
            "outcome: "
            f"{result.get('agent_execution_outcome') or 'unknown'}; "
            f"verifier={result.get('verifier_status') or 'unknown'}; "
            f"reward={result.get('reward')}"
        )
        artifact_bundle = report.get("artifact_bundle", {})
        print(
            "artifacts: "
            f"complete={artifact_bundle.get('complete')}; "
            f"statuses={artifact_bundle.get('status_counts')}"
        )
    return 0
