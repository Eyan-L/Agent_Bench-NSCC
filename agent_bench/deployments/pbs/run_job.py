from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Sequence

from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments.pbs.job_spec import PbsJobSpec, build_pbs_job_spec, dump_spec_json
from agent_bench.deployments.pbs.stages import (
    image_preparation_commands,
    otel_command,
    prepare_environment as prepare_pbs_environment,
    prepare_openclaw_plugin_cache,
    run_stage_commands,
)
from agent_bench.deployments.processes import ProcessError, terminate_process_group, wait_for_http_ready
from agent_bench.serving.launch_plan import LaunchPlanError, build_launch_environment, prepare_launch_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Agent Bench experiment inside a PBS allocation.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the PBS job spec without launching work.")
    parser.add_argument("--print-spec", action="store_true", help="Emit redacted PBS job spec JSON.")
    parser.add_argument("--no-retry-errors", action="store_true", help="Do not retry ERROR runs.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EVAL_MAX_WORKERS", "1")))
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
        spec = build_pbs_job_spec(resolved, site=os.environ.get("AGENT_BENCH_PBS_SITE", "NSCC"))
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.print_spec:
        print(dump_spec_json(spec))
        return 0
    if args.dry_run:
        print(_format_spec(spec))
        return 0

    try:
        _run_pbs_job(
            resolved=resolved,
            spec=spec,
            retry_errors=not args.no_retry_errors,
            max_workers=args.max_workers,
        )
    except (ConfigError, LaunchPlanError, ProcessError) as exc:
        print(f"pbs job failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_pbs_job(*, resolved, spec: PbsJobSpec, retry_errors: bool, max_workers: int) -> None:
    _prepare_environment(resolved, spec)
    if spec.stages["materialize_benchmark"]:
        _materialize_benchmark_if_needed(resolved, spec)
    if spec.stages["prepare_images"]:
        _prepare_images(resolved, spec)
    with ExitStack() as stack:
        if spec.stages["start_serving"]:
            stack.enter_context(_serving_sidecar(spec))
        if spec.stages["start_otel"]:
            stack.enter_context(_otel_collector(spec))
        if spec.stages["run_benchmark"]:
            from agent_bench.core.commands import _execute_experiment

            _execute_experiment(resolved, retry_errors=retry_errors, max_workers=max_workers)


def _prepare_environment(resolved, spec: PbsJobSpec) -> None:
    prepare_pbs_environment(resolved, spec, os.environ)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "pbs_job_spec.json").write_text(dump_spec_json(spec), encoding="utf-8")
    os.environ.setdefault("AB_EXPERIMENT_DIR", str(spec.run_dir))
    os.environ.setdefault("AGENT_BENCH_LOG_DIR", str(spec.log_dir))


def _materialize_benchmark_if_needed(resolved, spec: PbsJobSpec) -> None:
    if os.environ.get("INSTANCE_DATA") or isinstance(resolved.spec.raw.get("instance_data"), str):
        return
    if resolved.spec.benchmark_name == "terminal_bench":
        out = spec.run_dir / "terminal_bench_tasks.json"
        from agent_bench.benchmarks.terminal_bench import loader

        task_ids = [str(item) for item in (resolved.benchmark.get("instances") or [])]
        if not task_ids:
            task_ids = loader.list_task_ids()
        if not task_ids:
            raise ConfigError("Terminal-Bench PBS materialization found no task ids")
        loader.materialize_dataset(task_ids, out)
        os.environ["INSTANCE_DATA"] = str(out)
        os.environ["AB_INSTANCE_DATA"] = str(out)
        return
    if resolved.spec.benchmark_name != "swebench":
        return

    out = spec.run_dir / "swebench_instances.json"
    instances = resolved.benchmark.get("instances") or []
    if not instances:
        raise ConfigError("SWE-bench PBS materialization requires benchmark.instances")
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional cluster env
        raise ConfigError(
            "SWE-bench PBS materialization requires the `datasets` package or INSTANCE_DATA"
        ) from exc

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    wanted = set(instances)
    selected = [dict(row) for row in dataset if row.get("instance_id") in wanted]
    missing = wanted - {row.get("instance_id") for row in selected}
    if missing:
        raise ConfigError(f"SWE-bench dataset is missing requested instances: {sorted(missing)}")
    out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    os.environ["INSTANCE_DATA"] = str(out)
    os.environ["AB_INSTANCE_DATA"] = str(out)


def _prepare_images(resolved, spec: PbsJobSpec) -> None:
    commands = image_preparation_commands(resolved, spec, os.environ)
    cache_warm = prepare_openclaw_plugin_cache(resolved, spec, os.environ)
    if cache_warm is not None:
        commands.append(cache_warm)
    if not commands:
        _note("image/runtime preparation already satisfied")
        return
    for command in commands:
        _note(f"running stage command: {command.label}")
    run_stage_commands(commands, base_env=os.environ)


@contextmanager
def _serving_sidecar(spec: PbsJobSpec) -> Iterator[None]:
    plan = spec.serving_plan
    if plan is None:
        yield
        return
    if _endpoint_already_ready(plan):
        yield
        return

    env = build_launch_environment(plan, base_env=os.environ)
    env.setdefault("KIT_ROOT", str(Path.cwd()))
    prepare_launch_artifacts(plan, launch_env=env)
    for key, value in plan.env.items():
        os.environ.setdefault(key, value)

    log_path = Path(plan.log_path or spec.log_dir / f"{plan.backend}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            plan.argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_for_http_ready(
                url=plan.readiness_url,
                headers=plan.readiness_headers,
                proc=proc,
                timeout_seconds=int(os.environ.get("SERVE_READY_TIMEOUT", "600")),
                log_path=log_path,
            )
            yield
        finally:
            terminate_process_group(proc)


@contextmanager
def _otel_collector(spec: PbsJobSpec) -> Iterator[None]:
    start = otel_command(spec, os.environ, "start")
    stop = otel_command(spec, os.environ, "stop")
    try:
        try:
            run_stage_commands([start], base_env=os.environ)
            _note(f"OTel collector requested; trace file -> {os.environ.get('OC_OTEL_TRACE_FILE')}")
        except ConfigError as exc:
            _note(f"WARN: OTel collector failed to start; run continues without spans ({exc})")
        yield
    finally:
        try:
            run_stage_commands([stop], base_env=os.environ)
        except ConfigError:
            pass


def _endpoint_already_ready(plan) -> bool:
    from agent_bench.deployments.processes import http_ready

    return http_ready(plan.readiness_url, headers=plan.readiness_headers)


def _format_spec(spec: PbsJobSpec) -> str:
    lines = [
        f"pbs_job: {spec.resources.job_name}",
        f"site: {spec.site}",
        f"resources: {spec.resources.select_value()} walltime={spec.resources.walltime}",
        f"stages: {spec.stage_summary()}",
        f"run_dir: {spec.run_dir}",
        f"log_dir: {spec.log_dir}",
    ]
    if spec.serving_plan:
        lines.append(
            f"serving: {spec.serving_plan.backend}/{spec.serving_plan.mode} "
            f"{spec.serving_plan.endpoint}"
        )
    if spec.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in spec.warnings)
    return "\n".join(lines)


def _note(message: str) -> None:
    print(f"[agent-bench:pbs] {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
