from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import importlib
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from agent_bench.core.spec import ConfigError, ResolvedExperiment, STAGE_NAMES
from agent_bench.deployments.processes import (
    ProcessError,
    http_ready,
    tail_text,
    terminate_process_group,
    wait_for_http_ready,
)
from agent_bench.serving.launch_plan import (
    LaunchPlanError,
    ServingLaunchPlan,
    build_launch_environment,
    build_launch_plan,
    prepare_launch_artifacts,
)


STAGE_ORDER = STAGE_NAMES

_SWEBENCH_PYTHON_DEPENDENCIES = (
    ("datasets", "datasets"),
    ("swebench.harness.test_spec.test_spec", "swebench"),
    ("swebench.harness.grading", "swebench"),
)
_LITELLM_PYTHON_DEPENDENCIES = (("litellm", "litellm"),)

_STAGE_DESCRIPTIONS = {
    "prepare_environment": "prepare run directories and runtime environment",
    "materialize_benchmark": "resolve benchmark instance_data",
    "prepare_images": "prepare benchmark, serving, and harness runtime images",
    "start_serving": "start or attach to serving sidecar",
    "start_otel": "start telemetry collector",
    "run_benchmark": "run benchmark driver",
    "cleanup": "stop sidecars and flush telemetry",
}


def resolve_max_workers(
    resolved: ResolvedExperiment,
    *,
    cli_max_workers: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    if cli_max_workers is not None:
        return _positive_int(cli_max_workers, "--max-workers")

    target_env = os.environ if env is None else env
    env_value = target_env.get("AB_RUN_MAX_WORKERS")
    if env_value is not None:
        try:
            parsed = int(env_value)
        except ValueError as exc:
            raise ConfigError("AB_RUN_MAX_WORKERS must be a positive integer") from exc
        return _positive_int(parsed, "AB_RUN_MAX_WORKERS")

    return _positive_int(
        resolved.spec.limits.get("max_workers", 1),
        "limits.max_workers",
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExecutionStage:
    name: str
    enabled: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    experiment_name: str
    run_id: str
    deployment: str
    run_dir: Path
    log_dir: Path
    stages: tuple[ExecutionStage, ...]
    serving_plan: ServingLaunchPlan | None = None
    pbs_spec: Any | None = None
    warnings: tuple[str, ...] = ()

    def stage_enabled(self, name: str) -> bool:
        for stage in self.stages:
            if stage.name == name:
                return stage.enabled
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "deployment": self.deployment,
            "run_dir": str(self.run_dir),
            "log_dir": str(self.log_dir),
            "stages": [stage.to_dict() for stage in self.stages],
            "serving_plan": self.serving_plan.to_redacted_dict() if self.serving_plan else None,
            "warnings": list(self.warnings),
        }
        if self.pbs_spec is not None:
            data["pbs_job"] = self.pbs_spec.to_redacted_dict()
        return data


@dataclass(frozen=True)
class ExecutionResult:
    plan: ExecutionPlan
    completed_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "completed_stages": list(self.completed_stages),
            "skipped_stages": list(self.skipped_stages),
        }


@dataclass
class ExecutionContext:
    resolved: ResolvedExperiment
    plan: ExecutionPlan
    retry_errors: bool
    max_workers: int
    env: dict[str, str]
    instance_data_path: Path | None = None


def build_execution_plan(
    resolved: ResolvedExperiment,
    *,
    env: Mapping[str, str] | None = None,
) -> ExecutionPlan:
    source_env = dict(os.environ if env is None else env)
    deployment = str(resolved.deployment.get("kind") or resolved.deployment.get("name"))
    warnings: list[str] = []
    pbs_spec = None
    serving_plan: ServingLaunchPlan | None = None

    if deployment == "nscc_pbs":
        from agent_bench.deployments.pbs.job_spec import build_pbs_job_spec

        pbs_spec = build_pbs_job_spec(
            resolved,
            site=source_env.get("AB_PBS_SITE", "NSCC"),
            env=source_env,
        )
        run_dir = pbs_spec.run_dir
        log_dir = pbs_spec.log_dir
        execution_run_id = pbs_spec.metadata.get("run_stamp", pbs_spec.run_dir.name)
        stage_flags = dict(pbs_spec.stages)
        serving_plan = pbs_spec.serving_plan
        warnings.extend(pbs_spec.warnings)
    else:
        execution_run_id, run_dir = _resolve_execution_run_dir(resolved, source_env)
        log_dir = _path_from_env_or_default(source_env, "AB_LOG_DIR", run_dir / "logs")
        stage_flags = _default_stage_flags(resolved, source_env)
        serving_plan = _build_serving_plan_if_needed(resolved, source_env, log_dir=log_dir, warnings=warnings)

    stages = tuple(
        ExecutionStage(
            name=name,
            enabled=bool(stage_flags.get(name, True)),
            description=_STAGE_DESCRIPTIONS[name],
        )
        for name in STAGE_ORDER
    )
    return ExecutionPlan(
        experiment_name=resolved.spec.name,
        run_id=execution_run_id,
        deployment=deployment,
        run_dir=run_dir,
        log_dir=log_dir,
        stages=stages,
        serving_plan=serving_plan,
        pbs_spec=pbs_spec,
        warnings=tuple(warnings),
    )


def run_experiment_stages(
    resolved: ResolvedExperiment,
    *,
    retry_errors: bool,
    max_workers: int,
    dry_run: bool = False,
) -> ExecutionResult:
    plan = build_execution_plan(resolved)
    if dry_run:
        return ExecutionResult(plan=plan, completed_stages=(), skipped_stages=())

    ctx = ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=retry_errors,
        max_workers=max_workers,
        env=os.environ,
    )
    completed: list[str] = []
    skipped: list[str] = []
    stack = ExitStack()
    cleanup_enabled = plan.stage_enabled("cleanup")

    try:
        for stage in plan.stages:
            if stage.name == "cleanup":
                continue
            if not stage.enabled:
                _note(stage.name, "skipped")
                skipped.append(stage.name)
                continue
            _note(stage.name, "start")
            _run_stage(ctx, stage.name, stack)
            completed.append(stage.name)
            _note(stage.name, "done")
    finally:
        if cleanup_enabled:
            _note("cleanup", "start")
            stack.close()
            completed.append("cleanup")
            _note("cleanup", "done")
        else:
            _note("cleanup", "skipped")
            stack.pop_all()
            skipped.append("cleanup")

    return ExecutionResult(
        plan=plan,
        completed_stages=tuple(completed),
        skipped_stages=tuple(skipped),
    )


def format_execution_plan(plan: ExecutionPlan) -> str:
    lines = [
        f"experiment: {plan.experiment_name}",
        f"run_id: {plan.run_id}",
        f"deployment: {plan.deployment}",
        f"run_dir: {plan.run_dir}",
        f"log_dir: {plan.log_dir}",
        "stages:",
    ]
    for stage in plan.stages:
        state = "enabled" if stage.enabled else "skipped"
        lines.append(f"  - {stage.name}: {state} - {stage.description}")
    if plan.serving_plan is not None:
        lines.append(
            f"serving: {plan.serving_plan.backend}/{plan.serving_plan.mode} "
            f"{plan.serving_plan.endpoint}"
        )
    if plan.pbs_spec is not None:
        lines.append(f"pbs_job: {plan.pbs_spec.resources.job_name}")
        lines.append(
            f"pbs_resources: {plan.pbs_spec.resources.select_value()} "
            f"walltime={plan.pbs_spec.resources.walltime}"
        )
    if plan.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def materialize_benchmark_data(
    resolved: ResolvedExperiment,
    *,
    pbs_spec: Any | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    target_env = os.environ if env is None else env
    explicit = _optional_path_from_raw_or_env(resolved, "benchmark_instance_data", "AB_BENCHMARK_INSTANCE_DATA", target_env)
    if explicit is not None:
        _set_instance_data_env(target_env, explicit)
        return explicit

    if resolved.spec.benchmark_name == "swebench":
        from agent_bench.benchmarks.swebench.instance_data import (
            default_materialized_path,
            materialize_profile_dataset,
            resolve_profile_instance_data,
        )

        profile_path = resolve_profile_instance_data(resolved)
        out = profile_path
        if out is None:
            out = pbs_spec.run_dir / "swebench_instances.json" if pbs_spec else default_materialized_path(resolved)
            out = materialize_profile_dataset(resolved, out)
        _set_instance_data_env(target_env, out)
        return out

    if resolved.spec.benchmark_name == "terminal_bench":
        out = pbs_spec.run_dir / "terminal_bench_tasks.json" if pbs_spec else Path("data/terminal_bench/tasks.json")
        from agent_bench.benchmarks.terminal_bench import loader

        task_ids = [str(item) for item in (resolved.benchmark.get("instances") or [])]
        if not task_ids:
            task_ids = loader.list_task_ids()
        if not task_ids:
            raise ConfigError("Terminal-Bench materialization found no task ids")
        loader.materialize_dataset(task_ids, out)
        out = out.resolve()
        _set_instance_data_env(target_env, out)
        return out

    if resolved.spec.benchmark_name == "deepresearch_bench":
        out = Path("data/deepresearch_bench/instances.json").resolve()
        from agent_bench.benchmarks.deepresearch_bench import loader

        loader.materialize_dataset(loader.list_task_ids(), out)
        _set_instance_data_env(target_env, out)
        return out

    if resolved.spec.benchmark_name == "longbench":
        out = Path("data/longbench/instances.json").resolve()
        from agent_bench.benchmarks.longbench import loader

        loader.materialize_dataset(loader.list_task_ids(), out)
        _set_instance_data_env(target_env, out)
        return out

    if resolved.spec.benchmark_name == "request_bench":
        sweep_name = str(resolved.benchmark.get("sweep_name") or resolved.spec.name)
        out = Path("data/request_bench") / f"{sweep_name}_instances.json"
        from agent_bench.benchmarks.request_bench import loader

        loader.materialize_profile(resolved.benchmark, out)
        out = out.resolve()
        _set_instance_data_env(target_env, out)
        return out

    raise ConfigError(f"unsupported benchmark for instance_data resolution: {resolved.spec.benchmark_name}")


def _run_stage(ctx: ExecutionContext, stage_name: str, stack: ExitStack) -> None:
    if stage_name == "prepare_environment":
        _prepare_environment(ctx)
    elif stage_name == "materialize_benchmark":
        ctx.instance_data_path = materialize_benchmark_data(
            ctx.resolved,
            pbs_spec=ctx.plan.pbs_spec,
            env=ctx.env,
        )
    elif stage_name == "prepare_images":
        _prepare_images(ctx)
    elif stage_name == "start_serving":
        _start_serving(ctx, stack)
    elif stage_name == "start_otel":
        _start_otel(ctx, stack)
    elif stage_name == "run_benchmark":
        _run_benchmark(ctx)
    else:
        raise ConfigError(f"unknown execution stage: {stage_name}")


def _prepare_environment(ctx: ExecutionContext) -> None:
    if ctx.plan.pbs_spec is not None:
        from agent_bench.deployments.pbs.job_spec import dump_spec_json
        from agent_bench.deployments.pbs.stages import prepare_environment as prepare_pbs_environment

        spec = ctx.plan.pbs_spec
        prepare_pbs_environment(ctx.resolved, spec, ctx.env)
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "pbs_job_spec.json").write_text(dump_spec_json(spec), encoding="utf-8")
    else:
        ctx.plan.run_dir.mkdir(parents=True, exist_ok=True)
        ctx.plan.log_dir.mkdir(parents=True, exist_ok=True)
        ctx.env.setdefault("AB_REPO_ROOT", str(Path.cwd()))
        ctx.env.setdefault("AB_LOG_DIR", str(ctx.plan.log_dir))

    ctx.env.setdefault("AB_STAGE_SUMMARY", _stage_summary(ctx.plan))
    ctx.env.setdefault("AB_RUN_STAMP", ctx.plan.run_id)
    _set_aliased_run_dir_env(ctx.env, ctx.plan.run_dir)
    ctx.env.setdefault("AB_LOG_DIR", str(ctx.plan.log_dir))
    ctx.env.setdefault("AB_EXPERIMENT_CONFIG", str(ctx.resolved.spec.config_path))
    ctx.env.setdefault("AGENT_BENCH_RUN_DIR", str(ctx.plan.run_dir))
    ctx.env.setdefault("AGENT_BENCH_LOG_DIR", str(ctx.plan.log_dir))
    _prepare_otel_environment(ctx)
    _prepare_swebench_runtime_root(ctx)
    _prepare_python_dependencies(ctx)


def _set_aliased_run_dir_env(env: dict[str, str], run_dir: Path) -> None:
    value = str(run_dir)
    env["AB_RUN_DIR"] = value
    env["AB_EXPERIMENT_DIR"] = value
    os.environ["AB_RUN_DIR"] = value
    os.environ["AB_EXPERIMENT_DIR"] = value


def _prepare_otel_environment(ctx: ExecutionContext) -> None:
    data_dir = Path(ctx.env.get("AB_TELEMETRY_DATA_DIR") or ctx.plan.log_dir / "otel").expanduser()
    trace_file = Path(ctx.env.get("AB_TELEMETRY_TRACE_FILE") or data_dir / "otel-traces.jsonl").expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.touch(exist_ok=True)
    ctx.env["AB_TELEMETRY_DATA_DIR"] = str(data_dir)
    ctx.env["AB_TELEMETRY_TRACE_FILE"] = str(trace_file)
    ctx.env.setdefault("AB_TELEMETRY_NAME", f"agent-bench-otel-{_sanitize_run_stamp_suffix(ctx.plan.run_id)}")


def _prepare_swebench_runtime_root(ctx: ExecutionContext) -> None:
    if ctx.resolved.spec.benchmark_name != "swebench":
        return

    root = _optional_path_from_raw_or_env(
        ctx.resolved,
        "benchmark_data_root",
        "AB_BENCHMARK_DATA_ROOT",
        ctx.env,
    )
    if root is None:
        root = ctx.plan.run_dir / "benchmark_data_root"
        _note("prepare_environment", f"AB_BENCHMARK_DATA_ROOT defaulting to {root}")

    root.mkdir(parents=True, exist_ok=True)
    value = str(root.resolve())
    ctx.env["AB_BENCHMARK_DATA_ROOT"] = value
    os.environ["AB_BENCHMARK_DATA_ROOT"] = value


def _prepare_python_dependencies(ctx: ExecutionContext) -> None:
    dependencies = _python_dependency_modules_for_context(ctx)
    needs_litellm_cli = _needs_litellm_cli(ctx)
    if not dependencies and not needs_litellm_cli:
        return

    install_target = _python_dependency_install_target(ctx)
    if install_target is not None:
        install_target.mkdir(parents=True, exist_ok=True)
        _prepend_python_path(ctx.env, install_target)
    if needs_litellm_cli:
        _prepend_current_python_bin_to_path(ctx.env)

    missing_packages = _missing_python_dependency_packages(dependencies)
    if needs_litellm_cli and not _has_executable("litellm", ctx.env):
        _append_unique(missing_packages, "litellm")

    if not missing_packages:
        _note("prepare_environment", f"Python deps available in {_python_environment_label(ctx.env)}")
        return

    if not _parse_bool(ctx.env.get("AB_INSTALL_PYTHON_DEPS", "1")):
        missing = ", ".join(missing_packages)
        raise ConfigError(
            f"missing Python dependencies ({missing}); "
            "install them in the active Python environment or set AB_INSTALL_PYTHON_DEPS=1"
        )

    target_label = str(install_target) if install_target is not None else _python_environment_label(ctx.env)
    _note("prepare_environment", f"installing Python deps into {target_label}: {', '.join(missing_packages)}")
    _run_python_dependency_install(
        missing_packages,
        log_path=ctx.plan.log_dir / "python-deps.log",
        env=ctx.env,
        install_target=install_target,
    )
    importlib.invalidate_caches()
    if needs_litellm_cli:
        _prepend_current_python_bin_to_path(ctx.env)

    still_missing = _missing_python_dependency_packages(dependencies)
    if needs_litellm_cli and not _has_executable("litellm", ctx.env):
        _append_unique(still_missing, "litellm")
    if still_missing:
        missing = ", ".join(still_missing)
        raise ConfigError(
            f"Python dependencies still missing after install ({missing}); "
            f"see {ctx.plan.log_dir / 'python-deps.log'}"
        )


def _python_dependency_modules_for_context(ctx: ExecutionContext) -> tuple[tuple[str, str], ...]:
    dependencies: list[tuple[str, str]] = []
    if ctx.resolved.spec.benchmark_name == "swebench":
        dependencies.extend(_SWEBENCH_PYTHON_DEPENDENCIES)
    if _needs_litellm_cli(ctx):
        dependencies.extend(_LITELLM_PYTHON_DEPENDENCIES)
    return tuple(dependencies)


def _needs_litellm_cli(ctx: ExecutionContext) -> bool:
    plan = ctx.plan.serving_plan
    return plan is not None and plan.backend == "litellm" and plan.mode == "native"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _python_dependency_install_target(ctx: ExecutionContext) -> Path | None:
    value = ctx.env.get("AB_RUNTIME_PYLIB")
    if value:
        return Path(value).expanduser()
    if ctx.plan.pbs_spec is None:
        return ctx.plan.run_dir / "python_deps"
    return None


def _missing_python_dependency_packages(dependencies: tuple[tuple[str, str], ...]) -> list[str]:
    missing: list[str] = []
    for module, package in dependencies:
        if not _has_python_module(module) and package not in missing:
            missing.append(package)
    return missing


def _has_python_module(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _run_python_dependency_install(
    packages: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    install_target: Path | None,
) -> None:
    command = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if install_target is not None:
        command.extend(["--target", str(install_target)])
        cache_dir = env.get("AB_RUNTIME_PIP_CACHE")
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            command.extend(["--cache-dir", cache_dir])
    command.extend(packages)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if proc.returncode:
        raise ConfigError(
            f"failed to install Python dependencies with {sys.executable}; "
            f"see {log_path}"
        )


def _prepend_python_path(env: dict[str, str], path: Path) -> None:
    value = str(path.resolve())
    entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if value not in entries:
        entries.insert(0, value)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    if value not in sys.path:
        sys.path.insert(0, value)


def _has_executable(name: str, env: Mapping[str, str]) -> bool:
    return shutil.which(name, path=env.get("PATH")) is not None


def _prepend_current_python_bin_to_path(env: dict[str, str]) -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    entries = [entry for entry in env.get("PATH", os.environ.get("PATH", "")).split(os.pathsep) if entry]
    if bin_dir not in entries:
        entries.insert(0, bin_dir)
    env["PATH"] = os.pathsep.join(entries)
    os.environ["PATH"] = env["PATH"]


def _python_environment_label(env: Mapping[str, str]) -> str:
    conda_env = env.get("CONDA_DEFAULT_ENV")
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_env and conda_prefix:
        return f"conda env {conda_env} ({conda_prefix})"
    if conda_env:
        return f"conda env {conda_env}"
    return sys.executable


def _prepare_images(ctx: ExecutionContext) -> None:
    if ctx.plan.pbs_spec is None:
        return
    from agent_bench.deployments.pbs.stages import (
        image_preparation_commands,
        prepare_openclaw_plugin_cache,
        run_stage_commands,
    )

    commands = image_preparation_commands(ctx.resolved, ctx.plan.pbs_spec, ctx.env)
    cache_warm = prepare_openclaw_plugin_cache(ctx.resolved, ctx.plan.pbs_spec, ctx.env)
    if cache_warm is not None:
        commands.append(cache_warm)
    if not commands:
        _note("prepare_images", "no image/runtime commands needed")
        return
    for command in commands:
        _note("prepare_images", f"running command: {command.label}")
    run_stage_commands(commands, base_env=ctx.env)


def _start_serving(ctx: ExecutionContext, stack: ExitStack) -> None:
    plan = ctx.plan.serving_plan
    if plan is None:
        return
    _export_serving_agent_env(plan, ctx.env)
    if plan.backend == "litellm":
        agent_provider = (ctx.resolved.serving.get("proxy") or {}).get("agent_provider")
        if agent_provider:
            ctx.env.setdefault("AB_MODEL_PROVIDER", str(agent_provider))
            os.environ.setdefault("AB_MODEL_PROVIDER", str(agent_provider))
    if http_ready(plan.readiness_url, headers=plan.readiness_headers):
        _note("start_serving", f"endpoint already ready: {plan.readiness_url}")
        return

    log_path = Path(plan.log_path or ctx.plan.log_dir / f"{plan.backend}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        launch_env = build_launch_environment(plan, base_env=ctx.env)
        launch_env.setdefault("AB_REPO_ROOT", str(Path.cwd()))
        prepare_launch_artifacts(plan, launch_env=launch_env)
    except LaunchPlanError as exc:
        raise ConfigError(str(exc)) from exc

    with log_path.open("ab", buffering=0) as log:
        try:
            proc = subprocess.Popen(
                plan.argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=launch_env,
                start_new_session=True,
            )
        except OSError as exc:
            raise ConfigError(f"failed to launch serving backend {plan.backend}: {exc}") from exc
        readiness_proc = proc
        cleanup_proc = proc
        if _is_detached_docker_launch(plan):
            readiness_proc = _wait_for_detached_docker_launch(proc, log_path)
            cleanup_proc = readiness_proc

        container_name = _docker_container_name_for_cleanup(plan)
        if _is_detached_docker_launch(plan) and readiness_proc is None and container_name:
            readiness_proc = _start_detached_docker_log_follow(container_name, log_path)
            cleanup_proc = readiness_proc
        if cleanup_proc is not None:
            stack.callback(terminate_process_group, cleanup_proc)
        if container_name:
            stack.callback(_remove_docker_container, container_name)
        try:
            wait_for_http_ready(
                url=plan.readiness_url,
                headers=plan.readiness_headers,
                proc=readiness_proc,
                timeout_seconds=int(ctx.env.get("SERVE_READY_TIMEOUT", "600")),
                log_path=log_path,
            )
        except ProcessError as exc:
            raise ConfigError(str(exc)) from exc


def _docker_container_name_for_cleanup(plan: ServingLaunchPlan) -> str | None:
    if plan.mode == "docker":
        return plan.metadata.get("container_name")
    return None


def _is_detached_docker_launch(plan: ServingLaunchPlan) -> bool:
    return plan.mode == "docker" and plan.argv[:2] == ("docker", "run") and (
        "-d" in plan.argv or "--detach" in plan.argv
    )


def _wait_for_detached_docker_launch(
    proc: subprocess.Popen[bytes],
    log_path: Path,
) -> subprocess.Popen[bytes] | None:
    try:
        returncode = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        return proc
    if returncode:
        details = f"; see {log_path}"
        tail = tail_text(log_path, lines=20)
        if tail:
            details = f"{details}\n{tail}"
        raise ConfigError(f"detached docker serving command failed with exit code {returncode}{details}")
    return None


def _start_detached_docker_log_follow(
    container_name: str,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    log.close()
    return proc


def _remove_docker_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _start_otel(ctx: ExecutionContext, stack: ExitStack) -> None:
    _prepare_otel_environment(ctx)
    if ctx.plan.pbs_spec is None:
        _start_local_otel(ctx, stack)
        return
    from agent_bench.deployments.pbs.stages import otel_command, run_stage_commands

    start = otel_command(ctx.plan.pbs_spec, ctx.env, "start")
    stop = otel_command(ctx.plan.pbs_spec, ctx.env, "stop")
    try:
        run_stage_commands([start], base_env=ctx.env)
        _note("start_otel", f"trace file -> {ctx.env.get('AB_TELEMETRY_TRACE_FILE')}")
    except ConfigError as exc:
        _note("start_otel", f"WARN: collector failed to start; run continues without spans ({exc})")
        return

    def _stop_otel() -> None:
        try:
            run_stage_commands([stop], base_env=ctx.env)
        except ConfigError:
            pass

    stack.callback(_stop_otel)


def _start_local_otel(ctx: ExecutionContext, stack: ExitStack) -> None:
    mode = ctx.env.get("AB_TELEMETRY_MODE", "docker").strip().lower()
    if mode in {"0", "false", "no", "off", "none", "skip"}:
        _note("start_otel", "skipped by AB_TELEMETRY_MODE")
        return
    if mode not in {"docker", "host"}:
        raise ConfigError(f"unsupported AB_TELEMETRY_MODE={mode!r}; expected docker, host, or off")

    start_script = _local_otel_script("host_up.sh" if mode == "host" else "otel_collector_up.sh", ctx)
    stop_script = _local_otel_script("host_down.sh" if mode == "host" else "otel_collector_down.sh", ctx)
    try:
        _run_local_otel_script(ctx, start_script, "start")
        _note("start_otel", f"{mode} collector trace file -> {ctx.env.get('AB_TELEMETRY_TRACE_FILE')}")
    except ConfigError as exc:
        _note("start_otel", f"WARN: collector failed to start; run continues without spans ({exc})")
        return

    def _stop_local_otel() -> None:
        try:
            _run_local_otel_script(ctx, stop_script, "stop")
        except ConfigError:
            pass

    stack.callback(_stop_local_otel)


def _local_otel_script(name: str, ctx: ExecutionContext) -> Path:
    repo_root = Path(ctx.env.get("AB_REPO_ROOT") or Path.cwd())
    return repo_root / "agent_bench" / "telemetry" / "otel" / name


def _run_local_otel_script(ctx: ExecutionContext, script: Path, action: str) -> None:
    if not script.is_file():
        raise ConfigError(f"OTel script not found: {script}")
    log_path = ctx.plan.log_dir / f"otel-{action}.log"
    env = os.environ.copy()
    env.update(ctx.env)
    with log_path.open("ab") as log:
        proc = subprocess.run(
            ["bash", str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise ConfigError(f"{script.name} failed with rc={proc.returncode}; see {log_path}")


def _run_benchmark(ctx: ExecutionContext) -> None:
    from agent_bench.runtime.driver import run_sweep

    resolved = ctx.resolved
    benchmark_path = resolved.profile_paths.get("benchmark")
    harness_path = resolved.profile_paths.get("harness")
    model_path = resolved.profile_paths["model"]
    serving_path = resolved.profile_paths["serving"]
    if benchmark_path is None or harness_path is None:
        raise ConfigError("experiments must reference benchmark.profile and harness.profile to execute")

    instance_data = ctx.instance_data_path or materialize_benchmark_data(
        resolved,
        pbs_spec=ctx.plan.pbs_spec,
        env=ctx.env,
    )
    benchmark_data_root = _optional_path_from_raw_or_env(resolved, "benchmark_data_root", "AB_BENCHMARK_DATA_ROOT", ctx.env)

    ctx.env["AB_BENCHMARK_CONFIG"] = str(benchmark_path.resolve())
    ctx.env["AB_HARNESS_CONFIG"] = str(harness_path.resolve())
    ctx.env["AB_MODEL_CONFIG"] = str(model_path.resolve())
    ctx.env["AB_SERVING_CONFIG"] = str(serving_path.resolve())
    ctx.env["AB_BENCHMARK_INSTANCE_DATA"] = str(instance_data.resolve())
    ctx.env["AB_INSTANCE_DATA"] = str(instance_data.resolve())
    ctx.env["AB_EXPERIMENT_CONFIG"] = str(resolved.spec.config_path.resolve())

    run_sweep(
        benchmark_path=benchmark_path,
        harness_path=harness_path,
        model_path=model_path,
        serving_path=serving_path,
        benchmark_data_root=benchmark_data_root,
        instance_data_path=instance_data,
        retry_errors=ctx.retry_errors,
        max_workers=ctx.max_workers,
    )


def _build_serving_plan_if_needed(
    resolved: ResolvedExperiment,
    env: Mapping[str, str],
    *,
    log_dir: Path,
    warnings: list[str],
) -> ServingLaunchPlan | None:
    backend = resolved.serving.get("backend")
    deployment = resolved.deployment.get("kind") or resolved.deployment.get("name")
    if backend not in {"vllm", "sglang", "litellm"}:
        return None
    if deployment not in {"client_only", "local_docker"}:
        return None
    try:
        return build_launch_plan(
            model_path=resolved.profile_paths["model"],
            serving_path=resolved.profile_paths["serving"],
            backend=str(backend),
            mode=_local_serving_mode(resolved, str(backend), env),
            env=env,
            log_path=log_dir / f"{backend}.log",
        )
    except LaunchPlanError as exc:
        warnings.append(f"serving launch plan not complete: {exc}")
        return None


def _local_serving_mode(resolved: ResolvedExperiment, backend: str, env: Mapping[str, str]) -> str | None:
    backend_key = f"{backend.upper()}_MODE"
    if backend_key in env:
        return env[backend_key]
    if "AB_SERVING_MODE" in env:
        return env["AB_SERVING_MODE"]
    profile_mode = _serving_mode_from_profile(resolved)
    if profile_mode:
        return profile_mode
    deployment = resolved.deployment.get("kind") or resolved.deployment.get("name")
    if backend == "litellm" and deployment in {"client_only", "local_docker"}:
        return "docker"
    return None


def _serving_mode_from_profile(resolved: ResolvedExperiment) -> str | None:
    value = resolved.serving.get("serving_mode") or resolved.serving.get("servingmode")
    runtime = resolved.serving.get("runtime")
    if not value and isinstance(runtime, dict):
        value = runtime.get("serving_mode") or runtime.get("servingmode")
    return str(value).lower() if value else None


def _default_stage_flags(resolved: ResolvedExperiment, env: Mapping[str, str]) -> dict[str, bool]:
    backend = resolved.serving.get("backend")
    deployment = resolved.deployment.get("kind") or resolved.deployment.get("name")
    local_deployment = deployment in {"client_only", "local_docker"}
    stages = {stage: True for stage in STAGE_ORDER}
    stages["prepare_images"] = False
    stages["start_otel"] = local_deployment and _harness_uses_otel(resolved)
    stages["start_serving"] = backend in {"vllm", "sglang", "litellm"} and local_deployment
    _apply_stage_overrides(stages, resolved, env)
    return stages


def _harness_uses_otel(resolved: ResolvedExperiment) -> bool:
    diagnostics = (resolved.harness.get("openclaw") or {}).get("diagnostics") or {}
    otel = diagnostics.get("otel") or {}
    if isinstance(otel, dict) and otel.get("enabled"):
        return True
    return resolved.harness.get("harness") in {"openclaw", "openhands", "hermes"}


def _apply_stage_overrides(
    stages: dict[str, bool],
    resolved: ResolvedExperiment,
    env: Mapping[str, str],
) -> None:
    deployment_stages = resolved.deployment.get("stages") or {}
    if not isinstance(deployment_stages, dict):
        raise ConfigError("deployment.stages must be a table/object when present")
    experiment_stages = resolved.spec.stages
    for stage in STAGE_ORDER:
        if stage in deployment_stages:
            stages[stage] = bool(deployment_stages[stage])
        if stage in experiment_stages:
            stages[stage] = experiment_stages[stage]
        new_key = f"AB_STAGE_{stage.upper()}"
        if new_key in env:
            stages[stage] = _parse_bool(env[new_key])


def _instance_data_path(resolved: ResolvedExperiment) -> Path:
    return materialize_benchmark_data(resolved)


def _optional_path_from_raw_or_env(
    resolved: ResolvedExperiment,
    raw_key: str,
    env_key: str,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    raw_value = resolved.spec.raw.get(raw_key)
    if isinstance(raw_value, str) and raw_value:
        return _resolve_user_path(resolved.spec.config_path.parent, raw_value)
    source_env = os.environ if env is None else env
    env_value = source_env.get(env_key)
    if env_value:
        return _resolve_user_path(Path.cwd(), env_value)
    return None


def _resolve_user_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config = (base_dir / path).resolve()
    if from_config.exists():
        return from_config
    return (Path.cwd() / path).resolve()


def _set_instance_data_env(env: dict[str, str], path: Path) -> None:
    value = str(path.resolve())
    env["AB_BENCHMARK_INSTANCE_DATA"] = value
    os.environ["AB_BENCHMARK_INSTANCE_DATA"] = value


def _export_serving_agent_env(plan: ServingLaunchPlan, env: dict[str, str]) -> None:
    for key in ("AB_SERVING_BASE_URL", "OPENAI_BASE_URL", "AB_SERVING_BACKEND"):
        if key in plan.env:
            env.setdefault(key, plan.env[key])
            os.environ.setdefault(key, plan.env[key])
    if plan.backend == "litellm":
        for key in ("AB_SERVING_API_KEY", "OPENAI_API_KEY", "HERMES_INFERENCE_PROVIDER"):
            if key in plan.env:
                env.setdefault(key, plan.env[key])
                os.environ.setdefault(key, plan.env[key])


def _path_from_env_or_default(env: Mapping[str, str], key: str, default: str | Path) -> Path:
    value = env.get(key)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser()


def _resolve_execution_run_dir(
    resolved: ResolvedExperiment,
    env: Mapping[str, str],
) -> tuple[str, Path]:
    explicit = _aliased_run_dir(env)
    if explicit is not None:
        return explicit.name, explicit

    stamp = _run_stamp(env)
    return stamp, resolved.spec.output_dir / stamp


def _aliased_run_dir(env: Mapping[str, str]) -> Path | None:
    run_dir = env.get("AB_RUN_DIR")
    experiment_dir = env.get("AB_EXPERIMENT_DIR")
    if run_dir and experiment_dir:
        left = Path(run_dir).expanduser()
        right = Path(experiment_dir).expanduser()
        if left != right:
            raise ConfigError(
                "AB_RUN_DIR and AB_EXPERIMENT_DIR are aliases and must match "
                f"when both are set (got {left} and {right})"
            )
        return left
    value = run_dir or experiment_dir
    return Path(value).expanduser() if value else None


def _run_stamp(env: Mapping[str, str]) -> str:
    explicit = env.get("AB_RUN_STAMP")
    if explicit:
        return explicit
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d_%H-%M-%S")
    job_id = env.get("PBS_JOBID")
    if job_id:
        stamp = f"{stamp}_{_sanitize_run_stamp_suffix(job_id)}"
    return stamp


def _sanitize_run_stamp_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _stage_summary(plan: ExecutionPlan) -> str:
    return ",".join(f"{stage.name}={int(stage.enabled)}" for stage in plan.stages)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"invalid boolean value: {value!r}")


def _note(stage: str, message: str) -> None:
    print(f"[agent-bench:{stage}] {message}", file=sys.stderr)
