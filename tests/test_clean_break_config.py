from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import types
import subprocess
import sys

import pytest

import agent_bench.core.execution as execution

from agent_bench.core.execution import (
    STAGE_ORDER,
    _instance_data_path,
    build_execution_plan,
    resolve_max_workers,
)
from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments import plan_deployment


ROOT = Path(__file__).resolve().parents[1]


def _write_client_only_experiment(tmp_path: Path, *, stage_lines: list[str]) -> Path:
    config = tmp_path / "client_only_stages.toml"
    config.write_text(
        "\n".join(
            [
                'name = "client_only_stages"',
                'output_dir = "runs/client_only_stages"',
                "seeds = [42]",
                *stage_lines,
                "[benchmark]",
                'name = "terminal_bench"',
                f'profile = "{ROOT / "profiles/benchmarks/terminal_bench/kv_store.json"}"',
                "[harness]",
                'name = "openclaw"',
                f'profile = "{ROOT / "profiles/harnesses/openclaw_terminal_bench.yaml"}"',
                "[model]",
                f'profile = "{ROOT / "profiles/models/qwen36_35b_a3b.json"}"',
                "[serving]",
                f'profile = "{ROOT / "profiles/serving/litellm/qwen36_35b_a3b.yaml"}"',
                "[deployment]",
                f'profile = "{ROOT / "profiles/deployments/client_only.toml"}"',
                "[limits]",
                "max_workers = 1",
            ]
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.parametrize(
    ("experiment", "deployment", "backend", "deployment_profile"),
    [
        ("client_only", "client_only", "litellm", "client_only.toml"),
        ("local_vllm", "local_docker", "vllm", "local_docker.toml"),
        ("nscc_vllm", "nscc_pbs", "vllm", "nscc_pbs.toml"),
        ("nscc_terminal_bench", "nscc_pbs", "vllm", "nscc_pbs.toml"),
        ("nscc_litellm", "nscc_pbs", "litellm", "nscc_pbs_cpu.toml"),
        ("ccds_vllm", "ccds_cloud", "vllm", "ccds_cloud.toml"),
        ("roadmapbench.example", "client_only", "existing_endpoint", "client_only.toml"),
    ],
)
def test_example_experiments_resolve(
    experiment: str,
    deployment: str,
    backend: str,
    deployment_profile: str,
) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / f"{experiment}.toml", root=ROOT)

    assert resolved.run_id
    assert resolved.deployment["kind"] == deployment
    assert resolved.serving["backend"] == backend
    assert resolved.profile_paths["model"].is_absolute()
    assert resolved.to_dict()["profiles"]["deployment"].endswith(deployment_profile)


def test_roadmap_example_materializes_environment_root_and_uses_external_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_bench.core.config import runtime_endpoint_for_container
    from scripts.download_roadmapbench_tasks import DEFAULT_REVISION

    monkeypatch.chdir(tmp_path)
    task_id = "fyn-2.5.0-roadmap"
    task_root = tmp_path / "roadmap-tasks"
    task = task_root / task_id
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Implement the example roadmap.\n")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "example/roadmap:latest"\nworkdir = "/app"\n'
    )
    monkeypatch.setenv("ROADMAP_TASKS_ROOT", str(task_root))
    monkeypatch.setenv("AB_SERVING_BASE_URL", "http://model.example:8123/v1")
    monkeypatch.delenv("AB_BENCHMARK_INSTANCE_DATA", raising=False)
    resolved = load_resolved_experiment(
        ROOT / "experiments/roadmapbench.example.toml", root=ROOT
    )

    assert resolved.spec.benchmark_name == "roadmap_bench"
    assert resolved.benchmark["application"] == "roadmap_bench"
    assert resolved.benchmark["instances"] == [task_id]
    assert resolved.benchmark["source_commit"] == DEFAULT_REVISION
    assert resolved.benchmark["repo_mapping"][task_id]["base_commit"] == DEFAULT_REVISION
    assert resolved.harness["harness"] == "codex"
    assert resolved.model["name"] == "Qwen/Qwen3.6-35B-A3B"
    assert resolved.profile_paths["benchmark"] == (
        ROOT / "profiles/benchmarks/roadmap/example.json"
    )
    env = {"AB_REPO_ROOT": str(ROOT), "AB_RUN_DIR": str(tmp_path / "run")}
    plan = build_execution_plan(resolved, env=env)
    assert not plan.stage_enabled("start_serving")
    assert plan.serving_plan is None
    assert runtime_endpoint_for_container(
        "http://127.0.0.1:8000/v1", "vllm-host"
    ) == "http://model.example:8123/v1"

    materialized = execution.materialize_benchmark_data(resolved, env=env)
    rows = json.loads(materialized.read_text())
    assert len(rows) == 1
    assert rows[0]["instance_id"] == task_id
    assert rows[0]["benchmark"] == "roadmap_bench"
    assert rows[0]["benchmark_variant"] == "roadmapbench"
    assert rows[0]["docker_image"] == "example/roadmap:latest"
    assert env["AB_BENCHMARK_TASKS_ROOT"] == str(task_root)


def test_wrapped_configs_are_expanded() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)

    assert resolved.model["config_id"] == "qwen36_35b_a3b_local"
    assert resolved.serving["config_id"] == "vllm-default"
    assert resolved.benchmark["application"] == "swebench"
    assert resolved.harness["harness"] == "openclaw"


def test_run_id_is_stable() -> None:
    first = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    second = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)

    assert first.run_id == second.run_id
    assert len(first.run_id) == 12


def test_max_workers_controls_default_worker_limit() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)

    assert resolved.spec.limits["max_workers"] == 1
    assert resolve_max_workers(resolved, env={}) == 1
    assert resolve_max_workers(resolved, env={"AB_RUN_MAX_WORKERS": "3"}) == 3
    assert resolve_max_workers(resolved, cli_max_workers=4, env={"AB_RUN_MAX_WORKERS": "3"}) == 4


def test_execution_plan_uses_ab_run_stamp_for_run_dir() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    stamp = "2026-05-28_14-52-37"

    plan = build_execution_plan(resolved, env={"AB_REPO_ROOT": str(ROOT), "AB_RUN_STAMP": stamp})

    assert resolved.run_id != stamp
    assert plan.run_id == stamp
    assert plan.run_dir == ROOT / "runs" / "client_only" / stamp
    assert plan.log_dir == plan.run_dir / "logs"


def test_execution_plan_ab_experiment_dir_overrides_stamp(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    experiment_dir = tmp_path / "custom-exp"

    plan = build_execution_plan(
        resolved,
        env={
            "AB_REPO_ROOT": str(ROOT),
            "AB_EXPERIMENT_DIR": str(experiment_dir),
            "AB_RUN_STAMP": "ignored-stamp",
        },
    )

    assert plan.run_id == "custom-exp"
    assert plan.run_dir == experiment_dir
    assert plan.log_dir == experiment_dir / "logs"


def test_execution_plan_ab_run_dir_and_experiment_dir_are_aliases(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    run_dir = tmp_path / "explicit-run"

    run_plan = build_execution_plan(
        resolved,
        env={
            "AB_REPO_ROOT": str(ROOT),
            "AB_RUN_DIR": str(run_dir),
            "AB_RUN_STAMP": "ignored-stamp",
        },
    )
    experiment_plan = build_execution_plan(
        resolved,
        env={
            "AB_REPO_ROOT": str(ROOT),
            "AB_EXPERIMENT_DIR": str(run_dir),
            "AB_RUN_STAMP": "ignored-stamp",
        },
    )
    both_plan = build_execution_plan(
        resolved,
        env={
            "AB_REPO_ROOT": str(ROOT),
            "AB_RUN_DIR": str(run_dir),
            "AB_EXPERIMENT_DIR": str(run_dir),
            "AB_RUN_STAMP": "ignored-stamp",
        },
    )

    for plan in (run_plan, experiment_plan, both_plan):
        assert plan.run_id == "explicit-run"
        assert plan.run_dir == run_dir
        assert plan.log_dir == run_dir / "logs"


def test_execution_plan_rejects_conflicting_run_dir_aliases(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)

    with pytest.raises(ConfigError, match="AB_RUN_DIR and AB_EXPERIMENT_DIR"):
        build_execution_plan(
            resolved,
            env={
                "AB_REPO_ROOT": str(ROOT),
                "AB_RUN_DIR": str(tmp_path / "run"),
                "AB_EXPERIMENT_DIR": str(tmp_path / "experiment"),
            },
        )


@pytest.mark.parametrize(
    ("experiment", "deployment", "start_serving", "prepare_images", "start_otel"),
    [
        ("client_only", "client_only", True, False, True),
        ("local_vllm", "local_docker", True, False, True),
        ("nscc_vllm", "nscc_pbs", True, True, True),
        ("nscc_terminal_bench", "nscc_pbs", True, True, True),
    ],
)
def test_execution_plan_uses_canonical_stage_order(
    experiment: str,
    deployment: str,
    start_serving: bool,
    prepare_images: bool,
    start_otel: bool,
) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / f"{experiment}.toml", root=ROOT)

    plan = build_execution_plan(resolved, env={"AB_REPO_ROOT": str(ROOT), "USER": "alice"})

    assert tuple(stage.name for stage in plan.stages) == STAGE_ORDER
    assert plan.deployment == deployment
    assert plan.stage_enabled("start_serving") is start_serving
    assert plan.stage_enabled("prepare_images") is prepare_images
    assert plan.stage_enabled("start_otel") is start_otel


def test_otel_mode_off_does_not_disable_stage_in_plan() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)

    plan = build_execution_plan(
        resolved,
        env={"AB_REPO_ROOT": str(ROOT), "AB_TELEMETRY_MODE": "off"},
    )

    assert plan.stage_enabled("start_otel") is True


def test_experiment_stage_override_disables_local_otel(tmp_path: Path) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=["[stages]", "start_otel = false"])

    resolved = load_resolved_experiment(config, root=ROOT)
    plan = build_execution_plan(resolved, env={"AB_REPO_ROOT": str(ROOT)})

    assert resolved.spec.stages == {"start_otel": False}
    assert plan.stage_enabled("start_otel") is False


def test_experiment_stage_override_enables_prepare_images(tmp_path: Path) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=["[stages]", "prepare_images = true"])

    resolved = load_resolved_experiment(config, root=ROOT)
    plan = build_execution_plan(resolved, env={"AB_REPO_ROOT": str(ROOT)})

    assert plan.stage_enabled("prepare_images") is True


def test_experiment_stage_override_enables_rebuild_images(tmp_path: Path) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=["[stages]", "rebuild_images = true"])

    resolved = load_resolved_experiment(config, root=ROOT)
    plan = build_execution_plan(resolved, env={"AB_REPO_ROOT": str(ROOT)})

    assert resolved.spec.stages == {"rebuild_images": True}
    assert tuple(stage.name for stage in plan.stages) == STAGE_ORDER
    assert "rebuild_images" not in STAGE_ORDER
    assert plan.stage_enabled("prepare_images") is True


def test_env_stage_override_wins_over_experiment_stage_override(tmp_path: Path) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=["[stages]", "start_otel = false"])

    resolved = load_resolved_experiment(config, root=ROOT)
    plan = build_execution_plan(
        resolved,
        env={"AB_REPO_ROOT": str(ROOT), "AB_STAGE_START_OTEL": "1"},
    )

    assert plan.stage_enabled("start_otel") is True


@pytest.mark.parametrize(
    ("stage_lines", "message"),
    [
        (["stages = []"], "stages must be a table"),
        (["[stages]", "strat_otel = false"], "unknown stage"),
        (["[stages]", 'start_otel = "false"'], "stages.start_otel must be a boolean"),
    ],
)
def test_experiment_stages_reject_invalid_values(
    tmp_path: Path,
    stage_lines: list[str],
    message: str,
) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=stage_lines)

    with pytest.raises(ConfigError, match=message):
        load_resolved_experiment(config, root=ROOT)


def test_prepare_environment_installs_missing_swebench_deps_in_active_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)
    env = {
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
        "CONDA_DEFAULT_ENV": "agent-bench",
        "CONDA_PREFIX": str(tmp_path / "conda/envs/agent-bench"),
    }
    plan = execution.build_execution_plan(resolved, env=env)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    installed: list[str] = []
    targets: list[Path | None] = []

    monkeypatch.setattr(execution, "_has_python_module", lambda module: bool(installed))
    monkeypatch.setattr(execution, "_has_executable", lambda name, env: bool(installed))

    def fake_install(packages, *, log_path, env, install_target):
        installed.extend(packages)
        targets.append(install_target)

    monkeypatch.setattr(execution, "_run_python_dependency_install", fake_install)

    execution._prepare_environment(ctx)

    assert installed == ["datasets", "swebench"]
    assert targets == [None]
    assert Path(env["AB_LOG_DIR"]).is_dir()
    assert env["AB_TELEMETRY_DATA_DIR"] == str(tmp_path / "run/logs/otel")
    assert env["AB_TELEMETRY_TRACE_FILE"] == str(tmp_path / "run/logs/otel/otel-traces.jsonl")
    assert Path(env["AB_TELEMETRY_TRACE_FILE"]).is_file()
    assert env["AB_BENCHMARK_DATA_ROOT"] == str((tmp_path / "run" / "benchmark_data_root").resolve())
    assert Path(env["AB_BENCHMARK_DATA_ROOT"]).is_dir()


def test_local_otel_stage_runs_docker_scripts(tmp_path: Path, monkeypatch) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
    }
    plan = execution.build_execution_plan(resolved, env=env)
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs["env"])))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    with ExitStack() as stack:
        execution._start_otel(ctx, stack)

    assert len(calls) == 2
    assert calls[0][0][-1].endswith("otel_collector_up.sh")
    assert calls[1][0][-1].endswith("otel_collector_down.sh")
    assert calls[0][1]["AB_TELEMETRY_DATA_DIR"] == str(tmp_path / "run/logs/otel")
    assert calls[0][1]["AB_TELEMETRY_TRACE_FILE"] == str(tmp_path / "run/logs/otel/otel-traces.jsonl")
    assert calls[0][1]["AB_TELEMETRY_NAME"].startswith("agent-bench-otel-")


def test_rebuild_images_option_runs_local_harness_builder(tmp_path: Path, monkeypatch) -> None:
    config = _write_client_only_experiment(tmp_path, stage_lines=["[stages]", "rebuild_images = true"])
    resolved = load_resolved_experiment(config, root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
        "AB_AGENT_RUNTIME_IMAGE": "agent-bench/openclaw:test",
    }
    plan = execution.build_execution_plan(resolved, env=env)
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    build_script = tmp_path / "build.sh"
    build_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setitem(execution._LOCAL_HARNESS_IMAGE_BUILDERS, "openclaw", build_script)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs["env"])))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    execution._prepare_images(ctx)

    assert calls[0][0] == ["bash", str(build_script)]
    assert calls[0][1]["AB_AGENT_RUNTIME_IMAGE"] == "agent-bench/openclaw:test"
    assert env["AB_AGENT_RUNTIME_REBUILD"] == "1"
    assert (plan.log_dir / "runtime-openclaw-build.log").exists()


def test_prepare_environment_preserves_explicit_benchmark_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)
    explicit_root = tmp_path / "custom-benchmark-data-root"
    env = {
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
        "AB_BENCHMARK_DATA_ROOT": str(explicit_root),
    }
    plan = execution.build_execution_plan(resolved, env=env)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    monkeypatch.setattr(execution, "_has_python_module", lambda module: True)

    execution._prepare_environment(ctx)

    assert env["AB_BENCHMARK_DATA_ROOT"] == str(explicit_root.resolve())
    assert env["AB_BENCHMARK_DATA_ROOT"] == str(explicit_root.resolve())
    assert explicit_root.is_dir()


def test_prepare_environment_installs_missing_swebench_deps_in_pbs_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    resolved = load_resolved_experiment(ROOT / "experiments" / "nscc_vllm.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
    }
    plan = execution.build_execution_plan(resolved, env=env)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    installed: list[str] = []
    targets: list[Path | None] = []

    monkeypatch.setattr(execution, "_has_python_module", lambda module: bool(installed))
    monkeypatch.setattr(execution, "_has_executable", lambda name, env: bool(installed))

    def fake_install(packages, *, log_path, env, install_target):
        installed.extend(packages)
        targets.append(install_target)

    monkeypatch.setattr(execution, "_run_python_dependency_install", fake_install)

    execution._prepare_environment(ctx)

    assert installed == ["datasets", "swebench"]
    assert targets == [Path(env["AB_RUNTIME_PYLIB"])]
    assert Path(env["AB_RUNTIME_PIP_CACHE"]).is_dir()
    assert env["PYTHONPATH"].split(":")[0] == str(Path(env["AB_RUNTIME_PYLIB"]).resolve())


def test_prepare_environment_installs_missing_swebench_deps_in_local_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AB_RUNTIME_PYLIB", raising=False)
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
    }
    plan = execution.build_execution_plan(resolved, env=env)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    installed: list[str] = []
    targets: list[Path | None] = []

    monkeypatch.setattr(execution, "_has_python_module", lambda module: bool(installed))

    def fake_install(packages, *, log_path, env, install_target):
        installed.extend(packages)
        targets.append(install_target)

    monkeypatch.setattr(execution, "_run_python_dependency_install", fake_install)

    execution._prepare_environment(ctx)

    target = plan.run_dir / "python_deps"
    assert installed == ["datasets", "swebench"]
    assert targets == [target]
    assert env["PYTHONPATH"].split(":")[0] == str(target.resolve())


def test_prepare_environment_installs_litellm_for_native_litellm_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    monkeypatch.setenv("AB_BENCHMARK_DATA_ROOT", "")
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    env = {
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
        "CONDA_DEFAULT_ENV": "agent-bench",
        "CONDA_PREFIX": str(tmp_path / "conda/envs/agent-bench"),
        "AB_SERVING_MODE": "native",
    }
    plan = execution.build_execution_plan(resolved, env=env)
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    installed: list[str] = []

    monkeypatch.setattr(execution, "_has_python_module", lambda module: bool(installed))
    monkeypatch.setattr(execution, "_has_executable", lambda name, env: bool(installed))
    monkeypatch.setattr(
        execution,
        "_run_python_dependency_install",
        lambda packages, **kwargs: installed.extend(packages),
    )

    execution._prepare_environment(ctx)

    assert plan.serving_plan is not None
    assert plan.serving_plan.mode == "native"
    assert installed == ["litellm"]


def test_litellm_execution_plan_modes_follow_deployment_defaults() -> None:
    local = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    local_plan = build_execution_plan(local, env={"AB_REPO_ROOT": str(ROOT)})

    assert local_plan.serving_plan is not None
    assert local_plan.serving_plan.backend == "litellm"
    assert local_plan.serving_plan.mode == "docker"
    assert local_plan.serving_plan.argv[:2] == ("docker", "run")

    pbs = load_resolved_experiment(ROOT / "experiments" / "nscc_litellm.toml", root=ROOT)
    pbs_plan = build_execution_plan(pbs, env={"AB_REPO_ROOT": str(ROOT), "USER": "alice"})

    assert pbs_plan.serving_plan is not None
    assert pbs_plan.serving_plan.backend == "litellm"
    assert pbs_plan.serving_plan.mode == "singularity"


def test_litellm_docker_start_registers_container_cleanup(tmp_path: Path, monkeypatch) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
        "QWEN_API_KEY": "sk-upstream",
    }
    plan = build_execution_plan(resolved, env=env)
    assert plan.serving_plan is not None
    assert plan.serving_plan.mode == "docker"
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    calls: list[tuple[str, object]] = []

    class FakeProc:
        pass

    fake_proc = FakeProc()
    monkeypatch.setattr(execution, "http_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(execution, "prepare_launch_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(execution, "wait_for_http_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(execution, "terminate_process_group", lambda proc: calls.append(("terminate", proc)))
    monkeypatch.setattr(execution, "_remove_docker_container", lambda name: calls.append(("rm", name)))

    stack = ExitStack()
    execution._start_serving(ctx, stack)
    stack.close()

    assert calls == [
        ("terminate", fake_proc),
        ("rm", plan.serving_plan.metadata["container_name"]),
    ]


def test_detached_docker_serving_follows_container_logs_after_docker_cli_exits(tmp_path: Path, monkeypatch) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
    }
    plan = build_execution_plan(resolved, env=env)
    assert plan.serving_plan is not None
    assert plan.serving_plan.mode == "docker"
    assert "-d" in plan.serving_plan.argv
    ctx = execution.ExecutionContext(
        resolved=resolved,
        plan=plan,
        retry_errors=True,
        max_workers=1,
        env=env,
    )
    calls: list[tuple[str, object]] = []
    popen_argvs: list[tuple[str, ...]] = []

    class FakeProc:
        def __init__(self, label: str):
            self.label = label

        def wait(self, timeout=None):
            calls.append((f"{self.label}_wait", timeout))
            return 0

    docker_run_proc = FakeProc("docker_run")
    docker_logs_proc = FakeProc("docker_logs")
    popen_results = [docker_run_proc, docker_logs_proc]

    def fake_popen(argv, *args, **kwargs):
        popen_argvs.append(tuple(argv))
        return popen_results.pop(0)

    monkeypatch.setattr(execution, "http_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(execution, "prepare_launch_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(execution.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        execution,
        "wait_for_http_ready",
        lambda *args, **kwargs: calls.append(("ready_proc", kwargs["proc"])),
    )
    monkeypatch.setattr(execution, "terminate_process_group", lambda proc: calls.append(("terminate", proc)))
    monkeypatch.setattr(execution, "_remove_docker_container", lambda name: calls.append(("rm", name)))

    stack = ExitStack()
    execution._start_serving(ctx, stack)
    stack.close()

    container_name = plan.serving_plan.metadata["container_name"]
    assert ("docker_run_wait", 30) in calls
    assert popen_argvs[0] == plan.serving_plan.argv
    assert popen_argvs[1] == ("docker", "logs", "-f", container_name)
    assert ("ready_proc", docker_logs_proc) in calls
    assert ("terminate", docker_run_proc) not in calls
    assert ("rm", container_name) in calls
    assert ("terminate", docker_logs_proc) in calls


def test_swebench_instance_data_materializes_from_benchmark_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AB_BENCHMARK_INSTANCE_DATA", raising=False)

    def fake_load_dataset(dataset: str, *, split: str):
        assert dataset == "princeton-nlp/SWE-bench_Verified"
        assert split == "test"
        return [
            {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "problem_statement": "Fix astropy",
            },
            {"instance_id": "other__repo-1", "repo": "other/repo"},
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)

    path = _instance_data_path(resolved)

    assert path == (tmp_path / "data/swebench/phase0_sanity_instances.json").resolve()
    assert json.loads(path.read_text()) == [
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "problem_statement": "Fix astropy",
        }
    ]


def test_swebench_instance_data_top_level_override_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AB_BENCHMARK_INSTANCE_DATA", raising=False)
    instance_data = tmp_path / "instances.json"
    instance_data.write_text("[]", encoding="utf-8")
    config = tmp_path / "override.toml"
    config.write_text(
        "\n".join(
            [
                'name = "override"',
                'output_dir = "runs/override"',
                f'benchmark_instance_data = "{instance_data.name}"',
                "seeds = [42]",
                "[benchmark]",
                'name = "swebench"',
                f'profile = "{ROOT / "profiles/benchmarks/swebench/phase0.json"}"',
                "[harness]",
                'name = "openclaw"',
                f'profile = "{ROOT / "profiles/harnesses/openclaw_swebench.yaml"}"',
                "[model]",
                f'profile = "{ROOT / "profiles/models/gemini_3_5_flash.json"}"',
                "[serving]",
                f'profile = "{ROOT / "profiles/serving/litellm/gemini_flash_3_5.yaml"}"',
                "[deployment]",
                f'profile = "{ROOT / "profiles/deployments/client_only.toml"}"',
            ]
        ),
        encoding="utf-8",
    )
    resolved = load_resolved_experiment(config, root=ROOT)

    assert _instance_data_path(resolved) == instance_data.resolve()


def test_swebench_instance_data_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    instance_data = tmp_path / "env_instances.json"
    instance_data.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AB_BENCHMARK_INSTANCE_DATA", str(instance_data))
    resolved = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)

    assert _instance_data_path(resolved) == instance_data.resolve()


def test_client_only_rejects_local_serving(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        "\n".join(
            [
                'name = "bad"',
                'output_dir = "runs/bad"',
                "seeds = [42]",
                "[benchmark]",
                'name = "swebench"',
                f'profile = "{ROOT / "profiles/benchmarks/swebench/phase0.json"}"',
                "[harness]",
                'name = "openclaw"',
                f'profile = "{ROOT / "profiles/harnesses/openclaw_swebench.yaml"}"',
                "[model]",
                f'profile = "{ROOT / "profiles/models/qwen35_9b.json"}"',
                "[serving]",
                f'profile = "{ROOT / "profiles/serving/vllm/default.yaml"}"',
                "[deployment]",
                f'profile = "{ROOT / "profiles/deployments/client_only.toml"}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="client_only deployment requires"):
        load_resolved_experiment(config, root=ROOT)


def test_deployment_plans_render_static_scripts() -> None:
    nscc = load_resolved_experiment(ROOT / "experiments" / "nscc_vllm.toml", root=ROOT)
    nscc_plan = plan_deployment(nscc)

    assert nscc_plan.deployment == "nscc_pbs"
    assert nscc_plan.job_script is not None
    assert "#PBS -N agent-bench-nscc_vllm" in nscc_plan.job_script
    assert "python3 run.py --config experiments/nscc_vllm.toml" in nscc_plan.job_script
    assert "--dry-run" not in nscc_plan.job_script

    ccds = load_resolved_experiment(ROOT / "experiments" / "ccds_vllm.toml", root=ROOT)
    ccds_plan = plan_deployment(ccds)

    assert ccds_plan.deployment == "ccds_cloud"
    assert ccds_plan.job_script is not None
    assert "AB_DEPLOYMENT_PROJECT=agent-bench" in ccds_plan.job_script


def test_validate_and_run_entrypoints() -> None:
    validate = subprocess.run(
        [sys.executable, str(ROOT / "validate.py"), "--config", str(ROOT / "experiments/client_only.toml")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert validate.stdout.startswith("valid: client_only")

    dry_run = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--config", str(ROOT / "experiments/client_only.toml"), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "deployment: client_only" in dry_run.stdout
    assert "stages:" in dry_run.stdout
    assert "prepare_environment: enabled" in dry_run.stdout


def test_run_entrypoint_cli_env_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli-run"
    env = dict(os.environ)
    env["AB_RUN_DIR"] = str(tmp_path / "stale-run")
    env["AB_EXPERIMENT_DIR"] = str(tmp_path / "stale-experiment")
    env["VLLM_MODE"] = "native"

    dry_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--config",
            str(ROOT / "experiments/local_vllm.toml"),
            "--ab-run-dir",
            str(run_dir),
            "--ab-serving-mode",
            "docker",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    assert f"run_dir: {run_dir}" in dry_run.stdout
    assert "serving: vllm/docker" in dry_run.stdout
