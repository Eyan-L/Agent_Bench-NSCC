from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent_bench.core.spec import load_resolved_experiment
from agent_bench.deployments.pbs.job_spec import RUNTIME_STAGES, build_pbs_job_spec, dump_spec_json
from agent_bench.deployments.pbs.stages import (
    image_preparation_commands,
    otel_command,
    prepare_environment,
    prepare_openclaw_plugin_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def test_nscc_pbs_job_spec_embeds_serving_launch_plan() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    spec = build_pbs_job_spec(resolved, env={"KIT_ROOT": str(ROOT), "USER": "alice"})

    assert tuple(spec.stages) == RUNTIME_STAGES
    assert all(spec.stages.values())
    assert spec.resources.gpus == 1
    assert spec.resources.select_value() == "select=1:ngpus=1"
    assert spec.serving_plan is not None
    assert spec.serving_plan.backend == "vllm"
    assert spec.serving_plan.mode == "singularity"
    assert "/scratch/users/ntu/alice/sif_images/nscc_vllm_openai_cu128.sif" in spec.serving_plan.argv
    assert spec.serving_plan.readiness_url == "http://127.0.0.1:8000/v1/models"


def test_pbs_render_job_uses_short_python_worker() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    script = build_pbs_job_spec(resolved, env={"KIT_ROOT": str(ROOT)}).render_script()

    assert "#PBS -N agent-bench-nscc_vllm" in script
    assert "#PBS -l select=1:ngpus=1" in script
    assert 'cd "${PBS_O_WORKDIR:-$PWD}"' in script
    assert "python3 -m agent_bench.deployments.pbs.run_job --config experiments/nscc_vllm.toml" in script
    assert "run_benchmark.pbs" not in script
    assert "STAGE_SWEBENCH" not in script


def test_legacy_monolithic_pbs_entrypoints_are_removed() -> None:
    for name in (
        "run_benchmark.pbs",
        "run_benchmark_tb.pbs",
        "run_benchmark_singularity.pbs",
        "pull_images.pbs",
    ):
        assert not (ROOT / "agent_bench/deployments/pbs" / name).exists()


def test_terminal_bench_pbs_render_job_uses_same_python_worker() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_terminal_bench.toml", root=ROOT)
    spec = build_pbs_job_spec(resolved, env={"KIT_ROOT": str(ROOT), "USER": "alice"})
    script = spec.render_script()

    assert "#PBS -N agent-bench-nscc_terminal_bench" in script
    assert "python3 -m agent_bench.deployments.pbs.run_job --config experiments/nscc_terminal_bench.toml" in script
    assert "run_benchmark_tb.pbs" not in script
    assert spec.metadata["benchmark"] == "terminal_bench"
    assert spec.serving_plan is not None


def test_pbs_stage_env_overrides_are_resolved_once() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    spec = build_pbs_job_spec(
        resolved,
        env={
            "KIT_ROOT": str(ROOT),
            "STAGE_SERVING": "0",
            "AB_STAGE_START_OTEL": "false",
            "PBS_WALLTIME": "01:00:00",
            "PBS_CPUS": "8",
            "PBS_MEMORY": "64gb",
        },
    )

    assert not spec.stages["start_serving"]
    assert not spec.stages["start_otel"]
    assert spec.resources.walltime == "01:00:00"
    assert spec.resources.select_value() == "select=1:ncpus=8:ngpus=1:mem=64gb"


def test_litellm_pbs_spec_does_not_emit_upstream_secret(tmp_path: Path, monkeypatch) -> None:
    deployment = tmp_path / "pbs_cpu.toml"
    deployment.write_text(
        "\n".join(
            [
                "[deployment]",
                'name = "pbs_cpu"',
                'kind = "nscc_pbs"',
                "requires_gpu = false",
                'container_runtime = "singularity"',
                'walltime = "00:30:00"',
                "nodes = 1",
                "gpus = 0",
            ]
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "litellm_pbs.toml"
    experiment.write_text(
        "\n".join(
            [
                'name = "litellm_pbs"',
                'output_dir = "runs/litellm_pbs"',
                "seeds = [42]",
                "[benchmark]",
                'name = "terminal_bench"',
                f'profile = "{ROOT / "profiles/benchmarks/terminal_bench/kv_store.json"}"',
                "[harness]",
                'name = "react"',
                f'profile = "{ROOT / "profiles/harnesses/tb_react_terminal_bench.json"}"',
                "[model]",
                f'profile = "{ROOT / "profiles/models/gemini_3_5_flash.json"}"',
                "[serving]",
                f'profile = "{ROOT / "profiles/serving/litellm_gemini.toml"}"',
                "[deployment]",
                f'profile = "{deployment}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini-secret")

    resolved = load_resolved_experiment(experiment, root=ROOT)
    spec = build_pbs_job_spec(
        resolved,
        env={
            "KIT_ROOT": str(ROOT),
            "GEMINI_API_KEY": "sk-gemini-secret",
            "SERVING_MODE": "native",
        },
    )
    rendered = spec.render_script()
    payload = dump_spec_json(spec)

    assert spec.resources.select_value() == "select=1"
    assert "ngpus" not in rendered
    assert "sk-gemini-secret" not in rendered
    assert "sk-gemini-secret" not in payload
    assert json.loads(payload)["serving_plan"]["readiness_headers"]["Authorization"] == "<redacted>"


def test_pbs_run_job_dry_run_entrypoint() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_bench.deployments.pbs.run_job",
            "--config",
            str(ROOT / "experiments/nscc_vllm.toml"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "pbs_job: agent-bench-nscc_vllm" in proc.stdout
    assert "serving: vllm/singularity" in proc.stdout


def test_pbs_stage3_image_commands_are_planned_without_shell_fragments(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    env = {
        "KIT_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
        "VLLM_SIF": str(tmp_path / "scratch/sif_images/vllm.sif"),
    }
    spec = build_pbs_job_spec(resolved, env=env)

    commands = image_preparation_commands(resolved, spec, env)
    labels = [command.label for command in commands]

    assert "pull vllm serving SIF" in labels
    assert "pull SWE-bench task SIFs" in labels
    assert "pull OpenClaw base runtime SIF" in labels
    assert "build patched OpenClaw runtime SIF" in labels
    assert any("pull_images.sh" in " ".join(command.argv) for command in commands)
    assert any("build_oc_sif.sh" in " ".join(command.argv) for command in commands)


def test_terminal_bench_stage4_image_commands_are_python_owned(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_terminal_bench.toml", root=ROOT)
    env = {
        "KIT_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
        "VLLM_SIF": str(tmp_path / "scratch/sif_images/vllm.sif"),
    }
    spec = build_pbs_job_spec(resolved, env=env)

    commands = image_preparation_commands(resolved, spec, env)
    joined = [" ".join(command.argv) for command in commands]

    assert any(command.label == "pull Terminal-Bench task SIFs" for command in commands)
    assert any("agent_bench.deployments.pbs.terminal_bench_images" in argv for argv in joined)
    assert any("--task-id kv-store-grpc" in argv for argv in joined)


def test_pbs_prepare_environment_exports_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    env = {
        "KIT_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
        "AGENT_BENCH_RUN_DIR": str(tmp_path / "run"),
        "AGENT_BENCH_LOG_DIR": str(tmp_path / "run/logs"),
    }
    spec = build_pbs_job_spec(resolved, env=env)
    monkeypatch.delenv("OC_OTEL_TRACE_FILE", raising=False)

    prepare_environment(resolved, spec, env)

    assert Path(env["SIF_DIR"]).is_dir()
    assert Path(env["OC_OTEL_DATA_DIR"]).is_dir()
    assert env["OC_OTEL_TRACE_FILE"].endswith("otel-traces.jsonl")
    assert env["OC_OC_SIF"] == env["OC_RUNTIME_SIF"]
    assert env["HARNESS_BACKEND"] == "singularity"
    assert env["SWEBENCH_ROOT"] == env["OC_PYLIB"]


def test_pbs_openclaw_plugin_cache_warm_command_writes_state_template(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    env = {
        "KIT_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
        "AGENT_BENCH_RUN_DIR": str(tmp_path / "run"),
        "AGENT_BENCH_LOG_DIR": str(tmp_path / "run/logs"),
        "MODEL_ENDPOINT": "http://127.0.0.1:8000/v1",
        "MODEL_API_KEY": "sk-local",
        "MODEL": "Qwen/Qwen3.5-9B",
    }
    spec = build_pbs_job_spec(resolved, env=env)

    command = prepare_openclaw_plugin_cache(resolved, spec, env)

    assert command is not None
    assert command.label == "warm OpenClaw plugin cache"
    assert "warm_oc_plugins.sh" in " ".join(command.argv)
    state_config = spec.run_dir / "openclaw_state_template/openclaw.json"
    assert state_config.exists()
    rendered = state_config.read_text()
    assert "Qwen/Qwen3.5-9B" in rendered
    assert "sk-local" in rendered


def test_pbs_otel_stage_command_is_best_effort(tmp_path: Path) -> None:
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    env = {
        "KIT_ROOT": str(ROOT),
        "SCRATCH": str(tmp_path / "scratch"),
        "USER": "alice",
    }
    spec = build_pbs_job_spec(resolved, env=env)

    start = otel_command(spec, env, "start")
    stop = otel_command(spec, env, "stop")

    assert start.required is False
    assert stop.required is False
    assert start.argv[-1] == "start"
    assert stop.argv[-1] == "stop"
    assert "otel_collector.sh" in " ".join(start.argv)
