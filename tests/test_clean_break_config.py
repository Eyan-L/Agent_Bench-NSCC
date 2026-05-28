from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments import plan_deployment


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("experiment", "deployment", "backend", "deployment_profile"),
    [
        ("client_only", "client_only", "litellm", "client_only.toml"),
        ("local_vllm", "local_docker", "vllm", "local_docker.toml"),
        ("nscc_vllm", "nscc_pbs", "vllm", "nscc_pbs.toml"),
        ("nscc_terminal_bench", "nscc_pbs", "vllm", "nscc_pbs.toml"),
        ("nscc_litellm", "nscc_pbs", "litellm", "nscc_pbs_cpu.toml"),
        ("ccds_vllm", "ccds_cloud", "vllm", "ccds_cloud.toml"),
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


def test_legacy_wrapped_profiles_are_expanded() -> None:
    resolved = load_resolved_experiment(ROOT / "experiments" / "local_vllm.toml", root=ROOT)

    assert resolved.model["config_id"] == "qwen35_9b"
    assert resolved.serving["config_id"] == "vllm-default"
    assert resolved.benchmark["application"] == "swebench"
    assert resolved.harness["harness"] == "openclaw"


def test_run_id_is_stable() -> None:
    first = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)
    second = load_resolved_experiment(ROOT / "experiments" / "client_only.toml", root=ROOT)

    assert first.run_id == second.run_id
    assert len(first.run_id) == 12


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
    assert "python3 -m agent_bench.deployments.pbs.run_job --config experiments/nscc_vllm.toml" in nscc_plan.job_script
    assert "--dry-run" not in nscc_plan.job_script

    ccds = load_resolved_experiment(ROOT / "experiments" / "ccds_vllm.toml", root=ROOT)
    ccds_plan = plan_deployment(ccds)

    assert ccds_plan.deployment == "ccds_cloud"
    assert ccds_plan.job_script is not None
    assert "AGENT_BENCH_PROJECT=agent-bench" in ccds_plan.job_script


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
