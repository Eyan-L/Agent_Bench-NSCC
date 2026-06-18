from __future__ import annotations

from pathlib import Path

from agent_bench.deployments.containers.limits import (
    configured_task_container_cpus,
    docker_cpus_args,
)


def test_configured_task_container_cpus_reads_experiment_config(tmp_path: Path) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(
        "\n".join(
            [
                'name = "cpu_limit"',
                "[limits]",
                "task_container_cpus = 0.25",
            ]
        ),
        encoding="utf-8",
    )

    assert configured_task_container_cpus({"AB_EXPERIMENT_CONFIG": str(config)}) == 0.25


def test_configured_task_container_cpus_env_override_wins() -> None:
    assert configured_task_container_cpus({"AB_TASK_CONTAINER_CPUS": "0.5"}) == 0.5


def test_docker_cpus_args_only_emits_when_configured() -> None:
    assert docker_cpus_args(None) == []
    assert docker_cpus_args(0.1) == ["--cpus", "0.1"]
