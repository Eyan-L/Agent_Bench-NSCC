from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any, Protocol

from agent_bench.core.spec import ConfigError, ResolvedExperiment


@dataclass(frozen=True)
class DeploymentPlan:
    deployment: str
    summary: str
    steps: tuple[str, ...]
    job_script: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment": self.deployment,
            "summary": self.summary,
            "steps": list(self.steps),
            "job_script": self.job_script,
            "warnings": list(self.warnings),
        }


class DeploymentAdapter(Protocol):
    def plan(self, resolved: ResolvedExperiment) -> DeploymentPlan:
        ...


def plan_deployment(resolved: ResolvedExperiment) -> DeploymentPlan:
    kind = resolved.deployment.get("kind") or resolved.deployment.get("name")
    adapters: dict[str, DeploymentAdapter] = {
        "client_only": ClientOnlyDeployment(),
        "local_docker": LocalDockerDeployment(),
        "nscc_pbs": PbsDeployment(site="NSCC"),
        "ccds_cloud": CloudDeployment(site="CCDS"),
    }
    adapter = adapters.get(kind)
    if adapter is None:
        raise ConfigError(f"unsupported deployment kind: {kind}")
    return adapter.plan(resolved)


class ClientOnlyDeployment:
    def plan(self, resolved: ResolvedExperiment) -> DeploymentPlan:
        backend = resolved.serving.get("backend")
        warnings: list[str] = []
        if backend in {"vllm", "sglang"}:
            warnings.append("client_only does not start local model servers; use litellm or an existing endpoint")
        if backend == "litellm":
            summary = "Run benchmark and harness through a local LiteLLM proxy to a provider API."
            serving_step = "start or attach to the local LiteLLM proxy"
        else:
            summary = "Run benchmark and harness against an already reachable OpenAI-compatible endpoint."
            serving_step = "verify serving endpoint/API proxy is reachable"
        return DeploymentPlan(
            deployment="client_only",
            summary=summary,
            steps=(
                "resolve experiment and profiles",
                serving_step,
                "run benchmark adapter",
                "write manifest, traces, and results under output_dir",
            ),
            warnings=tuple(warnings),
        )


class LocalDockerDeployment:
    def plan(self, resolved: ResolvedExperiment) -> DeploymentPlan:
        backend = resolved.serving.get("backend")
        return DeploymentPlan(
            deployment="local_docker",
            summary=f"Run locally with Docker-managed benchmark/harness containers and {backend} serving.",
            steps=(
                "build or pull harness container images",
                f"start serving adapter: {backend}",
                "wait for /v1/models readiness",
                "run benchmark adapter with local container runtime",
                "stop serving adapter and collect artifacts",
            ),
        )


class PbsDeployment:
    def __init__(self, *, site: str) -> None:
        self.site = site

    def plan(self, resolved: ResolvedExperiment) -> DeploymentPlan:
        job_script = _pbs_script(resolved, site=self.site)
        return DeploymentPlan(
            deployment="nscc_pbs",
            summary=f"Render a short {self.site} PBS job that delegates runtime behavior to Python adapters.",
            steps=(
                "render PBS resource header from deployment profile",
                "stage profiles and experiment manifest",
                "start serving adapter when deployment requires it",
                "run benchmark and harness adapters inside the job",
            ),
            job_script=job_script,
        )


class CloudDeployment:
    def __init__(self, *, site: str) -> None:
        self.site = site

    def plan(self, resolved: ResolvedExperiment) -> DeploymentPlan:
        job_script = _cloud_script(resolved, site=self.site)
        return DeploymentPlan(
            deployment="ccds_cloud",
            summary=f"Render a {self.site} cloud launch script that keeps provider secrets in deployment scope.",
            steps=(
                "resolve cloud deployment profile",
                "prepare environment and secret references",
                "start serving adapter or attach to provider proxy",
                "run benchmark and harness adapters",
            ),
            job_script=job_script,
        )


def _config_arg(path: Path) -> str:
    try:
        value = str(path.relative_to(Path.cwd()))
    except ValueError:
        value = str(path)
    return shlex.quote(value)


def _pbs_script(resolved: ResolvedExperiment, *, site: str) -> str:
    deployment = resolved.deployment
    job_name = deployment.get("job_name") or f"agent-bench-{resolved.spec.name}"
    walltime = deployment.get("walltime", "02:00:00")
    nodes = deployment.get("nodes", 1)
    gpus = deployment.get("gpus", 0)
    queue = deployment.get("queue")
    config_arg = _config_arg(resolved.spec.config_path)
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by Agent Bench deployment renderer for {site}.",
        f"#PBS -N {job_name}",
        f"#PBS -l select={nodes}:ngpus={gpus}",
        f"#PBS -l walltime={walltime}",
    ]
    if queue:
        lines.append(f"#PBS -q {queue}")
    lines.extend(
        [
            "set -euo pipefail",
            'cd "${PBS_O_WORKDIR:-$PWD}"',
            f"python3 run.py --config {config_arg}",
        ]
    )
    return "\n".join(lines) + "\n"


def _cloud_script(resolved: ResolvedExperiment, *, site: str) -> str:
    config_arg = _config_arg(resolved.spec.config_path)
    project = resolved.deployment.get("project", "agent-bench")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"# Generated by Agent Bench deployment renderer for {site}.",
            "set -euo pipefail",
            f"export AGENT_BENCH_PROJECT={shlex.quote(str(project))}",
            f"python3 run.py --config {config_arg}",
            "",
        ]
    )
