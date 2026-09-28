from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from agent_bench.core.profile_io import load_profile
from agent_bench.core.spec import ConfigError, ResolvedExperiment
from agent_bench.deployments.pbs.job_spec import PbsJobSpec


@dataclass(frozen=True)
class StageCommand:
    label: str
    argv: tuple[str, ...]
    log_path: Path
    env: dict[str, str] = field(default_factory=dict)
    required: bool = True


def prepare_environment(resolved: ResolvedExperiment, spec: PbsJobSpec, env: dict[str, str]) -> None:
    runtime_env = runtime_environment(resolved, spec, env)
    for key, value in runtime_env.items():
        env.setdefault(key, value)
        os.environ.setdefault(key, value)

    spec.run_dir.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    Path(env["AB_SCORING_IMAGE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["SINGULARITY_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_RUNTIME_PYLIB"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_RUNTIME_PIP_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_HARNESS_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_DEPLOYMENT_TASK_IMAGE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_TELEMETRY_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["AB_HARNESS_PLUGIN_CACHE"]).mkdir(parents=True, exist_ok=True)


def runtime_environment(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> dict[str, str]:
    repo_root = str(Path(env.get("AB_REPO_ROOT") or Path.cwd()).resolve())
    user = env.get("USER") or os.environ.get("USER") or "user"
    scratch = env.get("SCRATCH") or f"/scratch/users/ntu/{user}"
    scoring_image_dir = env.get("AB_SCORING_IMAGE_DIR") or f"{scratch}/swebench"
    singularity_cache = env.get("SINGULARITY_CACHEDIR") or env.get("SINGULARITY_CACHE") or f"{scratch}/.singularity"
    harness = _harness_choice(resolved)

    roadmapbench = (
        resolved.spec.benchmark_name == "roadmap_bench"
        or (
            resolved.spec.benchmark_name == "terminal_bench"
            and resolved.benchmark.get("source_dataset") == "UnipatAI/RoadmapBench"
        )
    )
    runtime_defaults = _agent_runtime_defaults(
        harness, repo_root, scratch, roadmapbench=roadmapbench
    )

    out = {
        "AB_REPO_ROOT": repo_root,
        "SCRATCH": scratch,
        "SINGULARITY_CACHE": singularity_cache,
        "SINGULARITY_CACHEDIR": singularity_cache,
        "AB_EXPERIMENT_DIR": env.get("AB_EXPERIMENT_DIR") or str(spec.run_dir),
        "AB_RUN_DIR": env.get("AB_RUN_DIR") or str(spec.run_dir),
        "AB_LOG_DIR": env.get("AB_LOG_DIR") or str(spec.log_dir),
        "AB_RUNTIME_PYLIB": env.get("AB_RUNTIME_PYLIB") or f"{scratch}/pylib",
        "AB_RUNTIME_PIP_CACHE": env.get("AB_RUNTIME_PIP_CACHE") or f"{scratch}/.pip-cache",
        "AB_SCORING_IMAGE_DIR": scoring_image_dir,
        "AB_BENCHMARK_DATA_ROOT": env.get("AB_BENCHMARK_DATA_ROOT") or f"{scratch}/pylib",
        "AB_DEPLOYMENT_CONTAINER_BACKEND": env.get("AB_DEPLOYMENT_CONTAINER_BACKEND")
        or ("singularity" if spec.container_runtime == "singularity" else "docker"),
        "AB_HARNESS_WORKSPACE_BASE": env.get("AB_HARNESS_WORKSPACE_BASE") or f"{scratch}/agent-bench/workspaces",
        "AB_DEPLOYMENT_TASK_IMAGE_DIR": env.get("AB_DEPLOYMENT_TASK_IMAGE_DIR") or f"{scratch}/task-images",
        "AB_TELEMETRY_DATA_DIR": env.get("AB_TELEMETRY_DATA_DIR") or str(spec.log_dir / "otel"),
        "AB_HARNESS_PLUGIN_CACHE": env.get("AB_HARNESS_PLUGIN_CACHE") or f"{scratch}/agent-bench/plugin-cache",
        "AB_DEPLOYMENT_IMAGE_PULL_PARALLEL": env.get("AB_DEPLOYMENT_IMAGE_PULL_PARALLEL", "1"),
        "AB_TELEMETRY_SIF": env.get("AB_TELEMETRY_SIF") or f"{scratch}/sif_images/otelcol-contrib.sif",
        "AB_TELEMETRY_DOCKER": env.get("AB_TELEMETRY_DOCKER", "docker://otel/opentelemetry-collector-contrib:latest"),
    }
    out.update(runtime_defaults)
    if resolved.spec.benchmark_name in {"terminal_bench", "roadmap_bench"}:
        configured_root = resolved.benchmark.get("tasks_root")
        if configured_root:
            tasks_root = Path(
                os.path.expandvars(os.path.expanduser(str(configured_root)))
            ).resolve()
            if not tasks_root.is_dir():
                raise ConfigError(
                    f"Terminal-Bench tasks_root does not exist: {tasks_root}"
                )
            out["AB_BENCHMARK_TASKS_ROOT"] = str(tasks_root)
    for key in tuple(runtime_defaults):
        if key in env and env[key]:
            out[key] = env[key]
    out["AB_TELEMETRY_TRACE_FILE"] = env.get("AB_TELEMETRY_TRACE_FILE") or f"{out['AB_TELEMETRY_DATA_DIR']}/otel-traces.jsonl"
    out["AB_AGENT_RUNTIME_IMAGE"] = env.get("AB_AGENT_RUNTIME_IMAGE") or out["AB_AGENT_RUNTIME_SIF"]
    return out


def _agent_runtime_defaults(
    harness: str, repo_root: str, scratch: str, *, roadmapbench: bool = False
) -> dict[str, str]:
    if harness == "codex":
        version = _version_from_file(
            Path(repo_root) / "agent_bench/harnesses/codex/VERSION",
            "codex_version", "0.157.1",
        )
        runtime_sif = f"{scratch}/sif_images/codex_{version}.sif"
        return {
            "AB_AGENT_RUNTIME_VERSION": version,
            "AB_AGENT_RUNTIME_SIF": runtime_sif,
            "AB_AGENT_RUNTIME_DOCKER": f"agent-bench/codex:{version}",
            "AB_AGENT_RUNTIME_BASE_SIF": runtime_sif,
            "AB_AGENT_RUNTIME_BASE_DOCKER": "",
            "AB_AGENT_RUNTIME_REBUILD": "0",
            "AB_AGENT_RUNTIME_OVERLAY": "",
            "AB_AGENT_RUNTIME_INSIDE_PKG": "/opt/codex",
        }
    if harness == "claude_code":
        version_file = Path(repo_root) / "agent_bench/harnesses/claude_code/VERSION"
        claude_version = _version_from_file(
            version_file, "claude_code_version", "2.1.278"
        )
        litellm_version = _version_from_file(
            version_file, "litellm_version", "1.102.0"
        )
        runtime_sif = (
            f"{scratch}/sif_images/claude_code_{claude_version}_"
            f"litellm_{litellm_version}.sif"
        )
        return {
            "AB_AGENT_RUNTIME_VERSION": claude_version,
            "AB_AGENT_RUNTIME_SIF": runtime_sif,
            "AB_AGENT_RUNTIME_DOCKER": f"agent-bench/claude-code:{claude_version}",
            "AB_AGENT_RUNTIME_BASE_SIF": runtime_sif,
            "AB_AGENT_RUNTIME_BASE_DOCKER": "",
            "AB_AGENT_RUNTIME_REBUILD": "0",
            "AB_AGENT_RUNTIME_OVERLAY": "",
            "AB_AGENT_RUNTIME_INSIDE_PKG": "/opt/claude-code",
        }
    if harness == "openhands":
        version = _version_from_file(
            Path(repo_root) / "agent_bench/harnesses/openhands/VERSION",
            "upstream_version",
            "1.21.0",
        )
        return {
            "AB_AGENT_RUNTIME_VERSION": version,
            "AB_AGENT_RUNTIME_SIF": f"{scratch}/sif_images/openhands_runtime.sif",
            "AB_AGENT_RUNTIME_DOCKER": "",
            "AB_AGENT_RUNTIME_BASE_SIF": f"{scratch}/sif_images/openhands_python3.12.sif",
            "AB_AGENT_RUNTIME_BASE_DOCKER": "docker://python:3.12",
            "AB_AGENT_RUNTIME_REBUILD": "0",
            "AB_AGENT_RUNTIME_OVERLAY": "",
            "AB_AGENT_RUNTIME_INSIDE_PKG": "/app",
        }
    if harness == "hermes":
        version = _version_from_file(
            Path(repo_root) / "agent_bench/harnesses/hermes/patch/VERSION",
            "upstream_version",
            "v2026.5.16",
        )
        if roadmapbench:
            runtime_sif = f"{scratch}/sif_images/hermes_roadmap_{version}.sif"
            return {
                "AB_AGENT_RUNTIME_VERSION": version,
                "AB_AGENT_RUNTIME_SIF": runtime_sif,
                "AB_AGENT_RUNTIME_DOCKER": f"agent-bench/hermes-roadmap:{version}",
                "AB_AGENT_RUNTIME_BASE_SIF": runtime_sif,
                "AB_AGENT_RUNTIME_BASE_DOCKER": "",
                "AB_AGENT_RUNTIME_REBUILD": "0",
                "AB_AGENT_RUNTIME_OVERLAY": "",
                "AB_AGENT_RUNTIME_INSIDE_PKG": "/opt/hermes",
            }
        return {
            "AB_AGENT_RUNTIME_VERSION": version,
            "AB_AGENT_RUNTIME_SIF": f"{scratch}/sif_images/hermes_runtime.sif",
            "AB_AGENT_RUNTIME_DOCKER": "",
            "AB_AGENT_RUNTIME_BASE_SIF": f"{scratch}/sif_images/hermes_{version}.sif",
            "AB_AGENT_RUNTIME_BASE_DOCKER": f"docker://nousresearch/hermes-agent:{version}",
            "AB_AGENT_RUNTIME_REBUILD": "0",
            "AB_AGENT_RUNTIME_OVERLAY": "",
            "AB_AGENT_RUNTIME_INSIDE_PKG": "/app",
        }

    version = _version_from_file(
        Path(repo_root) / "agent_bench/harnesses/openclaw/patch/VERSION",
        "upstream_version",
        "2026.4.22",
    )
    return {
        "AB_AGENT_RUNTIME_VERSION": version,
        "AB_AGENT_RUNTIME_BASE_SIF": f"{scratch}/sif_images/openclaw_{version}.sif",
        "AB_AGENT_RUNTIME_SIF": f"{scratch}/sif_images/openclaw_{version}_patched.sif",
        "AB_AGENT_RUNTIME_DOCKER": f"docker://ghcr.io/openclaw/openclaw:{version}",
        "AB_AGENT_RUNTIME_REBUILD": "0",
        "AB_AGENT_RUNTIME_OVERLAY": "",
        "AB_AGENT_RUNTIME_INSIDE_PKG": "/app",
        "AB_AGENT_RUNTIME_BASE_DOCKER": f"docker://ghcr.io/openclaw/openclaw:{version}",
    }


def image_preparation_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> list[StageCommand]:
    commands: list[StageCommand] = []
    runtime_env = runtime_environment(resolved, spec, env)
    repo_root = Path(runtime_env["AB_REPO_ROOT"])

    commands.extend(_serving_image_commands(resolved, spec, runtime_env))
    commands.extend(_benchmark_image_commands(resolved, spec, runtime_env, repo_root))
    commands.extend(_runtime_image_commands(resolved, spec, runtime_env, repo_root))
    return commands


def run_stage_commands(commands: list[StageCommand], *, base_env: Mapping[str, str]) -> None:
    for command in commands:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(base_env)
        env.update(command.env)
        with command.log_path.open("ab", buffering=0) as log:
            proc = subprocess.run(
                list(command.argv),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if proc.returncode and command.required:
            raise ConfigError(
                f"{command.label} failed with exit code {proc.returncode}; see {command.log_path}"
            )


def prepare_openclaw_plugin_cache(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> StageCommand | None:
    if _harness_choice(resolved) != "openclaw":
        return None
    runtime_env = runtime_environment(resolved, spec, env)
    cache = Path(runtime_env["AB_HARNESS_PLUGIN_CACHE"])
    if _plugin_cache_warm(cache):
        return None

    state_template = spec.run_dir / "openclaw_state_template"
    _write_openclaw_state_template(resolved, state_template, env)
    argv = [
        "bash",
        str(Path(runtime_env["AB_REPO_ROOT"]) / "agent_bench/deployments/pbs/warm_oc_plugins.sh"),
        "--sif",
        runtime_env["AB_AGENT_RUNTIME_SIF"],
        "--cache",
        runtime_env["AB_HARNESS_PLUGIN_CACHE"],
        "--state-template",
        str(state_template),
    ]
    if runtime_env.get("AB_AGENT_RUNTIME_OVERLAY"):
        argv[4:4] = ["--overlay", runtime_env["AB_AGENT_RUNTIME_OVERLAY"]]
    return StageCommand(
        label="warm OpenClaw plugin cache",
        argv=tuple(argv),
        log_path=spec.log_dir / "openclaw-plugin-cache-warm.log",
        env=runtime_env,
        required=True,
    )


def otel_command(spec: PbsJobSpec, env: Mapping[str, str], mode: str) -> StageCommand:
    runtime_env = runtime_environment_for_spec(spec, env)
    return StageCommand(
        label=f"otel collector {mode}",
        argv=(
            "bash",
            str(Path(runtime_env["AB_REPO_ROOT"]) / "agent_bench/deployments/pbs/otel_collector.sh"),
            mode,
        ),
        log_path=spec.log_dir / f"otel-{mode}.log",
        env=runtime_env,
        required=False,
    )


def runtime_environment_for_spec(spec: PbsJobSpec, env: Mapping[str, str]) -> dict[str, str]:
    fake = _FakeResolved()
    return runtime_environment(fake, spec, env)


def _serving_image_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> list[StageCommand]:
    if not spec.serving_plan or spec.serving_plan.mode != "singularity":
        return []
    sif = _serving_sif_from_plan(spec.serving_plan.argv)
    image = resolved.serving.get("image")
    if not sif or not image or Path(sif).exists():
        return []
    return [
        StageCommand(
            label=f"pull {spec.serving_plan.backend} serving SIF",
            argv=(
                "singularity",
                "pull",
                "--dir",
                str(Path(sif).parent),
                "--name",
                Path(sif).name,
                str(image),
            ),
            log_path=spec.log_dir / f"{spec.serving_plan.backend}-pull.log",
            env=dict(env),
            required=True,
        )
    ]


def _benchmark_image_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
    repo_root: Path,
) -> list[StageCommand]:
    if resolved.spec.benchmark_name in {"terminal_bench", "roadmap_bench"}:
        return _terminal_bench_image_commands(resolved, spec, env)
    if resolved.spec.benchmark_name != "swebench":
        return []
    instances = [str(item) for item in (resolved.benchmark.get("instances") or [])]
    if not instances:
        return []
    return [
        StageCommand(
            label="pull SWE-bench task SIFs",
            argv=(
                "bash",
                str(repo_root / "agent_bench/deployments/pbs/pull_images.sh"),
                "--dir",
                env["AB_SCORING_IMAGE_DIR"],
                "--parallel",
                env["AB_DEPLOYMENT_IMAGE_PULL_PARALLEL"],
                "--instance-ids",
                *instances,
            ),
            log_path=spec.log_dir / "swebench-image-pull.log",
            env=dict(env),
            required=False,
        )
    ]


def _terminal_bench_image_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> list[StageCommand]:
    argv = [
        "python3",
        "-m",
        "agent_bench.deployments.pbs.terminal_bench_images",
        "--sif-dir",
        env["AB_DEPLOYMENT_TASK_IMAGE_DIR"],
    ]
    for task_id in [str(item) for item in (resolved.benchmark.get("instances") or [])]:
        argv.extend(["--task-id", task_id])
    return [
        StageCommand(
            label="pull Terminal-Bench task SIFs",
            argv=tuple(argv),
            log_path=spec.log_dir / "terminal-bench-image-pull.log",
            env=dict(env),
            required=True,
        )
    ]


def _runtime_image_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
    repo_root: Path,
) -> list[StageCommand]:
    if spec.container_runtime != "singularity":
        return []
    harness = _harness_choice(resolved)
    if harness == "openclaw":
        return _openclaw_runtime_commands(spec, env, repo_root)
    if harness == "openhands":
        return _openhands_runtime_commands(spec, env, repo_root)
    if harness == "hermes":
        roadmapbench = (
            resolved.spec.benchmark_name == "roadmap_bench"
            or (
                resolved.spec.benchmark_name == "terminal_bench"
                and resolved.benchmark.get("source_dataset") == "UnipatAI/RoadmapBench"
            )
        )
        return _hermes_runtime_commands(spec, env, repo_root, roadmapbench=roadmapbench)
    if harness == "claude_code":
        return _claude_code_runtime_commands(spec, env, repo_root)
    if harness == "codex":
        runtime_sif = Path(env["AB_AGENT_RUNTIME_SIF"])
        if not runtime_sif.is_file() or runtime_sif.stat().st_size == 0:
            raise FileNotFoundError(
                f"Codex runtime SIF missing: {runtime_sif}. Build it on local "
                "Ubuntu with scripts/build_codex_sif_local.sh and upload it."
            )
        return [StageCommand(
            label="check uploaded Codex CLI runtime",
            argv=("singularity", "exec", str(runtime_sif), "/opt/codex/bin/codex", "--version"),
            log_path=spec.log_dir / "runtime-codex-preflight.log",
            env=dict(env), required=True,
        )]
    return []


def _claude_code_runtime_commands(
    spec: PbsJobSpec,
    env: Mapping[str, str],
    repo_root: Path,
) -> list[StageCommand]:
    runtime_sif = Path(env["AB_AGENT_RUNTIME_SIF"])
    if runtime_sif.is_file() and runtime_sif.stat().st_size > 0:
        return []
    message = (
        f"Claude Code runtime SIF missing: {runtime_sif}. "
        "Build it locally with "
        f"{repo_root}/agent_bench/harnesses/claude_code/build_sif.sh "
        "and upload it before submitting this benchmark."
    )
    return [
        StageCommand(
            label="check uploaded Claude Code runtime SIF",
            argv=(
                "bash",
                "-lc",
                f"printf '%s\\n' {json.dumps(message)} >&2; exit 2",
            ),
            log_path=spec.log_dir / "runtime-claude-code-preflight.log",
            env=dict(env),
            required=True,
        )
    ]


def _openclaw_runtime_commands(spec: PbsJobSpec, env: Mapping[str, str], repo_root: Path) -> list[StageCommand]:
    if str(env.get("AB_AGENT_RUNTIME_VERSION") or "").lstrip("v") == "2026.9.2":
        candidate = Path(env["AB_AGENT_RUNTIME_SIF"])
        if env.get("AB_AGENT_RUNTIME_REBUILD") == "1":
            raise ConfigError(
                "OpenClaw v2026.9.2 requires a prebuilt audited runtime; "
                "prepare it with agent_bench/deployments/pbs/build_oc_v2026_9_2_sif.sh."
            )
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ConfigError(f"missing prebuilt OpenClaw v2026.9.2 runtime SIF: {candidate}")
        return []
    commands = _pull_if_missing(
        label="pull OpenClaw base runtime SIF",
        sif=env["AB_AGENT_RUNTIME_BASE_SIF"],
        image=env["AB_AGENT_RUNTIME_DOCKER"],
        log_path=spec.log_dir / "runtime-openclaw-base-pull.log",
        env=env,
    )
    if env.get("AB_AGENT_RUNTIME_REBUILD") == "1" or not Path(env["AB_AGENT_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build patched OpenClaw runtime SIF",
                argv=(
                    "bash",
                    str(repo_root / "agent_bench/deployments/pbs/build_oc_sif.sh"),
                    "--base",
                    env["AB_AGENT_RUNTIME_BASE_SIF"],
                    "--out",
                    env["AB_AGENT_RUNTIME_SIF"],
                    "--repo-root",
                    str(repo_root),
                    "--inside-pkg",
                    env["AB_AGENT_RUNTIME_INSIDE_PKG"],
                ),
                log_path=spec.log_dir / "runtime-openclaw-build.log",
                env=dict(env),
                required=True,
            )
        )
    return commands


def _openhands_runtime_commands(spec: PbsJobSpec, env: Mapping[str, str], repo_root: Path) -> list[StageCommand]:
    if env.get("AB_AGENT_RUNTIME_DOCKER"):
        return _pull_if_missing(
            label="pull OpenHands runtime SIF",
            sif=env["AB_AGENT_RUNTIME_SIF"],
            image=env["AB_AGENT_RUNTIME_DOCKER"],
            log_path=spec.log_dir / "runtime-openhands-pull.log",
            env=env,
        )
    commands = _pull_if_missing(
        label="pull OpenHands base SIF",
        sif=env["AB_AGENT_RUNTIME_BASE_SIF"],
        image=env["AB_AGENT_RUNTIME_BASE_DOCKER"],
        log_path=spec.log_dir / "runtime-openhands-base-pull.log",
        env=env,
    )
    if env.get("AB_AGENT_RUNTIME_REBUILD") == "1" or not Path(env["AB_AGENT_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build OpenHands runtime SIF",
                argv=(
                    "bash",
                    str(repo_root / "agent_bench/deployments/pbs/build_oh_sif.sh"),
                    "--base",
                    env["AB_AGENT_RUNTIME_BASE_SIF"],
                    "--out",
                    env["AB_AGENT_RUNTIME_SIF"],
                    "--repo-root",
                    str(repo_root),
                    "--version",
                    env["AB_AGENT_RUNTIME_VERSION"],
                ),
                log_path=spec.log_dir / "runtime-openhands-build.log",
                env=dict(env),
                required=True,
            )
        )
    return commands


def _hermes_runtime_commands(
    spec: PbsJobSpec, env: Mapping[str, str], repo_root: Path,
    *, roadmapbench: bool = False,
) -> list[StageCommand]:
    if roadmapbench:
        runtime_sif = Path(env["AB_AGENT_RUNTIME_SIF"])
        if runtime_sif.is_file() and runtime_sif.stat().st_size > 0:
            return []
        message = (
            f"Roadmap Hermes runtime SIF missing: {runtime_sif}. "
            "Run scripts/"
            "build_roadmap_hermes_sif_local.sh and upload the SIF "
            "before submitting this experiment."
        )
        return [StageCommand(
            label="check uploaded Roadmap Hermes SIF",
            argv=("bash", "-c", 'printf "%s\\n" "$1" >&2; exit 2', "bash", message),
            log_path=spec.log_dir / "runtime-roadmap-hermes-preflight.log",
            env=dict(env), required=True,
        )]
    if env.get("AB_AGENT_RUNTIME_DOCKER"):
        return _pull_if_missing(
            label="pull Hermes runtime SIF",
            sif=env["AB_AGENT_RUNTIME_SIF"],
            image=env["AB_AGENT_RUNTIME_DOCKER"],
            log_path=spec.log_dir / "runtime-hermes-pull.log",
            env=env,
        )
    commands = _pull_if_missing(
        label="pull Hermes base SIF",
        sif=env["AB_AGENT_RUNTIME_BASE_SIF"],
        image=env["AB_AGENT_RUNTIME_BASE_DOCKER"],
        log_path=spec.log_dir / "runtime-hermes-base-pull.log",
        env=env,
    )
    if env.get("AB_AGENT_RUNTIME_REBUILD") == "1" or not Path(env["AB_AGENT_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build patched Hermes runtime SIF",
                argv=(
                    "bash",
                    str(repo_root / "agent_bench/deployments/pbs/build_hm_sif.sh"),
                    "--base",
                    env["AB_AGENT_RUNTIME_BASE_SIF"],
                    "--out",
                    env["AB_AGENT_RUNTIME_SIF"],
                    "--repo-root",
                    str(repo_root),
                    "--version",
                    env["AB_AGENT_RUNTIME_VERSION"],
                ),
                log_path=spec.log_dir / "runtime-hermes-build.log",
                env=dict(env),
                required=True,
            )
        )
    return commands


def _pull_if_missing(
    *,
    label: str,
    sif: str,
    image: str,
    log_path: Path,
    env: Mapping[str, str],
) -> list[StageCommand]:
    if Path(sif).exists():
        return []
    return [
        StageCommand(
            label=label,
            argv=(
                "singularity",
                "pull",
                "--dir",
                str(Path(sif).parent),
                "--name",
                Path(sif).name,
                image,
            ),
            log_path=log_path,
            env=dict(env),
            required=True,
        )
    ]


def _write_openclaw_state_template(
    resolved: ResolvedExperiment,
    state_template: Path,
    env: Mapping[str, str],
) -> None:
    state_template.mkdir(parents=True, exist_ok=True)
    harness_path = resolved.profile_paths.get("harness")
    if not harness_path:
        raise ConfigError("OpenClaw plugin warm requires a harness profile path")
    harness = load_profile(harness_path)
    oc_cfg = dict(harness.get("openclaw") or {})
    if not oc_cfg:
        raise ConfigError(f"harness config {harness_path} has no openclaw block")

    model_name = env.get("AB_MODEL_NAME") or resolved.model.get("name") or "model"
    providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
    agent_model_provider = providers.setdefault("agent_model", {})
    agent_model_provider["api"] = "openai-completions"
    agent_model_provider.setdefault("baseUrl", env.get("AB_SERVING_BASE_URL", "http://127.0.0.1:8000/v1"))
    agent_model_provider.setdefault("apiKey", env.get("AB_SERVING_API_KEY") or "sk-dummy")
    agent_model_provider["models"] = [
        {
            "id": model_name,
            "name": model_name,
            "api": "openai-completions",
            "contextWindow": int(env.get("AB_MODEL_CONTEXT_WINDOW") or resolved.model.get("context_window") or 0),
            "maxTokens": int(env.get("AB_MODEL_MAX_OUTPUT_TOKENS") or resolved.model.get("max_tokens") or (min(32768, int(resolved.model.get("context_window"))) if resolved.model.get("context_window") and int(resolved.model.get("context_window")) > 0 else 32768)),
        }
    ]
    agents_defaults = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
    agents_defaults.setdefault("model", {})["primary"] = f"agent_model/{model_name}"
    (state_template / "openclaw.json").write_text(json.dumps(oc_cfg, indent=2), encoding="utf-8")

    agent_src = Path(env.get("AB_REPO_ROOT") or Path.cwd()) / "agent_bench/benchmarks/swebench/openclaw/agents"
    if agent_src.exists():
        agent_dst = state_template / "agents" / "swebench"
        agent_dst.mkdir(parents=True, exist_ok=True)
        for src in agent_src.glob("*.md"):
            (agent_dst / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _plugin_cache_warm(cache: Path) -> bool:
    return cache.exists() and (
        any(cache.glob("*/package.json")) or any(cache.glob("*/*/package.json"))
    )


def _serving_sif_from_plan(argv: tuple[str, ...]) -> str | None:
    if not argv or argv[0] != "singularity":
        return None
    for index, value in enumerate(argv):
        if value.endswith(".sif"):
            return value
        if index > 0 and argv[index - 1] == "--bind":
            continue
    return None


def _harness_choice(resolved: ResolvedExperiment) -> str:
    return str(
        resolved.harness.get("harness")
        or resolved.harness.get("name")
        or resolved.spec.harness_name
    )


def _version_from_file(path: Path, key: str, default: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            left, _, right = line.partition(":")
            if left.strip() == key and right.strip():
                return right.strip()
    except OSError:
        pass
    return default


class _FakeResolved:
    spec = type(
        "_Spec",
        (),
        {"harness_name": "", "benchmark_name": ""},
    )()
    harness: dict[str, Any] = {}
    benchmark: dict[str, Any] = {}
    model: dict[str, Any] = {}
    profile_paths: dict[str, Path] = {}
