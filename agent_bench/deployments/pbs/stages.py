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
    Path(env["SIF_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["SINGULARITY_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["OC_PYLIB"]).mkdir(parents=True, exist_ok=True)
    Path(env["OC_HOST_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(env["OC_TB_SIF_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["OC_OTEL_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["OC_PLUGIN_CACHE"]).mkdir(parents=True, exist_ok=True)


def runtime_environment(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> dict[str, str]:
    kit_root = str(Path(env.get("KIT_ROOT") or Path.cwd()).resolve())
    user = env.get("USER") or os.environ.get("USER") or "user"
    scratch = env.get("SCRATCH") or f"/scratch/users/ntu/{user}"
    sif_dir = env.get("SIF_DIR") or f"{scratch}/swebench"
    singularity_cache = env.get("SINGULARITY_CACHEDIR") or env.get("SINGULARITY_CACHE") or f"{scratch}/.singularity"

    oc_version = env.get("OC_RUNTIME_VERSION") or _version_from_file(
        Path(kit_root) / "agent_bench/harnesses/openclaw/patch/VERSION",
        "upstream_version",
        "2026.4.22",
    )
    oh_version = env.get("OH_RUNTIME_VERSION") or _version_from_file(
        Path(kit_root) / "agent_bench/harnesses/openhands/VERSION",
        "upstream_version",
        "1.21.0",
    )
    hm_version = env.get("HM_RUNTIME_VERSION") or _version_from_file(
        Path(kit_root) / "agent_bench/harnesses/hermes/patch/VERSION",
        "upstream_version",
        "v2026.5.16",
    )

    out = {
        "KIT_ROOT": kit_root,
        "SCRATCH": scratch,
        "SIF_DIR": sif_dir,
        "SINGULARITY_CACHE": singularity_cache,
        "SINGULARITY_CACHEDIR": singularity_cache,
        "AB_EXPERIMENT_DIR": env.get("AB_EXPERIMENT_DIR") or str(spec.run_dir),
        "AGENT_BENCH_LOG_DIR": env.get("AGENT_BENCH_LOG_DIR") or str(spec.log_dir),
        "OC_LOG_DIR": env.get("OC_LOG_DIR") or str(spec.log_dir),
        "OC_PYLIB": env.get("OC_PYLIB") or f"{scratch}/pylib",
        "OC_SCORING_SIF_DIR": sif_dir,
        "SWEBENCH_ROOT": env.get("SWEBENCH_ROOT") or f"{scratch}/pylib",
        "HARNESS_BACKEND": env.get("HARNESS_BACKEND")
        or ("singularity" if spec.container_runtime == "singularity" else "docker"),
        "OC_HOST_WORKSPACE_BASE": env.get("OC_HOST_WORKSPACE_BASE") or f"{scratch}/oc-eval/workspaces",
        "OC_TB_SIF_DIR": env.get("OC_TB_SIF_DIR") or f"{scratch}/tb-sif",
        "OC_OTEL_DATA_DIR": env.get("OC_OTEL_DATA_DIR") or str(spec.log_dir / "otel"),
        "OC_PLUGIN_CACHE": env.get("OC_PLUGIN_CACHE") or f"{scratch}/oc-eval/plugin-cache",
        "OC_PULL_PARALLEL": env.get("OC_PULL_PARALLEL", "1"),
        "OC_RUNTIME_VERSION": oc_version,
        "OC_RUNTIME_BASE_SIF": env.get("OC_RUNTIME_BASE_SIF") or f"{scratch}/sif_images/openclaw_{oc_version}.sif",
        "OC_RUNTIME_SIF": env.get("OC_RUNTIME_SIF") or f"{scratch}/sif_images/openclaw_{oc_version}_patched.sif",
        "OC_RUNTIME_DOCKER": env.get("OC_RUNTIME_DOCKER") or f"docker://ghcr.io/openclaw/openclaw:{oc_version}",
        "OC_RUNTIME_REBUILD": env.get("OC_RUNTIME_REBUILD", "0"),
        "OC_RUNTIME_OVERLAY": env.get("OC_RUNTIME_OVERLAY", ""),
        "OC_INSIDE_PKG": env.get("OC_INSIDE_PKG", "/app"),
        "OH_RUNTIME_VERSION": oh_version,
        "OH_RUNTIME_SIF": env.get("OH_RUNTIME_SIF") or f"{scratch}/sif_images/openhands_runtime.sif",
        "OH_RUNTIME_DOCKER": env.get("OH_RUNTIME_DOCKER", ""),
        "OH_RUNTIME_BASE_SIF": env.get("OH_RUNTIME_BASE_SIF") or f"{scratch}/sif_images/openhands_python3.12.sif",
        "OH_RUNTIME_BASE_DOCKER": env.get("OH_RUNTIME_BASE_DOCKER", "docker://python:3.12"),
        "OH_RUNTIME_REBUILD": env.get("OH_RUNTIME_REBUILD", "0"),
        "HM_RUNTIME_VERSION": hm_version,
        "HM_RUNTIME_SIF": env.get("HM_RUNTIME_SIF") or f"{scratch}/sif_images/hermes_runtime.sif",
        "HM_RUNTIME_DOCKER": env.get("HM_RUNTIME_DOCKER", ""),
        "HM_RUNTIME_BASE_SIF": env.get("HM_RUNTIME_BASE_SIF") or f"{scratch}/sif_images/hermes_{hm_version}.sif",
        "HM_RUNTIME_BASE_DOCKER": env.get("HM_RUNTIME_BASE_DOCKER") or f"docker://nousresearch/hermes-agent:{hm_version}",
        "HM_RUNTIME_REBUILD": env.get("HM_RUNTIME_REBUILD", "0"),
        "OC_OTEL_SIF": env.get("OC_OTEL_SIF") or f"{scratch}/sif_images/otelcol-contrib.sif",
        "OC_OTEL_DOCKER": env.get("OC_OTEL_DOCKER", "docker://otel/opentelemetry-collector-contrib:latest"),
    }
    out["OC_OTEL_TRACE_FILE"] = env.get("OC_OTEL_TRACE_FILE") or f"{out['OC_OTEL_DATA_DIR']}/otel-traces.jsonl"
    out["OH_IMAGE"] = env.get("OH_IMAGE") or out["OH_RUNTIME_SIF"]
    out["HM_IMAGE"] = env.get("HM_IMAGE") or out["HM_RUNTIME_SIF"]
    if _harness_choice(resolved) == "openclaw":
        out["OC_OC_SIF"] = env.get("OC_OC_SIF") or out["OC_RUNTIME_SIF"]
        if env.get("OC_OC_OVERLAY") or out["OC_RUNTIME_OVERLAY"]:
            out["OC_OC_OVERLAY"] = env.get("OC_OC_OVERLAY") or out["OC_RUNTIME_OVERLAY"]
    return out


def image_preparation_commands(
    resolved: ResolvedExperiment,
    spec: PbsJobSpec,
    env: Mapping[str, str],
) -> list[StageCommand]:
    commands: list[StageCommand] = []
    runtime_env = runtime_environment(resolved, spec, env)
    kit_root = Path(runtime_env["KIT_ROOT"])

    commands.extend(_serving_image_commands(resolved, spec, runtime_env))
    commands.extend(_benchmark_image_commands(resolved, spec, runtime_env, kit_root))
    commands.extend(_runtime_image_commands(resolved, spec, runtime_env, kit_root))
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
    cache = Path(runtime_env["OC_PLUGIN_CACHE"])
    if _plugin_cache_warm(cache):
        return None

    state_template = spec.run_dir / "openclaw_state_template"
    _write_openclaw_state_template(resolved, state_template, env)
    argv = [
        "bash",
        str(Path(runtime_env["KIT_ROOT"]) / "agent_bench/deployments/pbs/warm_oc_plugins.sh"),
        "--sif",
        runtime_env["OC_OC_SIF"],
        "--cache",
        runtime_env["OC_PLUGIN_CACHE"],
        "--state-template",
        str(state_template),
    ]
    if runtime_env.get("OC_OC_OVERLAY"):
        argv[4:4] = ["--overlay", runtime_env["OC_OC_OVERLAY"]]
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
            str(Path(runtime_env["KIT_ROOT"]) / "agent_bench/deployments/pbs/otel_collector.sh"),
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
    kit_root: Path,
) -> list[StageCommand]:
    if resolved.spec.benchmark_name == "terminal_bench":
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
                str(kit_root / "agent_bench/deployments/pbs/pull_images.sh"),
                "--dir",
                env["SIF_DIR"],
                "--parallel",
                env["OC_PULL_PARALLEL"],
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
        env["OC_TB_SIF_DIR"],
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
    kit_root: Path,
) -> list[StageCommand]:
    if spec.container_runtime != "singularity":
        return []
    harness = _harness_choice(resolved)
    if harness == "openclaw":
        return _openclaw_runtime_commands(spec, env, kit_root)
    if harness == "openhands":
        return _openhands_runtime_commands(spec, env, kit_root)
    if harness == "hermes":
        return _hermes_runtime_commands(spec, env, kit_root)
    return []


def _openclaw_runtime_commands(spec: PbsJobSpec, env: Mapping[str, str], kit_root: Path) -> list[StageCommand]:
    commands = _pull_if_missing(
        label="pull OpenClaw base runtime SIF",
        sif=env["OC_RUNTIME_BASE_SIF"],
        image=env["OC_RUNTIME_DOCKER"],
        log_path=spec.log_dir / "runtime-openclaw-base-pull.log",
        env=env,
    )
    if env.get("OC_RUNTIME_REBUILD") == "1" or not Path(env["OC_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build patched OpenClaw runtime SIF",
                argv=(
                    "bash",
                    str(kit_root / "agent_bench/deployments/pbs/build_oc_sif.sh"),
                    "--base",
                    env["OC_RUNTIME_BASE_SIF"],
                    "--out",
                    env["OC_RUNTIME_SIF"],
                    "--kit-root",
                    str(kit_root),
                    "--inside-pkg",
                    env["OC_INSIDE_PKG"],
                ),
                log_path=spec.log_dir / "runtime-openclaw-build.log",
                env=dict(env),
                required=True,
            )
        )
    return commands


def _openhands_runtime_commands(spec: PbsJobSpec, env: Mapping[str, str], kit_root: Path) -> list[StageCommand]:
    if env.get("OH_RUNTIME_DOCKER"):
        return _pull_if_missing(
            label="pull OpenHands runtime SIF",
            sif=env["OH_RUNTIME_SIF"],
            image=env["OH_RUNTIME_DOCKER"],
            log_path=spec.log_dir / "runtime-openhands-pull.log",
            env=env,
        )
    commands = _pull_if_missing(
        label="pull OpenHands base SIF",
        sif=env["OH_RUNTIME_BASE_SIF"],
        image=env["OH_RUNTIME_BASE_DOCKER"],
        log_path=spec.log_dir / "runtime-openhands-base-pull.log",
        env=env,
    )
    if env.get("OH_RUNTIME_REBUILD") == "1" or not Path(env["OH_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build OpenHands runtime SIF",
                argv=(
                    "bash",
                    str(kit_root / "agent_bench/deployments/pbs/build_oh_sif.sh"),
                    "--base",
                    env["OH_RUNTIME_BASE_SIF"],
                    "--out",
                    env["OH_RUNTIME_SIF"],
                    "--kit-root",
                    str(kit_root),
                    "--version",
                    env["OH_RUNTIME_VERSION"],
                ),
                log_path=spec.log_dir / "runtime-openhands-build.log",
                env=dict(env),
                required=True,
            )
        )
    return commands


def _hermes_runtime_commands(spec: PbsJobSpec, env: Mapping[str, str], kit_root: Path) -> list[StageCommand]:
    if env.get("HM_RUNTIME_DOCKER"):
        return _pull_if_missing(
            label="pull Hermes runtime SIF",
            sif=env["HM_RUNTIME_SIF"],
            image=env["HM_RUNTIME_DOCKER"],
            log_path=spec.log_dir / "runtime-hermes-pull.log",
            env=env,
        )
    commands = _pull_if_missing(
        label="pull Hermes base SIF",
        sif=env["HM_RUNTIME_BASE_SIF"],
        image=env["HM_RUNTIME_BASE_DOCKER"],
        log_path=spec.log_dir / "runtime-hermes-base-pull.log",
        env=env,
    )
    if env.get("HM_RUNTIME_REBUILD") == "1" or not Path(env["HM_RUNTIME_SIF"]).exists():
        commands.append(
            StageCommand(
                label="build patched Hermes runtime SIF",
                argv=(
                    "bash",
                    str(kit_root / "agent_bench/deployments/pbs/build_hm_sif.sh"),
                    "--base",
                    env["HM_RUNTIME_BASE_SIF"],
                    "--out",
                    env["HM_RUNTIME_SIF"],
                    "--kit-root",
                    str(kit_root),
                    "--version",
                    env["HM_RUNTIME_VERSION"],
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

    model_name = env.get("MODEL") or resolved.model.get("name") or "model"
    providers = oc_cfg.setdefault("models", {}).setdefault("providers", {})
    vllm_provider = providers.setdefault("vllm", {})
    vllm_provider["api"] = "openai-completions"
    vllm_provider.setdefault("baseUrl", env.get("MODEL_ENDPOINT", "http://127.0.0.1:8000/v1"))
    vllm_provider.setdefault("apiKey", env.get("MODEL_API_KEY") or "sk-dummy")
    vllm_provider["models"] = [
        {
            "id": model_name,
            "name": model_name,
            "api": "openai-completions",
            "contextWindow": int(env.get("MODEL_CONTEXT_WINDOW") or resolved.model.get("context_window") or 0),
            "maxTokens": int(env.get("MAX_TOKENS") or resolved.model.get("max_tokens") or 8192),
        }
    ]
    agents_defaults = oc_cfg.setdefault("agents", {}).setdefault("defaults", {})
    agents_defaults.setdefault("model", {})["primary"] = f"vllm/{model_name}"
    (state_template / "openclaw.json").write_text(json.dumps(oc_cfg, indent=2), encoding="utf-8")

    agent_src = Path(env.get("KIT_ROOT") or Path.cwd()) / "agent_bench/benchmarks/swebench/openclaw/agents"
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
    spec = type("_Spec", (), {"harness_name": ""})()
    harness: dict[str, Any] = {}
    model: dict[str, Any] = {}
    profile_paths: dict[str, Path] = {}
