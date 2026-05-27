from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from agent_bench.serving.launch_plan import (
    LaunchPlanError,
    build_launch_environment,
    build_launch_plan_from_env,
    prepare_launch_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch an Agent Bench serving backend from profiles.")
    parser.add_argument("--backend", choices=["vllm", "sglang", "litellm"], help="Serving backend to launch.")
    parser.add_argument("--mode", choices=["native", "docker", "singularity"], help="Launch mode.")
    parser.add_argument("--model-config", help="Path to profiles/models/*.json.")
    parser.add_argument("--serving-config", help="Path to a serving profile.")
    parser.add_argument("--model-profile", help="Model profile name under profiles/models.")
    parser.add_argument("--serving", help="Serving profile name under profiles/serving.")
    parser.add_argument("--preset", help="Serving preset override.")
    parser.add_argument("--model", help="Engine model path/name override.")
    parser.add_argument("--port", help="Serving port override.")
    parser.add_argument("--max-model-len", help="Max model length override.")
    parser.add_argument("--log-file", help="Redirect backend stdout/stderr to this file before exec.")
    parser.add_argument("--print-json", action="store_true", help="Print the redacted launch plan and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit without launching.")
    args = parser.parse_args(argv)

    env = os.environ.copy()
    if args.model_config:
        env["AB_MODEL_CONFIG"] = args.model_config
    if args.serving_config:
        env["AB_SERVING_CONFIG"] = args.serving_config
    if args.model_profile:
        env["MODEL_PROFILE"] = args.model_profile
    if args.serving:
        env["SERVING"] = args.serving
    if args.preset:
        env["SERVING_PRESET"] = args.preset
    if args.model:
        env["MODEL"] = args.model
    if args.port:
        env["PORT"] = args.port
    if args.max_model_len:
        env["MAX_MODEL_LEN"] = args.max_model_len

    try:
        plan = build_launch_plan_from_env(
            backend=args.backend,
            mode=args.mode,
            env=env,
            log_path=args.log_file,
        )
    except (LaunchPlanError, ValueError) as exc:
        print(f"invalid serving launch: {exc}", file=sys.stderr)
        return 2

    if args.print_json:
        print(json.dumps(plan.to_redacted_dict(), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(plan.command_for_display())
        return 0

    try:
        launch_env = build_launch_environment(plan, base_env=env)
        _prepare_runtime_environment(plan.backend, launch_env)
        prepare_launch_artifacts(plan, launch_env=launch_env)
        _write_pid_file_if_needed(plan.backend, launch_env)
        _redirect_output_if_needed(plan.log_path)
        _check_executable(plan.argv[0])
        os.execvpe(plan.argv[0], plan.argv, launch_env)
    except (LaunchPlanError, OSError) as exc:
        print(f"failed to launch {plan.backend}: {exc}", file=sys.stderr)
        return 1


def _prepare_runtime_environment(backend: str, env: dict[str, str]) -> None:
    if backend != "sglang":
        return
    if _python_can_import("sglang", env):
        _prepend_nvidia_libs(env)
        return
    conda_env = env.get("SGLANG_CONDA_ENV")
    if conda_env:
        for env_bin in (
            Path.home() / "anaconda3" / "envs" / conda_env / "bin",
            Path.home() / "miniconda3" / "envs" / conda_env / "bin",
            Path("/opt/conda/envs") / conda_env / "bin",
        ):
            if (env_bin / "python3").exists():
                env["PATH"] = f"{env_bin}:{env.get('PATH', '')}"
                env["CONDA_PREFIX"] = str(env_bin.parent)
                env["CONDA_DEFAULT_ENV"] = conda_env
                break
    _prepend_nvidia_libs(env)


def _python_can_import(module: str, env: dict[str, str]) -> bool:
    proc = subprocess.run(
        ["python3", "-c", f"import {module}"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _prepend_nvidia_libs(env: dict[str, str]) -> None:
    conda_prefix = env.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    libs = []
    pattern = str(Path(conda_prefix) / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
    for lib_dir in glob.glob(pattern):
        if Path(lib_dir).is_dir():
            libs.append(lib_dir)
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")


def _write_pid_file_if_needed(backend: str, env: dict[str, str]) -> None:
    if backend != "sglang":
        return
    pid_file = env.get("SGLANG_PID_FILE")
    if not pid_file:
        return
    path = Path(pid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


def _redirect_output_if_needed(log_path: str | None) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def _check_executable(executable: str) -> None:
    if "/" in executable:
        return
    if shutil.which(executable) is None:
        raise LaunchPlanError(f"{executable!r} is not on PATH")


if __name__ == "__main__":
    raise SystemExit(main())
