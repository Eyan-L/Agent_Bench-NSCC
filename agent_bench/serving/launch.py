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
    parser.add_argument("--preset", help="Serving preset override.")
    parser.add_argument("--model-name", help="Engine model path/name override.")
    parser.add_argument("--serving-port", help="Serving port override.")
    parser.add_argument("--serving-max-model-len", help="Max model length override.")
    parser.add_argument("--log-file", help="Redirect backend stdout/stderr to this file before exec.")
    parser.add_argument("--print-json", action="store_true", help="Print the redacted launch plan and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit without launching.")
    args = parser.parse_args(argv)

    env = os.environ.copy()
    if args.model_config:
        env["AB_MODEL_CONFIG"] = args.model_config
    if args.serving_config:
        env["AB_SERVING_CONFIG"] = args.serving_config
    if args.preset:
        env["AB_SERVING_PRESET"] = args.preset
    if args.model_name:
        env["AB_MODEL_NAME"] = args.model_name
    if args.serving_port:
        env["AB_SERVING_PORT"] = args.serving_port
    if args.serving_max_model_len:
        env["AB_SERVING_MAX_MODEL_LEN"] = args.serving_max_model_len

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
        _prepare_runtime_environment(plan, launch_env)
        prepare_launch_artifacts(plan, launch_env=launch_env)
        _write_pid_file_if_needed(plan.backend, launch_env)
        _redirect_output_if_needed(plan.log_path)
        _check_executable(plan.argv[0], launch_env)
        os.execvpe(plan.argv[0], plan.argv, launch_env)
    except (LaunchPlanError, OSError) as exc:
        print(f"failed to launch {plan.backend}: {exc}", file=sys.stderr)
        return 1


def _prepare_runtime_environment(plan, env: dict[str, str]) -> None:
    if plan.backend == "vllm":
        _prepare_vllm_runtime(plan, env)
        return
    if plan.backend != "sglang":
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


def _prepare_vllm_runtime(plan, env: dict[str, str]) -> None:
    """Make `vllm` resolvable and sanity-check the GPU topology before exec.

    On a serve-only GPU host, vLLM usually lives in a site-specific venv (VLLM_VENV)
    or a conda env (VLLM_CONDA_ENV) rather than on the system PATH. Activate whichever
    is configured by putting its bin dir first; also fall back to this interpreter's
    own bin dir, which covers the common case of being invoked by the venv's python
    (serve_vllm.sh does exactly that so tomllib/agent_bench import cleanly).
    """
    if shutil.which("vllm", path=env.get("PATH")) is None:
        bindirs: list[Path] = []
        venv = env.get("VLLM_VENV")
        if venv:
            venv_path = Path(venv).expanduser()
            bindirs.append(venv_path / "bin")
        conda_env = env.get("VLLM_CONDA_ENV")
        if conda_env:
            bindirs.extend(
                base / "envs" / conda_env / "bin"
                for base in (
                    Path.home() / "anaconda3",
                    Path.home() / "miniconda3",
                    Path("/opt/conda"),
                )
            )
        # The interpreter that launched us (e.g. the venv python serve_vllm.sh picked).
        bindirs.append(Path(sys.executable).resolve().parent)
        for bindir in bindirs:
            if (bindir / "vllm").exists():
                env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
                if venv and bindir == Path(venv).expanduser() / "bin":
                    env["VIRTUAL_ENV"] = str(Path(venv).expanduser())
                    env.pop("PYTHONHOME", None)
                break

    _vllm_gpu_preflight(plan, env)


def _vllm_gpu_preflight(plan, env: dict[str, str]) -> None:
    """Echo the vLLM version and fail fast if TP exceeds the visible GPU count."""
    vllm_bin = shutil.which("vllm", path=env.get("PATH"))
    if vllm_bin:
        try:
            ver = subprocess.run(
                [vllm_bin, "--version"], env=env, capture_output=True, text=True, check=False
            )
            line = (ver.stdout or ver.stderr).strip().splitlines()
            if line:
                print(f"[launch] vLLM version: {line[0]}", file=sys.stderr)
        except OSError:
            pass

    tp = int(plan.metadata.get("tensor_parallel_size", "1"))
    smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    if smi is None:
        print("[launch] WARN: nvidia-smi not found — vLLM will fail without a GPU.", file=sys.stderr)
        return
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    count = len([ln for ln in out.stdout.splitlines() if ln.strip()])
    print(f"[launch] detected {count} GPU(s); using TP={tp}", file=sys.stderr)
    if count and count < tp:
        raise LaunchPlanError(f"TP={tp} but only {count} GPU(s) visible")


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


def _check_executable(executable: str, env: dict[str, str]) -> None:
    if "/" in executable:
        return
    # Honor the launch env's PATH, not this process's — runtime prep may have
    # prepended a venv/conda bin dir there (e.g. the vLLM venv).
    if shutil.which(executable, path=env.get("PATH")) is None:
        raise LaunchPlanError(f"{executable!r} is not on PATH")


if __name__ == "__main__":
    raise SystemExit(main())
