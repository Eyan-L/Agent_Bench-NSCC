from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib import error, request

from agent_bench.core.spec import ConfigError, load_resolved_experiment
from agent_bench.deployments import plan_deployment


def validate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Agent Bench experiment TOML.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--json", action="store_true", help="Emit resolved experiment JSON.")
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(resolved.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"valid: {resolved.spec.name} ({resolved.run_id})")
    return 0


def run_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and run an Agent Bench experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    parser.add_argument("--dry-run", action="store_true", help="Render the execution plan without launching work.")
    parser.add_argument("--render-job", action="store_true", help="Print rendered job script when available.")
    parser.add_argument("--print-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-retry-errors", action="store_true", help="Do not retry ERROR runs.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EVAL_MAX_WORKERS", "1")))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    try:
        resolved = load_resolved_experiment(args.config)
        plan = plan_deployment(resolved)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2

    if args.json and (args.dry_run or args.render_job or args.print_job):
        print(json.dumps({"experiment": resolved.to_dict(), "plan": plan.to_dict()}, indent=2, sort_keys=True))
        return 0
    if (args.render_job or args.print_job) and plan.job_script:
        print(plan.job_script, end="")
        return 0
    if args.dry_run:
        print(_format_plan(resolved, plan.to_dict()))
        return 0

    try:
        _execute_experiment(resolved, retry_errors=not args.no_retry_errors, max_workers=args.max_workers)
    except ConfigError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2
    return 0


def analyze_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an Agent Bench run directory.")
    parser.add_argument("--run", required=True, help="Path to a run directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    args = parser.parse_args(argv)

    run_path = Path(args.run).expanduser()
    if not run_path.exists():
        print(f"missing run directory: {run_path}", file=sys.stderr)
        return 2
    if not run_path.is_dir():
        print(f"run path is not a directory: {run_path}", file=sys.stderr)
        return 2

    summary = {
        "run": str(run_path.resolve()),
        "manifest": str((run_path / "manifest.json").resolve()) if (run_path / "manifest.json").exists() else None,
        "result_files": sorted(path.name for path in run_path.glob("*.json")),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"run: {summary['run']}")
        print(f"manifest: {summary['manifest'] or 'not found'}")
        print(f"json files: {len(summary['result_files'])}")
    return 0


def _format_plan(resolved: Any, plan: dict[str, Any]) -> str:
    lines = [
        f"experiment: {resolved.spec.name}",
        f"run_id: {resolved.run_id}",
        f"deployment: {plan['deployment']}",
        f"summary: {plan['summary']}",
        "steps:",
    ]
    lines.extend(f"  - {step}" for step in plan["steps"])
    if plan["warnings"]:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in plan["warnings"])
    if plan["job_script"]:
        lines.append("job_script: available with --render-job")
    return "\n".join(lines)


def _execute_experiment(resolved: Any, *, retry_errors: bool, max_workers: int) -> None:
    from agent_bench.runtime.driver import run_sweep

    app_path = resolved.profile_paths.get("benchmark")
    harness_path = resolved.profile_paths.get("harness")
    model_path = resolved.profile_paths["model"]
    serving_path = resolved.profile_paths["serving"]
    if app_path is None or harness_path is None:
        raise ConfigError("experiments must reference benchmark.profile and harness.profile to execute")

    instance_data = _instance_data_path(resolved)
    swebench_root = _optional_path_from_raw_or_env(resolved, "swebench_root", "SWEBENCH_ROOT")

    os.environ["AB_APP_CONFIG"] = str(app_path.resolve())
    os.environ["AB_HARNESS_CONFIG"] = str(harness_path.resolve())
    os.environ["AB_MODEL_CONFIG"] = str(model_path.resolve())
    os.environ["AB_SERVING_CONFIG"] = str(serving_path.resolve())
    os.environ["AB_INSTANCE_DATA"] = str(instance_data.resolve())
    _export_serving_agent_env(resolved)

    with _serving_context(resolved):
        run_sweep(
            app_path=app_path,
            harness_path=harness_path,
            model_path=model_path,
            serving_path=serving_path,
            swebench_root=swebench_root,
            instance_data_path=instance_data,
            retry_errors=retry_errors,
            max_workers=max_workers,
        )


def _export_serving_agent_env(resolved: Any) -> None:
    backend = resolved.serving.get("backend")
    endpoint = _serving_endpoint(resolved)
    if endpoint:
        os.environ.setdefault("MODEL_ENDPOINT", endpoint)
        os.environ.setdefault("OPENAI_BASE_URL", endpoint)
    if backend:
        os.environ.setdefault("SERVING_BACKEND", str(backend))
    if backend == "litellm":
        proxy = resolved.serving.get("proxy") or {}
        proxy_key = proxy.get("api_key") or "sk-agent-bench-local"
        os.environ.setdefault("MODEL_API_KEY", proxy_key)
        os.environ.setdefault("OPENAI_API_KEY", proxy_key)
        os.environ.setdefault("OC_VLLM_API_KEY", proxy_key)
        agent_provider = proxy.get("agent_provider")
        if agent_provider:
            os.environ.setdefault("MODEL_PROVIDER", str(agent_provider))
            os.environ.setdefault("HERMES_INFERENCE_PROVIDER", str(agent_provider))


@contextmanager
def _serving_context(resolved: Any):
    backend = resolved.serving.get("backend")
    deployment_kind = resolved.deployment.get("kind") or resolved.deployment.get("name")
    if backend not in {"vllm", "sglang", "litellm"}:
        yield
        return
    if deployment_kind not in {"client_only", "local_docker"}:
        # Batch/cloud job wrappers own their serving sidecars.
        yield
        return

    endpoint = _serving_endpoint(resolved)
    auth_key = _serving_auth_key(resolved)
    if endpoint and _endpoint_ready(endpoint, auth_key=auth_key):
        yield
        return

    log_dir = resolved.spec.output_dir / resolved.run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{backend}.log"
    env = os.environ.copy()
    env.setdefault("SERVING_BACKEND", str(backend))
    env.setdefault("KIT_ROOT", str(Path.cwd()))

    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            ["bash", "agent_bench/serving/common/serve.sh"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            _wait_for_endpoint(proc, endpoint, auth_key=auth_key, log_path=log_path)
            yield
        finally:
            _terminate_process_group(proc)


def _serving_endpoint(resolved: Any) -> str | None:
    endpoint = resolved.serving.get("endpoint") or {}
    if isinstance(endpoint, str):
        return endpoint.rstrip("/")
    if not isinstance(endpoint, dict):
        return None
    direct = endpoint.get("base_url") or endpoint.get("url")
    if direct:
        return str(direct).rstrip("/")
    scheme = endpoint.get("scheme", "http")
    host = endpoint.get("client_host") or endpoint.get("host") or "127.0.0.1"
    port = endpoint.get("port", 443 if scheme == "https" else 8000)
    path = endpoint.get("path", "/v1")
    if not str(path).startswith("/"):
        path = "/" + str(path)
    return f"{scheme}://{host}:{port}{path}".rstrip("/")


def _serving_auth_key(resolved: Any) -> str | None:
    if resolved.serving.get("backend") != "litellm":
        return None
    proxy = resolved.serving.get("proxy") or {}
    return proxy.get("api_key") or "sk-agent-bench-local"


def _endpoint_ready(endpoint: str | None, *, auth_key: str | None) -> bool:
    if not endpoint:
        return False
    req = request.Request(endpoint.rstrip("/") + "/models")
    if auth_key:
        req.add_header("Authorization", f"Bearer {auth_key}")
    try:
        with request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except (OSError, error.URLError, error.HTTPError):
        return False


def _wait_for_endpoint(
    proc: subprocess.Popen[bytes],
    endpoint: str | None,
    *,
    auth_key: str | None,
    log_path: Path,
) -> None:
    timeout = int(os.environ.get("SERVE_READY_TIMEOUT", "600"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ConfigError(f"serving process exited before readiness; see {log_path}")
        if _endpoint_ready(endpoint, auth_key=auth_key):
            return
        time.sleep(2)
    raise ConfigError(f"serving endpoint was not ready after {timeout}s; see {log_path}")


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=10)


def _instance_data_path(resolved: Any) -> Path:
    explicit = _optional_path_from_raw_or_env(resolved, "instance_data", "INSTANCE_DATA")
    if explicit is not None:
        return explicit

    benchmark = resolved.spec.benchmark_name
    if benchmark == "terminal_bench":
        out = Path("data/terminal_bench/tasks.json").resolve()
        from agent_bench.benchmarks.terminal_bench import loader

        loader.materialize_dataset(loader.list_task_ids(), out)
        return out
    if benchmark == "deepresearch_bench":
        out = Path("data/deepresearch_bench/instances.json").resolve()
        from agent_bench.benchmarks.deepresearch_bench import loader

        loader.materialize_dataset(loader.list_task_ids(), out)
        return out
    if benchmark == "longbench":
        out = Path("data/longbench/instances.json").resolve()
        from agent_bench.benchmarks.longbench import loader

        loader.materialize_dataset(loader.list_task_ids(), out)
        return out

    raise ConfigError(
        "SWE-bench experiments require instance_data in the experiment TOML "
        "or INSTANCE_DATA in the environment"
    )


def _optional_path_from_raw_or_env(resolved: Any, raw_key: str, env_key: str) -> Path | None:
    raw_value = resolved.spec.raw.get(raw_key)
    if isinstance(raw_value, str) and raw_value:
        return _resolve_user_path(resolved.spec.config_path.parent, raw_value)
    env_value = os.environ.get(env_key)
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
