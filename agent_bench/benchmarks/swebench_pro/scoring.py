from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from agent_bench.benchmarks.swebench_pro.common import (
    TASK_WORKSPACE,
    as_list,
    get_dockerhub_image_uri,
    normalize_instance,
    selected_tests_arg,
)


def error_verdict(
    message: str,
    *,
    evaluation_dir: Optional[Path] = None,
    status: str = "ERROR",
) -> dict[str, Any]:
    return {
        "status": status,
        "test_pass_rate": 0.0,
        "passed": 0,
        "failed": 0,
        "error_message": message,
        "evaluation_dir": str(evaluation_dir) if evaluation_dir else None,
        "scoring_backend": "swebench_pro.docker",
    }


def image_for_instance(instance_data: Mapping[str, Any]) -> str:
    inst = normalize_instance(instance_data)
    return inst.get("dockerhub_image") or get_dockerhub_image_uri(
        inst["instance_id"], repo_name=inst.get("repo") or "",
    )


def score_patch(
    *,
    instance_id: str,
    patch: str,
    instance_data: Mapping[str, Any],
    timeout_seconds: int,
    run_id: str,
    benchmark_data_root: Path,
    run_dir: Optional[Path] = None,
) -> dict[str, Any]:
    if shutil.which("docker") is None:
        return error_verdict("docker is not on PATH; SWE-bench Pro local scoring requires Docker")

    inst = normalize_instance(instance_data)
    eval_root = Path(run_dir) / "swebench_pro_scoring" if run_dir else Path(benchmark_data_root) / "swebench_pro_scoring" / run_id
    eval_dir = eval_root / instance_id
    workspace = eval_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        files = assemble_workspace_files(inst, patch, benchmark_data_root)
    except Exception as exc:
        return error_verdict(f"failed to assemble SWE-bench Pro scoring files: {exc}", evaluation_dir=eval_dir)

    for name, content in files.items():
        path = workspace / name
        path.write_text(content, encoding="utf-8")
        if name.endswith(".sh"):
            path.chmod(0o755)

    image = image_for_instance(inst)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace.resolve()}:/workspace:rw",
        "--entrypoint", "/bin/bash",
    ]
    if os.environ.get("AB_SWEBENCH_PRO_BLOCK_NETWORK", "0") == "1":
        cmd.extend(["--network", "none"])
    platform = os.environ.get("AB_SWEBENCH_PRO_DOCKER_PLATFORM")
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend([image, "-c", "bash /workspace/entryscript.sh"])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 300,
        )
    except subprocess.TimeoutExpired as exc:
        _write_timeout_logs(eval_dir, exc)
        return error_verdict(
            f"SWE-bench Pro scoring timeout after {timeout_seconds + 300}s",
            evaluation_dir=eval_dir,
            status="TIMEOUT",
        )

    (eval_dir / "docker_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (eval_dir / "docker_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    output_path = workspace / "output.json"
    if not output_path.exists():
        return error_verdict(
            f"SWE-bench Pro entryscript produced no output.json (rc={proc.returncode}): {(proc.stderr or '')[-500:]}",
            evaluation_dir=eval_dir,
            status="FAIL" if proc.returncode != 0 else "ERROR",
        )

    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error_verdict(f"failed to parse output.json: {exc}", evaluation_dir=eval_dir)

    verdict = parse_output_verdict(output, inst, evaluation_dir=eval_dir, output_path=output_path)
    verdict["docker_image"] = image
    verdict["scoring_backend"] = "swebench_pro.docker"
    if proc.returncode != 0 and verdict["status"] != "PASS":
        verdict["error_message"] = (
            verdict.get("error_message")
            or f"entryscript exited {proc.returncode}: {(proc.stderr or '')[-500:]}"
        )
    return verdict


def assemble_workspace_files(
    instance_data: Mapping[str, Any],
    patch: str,
    benchmark_data_root: Path,
) -> dict[str, str]:
    root = Path(os.environ.get("AB_SWEBENCH_PRO_ROOT") or benchmark_data_root)
    instance_id = str(instance_data["instance_id"])
    return {
        "patch.diff": patch or "",
        "run_script.sh": _resource_text(instance_data, "run_script", root / "run_scripts" / instance_id / "run_script.sh"),
        "parser.py": _resource_text(instance_data, "parsing_script", root / "run_scripts" / instance_id / "parser.py"),
        "entryscript.sh": create_entryscript(instance_data, root),
        "raw_sample.json": json.dumps(dict(instance_data), indent=2),
    }


def create_entryscript(instance_data: Mapping[str, Any], resource_root: Path) -> str:
    base_commit = str(instance_data.get("base_commit") or "")
    if not base_commit:
        raise ValueError("instance_data is missing base_commit")

    pre_apply_cmds, post_apply_cmds = _split_before_repo_set_cmd(
        instance_data.get("before_repo_set_cmd"),
        base_commit=base_commit,
    )
    selected_arg = selected_tests_arg(instance_data.get("selected_test_files_to_run"))
    env_cmds = _dockerfile_env_exports(instance_data, resource_root)

    return "\n".join([
        env_cmds,
        "cd /app",
        f"git reset --hard {base_commit}",
        pre_apply_cmds,
        f"git checkout {base_commit}",
        "git apply -v /workspace/patch.diff",
        post_apply_cmds,
        f"bash /workspace/run_script.sh {selected_arg} > /workspace/stdout.log 2> /workspace/stderr.log",
        "python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
        "",
    ])


def _split_before_repo_set_cmd(value: Any, *, base_commit: str) -> tuple[str, str]:
    """Place SWE-bench Pro setup commands around candidate patch application.

    The dataset's before_repo_set_cmd commonly resets/cleans the repo, then
    checks out test files from the gold commit. Reset/clean must happen before
    applying the candidate patch; the test checkout must happen after it so the
    hidden tests are restored without wiping the candidate solution.
    """
    if not isinstance(value, str) or not value.strip():
        return "", ""

    pre_apply: list[str] = []
    post_apply: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_builtin_base_checkout_cmd(line, base_commit=base_commit):
            continue
        if line.startswith("git clean "):
            pre_apply.append(line)
        else:
            post_apply.append(line)
    return "\n".join(pre_apply), "\n".join(post_apply)


def _is_builtin_base_checkout_cmd(line: str, *, base_commit: str) -> bool:
    return (
        line == f"git reset --hard {base_commit}"
        or line == f"git checkout {base_commit}"
        or line.startswith(f"git reset --hard {base_commit} ")
        or line.startswith(f"git checkout {base_commit} ")
    )


def parse_output_verdict(
    output: Mapping[str, Any],
    instance_data: Mapping[str, Any],
    *,
    evaluation_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    tests = output.get("tests") or []
    passed_tests = {
        str(item.get("name"))
        for item in tests
        if isinstance(item, Mapping) and str(item.get("status")).upper() == "PASSED"
    }
    f2p_required = set(as_list(instance_data.get("fail_to_pass", instance_data.get("FAIL_TO_PASS"))))
    p2p_required = set(as_list(instance_data.get("pass_to_pass", instance_data.get("PASS_TO_PASS"))))

    f2p_success = sorted(f2p_required & passed_tests)
    f2p_failure = sorted(f2p_required - passed_tests)
    p2p_success = sorted(p2p_required & passed_tests)
    p2p_failure = sorted(p2p_required - passed_tests)
    required_total = len(f2p_required) + len(p2p_required)
    passed = len(f2p_success) + len(p2p_success)
    failed = len(f2p_failure) + len(p2p_failure)
    resolved = failed == 0

    report = {
        "resolved": resolved,
        "tests_status": {
            "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
            "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
        },
        "raw_tests": tests,
    }
    report_path = evaluation_dir / "report.json"
    report_path.write_text(json.dumps({instance_data["instance_id"]: report}, indent=2), encoding="utf-8")

    return {
        "status": "PASS" if resolved else "FAIL",
        "test_pass_rate": passed / required_total if required_total else 0.0,
        "passed": passed,
        "failed": failed,
        "error_message": None if resolved else f"{failed} required test(s) did not pass",
        "evaluation_dir": str(evaluation_dir),
        "raw_report_path": str(report_path),
        "raw_output_path": str(output_path),
        "tests_status": report["tests_status"],
    }


def _resource_text(instance_data: Mapping[str, Any], key: str, fallback: Path) -> str:
    value = instance_data.get(key)
    if isinstance(value, str) and value.strip():
        return value
    alt_key = "parser_script" if key == "parsing_script" else key
    value = instance_data.get(alt_key)
    if isinstance(value, str) and value.strip():
        return value
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError(f"missing {key}: instance data has no content and {fallback} does not exist")


def _dockerfile_env_exports(instance_data: Mapping[str, Any], resource_root: Path) -> str:
    instance_id = str(instance_data["instance_id"])
    chunks: list[str] = []
    for key, fallback in (
        ("base_dockerfile", resource_root / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"),
        ("instance_dockerfile", resource_root / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"),
    ):
        text = instance_data.get(key)
        if not isinstance(text, str) or not text.strip():
            text = fallback.read_text(encoding="utf-8") if fallback.is_file() else ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV "):
                parts = stripped.split(None, 2)
                if len(parts) == 3 and "=" not in parts[1]:
                    chunks.append(f"export {parts[1]}={parts[2]}")
                else:
                    chunks.append(stripped.replace("ENV", "export", 1))
    return "\n".join(chunks)


def _write_timeout_logs(eval_dir: Path, exc: subprocess.TimeoutExpired) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    stdout = exc.stdout or ""
    stderr = exc.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    (eval_dir / "docker_stdout.log").write_text(stdout, encoding="utf-8")
    (eval_dir / "docker_stderr.log").write_text(stderr, encoding="utf-8")

