"""Shared SWE-bench scoring helpers."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
_GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


def _error_verdict(
    message: str,
    *,
    evaluation_dir: Optional[Path | str] = None,
    backend: Optional[str] = None,
    status: str = "ERROR",
) -> Dict:
    verdict = {
        "status": status,
        "test_pass_rate": 0.0,
        "passed": 0,
        "failed": 0,
        "error_message": message,
        "evaluation_dir": str(evaluation_dir) if evaluation_dir else None,
    }
    if backend:
        verdict["scoring_backend"] = backend
    return verdict


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[dict] = None,
    timeout: float = 30.0,
) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"HTTP {e.code} from scoring server: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"scoring server request failed: {e}") from e
    return json.loads(body or "{}")


def score_via_scoring_server(
    *,
    instance_id: str,
    patch: str,
    model_name_or_path: str,
    timeout_seconds: int,
) -> Optional[Dict]:
    """Score through AB_SCORING_SERVICE_URL when configured.

    Returns None when no scoring server is configured, so callers can fall back
    to their local run_evaluation path. When AB_SCORING_SERVICE_URL is configured,
    failures are returned as ERROR verdicts instead of falling through to
    Docker-backed local scoring, which is exactly what NSCC cannot run.
    """
    base = (os.environ.get("AB_SCORING_SERVICE_URL") or "").rstrip("/")
    if not base:
        return None

    dataset = os.environ.get("AB_SCORING_DATASET", _DEFAULT_DATASET)
    request_timeout = float(os.environ.get("AB_SCORING_REQUEST_TIMEOUT_SECONDS", "30"))
    poll_s = float(os.environ.get("AB_SCORING_POLL_SECONDS", "10"))
    client_timeout = float(
        os.environ.get("AB_SCORING_CLIENT_TIMEOUT_SECONDS", str(timeout_seconds + 900))
    )

    submit_payload = {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name_or_path,
        "dataset_name": dataset,
        "timeout_seconds": timeout_seconds,
    }

    try:
        state = _request_json(
            f"{base}/v1/submit",
            method="POST",
            payload=submit_payload,
            timeout=request_timeout,
        )
        sid = state.get("submission_id")
        if not sid:
            raise RuntimeError(f"scoring server submit returned no submission_id: {state}")

        deadline = time.time() + client_timeout
        while True:
            status = state.get("status")
            if status == "done":
                verdict = dict(state.get("verdict") or {})
                verdict.setdefault("status", "ERROR")
                verdict.setdefault("test_pass_rate", 0.0)
                verdict.setdefault("passed", 0)
                verdict.setdefault("failed", 0)
                verdict["scoring_submission_id"] = sid
                verdict["scoring_url"] = base
                return verdict
            if status == "failed":
                return {
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": f"scoring server failed: {state.get('error')}",
                    "evaluation_dir": None,
                    "scoring_submission_id": sid,
                    "scoring_url": base,
                }
            if status not in ("queued", "running"):
                return {
                    "status": "ERROR",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": f"unexpected scoring server status: {status!r}",
                    "evaluation_dir": None,
                    "scoring_submission_id": sid,
                    "scoring_url": base,
                }
            if time.time() >= deadline:
                return {
                    "status": "TIMEOUT",
                    "test_pass_rate": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "error_message": f"scoring server client timeout after {client_timeout}s",
                    "evaluation_dir": None,
                    "scoring_submission_id": sid,
                    "scoring_url": base,
                }
            time.sleep(poll_s)
            state = _request_json(
                f"{base}/v1/submission/{sid}",
                timeout=request_timeout,
            )
    except Exception as e:
        logger.error("remote SWE-bench scoring failed: %s", e)
        return {
            "status": "ERROR",
            "test_pass_rate": 0.0,
            "passed": 0,
            "failed": 0,
            "error_message": f"remote scoring failed: {e}",
            "evaluation_dir": None,
            "scoring_url": base,
        }


def _singularity_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("SINGULARITYENV_HOME", None)
    return env


def _load_instance_from_file(path: Path, instance_id: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("instances"), list):
        rows = data["instances"]
    elif isinstance(data, dict) and instance_id in data:
        item = data[instance_id]
        return dict(item) if isinstance(item, dict) else None
    elif isinstance(data, dict) and data.get("instance_id") == instance_id:
        return dict(data)
    else:
        rows = []
    for row in rows:
        if isinstance(row, dict) and row.get("instance_id") == instance_id:
            return dict(row)
    return None


def _load_instance_data(
    instance_id: str,
    instance_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if instance_data:
        return dict(instance_data)

    for key in ("AB_BENCHMARK_INSTANCE_DATA", "AB_SCORING_DATASET_PATH"):
        value = os.environ.get(key)
        if not value:
            continue
        found = _load_instance_from_file(Path(value), instance_id)
        if found:
            return found

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "could not load SWE-bench instance metadata from instance_data, "
            "AB_BENCHMARK_INSTANCE_DATA/AB_SCORING_DATASET_PATH, or Hugging Face datasets"
        ) from e

    dataset = os.environ.get("AB_SCORING_DATASET", _DEFAULT_DATASET)
    split = os.environ.get("AB_SCORING_DATASET_SPLIT", "test")
    for row in load_dataset(dataset, split=split):
        if row["instance_id"] == instance_id:
            return dict(row)
    raise RuntimeError(f"instance {instance_id!r} not found in {dataset}:{split}")


def _copy_testbed_from_sif(sif: Path, dst_host: Path, *, timeout: float = 600.0) -> None:
    dst_host.mkdir(parents=True, exist_ok=True)
    producer = subprocess.Popen(
        ["singularity", "exec", str(sif), "tar", "-C", "/testbed", "-cf", "-", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_singularity_env(),
    )
    consumer = subprocess.Popen(
        ["tar", "-C", str(dst_host), "-xf", "-"],
        stdin=producer.stdout,
        stderr=subprocess.PIPE,
    )
    if producer.stdout is not None:
        producer.stdout.close()
    try:
        consumer_err = consumer.communicate(timeout=timeout)[1]
        producer_err = producer.communicate(timeout=10)[1]
    except subprocess.TimeoutExpired:
        consumer.kill()
        producer.kill()
        raise RuntimeError(f"copy /testbed from {sif} timed out after {timeout}s")
    if consumer.returncode != 0:
        raise RuntimeError(
            "host tar extraction failed: "
            f"{(consumer_err or b'').decode(errors='replace')[-500:]}"
        )
    if producer.returncode != 0:
        raise RuntimeError(
            "in-sif tar failed: "
            f"{(producer_err or b'').decode(errors='replace')[-500:]}"
        )


def _run_in_sif(
    sif: Path,
    workspace: Path,
    argv: list[str],
    *,
    extra_binds: tuple[str, ...] = (),
    home: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    cmd = ["singularity", "exec"]
    if os.environ.get("AB_SCORING_WRITABLE_TMPFS", "1") != "0":
        cmd += ["--writable-tmpfs"]
    if home is not None:
        cmd += ["--home", f"{home}:/root"]
    cmd += ["--bind", f"{workspace}:/testbed:rw"]
    for bind in extra_binds:
        cmd += ["--bind", bind]
    cmd += [str(sif), *argv]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_singularity_env(),
    )


def _parse_report(
    report: Dict[str, Any],
    instance_id: str,
    *,
    evaluation_dir: Path,
    report_path: Path,
) -> Dict:
    inst = report.get(instance_id, report)
    resolved = inst.get("resolved", False)
    ts = inst.get("tests_status", {})
    f2p = ts.get("FAIL_TO_PASS", {"success": [], "failure": []})
    p2p = ts.get("PASS_TO_PASS", {"success": [], "failure": []})
    passed = len(f2p.get("success", [])) + len(p2p.get("success", []))
    failed = len(f2p.get("failure", [])) + len(p2p.get("failure", []))
    total = passed + failed
    return {
        "status": "PASS" if resolved else "FAIL",
        "test_pass_rate": passed / total if total else 0.0,
        "passed": passed,
        "failed": failed,
        "evaluation_dir": str(evaluation_dir),
        "raw_report_path": str(report_path),
        "scoring_backend": "singularity",
    }


def score_via_singularity(
    *,
    instance_id: str,
    patch: str,
    model_name_or_path: str,
    timeout_seconds: int,
    run_id: str,
    benchmark_data_root: Path,
    instance_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    """Score one SWE-bench patch with the per-instance SIF on NSCC.

    Returns None when AB_SCORING_IMAGE_DIR is not configured. Once configured,
    failures are reported as verdicts instead of falling through to Docker,
    because NSCC compute nodes do not have Docker.
    """
    sif_dir = os.environ.get("AB_SCORING_IMAGE_DIR")
    if not sif_dir:
        return None
    if shutil.which("singularity") is None:
        return _error_verdict(
            "AB_SCORING_IMAGE_DIR is set but singularity is not on PATH",
            backend="singularity",
        )

    sif = Path(sif_dir) / f"{instance_id}.sif"
    if not sif.is_file():
        return _error_verdict(
            f"scoring SIF not found: {sif}",
            backend="singularity",
        )

    try:
        from swebench.harness.constants import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTION,
            LOG_INSTANCE,
            LOG_REPORT,
            LOG_TEST_OUTPUT,
            RUN_EVALUATION_LOG_DIR,
        )
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    except Exception as e:
        return _error_verdict(
            f"swebench scoring imports failed: {e}",
            backend="singularity",
        )

    model_dir = model_name_or_path.replace("/", "__")
    eval_dir = (
        Path(benchmark_data_root)
        / RUN_EVALUATION_LOG_DIR
        / run_id
        / model_dir
        / instance_id
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / LOG_REPORT
    instance_log_path = eval_dir / LOG_INSTANCE
    test_output_path = eval_dir / LOG_TEST_OUTPUT

    try:
        inst = _load_instance_data(instance_id, instance_data)
        test_spec = make_test_spec(inst, namespace=None)
    except Exception as e:
        return _error_verdict(
            f"failed to load SWE-bench test spec: {e}",
            evaluation_dir=eval_dir,
            backend="singularity",
        )

    pred = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: model_name_or_path,
        KEY_PREDICTION: patch,
    }

    log_lines: list[str] = [
        f"[agent-bench] scoring_backend=singularity",
        f"[agent-bench] instance_id={instance_id}",
        f"[agent-bench] sif={sif}",
    ]

    with tempfile.TemporaryDirectory(prefix=f"ab_score_{instance_id}_") as tmp:
        tmp_path = Path(tmp)
        workspace = tmp_path / "testbed"
        home = tmp_path / "home"
        home.mkdir()
        patch_path = eval_dir / "patch.diff"
        eval_script_path = eval_dir / "eval.sh"
        patch_path.write_text(patch or "")
        eval_script_path.write_text(test_spec.eval_script)

        try:
            _copy_testbed_from_sif(sif, workspace)
        except Exception as e:
            return _error_verdict(
                f"failed to copy /testbed from scoring SIF: {e}",
                evaluation_dir=eval_dir,
                backend="singularity",
            )

        patch_bind = f"{patch_path}:/tmp/patch.diff:ro"
        applied_patch = False
        last_apply_output = ""
        for apply_cmd in _GIT_APPLY_CMDS:
            proc = _run_in_sif(
                sif,
                workspace,
                ["bash", "-lc", f"cd /testbed && {apply_cmd} /tmp/patch.diff"],
                extra_binds=(patch_bind,),
                home=home,
                timeout=120,
            )
            last_apply_output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                log_lines.append(f"{APPLY_PATCH_PASS} via {apply_cmd}")
                log_lines.append(last_apply_output)
                applied_patch = True
                break
            log_lines.append(f"[agent-bench] patch apply failed via {apply_cmd}")
            log_lines.append(last_apply_output)

        if not applied_patch:
            log_lines.append(f"{APPLY_PATCH_FAIL}:\n{last_apply_output}")
            instance_log_path.write_text("\n".join(log_lines))
            return _error_verdict(
                f"{APPLY_PATCH_FAIL}: {last_apply_output[-500:]}",
                evaluation_dir=eval_dir,
                backend="singularity",
            )

        before = _run_in_sif(
            sif,
            workspace,
            ["git", "-C", "/testbed", "-c", "core.fileMode=false", "diff"],
            home=home,
            timeout=120,
        )
        log_lines.append("[agent-bench] git diff before:")
        log_lines.append(before.stdout or before.stderr or "")

        eval_bind = f"{eval_script_path}:/eval.sh:ro"
        try:
            proc = _run_in_sif(
                sif,
                workspace,
                ["/bin/bash", "/eval.sh"],
                extra_binds=(eval_bind,),
                home=home,
                timeout=timeout_seconds,
            )
            test_output = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as e:
            test_output = ""
            if e.stdout:
                test_output += (
                    e.stdout
                    if isinstance(e.stdout, str)
                    else e.stdout.decode(errors="replace")
                )
            if e.stderr:
                test_output += (
                    e.stderr
                    if isinstance(e.stderr, str)
                    else e.stderr.decode(errors="replace")
                )
            test_output += f"\n\nTimeout error: {timeout_seconds} seconds exceeded."
            test_output_path.write_text(test_output)
            log_lines.append(test_output)
            instance_log_path.write_text("\n".join(log_lines))
            return _error_verdict(
                f"Singularity scoring timeout after {timeout_seconds}s",
                evaluation_dir=eval_dir,
                backend="singularity",
                status="TIMEOUT",
            )

        test_output_path.write_text(test_output)
        log_lines.append(f"[agent-bench] eval rc={proc.returncode}")
        log_lines.append(test_output)

        after = _run_in_sif(
            sif,
            workspace,
            ["git", "-C", "/testbed", "-c", "core.fileMode=false", "diff"],
            home=home,
            timeout=120,
        )
        log_lines.append("[agent-bench] git diff after:")
        log_lines.append(after.stdout or after.stderr or "")
        instance_log_path.write_text("\n".join(log_lines))

    try:
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        report_path.write_text(json.dumps(report, indent=4))
        verdict = _parse_report(
            report,
            instance_id,
            evaluation_dir=eval_dir,
            report_path=report_path,
        )
        verdict["scoring_sif"] = str(sif)
        return verdict
    except Exception as e:
        return _error_verdict(
            f"failed to grade Singularity scoring output: {e}",
            evaluation_dir=eval_dir,
            backend="singularity",
        )
