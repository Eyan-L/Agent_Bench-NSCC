"""Self-hosted SWE-bench scoring server.

Sits on a docker-equipped host. Accepts predictions from clients that
don't have docker locally (typically remote/cluster runs), runs
`python -m swebench.harness.run_evaluation`, returns the parsed verdict.

Endpoints:
    POST /v1/submit                  body: SubmitRequest -> {submission_id}
    GET  /v1/submission/{id}         -> SubmissionState
    GET  /healthz                    -> {ok: true}

Concurrency: a single asyncio worker per process drains a FIFO queue and
runs one swebench evaluation at a time. Multiple submissions are accepted
immediately and processed in order. (run_evaluation has its own
--max_workers; serializing at the queue level keeps docker pressure
predictable.)

Run:
    python -m orchestration.scoring_server.main \\
        --swebench-root /path/to/SWE-bench \\
        --host 0.0.0.0 --port 8765 \\
        [--dataset princeton-nlp/SWE-bench_Verified]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    raise SystemExit(
        f"missing dep ({e}). Install: pip install fastapi uvicorn pydantic"
    )

logger = logging.getLogger("scoring_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------- API


class SubmitRequest(BaseModel):
    instance_id: str
    model_patch: str
    model_name_or_path: str = "oc-eval"
    dataset_name: Optional[str] = None  # falls back to server default
    timeout_seconds: int = 1800


class SubmissionState(BaseModel):
    submission_id: str
    instance_id: str
    status: str  # queued | running | done | failed
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    verdict: Optional[Dict] = None  # parsed report fields when done
    error: Optional[str] = None


# -------------------------------------------------------------- worker state


@dataclass
class _Job:
    submission_id: str
    request: SubmitRequest
    state: SubmissionState


@dataclass
class _Server:
    swebench_root: Path
    default_dataset: str
    queue: "asyncio.Queue[_Job]" = field(default_factory=asyncio.Queue)
    submissions: Dict[str, _Job] = field(default_factory=dict)
    submissions_lock: threading.Lock = field(default_factory=threading.Lock)


SERVER: Optional[_Server] = None


def _run_evaluation(job: _Job, swebench_root: Path, dataset: str) -> None:
    """Run swebench.harness.run_evaluation for one instance, parse report."""
    job.state.status = "running"
    job.state.started_at = time.time()
    req = job.request

    with tempfile.TemporaryDirectory(prefix="oc_score_") as tmp:
        tmp = Path(tmp)
        pred_path = tmp / "predictions.json"
        pred_path.write_text(
            json.dumps([{
                "instance_id": req.instance_id,
                "model_name_or_path": req.model_name_or_path,
                "model_patch": req.model_patch,
            }])
        )

        run_id = job.submission_id
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", req.dataset_name or dataset,
            "--predictions_path", str(pred_path),
            "--instance_ids", req.instance_id,
            "--max_workers", "1",
            "--run_id", run_id,
            "--cache_level", "env",
            "--timeout", str(req.timeout_seconds),
        ]

        logger.info("[%s] running: %s", req.instance_id, " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, cwd=str(swebench_root),
                capture_output=True, text=True,
                timeout=req.timeout_seconds + 600,
            )
        except subprocess.TimeoutExpired:
            job.state.status = "failed"
            job.state.error = "wall-clock timeout from server"
            job.state.finished_at = time.time()
            return

        # Locate report even if run_evaluation exits non-zero. SWE-bench can
        # write a valid report and then fail during container teardown.
        candidates = [
            swebench_root / "logs" / "run_evaluation" / run_id /
            req.model_name_or_path / req.instance_id / "report.json",
            swebench_root / "evaluation_results" / run_id /
            req.instance_id / "report.json",
        ]
        report_path = next((p for p in candidates if p.exists()), None)
        if report_path is None:
            # Fallback rglob.
            for root in (
                swebench_root / "logs" / "run_evaluation" / run_id,
                swebench_root / "evaluation_results" / run_id,
            ):
                if root.exists():
                    matches = list(root.rglob("report.json"))
                    if matches:
                        report_path = matches[0]
                        break

        if proc.returncode != 0 and report_path is None:
            job.state.status = "failed"
            job.state.error = (
                f"run_evaluation exit {proc.returncode}: {proc.stderr[-800:]}"
            )
            job.state.finished_at = time.time()
            return

        if report_path is None:
            job.state.status = "failed"
            job.state.error = f"report.json not found for {req.instance_id}"
            job.state.finished_at = time.time()
            return

        try:
            report = json.loads(report_path.read_text())
            inst = report.get(req.instance_id, report)
            resolved = inst.get("resolved", False)
            ts = inst.get("tests_status", {})
            f2p = ts.get("FAIL_TO_PASS", {"success": [], "failure": []})
            p2p = ts.get("PASS_TO_PASS", {"success": [], "failure": []})
            passed = len(f2p.get("success", [])) + len(p2p.get("success", []))
            failed = len(f2p.get("failure", [])) + len(p2p.get("failure", []))
            total = passed + failed
            job.state.verdict = {
                "status": "PASS" if resolved else "FAIL",
                "test_pass_rate": passed / total if total else 0.0,
                "passed": passed,
                "failed": failed,
                "evaluation_dir": str(report_path.parent),
                "raw_report_path": str(report_path),
            }
            job.state.status = "done"
        except Exception as e:
            job.state.status = "failed"
            job.state.error = f"failed to parse report: {e}"
        finally:
            job.state.finished_at = time.time()


async def _worker_loop():
    assert SERVER is not None
    loop = asyncio.get_running_loop()
    while True:
        job = await SERVER.queue.get()
        try:
            await loop.run_in_executor(
                None, _run_evaluation, job,
                SERVER.swebench_root, SERVER.default_dataset,
            )
        except Exception as e:
            logger.exception("worker error on %s", job.submission_id)
            job.state.status = "failed"
            job.state.error = f"worker exception: {e}"
            job.state.finished_at = time.time()
        finally:
            SERVER.queue.task_done()


# ------------------------------------------------------------------ FastAPI

app = FastAPI(title="oc-scoring-server", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "queue_depth": SERVER.queue.qsize() if SERVER else 0}


@app.post("/v1/submit", response_model=SubmissionState)
async def submit(req: SubmitRequest):
    if SERVER is None:
        raise HTTPException(500, "server not initialized")
    sid = uuid.uuid4().hex
    state = SubmissionState(
        submission_id=sid,
        instance_id=req.instance_id,
        status="queued",
        submitted_at=time.time(),
    )
    job = _Job(submission_id=sid, request=req, state=state)
    with SERVER.submissions_lock:
        SERVER.submissions[sid] = job
    await SERVER.queue.put(job)
    return state


@app.get("/v1/submission/{sid}", response_model=SubmissionState)
async def get_submission(sid: str):
    if SERVER is None:
        raise HTTPException(500, "server not initialized")
    with SERVER.submissions_lock:
        job = SERVER.submissions.get(sid)
    if job is None:
        raise HTTPException(404, f"unknown submission {sid}")
    return job.state


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_worker_loop())


# ---------------------------------------------------------------- entrypoint


def main():
    p = argparse.ArgumentParser(description="oc-scoring-server")
    p.add_argument("--swebench-root", required=True, type=Path)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified",
                   help="default dataset_name when client omits it")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    if not args.swebench_root.exists():
        raise SystemExit(f"swebench-root not found: {args.swebench_root}")

    global SERVER
    SERVER = _Server(
        swebench_root=args.swebench_root.resolve(),
        default_dataset=args.dataset,
    )
    logger.info(
        "scoring server starting on %s:%d (swebench_root=%s, default_dataset=%s)",
        args.host, args.port, SERVER.swebench_root, SERVER.default_dataset,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
