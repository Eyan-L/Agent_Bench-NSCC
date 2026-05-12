"""Sanity checks for the orchestration/trace/ instrumentation layer.

Covers:
  1. Synthetic step durations roll up within tolerance (StepLogger + JSONL).
  2. TTFT matches external stopwatch within tolerance (LLMWrapper + fake stream).
  3. Telemetry coverage >= configured minimum of step wall time
     (ResourceCollector + align_telemetry_to_steps).
  4. Forced crash (SIGTERM mid-trajectory) leaves a recoverable JSONL with
     a final manifest line carrying partial=true.

Thresholds load from config/stress/trace_thresholds.yaml; override path via
TRACE_THRESHOLDS_FILE env var.

Run: python -m tests.trace.test_trace_pipeline
Exit code 0 on pass, non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from orchestration.trace.schema import StepEvent, TelemetrySample
from orchestration.trace.logger import StepLogger
from orchestration.trace.writers import JSONLWriter
from orchestration.trace.telemetry import ResourceCollector
from orchestration.trace.llm_wrapper import LLMWrapper
from orchestration.trace.align import align_telemetry_to_steps


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THRESHOLDS = _REPO_ROOT / "config" / "stress" / "trace_thresholds.yaml"


def load_thresholds() -> dict:
    path = Path(os.environ.get("TRACE_THRESHOLDS_FILE", str(_DEFAULT_THRESHOLDS)))
    with path.open() as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (manifests, step_rows). Manifest entries are tagged _type=manifest."""
    manifests, steps = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("_type") == "manifest":
            manifests.append(row)
        else:
            steps.append(row)
    return manifests, steps


# -----------------------------------------------------------------------------
# 1. Rollup accuracy
# -----------------------------------------------------------------------------

def check_rollup(tmp: Path, cfg: dict) -> None:
    path = tmp / "steps.jsonl"
    durations = cfg["rollup_check"]["step_durations_s"]
    expected_total_ms = cfg["rollup_check"]["expected_total_ms"]
    tol = cfg["rollup_tolerance_ms"]

    logger = StepLogger(
        output_path=str(path),
        trajectory_id="t1", instance_id="demo", repo_id="demo",
        config_id="test", model_name="fake",
    )
    try:
        for d in durations:
            with logger.step(step_type="TOOL_EXEC", tool_name="sleep"):
                time.sleep(d)
    finally:
        logger.close()

    _, rows = _read_jsonl(path)
    assert len(rows) == len(durations), f"expected {len(durations)} rows, got {len(rows)}"
    measured = [r["duration_ms"] for r in rows]
    for d, m in zip(durations, measured):
        assert abs(m - d * 1000.0) < tol, f"step duration off: expected ~{d*1000}, got {m}"
    total = sum(measured)
    assert abs(total - expected_total_ms) < tol, f"rollup off: {total}"
    print(f"[PASS] rollup: durations={[f'{m:.2f}' for m in measured]}ms total={total:.2f}ms")


# -----------------------------------------------------------------------------
# 2. TTFT accuracy with fake streaming client
# -----------------------------------------------------------------------------

class _FakeDelta:
    def __init__(self, c: str) -> None:
        self.content = c


class _FakeChoice:
    def __init__(self, c: str) -> None:
        self.delta = _FakeDelta(c)


class _FakeChunk:
    def __init__(self, c: str, usage: Any = None) -> None:
        self.choices = [_FakeChoice(c)] if c else []
        self.usage = usage


class _FakeUsage:
    def __init__(self, pt: int, ct: int) -> None:
        self.prompt_tokens = pt
        self.completion_tokens = ct


class _FakeChatCompletions:
    def __init__(self, first_token_delay_s: float, inter_token_s: float, n_tokens: int) -> None:
        self.first_delay = first_token_delay_s
        self.inter = inter_token_s
        self.n = n_tokens

    def create(self, *, model: str, messages: List[Dict[str, Any]], stream: bool = True, **kw: Any) -> Iterable[Any]:
        if not stream:
            raise NotImplementedError

        def gen() -> Iterable[Any]:
            time.sleep(self.first_delay)
            yield _FakeChunk("hello")
            for _ in range(self.n - 1):
                time.sleep(self.inter)
                yield _FakeChunk(" world")
            yield _FakeChunk("", usage=_FakeUsage(10, self.n))

        return gen()


class _FakeClient:
    def __init__(self, c: _FakeChatCompletions) -> None:
        self.chat = type("X", (), {"completions": c})()


def check_ttft(tmp: Path, cfg: dict) -> None:
    fs = cfg["fake_stream"]
    expected_ttft_ms = fs["first_token_delay_s"] * 1000.0
    tol = cfg["ttft_tolerance_ms"]
    fake = _FakeClient(_FakeChatCompletions(
        first_token_delay_s=fs["first_token_delay_s"],
        inter_token_s=fs["inter_token_s"],
        n_tokens=fs["n_tokens"],
    ))
    wrapper = LLMWrapper(fake)

    ext_start = time.monotonic_ns()
    for _ in wrapper.stream(model="fake", messages=[{"role": "user", "content": "hi"}]):
        pass
    ext_elapsed_ms = (time.monotonic_ns() - ext_start) / 1e6

    metrics = wrapper.last_metrics
    assert metrics is not None, "metrics not captured"
    assert abs(metrics.ttft_ms - expected_ttft_ms) < tol, (
        f"ttft off: {metrics.ttft_ms} vs expected {expected_ttft_ms}"
    )
    assert metrics.token_source == "api"
    assert metrics.completion_tokens == fs["n_tokens"]
    print(f"[PASS] ttft: {metrics.ttft_ms:.2f}ms (ext~{ext_elapsed_ms:.2f}ms) tpot={metrics.tpot_ms:.2f}ms")


# -----------------------------------------------------------------------------
# 3. Telemetry coverage
# -----------------------------------------------------------------------------

def check_coverage(tmp: Path, cfg: dict) -> None:
    steps_path = tmp / "steps_cov.jsonl"
    cc = cfg["coverage_check"]
    cov_min = cfg["telemetry_coverage_min"]

    samples: List[TelemetrySample] = []
    collector = ResourceCollector(
        trajectory_id="t3",
        sample_callback=lambda s: samples.append(s),
        poll_interval_ms=int(cc["collector_interval_s"] * 1000),
    )

    logger = StepLogger(
        output_path=str(steps_path),
        trajectory_id="t3", instance_id="demo", repo_id="demo",
        config_id="test", model_name="fake",
    )
    try:
        collector.start()
        with logger.step(step_type="TOOL_EXEC", tool_name="sleep"):
            time.sleep(cc["step_duration_s"])
        collector.stop()
    finally:
        logger.close()

    _, rows = _read_jsonl(steps_path)
    events = [
        StepEvent(
            trajectory_id=r["trajectory_id"], attempt_id=r["attempt_id"],
            step_id=r["step_id"], parent_step_id=r.get("parent_step_id"),
            step_type=r["step_type"], tool_name=r.get("tool_name"),
            start_ts_mono=r["start_ts_mono"], end_ts_mono=r["end_ts_mono"],
            start_ts_wall=r["start_ts_wall"], end_ts_wall=r["end_ts_wall"],
            duration_ms=r["duration_ms"],
        )
        for r in rows
    ]
    profiles = align_telemetry_to_steps(events, samples)
    assert profiles, "no profiles produced"

    # Coverage = (sample_count * interval) / duration, clamped to [0, 1].
    interval_ms = cc["collector_interval_s"] * 1000.0
    covered_ms = sum(p.sample_count * interval_ms for p in profiles)
    total_ms = sum(p.duration_ms for p in profiles)
    cov = min(covered_ms / total_ms, 1.0) if total_ms else 0.0
    assert cov >= cov_min, f"coverage too low: {cov:.3f} < {cov_min}"
    print(f"[PASS] coverage: {cov:.3f} ({len(samples)} samples over {total_ms:.0f}ms)")


# -----------------------------------------------------------------------------
# 4. Crash recovery
# -----------------------------------------------------------------------------

_CRASH_SCRIPT = """
import os, sys, time, signal
sys.path.insert(0, {repo_root!r})
from orchestration.trace.logger import StepLogger

logger = StepLogger(
    output_path={path!r},
    trajectory_id="t4", instance_id="demo", repo_id="demo",
    config_id="test", model_name="fake",
)

def _fin(*a):
    logger.close(partial=True, error_message="SIGTERM mid-trajectory")
    os._exit(1)

signal.signal(signal.SIGTERM, _fin)

logger.start_step(step_type="TOOL_EXEC", tool_name="long")
time.sleep({pre_crash})
os.kill(os.getpid(), signal.SIGTERM)
time.sleep(5)
"""


def check_crash(tmp: Path, cfg: dict) -> None:
    path = tmp / "steps_crash.jsonl"
    repo_root = str(_REPO_ROOT)
    pre_crash = cfg["crash_check"]["pre_crash_sleep_s"]
    timeout = cfg["crash_recovery_timeout_s"]
    script = _CRASH_SCRIPT.format(repo_root=repo_root, path=str(path), pre_crash=pre_crash)
    subprocess.run([sys.executable, "-c", script], timeout=timeout)
    assert path.exists(), "no JSONL after crash"
    manifests, _ = _read_jsonl(path)
    assert manifests, "no manifest line in JSONL"
    last = manifests[-1]
    assert last.get("partial") is True, f"final manifest not marked partial: {last}"
    print(f"[PASS] crash recovery: {len(manifests)} manifest line(s); final partial=True")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> int:
    cfg = load_thresholds()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        try:
            check_rollup(tmp, cfg)
            check_ttft(tmp, cfg)
            check_coverage(tmp, cfg)
            check_crash(tmp, cfg)
        except AssertionError as e:
            print(f"[FAIL] {e}")
            return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
