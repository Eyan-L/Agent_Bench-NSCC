"""OpenHands agent condition.

Runtime contract: the harness (SWEBenchOpenHandsHarness) has prepared an
OpenHands container (oh-<rid>) with the driver baked at
/opt/openhands_driver.py and OPENHANDS_EXEC_BRIDGE_* env set. We invoke
the driver via the injected `Container` (docker exec / singularity exec
depending on backend), wait for it to finish, then parse the
events.jsonl it persisted to <state_dir>/events.jsonl (host-visible).

The events follow openhands-sdk 1.21 schema:
    {"kind": "ActionEvent",      source: "agent",       tool_name, tool_call_id, ...}
    {"kind": "ObservationEvent", source: "environment", tool_name, tool_call_id, ...}
    {"driver_event": "summary",  metrics: {prompt_tokens, completion_tokens, ...}}

The driver emits `agent.model.usage` and `agent.tool.call` spans to the shared
collector. This condition slices collector output by `oc.run_id`, then writes one
coarse LLM_INFER step row in <run_dir>/steps.jsonl and the candidate.patch
(collected by the harness).

Returns the dict shape the driver expects.
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_bench.benchmarks.conditions.agent_otel import (
    optional_int,
    otel_trace_offset,
    snapshot_otel_spans,
    wait_for_otel_flush,
)
from agent_bench.benchmarks.conditions.util_sampler import UtilSampler
from agent_bench.deployments.containers.base import Container, ContainerHandle

logger = logging.getLogger(__name__)


def _build_message(issue_text: str) -> str:
    return (
        "You are fixing a bug. The repository is in the current working "
        "directory. Read the issue, modify the source files to fix it, then "
        "stop.\n\n"
        f"Issue:\n{issue_text}"
    )


def _iso_to_unix_ns(t: Any) -> int:
    if t is None:
        return 0
    if isinstance(t, (int, float)):
        f = float(t)
        if f > 1e15:
            return int(f)
        return int(f * (1e6 if f > 1e12 else 1e9))
    if isinstance(t, str):
        try:
            iso = t.replace("Z", "+00:00")
            return int(datetime.fromisoformat(iso).timestamp() * 1e9)
        except ValueError:
            return 0
    return 0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ns_to_iso(ns: int) -> Optional[str]:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


class OpenHandsAgentCondition:
    """Run the OpenHands SDK driver against a SWE-bench task inside the container."""

    def __init__(self, model_config, agent_config, container: Optional[Container] = None):
        self.model_config = model_config
        self.agent_config = agent_config
        self.container = container

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id: Optional[str],
        env_info: Optional[Dict] = None,
    ) -> Dict:
        env_info = env_info or {}
        instance_id = instance_data["instance_id"]
        logger.info("Running OpenHands agent for %s", instance_id)

        if not container_id:
            return self._failure(run_dir, instance_id,
                                 "OpenHands condition requires a prepared container_id")

        config_path_in = env_info.get("config_path_in_container")
        events_path_in = env_info.get("events_path_in_container")
        events_path_host = env_info.get("events_path_on_host")
        if not (config_path_in and events_path_in and events_path_host):
            return self._failure(run_dir, instance_id,
                                 "env_info missing runtime config and events paths")

        message = self._build_message(instance_data)
        inner = (
            f"python /opt/openhands_driver.py "
            f"--config {shlex.quote(config_path_in)} "
            f"--task {shlex.quote(message)} "
            f"--events {shlex.quote(events_path_in)} "
            f"--workspace {shlex.quote(workspace_root)}"
        )
        timeout = getattr(self.agent_config, "max_time_seconds", 1800)
        otel_trace_file, pre_offset = otel_trace_offset()
        sampler_cm = UtilSampler(run_dir, gateway_pid=optional_int(env_info.get("bridge_pid")))
        sampler_cm.__enter__()
        step_start_ns = time.time_ns()
        try:
            try:
                if self.container is not None:
                    handle = ContainerHandle(
                        id=container_id, name=container_id,
                        backend=self.container.backend_id,
                    )
                    proc = self.container.exec(
                        handle, ["bash", "-lc", inner],
                        check=False, timeout=timeout + 120,
                    )
                else:
                    # Legacy fallback when no container injected.
                    argv = ["docker", "exec", container_id, "bash", "-lc", inner]
                    proc = subprocess.run(
                        argv, capture_output=True, text=True, timeout=timeout + 120,
                    )
                rc = proc.returncode
                stdout_tail = (proc.stdout or "")[-2000:]
                stderr_tail = (proc.stderr or "")[-2000:]
            except subprocess.TimeoutExpired as e:
                rc = -1
                stdout_tail = (e.stdout or b"").decode("utf-8", errors="replace")[-2000:] if e.stdout else ""
                stderr_tail = (e.stderr or b"").decode("utf-8", errors="replace")[-2000:] if e.stderr else ""
        finally:
            step_end_ns = time.time_ns()
            sampler_cm.__exit__(None, None, None)

        # 2. Parse events.jsonl for token totals; spans come from the collector.
        events = self._load_events(Path(events_path_host))
        run_id = run_dir.name
        wait_for_otel_flush()
        spans_path = snapshot_otel_spans(
            otel_trace_file, pre_offset, run_dir, expected_run_id=run_id,
        )
        if spans_path is None:
            spans_path = run_dir / "spans.jsonl"
            spans_path.touch()
            (run_dir / "spans_raw.jsonl").touch()
        logger.info("  sliced OpenHands OTel -> %s", spans_path.name)

        prompt_tokens, completion_tokens = self._aggregate_tokens(events)

        # 3. Single coarse LLM_INFER row (mirrors openclaw_agent).
        step = {
            "trajectory_id": run_id,
            "attempt_id": 1,
            "step_id": 1,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": "LLM_INFER_OPENHANDS_AGENTIC_COARSE",
            "tool_name": "openhands.agent",
            "command": "python /opt/openhands_driver.py ...",
            "file_path": None,
            "start_ts": step_start_ns,
            "end_ts": step_end_ns,
            "duration_ms": (step_end_ns - step_start_ns) / 1e6,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "exit_code": rc,
            "error_type": None if rc == 0 else f"openhands_exit_{rc}",
            "repo_id": instance_data.get("repo"),
            "instance_id": instance_id,
            "config_id": "openhands_agent",
            "model_name": self.model_config.name,
            "response_preview": stdout_tail[-500:] if stdout_tail else None,
            # Provenance for cross-harness comparison.
            "harness": "openhands",
            "agent_server_location": "openhands_harness_container",
            "workspace_mount": "bind_from_host",
            "bridge_active": bool(env_info.get("bridge_socket")),
        }
        step_path = run_dir / "steps.jsonl"
        step_path.write_text(json.dumps(step) + "\n")

        if stderr_tail:
            (run_dir / "openhands_stderr.log").write_text(stderr_tail)
        if stdout_tail:
            (run_dir / "openhands_stdout.log").write_text(stdout_tail)
        if Path(events_path_host).exists():
            try:
                shutil.copy2(events_path_host, run_dir / "openhands_events.jsonl")
            except OSError:
                pass

        # 4. Benchmark-specific artifact: SWE-bench writes a git patch;
        # Terminal-Bench records a marker because work lives in the task container.
        patch_text = self._collect_artifacts(env_info, container_id, workspace_root)
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text(patch_text)

        return {
            "step_trace_path": str(step_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "telemetry_path": str(spans_path),
        }

    def _build_message(self, instance_data: Dict) -> str:
        return _build_message(instance_data["problem_statement"])

    def _collect_artifacts(
        self,
        env_info: Dict,
        container_id: str,
        workspace_root: str,
    ) -> str:
        harness = env_info.get("_harness")
        if harness is None:
            return ""
        try:
            return harness.collect_patch(container_id, workspace_root)
        except Exception as e:
            logger.warning("collect_patch failed: %s", e)
            return ""

    @staticmethod
    def _resource_attrs(run_id: str, instance_id: str | None = None) -> Dict[str, Any]:
        resource: Dict[str, Any] = {
            "oc.run_id": run_id,
            "oh.run_id": run_id,
            "agent.run_id": run_id,
            "service.name": "agent-bench-openhands",
        }
        if instance_id:
            resource["instance_id"] = instance_id
        return resource

    # ------------------------------------------------------------------
    # Event parsing + span synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _load_events(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        events: List[Dict] = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("sdk_event"), dict):
                events.append(obj["sdk_event"])
            elif isinstance(obj, dict):
                events.append(obj)
        return events

    @staticmethod
    def _extract_summary_metrics(events: List[Dict]) -> Dict[str, Any]:
        for evt in reversed(events):
            if isinstance(evt, dict) and evt.get("driver_event") == "summary":
                m = evt.get("metrics") or {}
                if isinstance(m, dict):
                    return m
        return {}

    @classmethod
    def _aggregate_tokens(cls, events: List[Dict]) -> Tuple[int, int]:
        m = cls._extract_summary_metrics(events)
        return (
            int(m.get("prompt_tokens") or 0),
            int(m.get("completion_tokens") or 0),
        )

    def _build_spans(
        self, events: List[Dict], run_id: str,
        fallback_start_ns: int, fallback_end_ns: int,
    ) -> List[Dict]:
        usage_events = self._model_usage_events(events)
        if usage_events:
            spans: List[Dict] = [
                self._model_usage_event_to_span(
                    evt, run_id, fallback_start_ns, fallback_end_ns, idx,
                )
                for idx, evt in enumerate(usage_events, start=1)
            ]
        else:
            m = self._extract_summary_metrics(events)
            prompt = _as_int(m.get("prompt_tokens"))
            completion = _as_int(m.get("completion_tokens"))
            cache_read = _as_int(m.get("cache_read_tokens"))
            cache_write = _as_int(m.get("cache_creation_tokens"))
            total = _as_int(m.get("total_tokens")) or (prompt + completion)
            spans = [{
                "name": "agent.model.usage",
                "canonical_name": "agent.model.usage",
                "start_unix_nano": fallback_start_ns,
                "end_unix_nano": fallback_end_ns,
                "start_iso": _ns_to_iso(fallback_start_ns),
                "end_iso": _ns_to_iso(fallback_end_ns),
                "duration_ms": (fallback_end_ns - fallback_start_ns) / 1e6,
                "attrs": {
                    "agent.channel": "synth.agent_call",
                    "agent.provider": "openhands",
                    "agent.model": self.model_config.name,
                    "agent.run_id": run_id,
                    "agent.llm.request_index": 1,
                    "agent.tokens.input": prompt,
                    "agent.tokens.output": completion,
                    "agent.tokens.cache_read": cache_read,
                    "agent.tokens.cache_write": cache_write,
                    "agent.tokens.total": total,
                },
                "resource": self._resource_attrs(run_id),
                "synthesized": True,
            }]

        # Tool-call pairs by tool_call_id.
        pending: Dict[str, Dict] = {}
        for evt in events:
            if not isinstance(evt, dict):
                continue
            kind = evt.get("kind") or ""
            cid = evt.get("tool_call_id")
            if kind == "ActionEvent" and cid:
                pending[cid] = {
                    "tool_name": evt.get("tool_name") or "unknown",
                    "ts_ns": _iso_to_unix_ns(evt.get("timestamp")),
                }
            elif kind == "ObservationEvent" and cid:
                info = pending.pop(cid, None)
                if info is None:
                    continue
                end_ns = _iso_to_unix_ns(evt.get("timestamp")) or info["ts_ns"]
                start_ns = info["ts_ns"] or end_ns
                duration_ms = max((end_ns - start_ns) / 1e6, 0.0)
                is_error = bool(
                    evt.get("error") or evt.get("is_error")
                    or (evt.get("extras") or {}).get("error")
                )
                spans.append({
                    "name": "agent.tool.call",
                    "canonical_name": "agent.tool.call",
                    "start_unix_nano": start_ns,
                    "end_unix_nano": end_ns,
                    "start_iso": _ns_to_iso(start_ns),
                    "end_iso": _ns_to_iso(end_ns),
                    "duration_ms": duration_ms,
                    "attrs": {
                        "agent.tool.name": info["tool_name"],
                        "agent.tool.call_id": cid,
                        "agent.tool.is_error": is_error,
                        "agent.run_id": run_id,
                    },
                    "resource": self._resource_attrs(run_id),
                    "synthesized": True,
                })
        return spans

    @staticmethod
    def _model_usage_events(events: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for evt in events:
            if not isinstance(evt, dict):
                continue
            if evt.get("type") == "model.usage" or evt.get("driver_event") == "model_usage":
                out.append(evt)
        return out

    def _model_usage_event_to_span(
        self,
        evt: Dict,
        run_id: str,
        fallback_start_ns: int,
        fallback_end_ns: int,
        request_index: int,
    ) -> Dict:
        start_ns = (
            _as_int(evt.get("start_ns"))
            or _iso_to_unix_ns(evt.get("start_ts"))
            or _iso_to_unix_ns(evt.get("timestamp"))
            or _iso_to_unix_ns(evt.get("ts"))
            or fallback_start_ns
        )
        end_ns = (
            _as_int(evt.get("end_ns"))
            or _iso_to_unix_ns(evt.get("end_ts"))
            or fallback_end_ns
        )
        try:
            duration_ms = float(evt.get("duration_ms"))
        except (TypeError, ValueError):
            duration_ms = max((end_ns - start_ns) / 1e6, 0.0)
        request_index = _as_int(evt.get("request_index")) or request_index
        prompt = _as_int(evt.get("input_tokens") or evt.get("prompt_tokens"))
        completion = _as_int(evt.get("output_tokens") or evt.get("completion_tokens"))
        cache_read = _as_int(evt.get("cache_read_tokens") or evt.get("cached_tokens"))
        cache_write = _as_int(evt.get("cache_write_tokens") or evt.get("cache_creation_tokens"))
        total = _as_int(evt.get("total_tokens")) or (prompt + completion)
        attrs = {
            "agent.channel": evt.get("channel") or "openhands.events",
            "agent.provider": evt.get("provider") or "openhands",
            "agent.model": evt.get("model") or self.model_config.name,
            "agent.api": evt.get("api"),
            "agent.session_id": evt.get("session_id") or evt.get("sessionId") or run_id,
            "agent.run_id": evt.get("run_id") or evt.get("runId") or run_id,
            "agent.llm.request_index": request_index,
            "agent.llm.requestIndex": request_index,
            "agent.http.status_code": evt.get("http_status") or evt.get("httpStatus"),
            "agent.http.statusCode": evt.get("httpStatus") or evt.get("http_status"),
            "agent.tokens.input": prompt,
            "agent.tokens.output": completion,
            "agent.tokens.cache_read": cache_read,
            "agent.tokens.cache_write": cache_write,
            "agent.tokens.total": total,
            "agent.ttft_ms": evt.get("ttft_ms"),
            "agent.tpot_ms": evt.get("tpot_ms") or evt.get("tbt_avg_ms"),
            "agent.headers_ms": evt.get("headers_ms"),
            "agent.chunks.count": evt.get("chunks_count"),
            "agent.api_calls": _as_int(evt.get("api_calls")),
        }
        return {
            "name": "agent.model.usage",
            "canonical_name": "agent.model.usage",
            "start_unix_nano": start_ns,
            "end_unix_nano": end_ns,
            "start_iso": _ns_to_iso(start_ns),
            "end_iso": _ns_to_iso(end_ns),
            "duration_ms": duration_ms,
            "attrs": attrs,
            "resource": self._resource_attrs(run_id),
            "synthesized": False,
        }

    # ------------------------------------------------------------------

    def _failure(self, run_dir: Path, instance_id: str, msg: str) -> Dict:
        logger.error("OpenHands precondition failed for %s: %s", instance_id, msg)
        step_path = run_dir / "steps.jsonl"
        step_path.write_text(json.dumps({
            "trajectory_id": run_dir.name,
            "step_type": "LLM_INFER",
            "step_subtype": "LLM_INFER_OPENHANDS_AGENTIC_COARSE",
            "tool_name": "openhands.agent",
            "error_type": "precondition_failed",
            "instance_id": instance_id,
            "config_id": "openhands_agent",
            "model_name": self.model_config.name,
            "response_preview": msg[:500],
        }) + "\n")
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text("")
        return {
            "step_trace_path": str(step_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "telemetry_path": None,
        }
