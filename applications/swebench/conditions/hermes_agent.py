"""Hermes Agent condition for SWE-bench.

The harness prepares an `hm-*` container with the patched Hermes CLI and an
Agent_Bench bridge. This condition invokes Hermes in oneshot mode, captures
stdout/stderr, writes the standard artifacts, and asks the harness to collect
the final git diff from the shared workspace.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_message(issue_text: str) -> str:
    return (
        "You are fixing a bug. The repository is in the current working "
        "directory. Read the issue, modify the source files to fix it, then "
        "reply with the single word DONE.\n\n"
        f"Issue:\n{issue_text}"
    )


def _ns_to_iso(ns: int) -> Optional[str]:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def _resolve_provider() -> str:
    provider = os.environ.get("HERMES_INFERENCE_PROVIDER")
    if provider:
        return provider
    harness_path = os.environ.get("OC_HARNESS_YAML")
    if harness_path:
        try:
            with open(harness_path) as fh:
                cfg = json.load(fh) or {}
            provider = ((cfg.get("model") or {}).get("provider") or "").strip()
            if provider:
                return provider
        except Exception:
            logger.debug("Could not read Hermes provider from %s", harness_path, exc_info=True)
    return "custom"


class HermesAgentCondition:
    """Run Hermes Agent against a prepared SWE-bench task container."""

    def __init__(self, model_config, agent_config):
        self.model_config = model_config
        self.agent_config = agent_config

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
        logger.info("Running Hermes agent for %s", instance_id)

        if not container_id:
            return self._failure(run_dir, instance_id, "Hermes condition requires container_id")

        message = _build_message(instance_data["problem_statement"])
        model_name = getattr(self.model_config, "name", None) or "Qwen/Qwen3.6-27B"
        provider = _resolve_provider()
        timeout = getattr(self.agent_config, "max_time_seconds", 1800)
        argv = [
            "hermes",
            "--ignore-rules",
            "--yolo",
            "--model", model_name,
            "--provider", provider,
            "--toolsets", "terminal,file,code_execution",
            "-z", message,
        ]
        harness = env_info.get("_harness")

        start_ns = time.time_ns()
        try:
            if harness is not None and hasattr(harness, "exec_in_container"):
                proc = harness.exec_in_container(
                    container_id,
                    argv,
                    timeout=timeout + 120,
                    env=env_info.get("agent_env"),
                    check=False,
                )
            else:
                proc = subprocess.run(
                    ["docker", "exec", container_id, *argv],
                    capture_output=True,
                    text=True,
                    timeout=timeout + 120,
                )
            rc = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as e:
            rc = -1
            stdout = (e.stdout or b"").decode("utf-8", errors="replace") if e.stdout else ""
            stderr = (e.stderr or b"").decode("utf-8", errors="replace") if e.stderr else ""
            stderr += "\nHermes agent wall-clock timeout"
        end_ns = time.time_ns()

        if stdout:
            (run_dir / "hermes_stdout.log").write_text(stdout[-20000:])
        if stderr:
            (run_dir / "hermes_stderr.log").write_text(stderr[-20000:])

        self._copy_hermes_logs(env_info, run_dir)
        events = self._read_hermes_events(env_info, run_dir)
        usage = self._aggregate_usage(events)

        spans_path = run_dir / "spans.jsonl"
        self._write_spans(
            spans_path,
            run_id=run_dir.name,
            session_id=run_dir.name,
            start_ns=start_ns,
            end_ns=end_ns,
            model_name=model_name,
            provider=provider,
            exit_code=rc,
            events=events,
            usage=usage,
        )

        step = {
            "trajectory_id": run_dir.name,
            "attempt_id": 1,
            "step_id": 1,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": "LLM_INFER_HERMES_AGENTIC_COARSE",
            "tool_name": "hermes.agent",
            "command": "hermes --ignore-rules --yolo --toolsets terminal,file,code_execution -z ...",
            "file_path": None,
            "start_ts": start_ns,
            "end_ts": end_ns,
            "duration_ms": (end_ns - start_ns) / 1e6,
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "exit_code": rc,
            "error_type": None if rc == 0 else f"hermes_exit_{rc}",
            "repo_id": instance_data.get("repo"),
            "instance_id": instance_id,
            "config_id": "hermes_agent",
            "model_name": model_name,
            "provider": provider,
            "response_preview": stdout[-500:] if stdout else None,
            "harness": "hermes",
            "agent_server_location": "hermes_harness_container",
            "workspace_mount": "bind_from_host",
            "bridge_active": bool(env_info.get("bridge_socket")),
        }
        step_path = run_dir / "steps.jsonl"
        step_path.write_text(json.dumps(step) + "\n")

        patch_text = ""
        if harness is not None:
            try:
                patch_text = harness.collect_patch(container_id, workspace_root)
            except Exception as e:
                logger.warning("collect_patch failed: %s", e)
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text(patch_text)

        return {
            "step_trace_path": str(step_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": usage["input_tokens"],
            "total_completion_tokens": usage["output_tokens"],
            "telemetry_path": str(spans_path),
            "exit_code": rc,
        }

    def _failure(self, run_dir: Path, instance_id: str, msg: str) -> Dict:
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text("")
        step_path = run_dir / "steps.jsonl"
        step_path.write_text(json.dumps({
            "trajectory_id": run_dir.name,
            "step_type": "ERROR",
            "tool_name": "hermes.agent",
            "error_type": msg,
            "instance_id": instance_id,
            "config_id": "hermes_agent",
        }) + "\n")
        spans_path = run_dir / "spans.jsonl"
        spans_path.write_text("")
        return {
            "step_trace_path": str(step_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "telemetry_path": str(spans_path),
            "exit_code": 1,
        }

    def _write_spans(
        self,
        path: Path,
        run_id: str,
        session_id: str,
        start_ns: int,
        end_ns: int,
        model_name: str,
        provider: str,
        exit_code: int,
        events: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        events = events or []
        usage = usage or self._aggregate_usage(events)
        duration_ms = (end_ns - start_ns) / 1e6
        resource = {
            "oc.run_id": run_id,
            "hm.run_id": run_id,
            "service.name": "agent-bench-hermes",
        }
        model_start_ns = int(usage.get("start_ns") or start_ns)
        model_end_ns = int(usage.get("end_ns") or end_ns)
        model_duration_ms = (
            float(usage.get("duration_ms"))
            if usage.get("duration_ms") is not None
            else (model_end_ns - model_start_ns) / 1e6
        )
        model_span = {
            "name": "agent.model.usage",
            "canonical_name": "agent.model.usage",
            "start_unix_nano": model_start_ns,
            "end_unix_nano": model_end_ns,
            "start_iso": _ns_to_iso(model_start_ns),
            "end_iso": _ns_to_iso(model_end_ns),
            "duration_ms": model_duration_ms,
            "attrs": {
                "agent.channel": "hermes.oneshot",
                "agent.provider": usage.get("provider") or provider,
                "agent.model": usage.get("model") or model_name,
                "agent.run_id": run_id,
                "agent.session_id": session_id,
                "agent.tokens.input": usage["input_tokens"],
                "agent.tokens.output": usage["output_tokens"],
                "agent.tokens.cache_read": usage["cache_read_tokens"],
                "agent.tokens.cache_write": usage["cache_write_tokens"],
                "agent.tokens.total": usage["total_tokens"],
                "agent.api_calls": usage["api_calls"],
            },
            "resource": resource,
            "synthesized": not usage.get("from_events"),
        }
        with path.open("w") as fh:
            fh.write(json.dumps(model_span) + "\n")
            tool_events = [e for e in events if e.get("type") == "tool.call"]
            if tool_events:
                for evt in tool_events:
                    span = self._tool_event_to_span(evt, run_id, session_id, resource)
                    fh.write(json.dumps(span) + "\n")
            else:
                call_id = f"{session_id}:hermes-agent"
                is_error = exit_code != 0
                tool_span = {
                    "name": "agent.tool.call",
                    "canonical_name": "agent.tool.call",
                    "start_unix_nano": start_ns,
                    "end_unix_nano": end_ns,
                    "start_iso": _ns_to_iso(start_ns),
                    "end_iso": _ns_to_iso(end_ns),
                    "duration_ms": duration_ms,
                    "attrs": {
                        "agent.channel": "hermes.oneshot",
                        "agent.tool.name": "hermes.agent",
                        "agent.tool.call_id": call_id,
                        "agent.tool.callId": call_id,
                        "agent.tool.is_error": is_error,
                        "agent.tool.isError": is_error,
                        "agent.run_id": run_id,
                        "agent.runId": run_id,
                        "agent.session_id": session_id,
                        "agent.sessionId": session_id,
                        "agent.exit_code": exit_code,
                        "agent.tool.synthetic_run": True,
                    },
                    "resource": resource,
                    "synthesized": True,
                }
                fh.write(json.dumps(tool_span) + "\n")

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            try:
                return int(float(value or 0))
            except (TypeError, ValueError):
                return 0

    def _read_hermes_events(self, env_info: Dict, run_dir: Path) -> List[Dict[str, Any]]:
        state_dir = env_info.get("state_dir")
        if not state_dir:
            return []
        state_path = Path(state_dir)
        src = next(
            (
                candidate for candidate in (
                    state_path / "events.jsonl",
                    state_path / "agent_events.jsonl",
                )
                if candidate.is_file()
            ),
            None,
        )
        if src is None:
            return []
        dst = run_dir / "hermes_events.jsonl"
        events: List[Dict[str, Any]] = []
        try:
            dst.write_text(src.read_text(errors="replace"))
        except OSError:
            pass
        try:
            for line in src.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict):
                    events.append(evt)
        except OSError:
            return []
        return events

    def _aggregate_usage(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        usage_events = [e for e in events if e.get("type") == "model.usage"]
        out: Dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "provider": "",
            "model": "",
            "start_ns": None,
            "end_ns": None,
            "duration_ms": None,
            "from_events": False,
        }
        for evt in usage_events:
            out["from_events"] = True
            out["input_tokens"] += self._as_int(evt.get("input_tokens"))
            out["output_tokens"] += self._as_int(evt.get("output_tokens"))
            out["cache_read_tokens"] += self._as_int(evt.get("cache_read_tokens"))
            out["cache_write_tokens"] += self._as_int(evt.get("cache_write_tokens"))
            out["total_tokens"] += self._as_int(evt.get("total_tokens"))
            out["api_calls"] += self._as_int(evt.get("api_calls"))
            if evt.get("provider"):
                out["provider"] = evt["provider"]
            if evt.get("model"):
                out["model"] = evt["model"]
            start_ns = self._as_int(evt.get("start_ns"))
            end_ns = self._as_int(evt.get("end_ns"))
            if start_ns:
                out["start_ns"] = (
                    start_ns if out["start_ns"] is None
                    else min(out["start_ns"], start_ns)
                )
            if end_ns:
                out["end_ns"] = (
                    end_ns if out["end_ns"] is None
                    else max(out["end_ns"], end_ns)
                )
        if out["total_tokens"] == 0:
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        if out["start_ns"] and out["end_ns"]:
            out["duration_ms"] = (out["end_ns"] - out["start_ns"]) / 1e6
        return out

    def _tool_event_to_span(
        self,
        evt: Dict[str, Any],
        fallback_run_id: str,
        fallback_session_id: str,
        resource: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_ns = self._as_int(evt.get("start_ns"))
        end_ns = self._as_int(evt.get("end_ns")) or start_ns
        duration_ms = evt.get("duration_ms")
        try:
            duration_ms = float(duration_ms)
        except (TypeError, ValueError):
            duration_ms = max((end_ns - start_ns) / 1e6, 0.0)
        run_id = str(evt.get("run_id") or fallback_run_id)
        session_id = str(evt.get("session_id") or fallback_session_id)
        call_id = str(evt.get("tool_call_id") or f"{session_id}:tool")
        is_error = bool(evt.get("is_error"))
        return {
            "name": "agent.tool.call",
            "canonical_name": "agent.tool.call",
            "start_unix_nano": start_ns,
            "end_unix_nano": end_ns,
            "start_iso": _ns_to_iso(start_ns),
            "end_iso": _ns_to_iso(end_ns),
            "duration_ms": duration_ms,
            "attrs": {
                "agent.channel": "hermes.oneshot",
                "agent.tool.name": str(evt.get("tool_name") or "unknown"),
                "agent.tool.call_id": call_id,
                "agent.tool.callId": call_id,
                "agent.tool.is_error": is_error,
                "agent.tool.isError": is_error,
                "agent.run_id": run_id,
                "agent.runId": run_id,
                "agent.session_id": session_id,
                "agent.sessionId": session_id,
                "agent.tool.result_chars": self._as_int(evt.get("result_chars")),
            },
            "resource": resource,
            "synthesized": False,
        }

    def _copy_hermes_logs(self, env_info: Dict, run_dir: Path) -> None:
        state_dir = env_info.get("state_dir")
        if not state_dir:
            return
        src = Path(state_dir) / "logs"
        if src.is_dir():
            dst = run_dir / "hermes_logs"
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except OSError:
                pass
