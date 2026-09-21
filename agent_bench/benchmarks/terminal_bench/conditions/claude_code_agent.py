"""Claude Code condition for Terminal-Bench.

The unmodified Claude Code CLI runs inside the task instance. Its native
stream-json output and session files are retained so subagent use can be
verified without inferring it from prose.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agent_bench.benchmarks.conditions.agent_otel import (
    otel_trace_offset,
    snapshot_otel_spans,
    wait_for_otel_flush,
)
from agent_bench.benchmarks.conditions.util_sampler import UtilSampler
from agent_bench.deployments.containers import Container, ContainerHandle

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            yield value


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _usage_from_events(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Use Claude's final aggregate usage when present, else dedupe messages."""
    final_usage: Optional[dict[str, Any]] = None
    messages: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "result" and isinstance(event.get("usage"), dict):
            final_usage = event["usage"]
        for item in _walk(event):
            if item.get("role") != "assistant" or not isinstance(item.get("usage"), dict):
                continue
            message_id = str(item.get("id") or "")
            if message_id:
                messages[message_id] = item["usage"]
            else:
                anonymous.append(item["usage"])
    usage_rows = [final_usage] if final_usage is not None else [
        *messages.values(), *anonymous
    ]
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0
    for usage in usage_rows:
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        cache_read += int(usage.get("cache_read_input_tokens") or 0)
        cache_creation += int(usage.get("cache_creation_input_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _native_subagent_evidence(
    events: Iterable[dict[str, Any]], session_files: Iterable[Path]
) -> dict[str, Any]:
    calls_by_identity: dict[str, dict[str, Any]] = {}
    for event in events:
        for item in _walk(event):
            if item.get("type") != "tool_use":
                continue
            name = str(item.get("name") or "")
            if name.lower() not in {"agent", "task"}:
                continue
            call = {
                "tool_use_id": item.get("id"),
                "tool_name": name,
                "input": item.get("input"),
            }
            identity = str(item.get("id") or json.dumps(
                call, sort_keys=True, ensure_ascii=False
            ))
            calls_by_identity[identity] = call
    calls = list(calls_by_identity.values())
    materialized = sorted(
        {
            str(path)
            for path in session_files
            if "subagents" in path.parts or path.name.startswith("agent-")
        }
    )
    return {
        "subagent_tool_call_count": len(calls),
        "subagent_tool_calls": calls,
        "materialized_subagent_count": len(materialized),
        "materialized_subagent_files": materialized,
    }


class TerminalBenchClaudeCodeCondition:
    """Invoke Claude Code against a prepared Terminal-Bench task instance."""

    def __init__(self, model_config: Any, agent_config: Any, container: Container):
        self.model_config = model_config
        self.agent_config = agent_config
        self.container = container

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Lead the completion of this Terminal-Bench task in the current "
            "environment. Inspect the workspace and task data, identify "
            "independent workstreams and dependencies, coordinate parallel work "
            "where appropriate, integrate the complete result, and validate all "
            "requirements. The hidden verifier is not available during the agent "
            "run. Declare DONE only when the integrated task is complete.\n\n"
            f"Task:\n{instance_data['instruction']}"
        )

    @staticmethod
    def _archive_sessions(state_dir: Path, run_dir: Path) -> tuple[Path, list[Path]]:
        destination = run_dir / "conversation" / "claude-code"
        if destination.exists():
            shutil.rmtree(destination)
        if state_dir.is_dir():
            shutil.copytree(state_dir, destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
        return destination, sorted(destination.rglob("*.jsonl"))

    @staticmethod
    def _write_trajectory(
        run_dir: Path,
        stdout_events: list[dict[str, Any]],
        session_files: list[Path],
    ) -> Path:
        trajectory = run_dir / "trajectory.jsonl"
        seen: set[str] = set()
        with trajectory.open("w", encoding="utf-8") as stream:
            for index, event in enumerate(stdout_events, start=1):
                row = {
                    "source": "claude_code_stream_json",
                    "sequence": index,
                    "event": event,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            for path in session_files:
                for index, event in enumerate(_iter_jsonl(path), start=1):
                    identity = json.dumps(event, sort_keys=True, ensure_ascii=False)
                    key = f"{path}:{index}:{identity}"
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {
                        "source": "claude_code_session",
                        "session_file": str(path.relative_to(run_dir)),
                        "sequence": index,
                        "event": event,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return trajectory

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
        if not container_id:
            raise RuntimeError("Claude Code condition requires a task container")
        run_dir.mkdir(parents=True, exist_ok=True)
        handle = ContainerHandle(
            id=container_id,
            name=str(env_info.get("container_name") or container_id),
            backend=self.container.backend_id,
        )
        model_name = getattr(self.model_config, "name", None) or "default"
        timeout = int(getattr(self.agent_config, "max_time_seconds", 1800))
        max_turns = int(getattr(self.agent_config, "max_steps", 200))
        argv = [
            "/tmp/agent-bench-claude-code/bin/claude",
            "--verbose",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--model",
            model_name,
            "--max-turns",
            str(max_turns),
            "--print",
        ]
        prompt = self._build_message(instance_data)
        trace_file, trace_offset = otel_trace_offset()
        sampler = UtilSampler(
            run_dir,
            llm_container=env_info.get("llm_container_name"),
            client_container=env_info.get("task_container_name"),
            tools_container=None,
        )
        sampler.__enter__()
        start_ns = time.time_ns()
        timed_out = False
        try:
            try:
                proc = self.container.exec(
                    handle,
                    argv,
                    check=False,
                    timeout=timeout,
                    input_text=prompt,
                    workdir=workspace_root,
                    env=dict(env_info.get("agent_env") or {}),
                )
                rc = proc.returncode
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                rc = -1
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + f"\n[agent-bench] timeout after {timeout}s"
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
        finally:
            end_ns = time.time_ns()
            sampler.__exit__(None, None, None)

        (run_dir / "claude_code_stdout.jsonl").write_text(stdout, encoding="utf-8")
        (run_dir / "claude_code_stderr.log").write_text(stderr, encoding="utf-8")
        stdout_events = list(_iter_jsonl(run_dir / "claude_code_stdout.jsonl"))
        state_value = env_info.get("claude_state_dir")
        if not state_value:
            raise RuntimeError("Claude Code harness did not provide claude_state_dir")
        conversation_dir, session_files = self._archive_sessions(
            Path(str(state_value)), run_dir
        )
        all_events = list(stdout_events)
        for path in session_files:
            all_events.extend(_iter_jsonl(path))
        usage = _usage_from_events(all_events)
        subagents = _native_subagent_evidence(all_events, session_files)
        trajectory_path = self._write_trajectory(
            run_dir, stdout_events, session_files
        )

        runtime_observability = {
            "agent_runtime_mode": "claude_code_cli",
            "prompt_policy": "roadmap_adapted",
            "conversation_dir": str(conversation_dir),
            "session_file_count": len(session_files),
            **subagents,
        }
        observability_path = run_dir / "runtime_observability.json"
        observability_path.write_text(
            json.dumps(runtime_observability, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        token_accounting = {
            "source": "claude_code_native_usage",
            "trajectory_tokens": {
                "total_logical_input": usage["input_tokens"],
                "total_logical_output": usage["output_tokens"],
                "cache_read_input_tokens": usage["cache_read_input_tokens"],
                "cache_creation_input_tokens": usage[
                    "cache_creation_input_tokens"
                ],
            },
        }
        token_path = run_dir / "token_accounting.json"
        token_path.write_text(json.dumps(token_accounting, indent=2), encoding="utf-8")

        step_path = run_dir / "steps.jsonl"
        step = {
            "trajectory_id": run_dir.name,
            "attempt_id": 1,
            "step_id": 1,
            "step_type": "LLM_INFER",
            "step_subtype": "CLAUDE_CODE_NATIVE_CLI",
            "tool_name": "claude_code.agent",
            "start_ts": start_ns,
            "end_ts": end_ns,
            "duration_ms": (end_ns - start_ns) / 1e6,
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "exit_code": rc,
            "error_type": "timeout" if timed_out else (None if rc == 0 else f"claude_code_exit_{rc}"),
            "instance_id": instance_data["instance_id"],
            "config_id": "claude_code_agent",
            "model_name": model_name,
            "harness": "claude_code",
        }
        step_path.write_text(json.dumps(step) + "\n", encoding="utf-8")
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text("", encoding="utf-8")

        wait_for_otel_flush()
        spans_path = snapshot_otel_spans(
            trace_file, trace_offset, run_dir, expected_run_id=run_dir.name
        )
        if spans_path is None:
            spans_path = run_dir / "spans.jsonl"
            spans_path.touch()

        outcome = "timeout" if timed_out else (
            "normal_completion" if rc == 0 else "runtime_error"
        )
        return {
            "step_trace_path": str(step_path),
            "patch_path": str(patch_path),
            "trajectory_path": str(trajectory_path),
            "token_accounting_path": str(token_path),
            "runtime_observability_path": str(observability_path),
            "runtime_observability": runtime_observability,
            "trajectory_tokens": token_accounting["trajectory_tokens"],
            "serving_tokens": None,
            "total_prompt_tokens": usage["input_tokens"],
            "total_completion_tokens": usage["output_tokens"],
            "telemetry_path": str(spans_path),
            "exit_code": rc,
            "agent_runtime_mode": "claude_code_cli",
            "agent_completion_status": outcome,
            "agent_execution_outcome": outcome,
            "agent_completion_error": None if rc == 0 else stderr[-2000:],
            "agent_execution_seconds": (end_ns - start_ns) / 1e9,
            "phase_timing": {
                "agent_execution_start_ns": start_ns,
                "agent_execution_end_ns": end_ns,
                "agent_execution_seconds": (end_ns - start_ns) / 1e9,
            },
        }
