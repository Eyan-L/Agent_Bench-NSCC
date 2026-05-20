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

We collapse them into:
    - one `agent.model.usage` span (totals from the driver-emitted summary)
    - one `agent.tool.call` span per (Action, Observation) pair
written to <run_dir>/spans.jsonl. Plus one coarse LLM_INFER step row in
<run_dir>/steps.jsonl and the candidate.patch (collected by the harness).

Returns the dict shape the driver expects.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestration.container.base import Container, ContainerHandle

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
        return int(float(t) * 1e6) if float(t) > 1e12 else int(float(t) * 1e9)
    if isinstance(t, str):
        try:
            iso = t.replace("Z", "+00:00")
            return int(datetime.fromisoformat(iso).timestamp() * 1e9)
        except ValueError:
            return 0
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
                                 "env_info missing config/events paths")

        message = _build_message(instance_data["problem_statement"])
        inner = (
            f"python /opt/openhands_driver.py "
            f"--config {shlex.quote(config_path_in)} "
            f"--task {shlex.quote(message)} "
            f"--events {shlex.quote(events_path_in)} "
            f"--workspace {shlex.quote(workspace_root)}"
        )
        timeout = getattr(self.agent_config, "max_time_seconds", 1800)
        step_start_ns = time.time_ns()
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
        step_end_ns = time.time_ns()

        # 2. Parse events.jsonl + build spans.
        events = self._load_events(Path(events_path_host))
        run_id = run_dir.name
        spans = self._build_spans(events, run_id, step_start_ns, step_end_ns)
        spans_path = run_dir / "spans.jsonl"
        with spans_path.open("w") as fh:
            for s in spans:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("  wrote %d span(s) -> %s", len(spans), spans_path.name)

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

        # 4. Patch — harness collects it via git diff in the workspace.
        harness = env_info.get("_harness")
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
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "telemetry_path": str(spans_path),
        }

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
        m = self._extract_summary_metrics(events)
        prompt = int(m.get("prompt_tokens") or 0)
        completion = int(m.get("completion_tokens") or 0)
        cache_read = int(m.get("cache_read_tokens") or 0)
        cache_write = int(m.get("cache_creation_tokens") or 0)
        total = int(m.get("total_tokens") or 0) or (prompt + completion)

        spans: List[Dict] = [{
            "name": "agent.model.usage",
            "canonical_name": "agent.model.usage",
            "start_unix_nano": fallback_start_ns,
            "end_unix_nano": fallback_end_ns,
            "start_iso": _ns_to_iso(fallback_start_ns),
            "end_iso": _ns_to_iso(fallback_end_ns),
            "duration_ms": (fallback_end_ns - fallback_start_ns) / 1e6,
            "attrs": {
                "agent.channel": "openhands.driver",
                "agent.provider": "openhands",
                "agent.model": self.model_config.name,
                "agent.tokens.input": prompt,
                "agent.tokens.output": completion,
                "agent.tokens.cache_read": cache_read,
                "agent.tokens.cache_write": cache_write,
                "agent.tokens.total": total,
            },
            "resource": {
                "oc.run_id": run_id,
                "service.name": "agent-bench-openhands",
            },
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
                    "resource": {
                        "oc.run_id": run_id,
                        "service.name": "agent-bench-openhands",
                    },
                    "synthesized": True,
                })
        return spans

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
