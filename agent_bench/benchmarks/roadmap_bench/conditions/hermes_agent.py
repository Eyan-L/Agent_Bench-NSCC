"""RoadmapBench candidate and trace handling for the Hermes runtime."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)

logger = logging.getLogger(__name__)


class RoadmapBenchHermesCondition(HermesAgentCondition):
    """Run Hermes, fence all writers, and preserve a reproducible candidate."""

    toolsets = "terminal,file,code_execution,delegation"
    ignore_rules = False

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Implement the complete multi-target roadmap in the current repository. "
            "Read the repository and requirements, plan the dependent work, and "
            "delegate bounded parallel tasks when independent workstreams exist. "
            "Review and integrate each child result, then validate every target. "
            "Reply DONE only after the implementation is complete.\n\n"
            f"Roadmap:\n{instance_data['instruction']}"
        )

    def run(
        self, instance_data: Dict, run_dir: Path, seed: int,
        workspace_root: str, container_id: Optional[str],
        env_info: Optional[Dict] = None,
    ) -> Dict:
        runtime_env = env_info if env_info is not None else {}
        workspace = Path(str(runtime_env.get("workspace_host_path") or ""))
        baseline = RoadmapBenchOpenClawCondition._capture_host_workspace_baseline(workspace)
        if not baseline or not baseline.get("git_head"):
            raise RuntimeError("Roadmap Hermes task Git baseline was not captured")
        runtime_env["roadmap_workspace_baseline"] = baseline
        runtime_env["roadmap_workspace_quiesced"] = False
        self._roadmap_run_dir = run_dir
        self._roadmap_baseline = baseline
        trajectory = run_dir / "trajectory.jsonl"
        self._event(trajectory, "agent_start", instance_id=instance_data["instance_id"])
        try:
            result = super().run(
                instance_data, run_dir, seed, workspace_root, container_id, runtime_env
            )
        except Exception as exc:
            self._event(trajectory, "agent_error", error=str(exc))
            self._partial_execution_result = {
                "step_trace_path": str(run_dir / "steps.jsonl") if (run_dir / "steps.jsonl").is_file() else None,
                "telemetry_path": str(run_dir / "spans.jsonl") if (run_dir / "spans.jsonl").is_file() else None,
                "trajectory_path": str(trajectory),
                "agent_runtime_mode": "hermes",
                "agent_execution_outcome": "runtime_error",
                "agent_completion_error": str(exc),
                "agent_lifecycle": {
                    "workspaceQuiesced": runtime_env.get("roadmap_workspace_quiesced") is True,
                    "lifecycleError": str(exc),
                },
                "failure_phase": "candidate_freeze" if (run_dir / "steps.jsonl").is_file() else "agent_execution",
            }
            raise
        sessions = self._read_session_ledger(run_dir)
        runtime_observability = self._write_runtime_observability(run_dir, sessions)
        self._append_hermes_timeline(run_dir, trajectory, sessions)
        steps = run_dir / "steps.jsonl"
        step = json.loads(steps.read_text().splitlines()[0]) if steps.is_file() else {}
        start_ns = step.get("start_ts")
        end_ns = step.get("end_ts")
        quiesced_ns = getattr(self, "_roadmap_quiesced_ns", None)
        if isinstance(end_ns, int) and isinstance(getattr(self, "_roadmap_quiescence", None), dict):
            self._roadmap_quiescence["parentTerminalObservedNs"] = end_ns
        exit_code = result.get("exit_code")
        outcome = (
            "normal_completion" if exit_code == 0 else
            "timeout" if exit_code == -1 else "runtime_error"
        )
        self._event(trajectory, "agent_result", exit_code=exit_code, outcome=outcome)
        parent_prompt = int(result.get("total_prompt_tokens") or 0)
        parent_completion = int(result.get("total_completion_tokens") or 0)
        session_prompt = sum(int(s["input_tokens"] or 0) for s in sessions)
        session_completion = sum(int(s["output_tokens"] or 0) for s in sessions)
        use_sessions = bool(sessions) and bool(session_prompt or session_completion)
        tokens = {
            "source": "hermes_state_db_sessions" if use_sessions else "hermes_events_parent_only",
            "prompt_tokens": session_prompt if use_sessions else parent_prompt,
            "completion_tokens": session_completion if use_sessions else parent_completion,
            "session_count": len(sessions),
            "parent_event_prompt_tokens": parent_prompt,
            "parent_event_completion_tokens": parent_completion,
            "model_request_timeline_scope": "patched_parent_agent_only",
            "child_model_request_latency_available": False,
        }
        token_path = run_dir / "token_accounting.json"
        token_path.write_text(json.dumps(tokens, indent=2))
        result.update({
            "harness": "hermes",
            "agent_runtime_mode": "hermes",
            "agent_execution_outcome": outcome,
            "agent_completion_error": None if exit_code == 0 else f"Hermes exited with code {exit_code}",
            "agent_lifecycle": getattr(self, "_roadmap_quiescence", {}),
            "agent_execution_seconds": (
                (quiesced_ns - start_ns) / 1e9
                if isinstance(start_ns, int) and isinstance(quiesced_ns, int) else None
            ),
            "phase_timing": {
                "agent_execution_start_ns": start_ns,
                "parent_terminal_observed_ns": end_ns,
                "agent_execution_end_ns": quiesced_ns,
                "workspace_quiesced_ns": quiesced_ns,
                "agent_execution_seconds": (
                    (quiesced_ns - start_ns) / 1e9
                    if isinstance(start_ns, int) and isinstance(quiesced_ns, int) else None
                ),
                "parent_execution_seconds": (
                    (end_ns - start_ns) / 1e9
                    if isinstance(start_ns, int) and isinstance(end_ns, int) else None
                ),
            },
            "trajectory_path": str(trajectory),
            "token_accounting_path": str(token_path),
            "runtime_observability_path": (
                str(runtime_observability) if runtime_observability else None
            ),
            "total_prompt_tokens": tokens["prompt_tokens"],
            "total_completion_tokens": tokens["completion_tokens"],
        })
        return result

    @staticmethod
    def _event(path: Path, event: str, **fields) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps({"event_type": event, "timestamp_ns": time.time_ns(), **fields}) + "\n")

    @staticmethod
    def _read_session_ledger(run_dir: Path) -> list[Dict]:
        """Read the stopped Hermes runtime's per-session counters and lineage."""
        db_path = run_dir / "hermes_state" / "state.db"
        if not db_path.is_file():
            return []
        uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    "SELECT id, source, parent_session_id, model, started_at, "
                    "ended_at, end_reason, input_tokens, output_tokens, "
                    "cache_read_tokens, cache_write_tokens, reasoning_tokens, "
                    "api_call_count, message_count, tool_call_count "
                    "FROM sessions ORDER BY started_at, id"
                ).fetchall()
        except sqlite3.Error:
            logger.exception("Could not read Hermes session ledger from %s", db_path)
            return []
        sessions = [dict(row) for row in rows]
        (run_dir / "hermes_sessions.json").write_text(json.dumps({
            "source": "hermes_state/state.db",
            "sessions": sessions,
        }, indent=2))
        return sessions

    @staticmethod
    def _write_runtime_observability(run_dir: Path, sessions: list[Dict]) -> Optional[Path]:
        """Project durable Hermes child sessions into the Roadmap report schema."""
        if not (run_dir / "hermes_sessions.json").is_file():
            return None
        by_id = {str(item["id"]): item for item in sessions}
        # Hermes also links context-compression and branch continuations to
        # their prior session. A delegate starts while its parent is live.
        def is_continuation(item: Dict) -> bool:
            parent = by_id.get(str(item.get("parent_session_id") or ""))
            return bool(
                parent
                and parent.get("end_reason") in {"compression", "branched"}
                and isinstance(item.get("started_at"), (int, float))
                and isinstance(parent.get("ended_at"), (int, float))
                and item["started_at"] >= parent["ended_at"]
            )

        children = [
            item for item in sessions
            if item.get("parent_session_id") in by_id
            and not is_continuation(item)
        ]
        continuations = {
            str(item["parent_session_id"]): item for item in sessions
            if item.get("parent_session_id") and is_continuation(item)
            and by_id[item["parent_session_id"]].get("end_reason") == "compression"
        }
        child_runtime: Dict[str, Dict] = {}
        points: list[tuple[int, int]] = []
        sum_seconds = 0.0
        unknown = 0
        for child in children:
            tip = child
            chain = [str(child["id"])]
            while str(tip["id"]) in continuations and len(chain) <= len(sessions):
                tip = continuations[str(tip["id"])]
                if str(tip["id"]) in chain:
                    break
                chain.append(str(tip["id"]))
            started = child.get("started_at")
            ended = tip.get("ended_at")
            start_ns = int(started * 1e9) if isinstance(started, (int, float)) else None
            end_ns = int(ended * 1e9) if isinstance(ended, (int, float)) else None
            complete = start_ns is not None and end_ns is not None and end_ns >= start_ns
            runtime = (end_ns - start_ns) / 1e9 if complete else None
            if complete:
                points.extend(((start_ns, 1), (end_ns, -1)))
                sum_seconds += runtime
            else:
                unknown += 1
            child_runtime[str(child["id"])] = {
                "session_id": child["id"],
                "continuation_session_ids": chain[1:],
                "parent_session_id": child["parent_session_id"],
                "label": child["id"],
                "status": tip.get("end_reason") or ("ended" if end_ns else "unknown"),
                "start_timestamp_ns": start_ns,
                "terminal_timestamp_ns": end_ns if complete else None,
                "terminal_timestamp_source": "hermes_state.sessions.ended_at" if complete else None,
                "terminal_timestamp_quality": "durable_session_end" if complete else None,
                "runtime_seconds": runtime,
                "runtime_complete": complete,
                "runtime_censored": not complete,
                "start_timestamp_source": "hermes_state.sessions.started_at",
                "start_timestamp_quality": "durable_session_start" if start_ns else "unknown",
            }
        points.sort()
        active = maximum = 0
        union = overlap = excess = 0.0
        for index, (timestamp_ns, delta) in enumerate(points):
            active += delta
            maximum = max(maximum, active)
            if index + 1 == len(points):
                continue
            duration = (points[index + 1][0] - timestamp_ns) / 1e9
            if active:
                union += duration
            if active > 1:
                overlap += duration
                excess += (active - 1) * duration
        observation = {
            "source": "hermes_state/state.db",
            "materialized_child_session_count": len(children),
            "child_session_count": len(children),
            "child_count_source_quality": "session_parent_linkage_excluding_continuations",
            "child_derived_metrics_quality": "complete" if not unknown else "partial",
            "child_runtime_complete_count": len(children) - unknown,
            "child_runtime_unknown_count": unknown,
            "child_runtime_by_session": child_runtime,
            "max_concurrent_children": maximum if not unknown else None,
            "max_concurrent_children_lower_bound": maximum,
            "child_sum_runtime_seconds": sum_seconds if not unknown else None,
            "child_sum_runtime_lower_bound_seconds": sum_seconds,
            "child_union_active_seconds": union if not unknown else None,
            "child_union_active_lower_bound_seconds": union,
            "child_overlap_wall_seconds": overlap if not unknown else None,
            "child_concurrency_excess_seconds": excess if not unknown else None,
        }
        path = run_dir / "runtime_observability.json"
        path.write_text(json.dumps(observation, indent=2))
        return path

    @staticmethod
    def _append_hermes_timeline(
        run_dir: Path, trajectory: Path, sessions: Optional[list[Dict]] = None,
    ) -> None:
        raw = run_dir / "hermes_events.jsonl"
        events = []
        for session in sessions or []:
            for suffix, field in (("start", "started_at"), ("end", "ended_at")):
                value = session.get(field)
                if not isinstance(value, (int, float)):
                    continue
                events.append({
                    "event_type": f"session_{suffix}",
                    "timestamp_ns": int(value * 1e9),
                    "session_key": session["id"],
                    "parent_session_key": session.get("parent_session_id"),
                    "session_source": session.get("source"),
                })
        for line in raw.read_text(errors="replace").splitlines() if raw.is_file() else []:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind not in {"model.usage", "tool.call"}:
                continue
            entry = {
                "event_type": "model_request" if kind == "model.usage" else "tool_call",
                "timestamp_ns": item.get("start_ns") or item.get("timestamp_ns"),
                "end_ns": item.get("end_ns"),
                "session_key": item.get("session_id"),
            }
            if kind == "model.usage":
                entry.update({
                    "model": item.get("model"),
                    "request_latency_ms": item.get("duration_ms"),
                    "input_tokens": item.get("input_tokens"),
                    "output_tokens": item.get("output_tokens"),
                })
            else:
                entry.update({
                    "tool_name": item.get("tool_name"),
                    "tool_call_id": item.get("tool_call_id"),
                    "is_error": item.get("is_error"),
                })
            events.append(entry)
        existing = [
            json.loads(line) for line in trajectory.read_text().splitlines()
            if line.strip()
        ]
        combined = existing + events
        def timestamp_ns(entry: Dict) -> int:
            try:
                return int(entry.get("timestamp_ns") or 0)
            except (TypeError, ValueError):
                return 0

        combined.sort(key=timestamp_ns)
        trajectory.write_text("".join(json.dumps(entry) + "\n" for entry in combined))

    def _collect_artifacts(
        self, env_info: Dict, container_id: str, workspace_root: str,
    ) -> str:
        harness = env_info.get("_harness")
        if harness is None or not hasattr(harness, "quiesce_orphan_children"):
            raise RuntimeError("Roadmap Hermes candidate freeze requires its harness")
        run_dir = self._roadmap_run_dir
        state = Path(str(env_info.get("state_dir") or ""))
        workspace = Path(str(env_info.get("workspace_host_path") or ""))
        if not workspace.is_dir():
            raise RuntimeError("Roadmap Hermes workspace vanished before freeze")

        # Hermes has completed its oneshot invocation. Export the durable
        # conversation while its runtime is still available.
        try:
            export = harness.exec_in_container(
                container_id,
                ["hermes", "sessions", "export", "sessions.jsonl"],
                timeout=120, env=env_info.get("agent_env"),
                workdir="/opt/data", check=False,
            )
            if export.returncode != 0:
                logger.warning("Hermes session export failed: %s", export.stderr[-500:])
        except Exception:
            logger.exception("Hermes session export failed")

        quiescence = harness.quiesce_orphan_children(env_info)
        if quiescence.get("workspaceQuiesced") is not True:
            raise RuntimeError("Hermes writers were not quiesced before candidate freeze")
        self._roadmap_quiesced_ns = time.time_ns()
        quiescence["workspaceQuiescedNs"] = self._roadmap_quiesced_ns
        self._roadmap_quiescence = quiescence
        patch = RoadmapBenchOpenClawCondition._collect_workspace_patch(
            workspace, str(self._roadmap_baseline["git_head"])
        )
        candidate = run_dir / "candidate_workspace"
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(
            workspace, candidate, symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        if state.is_dir():
            archive = run_dir / "hermes_state"
            if archive.exists():
                shutil.rmtree(archive)
            shutil.copytree(
                state, archive,
                ignore=shutil.ignore_patterns(
                    "config.yaml", "exec-bridge.token", "*.sock"
                ),
            )
        metadata = {
            "baseline": self._roadmap_baseline,
            "quiescence": quiescence,
            "candidate_patch_bytes": len(patch.encode()),
        }
        (run_dir / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2))
        self._event(run_dir / "trajectory.jsonl", "candidate_frozen", patch_bytes=len(patch.encode()))
        return patch

    def _collect_sessions(self, *_args, **_kwargs) -> None:
        # Exported into the bound state directory before quiescence.
        return None

    def _collect_bridge_data(self, *_args, **_kwargs) -> None:
        # exec-bridge.log is archived with the Hermes state directory.
        return None
