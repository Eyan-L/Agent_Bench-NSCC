"""Native Codex rollouts retain parent/child identity and concurrent turns."""
from __future__ import annotations
import json
from pathlib import Path
from agent_bench.harnesses.codex.rollouts import read_rollout, timestamp_ns


def jsonl(path):
    if not Path(path).is_file():
        return
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def ns(value):
    return timestamp_ns(value)


def api_response(row):
    """Read the terminal response from either protocol adapter mode."""
    response = row.get("response") or {}
    for line in (row.get("response_sse") or "").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
            response = event.get("response") or response
    return response


def summarize(run_dir):
    sessions, trajectory, spans, expected_children, observations = [], [], [], set(), []
    for path in sorted((run_dir / "codex_state").rglob("rollout-*.jsonl")):
        session = read_rollout(list(jsonl(path)), path, run_dir, trajectory, spans, expected_children, observations)
        if session:
            sessions.append(session)
    pending_requests = set()
    for row in jsonl(run_dir / "codex_api.jsonl"):
        identity = (row.get("thread_id"), row.get("start_ns"))
        if row.get("event") == "request_start":
            pending_requests.add(identity)
        if "end_ns" in row:
            pending_requests.discard(identity)
        if not isinstance(row.get("end_ns"), int) or not isinstance(row.get("start_ns"), int):
            continue
        response = api_response(row)
        usage = response.get("usage") or {}
        trajectory.append({"timestamp_ns": row["end_ns"], "start_timestamp_ns": row["start_ns"],
                           "session_key": row.get("thread_id"), "event_type": "model_request",
                           "source": "codex_responses_adapter", "request_latency_ms": (row["end_ns"] - row["start_ns"]) / 1e6,
                           "usage": usage, "error": row.get("error")})
        spans.append({"name": "agent.model.usage", "canonical_name": "agent.model.usage", "start_unix_nano": row["start_ns"],
            "end_unix_nano": row["end_ns"], "duration_ms": (row["end_ns"] - row["start_ns"]) / 1e6,
            "attrs": {"agent.session_id": row.get("thread_id"), "agent.model": response.get("model"), "agent.tokens.input": usage.get("input_tokens"),
                      "agent.tokens.output": usage.get("output_tokens"), "agent.span.source": "codex.responses_adapter", "agent.error": row.get("error")}})
    pending_threads = {thread for thread, _start in pending_requests}
    children = [s for s in sessions if s["parent_thread_id"]]
    intervals = [i for s in children for i in s["turn_intervals"]]
    points = sorted([(i["start_timestamp_ns"], 1) for i in intervals] + [(i["end_timestamp_ns"], -1) for i in intervals])
    maximum = active = 0
    union = overlap_wall = excess = 0
    previous = points[0][0] if points else 0
    for stamp, delta in points:
        duration = stamp - previous
        if active:
            union += duration
        if active > 1:
            overlap_wall += duration
            excess += duration * (active - 1)
        active += delta
        maximum = max(maximum, active)
        previous = stamp
    runtimes, child_states = {}, {}
    child_by_id = {s["id"]: s for s in children}
    child_ids = expected_children | set(child_by_id)
    for key in sorted(child_ids):
        child = child_by_id.get(key, {})
        turns = child.get("turn_intervals") or []
        # Timing completeness says whether all active intervals are known.
        # An interrupted interval can have exact timing but unfinished work.
        complete = bool(child.get("timing_complete"))
        evidence = [r for r in observations if r["session_id"] == key]
        submissions = [r for r in evidence if r["source"] == "send_input"]
        if submissions:
            submission = max(submissions, key=lambda r: r["timestamp_ns"] or 0)
            queued = submission["timestamp_ns"]
            native = next((r for r in evidence if r["source"] == "native_latest_turn"), {})
            processed = native.get("turn_id") == submission.get("submission_id") or (
                queued is not None and native.get("work_activity_ns") is not None and native["work_activity_ns"] >= queued)
            # A late result from a wait begun before send_input, or an old
            # child turn finishing after that submission, cannot settle it.
            evidence = [r for r in evidence if r["status"] != "completed" or (
                (r["source"] == "native_latest_turn" and processed) or
                (r["source"] != "native_latest_turn" and queued is not None
                 and r.get("request_start_ns") is not None and r["request_start_ns"] >= queued
                 and (not child or processed)))]
        known = sorted((r for r in evidence if r["timestamp_ns"] is not None), key=lambda r: r["timestamp_ns"])
        terminal = known[-1] if known else (evidence[-1] if evidence else {})
        undated = [r for r in evidence if r["source"] == "native_latest_turn" and r["timestamp_ns"] is None]
        if undated and (not terminal or terminal.get("status") == "completed" or terminal.get("source") == "spawn_agent"):
            terminal = undated[-1]
        status = terminal.get("status", "unknown")
        if terminal.get("source") == "spawn_agent":
            status = "unknown"
        if child.get("pending_call_ids"):
            status = "pending_calls"
        if key in pending_threads:
            status = "pending_model_request"
        if status in {"running", "queued", "pending_calls", "pending_model_request"}:
            complete = False
        child_states[key] = {"session_id": key, "status": status, "execution_complete": status == "completed",
                             "evidence_source": terminal.get("source", "missing_lifecycle_evidence"),
                             "evidence_timestamp_ns": terminal.get("timestamp_ns")}
        runtimes[key] = {"session_id": key, "parent_session_id": child.get("parent_thread_id"),
            "start_timestamp_ns": turns[0]["start_timestamp_ns"] if turns else None,
            "terminal_timestamp_ns": turns[-1]["end_timestamp_ns"] if complete else None,
            "runtime_seconds": sum((i["end_timestamp_ns"] - i["start_timestamp_ns"]) / 1e9 for i in turns) if complete else None,
            "observed_runtime_seconds": sum((i["end_timestamp_ns"] - i["start_timestamp_ns"]) / 1e9 for i in turns),
            "runtime_complete": complete, "runtime_censored": not complete, "turn_intervals": turns,
            "start_timestamp_source": "codex_rollout.task_started" if turns else None,
            "start_timestamp_quality": "native_event" if turns else None,
            "terminal_timestamp_quality": "native_event" if complete else None,
            "terminal_timestamp_source": "codex_rollout.task_complete_or_turn_aborted" if complete else None,
            "status": status, "execution_complete": status == "completed"}
    roots = [s["id"] for s in sessions if not s["parent_thread_id"]]
    overlap = [s for s in spans if s["attrs"].get("agent.session_id") in roots and
               any(s["start_unix_nano"] < i["end_timestamp_ns"] and s["end_unix_nano"] > i["start_timestamp_ns"] for i in intervals)]
    missing_children = sorted(expected_children - {s["id"] for s in children})
    unknown = sum(not c["runtime_complete"] for c in runtimes.values())
    coordination = {"spawn_agent", "send_input", "wait_agent", "close_agent", "resume_agent"}
    parent_work = [s for s in overlap if s["name"] == "agent.tool.call" and
                   s["attrs"].get("agent.tool.name", "").split(".")[-1] not in coordination]
    parent_model = sum(s["name"] == "agent.model.usage" for s in overlap)
    parent_coordination = sum(s["name"] == "agent.tool.call" for s in overlap) - len(parent_work)
    unfinished = sorted(key for key, state in child_states.items() if not state["execution_complete"])
    obs = {"source": "codex_native_rollouts", "materialized_child_session_count": len(children), "child_session_count": len(child_ids),
        "child_lifecycle_by_session": child_states, "unfinished_child_session_ids": unfinished,
        "child_execution_complete_count": sum(s["execution_complete"] for s in child_states.values()),
        "child_lifecycle_unknown_count": sum(s["status"] == "unknown" for s in child_states.values()),
        "child_runtime_by_session": runtimes, "child_runtime_complete_count": sum(c["runtime_complete"] for c in runtimes.values()), "child_runtime_unknown_count": unknown,
        "spawned_child_count": len(expected_children), "missing_child_session_ids": missing_children,
        "child_count_source_quality": "native_thread_lineage" if not missing_children else "missing_rollouts",
        "max_concurrent_children": maximum if not unknown else None, "max_concurrent_children_lower_bound": maximum,
        "child_sum_runtime_seconds": sum(c["runtime_seconds"] for c in runtimes.values()) if not unknown else None,
        "child_union_active_seconds": union / 1e9 if not unknown else None,
        "child_overlap_wall_seconds": overlap_wall / 1e9 if not unknown else None,
        "child_concurrency_excess_seconds": excess / 1e9 if not unknown else None,
        "child_derived_metrics_quality": "complete" if not unknown else "censored",
        "parent_activity_during_child_execution": True if parent_work else (None if unknown else False),
        "parent_overlap_metrics_quality": "complete" if not unknown else "censored",
        "parent_overlapping_model_calls": parent_model if not unknown else None,
        "parent_overlapping_tool_calls": len(parent_work) if not unknown else None,
        "parent_overlapping_coordination_calls": parent_coordination if not unknown else None,
        "parent_overlapping_model_calls_lower_bound": parent_model,
        "parent_overlapping_tool_calls_lower_bound": len(parent_work),
        "parent_overlapping_coordination_calls_lower_bound": parent_coordination}
    for filename, payload in (("codex_sessions.json", {"sessions": sessions}), ("runtime_observability.json", obs)):
        (run_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for filename, rows in (("trajectory.jsonl", trajectory), ("spans.jsonl", spans)):
        rows.sort(key=lambda r: r.get("timestamp_ns") or r.get("start_unix_nano") or 0)
        (run_dir / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return sessions, obs
