"""Native Codex rollouts retain parent/child identity and concurrent turns."""
from __future__ import annotations
import datetime
import json
from pathlib import Path


def jsonl(path):
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def ns(value):
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1000


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
    sessions, trajectory, spans, expected_children = [], [], [], set()
    for path in sorted((run_dir / "codex_state").rglob("rollout-*.jsonl")):
        rows = list(jsonl(path))
        meta = next((r["payload"] for r in rows if r.get("type") == "session_meta"), None)
        if not meta:
            continue
        parent = meta.get("parent_thread_id")
        source = meta.get("source") or {}
        if not parent and isinstance(source, dict):
            sub = source.get("subagent") or {}
            if isinstance(sub, dict):
                parent = (sub.get("thread_spawn") or {}).get("parent_thread_id")
        thread = str(meta["id"])
        session = {"id": thread, "parent_thread_id": parent, "source": source, "cli_version": meta.get("cli_version"),
                   "rollout_file": str(path.relative_to(run_dir)), "turn_intervals": [], "final_assistant": None, "usage": {}}
        active, calls = None, {}
        for row in rows:
            payload, kind, stamp = row.get("payload") or {}, row.get("type"), ns(row["timestamp"])
            event = payload.get("type")
            canonical = {
                "function_call": "tool_call", "custom_tool_call": "tool_call",
                "function_call_output": "tool_result", "custom_tool_call_output": "tool_result",
                "task_started": "agent_start", "task_complete": "child_completion" if parent else "agent_result",
                "turn_aborted": "agent_interrupted",
            }.get(event, kind)
            trajectory.append({"timestamp_ns": stamp, "session_key": thread, "parent_session_key": parent,
                               "source": "codex_rollout", "event_type": canonical, "native_type": kind, "event": payload})
            if kind == "event_msg":
                if event == "task_started":
                    active = stamp
                elif event in {"task_complete", "turn_aborted"} and active is not None:
                    session["turn_intervals"].append({"start_timestamp_ns": active, "end_timestamp_ns": stamp,
                                                     "status": "completed" if event == "task_complete" else "interrupted"})
                    active = None
                    session["terminal_event"] = event
                elif event == "token_count":
                    session["usage"] = (payload.get("info") or {}).get("total_token_usage") or session["usage"]
            elif kind == "response_item":
                if event == "message" and payload.get("role") == "assistant":
                    final = "\n".join(p.get("text", "") for p in payload.get("content") or [] if p.get("type") == "output_text")
                    if final and payload.get("phase") != "commentary":
                        session["final_assistant"] = final
                elif event in {"function_call", "custom_tool_call"}:
                    calls[payload["call_id"]] = (stamp, payload)
                elif event in {"function_call_output", "custom_tool_call_output"}:
                    matched = calls.pop(payload["call_id"], None)
                    if matched:
                        start, call = matched
                        name = (call.get("namespace") + "." if call.get("namespace") else "") + call["name"]
                        if call["name"] == "spawn_agent":
                            try:
                                result = json.loads(payload.get("output") or "{}")
                            except (ValueError, TypeError):
                                result = {}
                            if isinstance(result, dict) and result.get("agent_id"):
                                expected_children.add(str(result["agent_id"]))
                                trajectory.append({"timestamp_ns": stamp, "session_key": thread, "event_type": "spawn",
                                                   "source": "codex_rollout", "child_session_key": result["agent_id"], "event": result})
                        spans.append({"name": "agent.tool.call", "canonical_name": "agent.tool.call", "start_unix_nano": start,
                            "end_unix_nano": stamp, "duration_ms": (stamp - start) / 1e6,
                            "attrs": {"agent.session_id": thread, "agent.tool.name": name, "agent.tool.call_id": call["call_id"],
                                "agent.tool.arguments_json": call.get("arguments") or call.get("input"),
                                "agent.tool.result_preview": payload.get("output"), "agent.span.source": "codex.native_rollout"}})
        session["unclosed_turn_start_ns"] = active
        sessions.append(session)
    for row in jsonl(run_dir / "codex_api.jsonl"):
        if "end_ns" not in row:
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
    runtimes = {}
    for child in children:
        turns = child["turn_intervals"]
        complete = child["unclosed_turn_start_ns"] is None and bool(turns)
        runtimes[child["id"]] = {"session_id": child["id"], "parent_session_id": child["parent_thread_id"],
            "start_timestamp_ns": turns[0]["start_timestamp_ns"] if turns else None,
            "terminal_timestamp_ns": turns[-1]["end_timestamp_ns"] if complete else None,
            "runtime_seconds": sum((i["end_timestamp_ns"] - i["start_timestamp_ns"]) / 1e9 for i in turns),
            "runtime_complete": complete, "runtime_censored": not complete, "turn_intervals": turns,
            "start_timestamp_source": "codex_rollout.task_started", "start_timestamp_quality": "native_event",
            "terminal_timestamp_quality": "native_event",
            "terminal_timestamp_source": "codex_rollout.task_complete_or_turn_aborted", "status": turns[-1]["status"] if complete else "incomplete"}
    roots = [s["id"] for s in sessions if not s["parent_thread_id"]]
    overlap = [s for s in spans if s["attrs"].get("agent.session_id") in roots and
               any(s["start_unix_nano"] < i["end_timestamp_ns"] and s["end_unix_nano"] > i["start_timestamp_ns"] for i in intervals)]
    missing_children = sorted(expected_children - {s["id"] for s in children})
    unknown = sum(not c["runtime_complete"] for c in runtimes.values()) + len(missing_children)
    coordination = {"spawn_agent", "send_input", "wait_agent", "close_agent", "resume_agent"}
    parent_work = [s for s in overlap if s["name"] == "agent.tool.call" and
                   s["attrs"].get("agent.tool.name", "").split(".")[-1] not in coordination]
    obs = {"source": "codex_native_rollouts", "materialized_child_session_count": len(children), "child_session_count": len(children),
        "child_runtime_by_session": runtimes, "child_runtime_complete_count": sum(c["runtime_complete"] for c in runtimes.values()), "child_runtime_unknown_count": unknown,
        "spawned_child_count": len(expected_children), "missing_child_session_ids": missing_children,
        "child_count_source_quality": "native_thread_lineage" if not missing_children else "missing_rollouts",
        "max_concurrent_children": maximum if not unknown else None, "max_concurrent_children_lower_bound": maximum,
        "child_sum_runtime_seconds": sum(c["runtime_seconds"] for c in runtimes.values()) if not unknown else None,
        "child_union_active_seconds": union / 1e9 if not unknown else None,
        "child_overlap_wall_seconds": overlap_wall / 1e9 if not unknown else None,
        "child_concurrency_excess_seconds": excess / 1e9 if not unknown else None,
        "child_derived_metrics_quality": "complete" if not unknown else "censored", "parent_activity_during_child_execution": bool(parent_work),
        "parent_overlapping_model_calls": sum(s["name"] == "agent.model.usage" for s in overlap),
        "parent_overlapping_tool_calls": len(parent_work),
        "parent_overlapping_coordination_calls": sum(s["name"] == "agent.tool.call" for s in overlap) - len(parent_work)}
    for filename, payload in (("codex_sessions.json", {"sessions": sessions}), ("runtime_observability.json", obs)):
        (run_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for filename, rows in (("trajectory.jsonl", trajectory), ("spans.jsonl", spans)):
        rows.sort(key=lambda r: r.get("timestamp_ns", r.get("start_unix_nano", 0)))
        (run_dir / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return sessions, obs
