"""Read turn-scoped native evidence without carrying terminal state forward."""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def timestamp_ns(value: str | None) -> int | None:
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1000
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def native_status(value: str | dict | None) -> str:
    if isinstance(value, dict):
        value = next(iter(value), "unknown")
    return {"pending_init": "running", "shutdown": "closed_incomplete",
            "errored": "failed", "not_found": "unknown"}.get(value, value or "unknown")


def as_object(value: str | dict | None) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def read_rollout(rows: list[dict], path: Path, run_dir: Path, trajectory: list[dict],
                 spans: list[dict], expected: set[str], observations: list[dict]) -> dict | None:
    meta = next((r["payload"] for r in rows if r.get("type") == "session_meta"), None)
    if not meta:
        return None
    parent, source = meta.get("parent_thread_id"), meta.get("source") or {}
    if not parent and isinstance(source, dict):
        subagent = source.get("subagent") or {}
        if isinstance(subagent, dict):
            parent = (subagent.get("thread_spawn") or {}).get("parent_thread_id")
    thread = str(meta["id"])
    session = {"id": thread, "parent_thread_id": parent, "source": source,
               "cli_version": meta.get("cli_version"), "rollout_file": str(path.relative_to(run_dir)),
               "turn_intervals": [], "turns": [], "usage": {}}
    turns, latest, calls = {}, None, {}

    def observe(child, status, stamp, evidence, **details):
        observations.append({"session_id": str(child), "status": native_status(status),
                             "timestamp_ns": stamp, "source": evidence, **details})

    def new_turn(identity, stamp=None, started=False):
        turn = {"id": identity, "index": len(session["turns"]), "start_timestamp_ns": stamp,
                "terminal_timestamp_ns": None, "terminal_event": None, "is_open": started,
                "start_observed": started, "final_assistant": None, "final_item_id": None,
                "last_assistant_message": None, "last_assistant_phase": None,
                "last_response_type": None, "last_input_timestamp_ns": None}
        session["turns"].append(turn)
        turns[identity] = turn
        return turn

    for row in rows:
        payload, kind = row.get("payload") or {}, row.get("type")
        stamp, event = timestamp_ns(row.get("timestamp")), payload.get("type")
        identity = payload.get("turn_id") or (payload.get("internal_chat_message_metadata_passthrough") or {}).get("turn_id")
        canonical = {"function_call": "tool_call", "custom_tool_call": "tool_call",
                     "function_call_output": "tool_result", "custom_tool_call_output": "tool_result",
                     "task_started": "agent_start", "task_complete": "child_completion" if parent else "agent_result",
                     "turn_aborted": "agent_interrupted"}.get(event, kind)
        trajectory.append({"timestamp_ns": stamp, "session_key": thread, "parent_session_key": parent,
                           "turn_id": identity, "source": "codex_rollout", "event_type": canonical,
                           "native_type": kind, "event": payload})
        if kind == "event_msg":
            if event == "task_started":
                identity = identity or f"rollout_turn_{len(session['turns'])}"
                latest = new_turn(identity, stamp, started=True)
            elif event in {"task_complete", "turn_aborted"}:
                turn = turns.get(identity) if identity else latest
                if turn is None:
                    turn = new_turn(identity or f"rollout_turn_{len(session['turns'])}")
                    latest = turn
                turn.update(terminal_event=event, terminal_timestamp_ns=stamp, is_open=False)
            elif event == "token_count":
                session["usage"] = (payload.get("info") or {}).get("total_token_usage") or session["usage"]
            continue
        if kind != "response_item":
            continue
        turn = turns.get(identity) if identity else latest
        if event == "message" and payload.get("role") == "user" and turn is not None:
            turn["last_input_timestamp_ns"] = stamp
            turn.update(last_response_type="user_message", final_assistant=None, final_item_id=None)
        elif event == "message" and payload.get("role") == "assistant" and turn is not None:
            # Reset even for blank/commentary messages. An older DONE cannot be
            # used after a new message or a subsequent tool result.
            turn["last_response_type"] = "message"
            turn["last_assistant_message"] = "\n".join(
                p.get("text", "") for p in payload.get("content") or [] if p.get("type") == "output_text")
            turn["last_assistant_phase"] = payload.get("phase")
            turn["final_assistant"] = None if payload.get("phase") == "commentary" else turn["last_assistant_message"]
            turn["final_item_id"] = payload.get("id")
        elif event in {"function_call", "custom_tool_call"}:
            calls[payload["call_id"]] = (stamp, payload, turn)
            if turn is not None:
                turn.update(last_response_type=event, final_assistant=None, final_item_id=None)
        elif event in {"function_call_output", "custom_tool_call_output"}:
            matched = calls.pop(payload["call_id"], None)
            if not matched:
                continue
            start, call, call_turn = matched
            if call_turn is not None:
                call_turn.update(last_response_type=event, final_assistant=None, final_item_id=None)
            name = (call.get("namespace") + "." if call.get("namespace") else "") + call["name"]
            result = as_object(payload.get("output"))
            args = as_object(call.get("arguments") or call.get("input"))
            target = args.get("target") or args.get("id")
            if call.get("namespace") == "multi_agent_v1":
                if call["name"] == "spawn_agent" and result.get("agent_id"):
                    child = str(result["agent_id"])
                    expected.add(child)
                    observe(child, "running", start, "spawn_agent")
                    trajectory.append({"timestamp_ns": stamp, "session_key": thread, "event_type": "spawn",
                                       "source": "codex_rollout", "child_session_key": child, "event": result})
                elif call["name"] == "send_input" and result.get("submission_id") and target:
                    observe(target, "queued", start, "send_input", submission_id=result["submission_id"])
                elif call["name"] == "wait_agent":
                    for child, status in (result.get("status") or {}).items():
                        observe(child, status, stamp, "wait_agent", request_start_ns=start)
                elif call["name"] in {"resume_agent", "close_agent"} and target and ("status" in result or "previous_status" in result):
                    status = native_status(result.get("previous_status", result.get("status")))
                    if call["name"] == "close_agent" and status != "completed":
                        status = "closed_incomplete"
                    observe(target, status, stamp, call["name"], request_start_ns=start)
            if start is not None and stamp is not None and stamp >= start:
                spans.append({"name": "agent.tool.call", "canonical_name": "agent.tool.call", "start_unix_nano": start,
                    "end_unix_nano": stamp, "duration_ms": (stamp - start) / 1e6,
                    "attrs": {"agent.session_id": thread, "agent.tool.name": name, "agent.tool.call_id": call["call_id"],
                              "agent.tool.arguments_json": call.get("arguments") or call.get("input"),
                              "agent.tool.result_preview": payload.get("output"), "agent.span.source": "codex.native_rollout"}})
    for turn in session["turns"]:
        start, end = turn["start_timestamp_ns"], turn["terminal_timestamp_ns"]
        if turn["terminal_event"] and start is not None and end is not None and end >= start:
            session["turn_intervals"].append({"turn_id": turn["id"], "start_timestamp_ns": start,
                "end_timestamp_ns": end, "status": "completed" if turn["terminal_event"] == "task_complete" else "interrupted"})
    session.update(latest_turn=latest, final_assistant=latest["final_assistant"] if latest else None,
                   terminal_event=latest["terminal_event"] if latest else None,
                   unclosed_turn_start_ns=latest["start_timestamp_ns"] if latest and latest["is_open"] else None,
                   pending_call_ids=sorted(calls),
                   timing_complete=bool(session["turns"]) and len(session["turn_intervals"]) == len(session["turns"]))
    status = "unknown"
    if latest:
        status = "running" if latest["is_open"] else {"task_complete": "completed", "turn_aborted": "interrupted"}.get(latest["terminal_event"], "unknown")
    if calls:
        status = "pending_calls"
    session["execution_status"] = status
    if parent:
        observed = latest["terminal_timestamp_ns"] if latest and latest["terminal_event"] else (latest["start_timestamp_ns"] if latest else None)
        observe(thread, status, observed,
                "native_latest_turn", turn_id=latest["id"] if latest else None,
                work_activity_ns=max((t for t in (latest["start_timestamp_ns"], latest["last_input_timestamp_ns"]) if t is not None), default=None) if latest else None)
    return session
