"""Bind CLI, rollout and API terminal evidence to the final parent turn."""
from __future__ import annotations

from collections.abc import Iterable

from agent_bench.harnesses.codex.telemetry import api_response
from agent_bench.harnesses.codex.messages import continuation_preamble, done_marker


def api_turn_id(row: dict) -> str | None:
    return row.get("turn_id") or ((row.get("request") or {}).get("client_metadata") or {}).get("turn_id")


def belongs_to_turn(row: dict, session: dict) -> bool:
    turn = session.get("latest_turn") or {}
    if row.get("thread_id") != session["id"]:
        return False
    identity = api_turn_id(row)
    if identity:
        return identity == turn.get("id")
    # Older adapter logs lack turn_id. Use only the final native turn window.
    start, end = turn.get("start_timestamp_ns"), turn.get("terminal_timestamp_ns")
    request_start = row.get("start_ns")
    return isinstance(start, int) and isinstance(request_start, int) and request_start >= start and (
        end is None or request_start <= end)


def parent_completion(sessions: list[dict], events: Iterable[dict], api_rows: Iterable[dict]) -> dict:
    roots = [s for s in sessions if not s["parent_thread_id"]]
    evidence = {"complete": False, "done_marker": False, "continuation_eligible": False, "terminal_evidence": "missing_parent_turn",
                "completion_error": "Codex has no unique final parent turn"}
    if len(roots) != 1 or not roots[0].get("latest_turn"):
        return evidence
    root, turn = roots[0], roots[0]["latest_turn"]
    evidence.update(parent_thread_id=root["id"], parent_turn_id=turn["id"],
                    final_assistant=turn.get("final_assistant"), pending_call_ids=root.get("pending_call_ids", []))

    def blocked(reason):
        evidence.update(terminal_evidence=reason, completion_error=f"Codex final parent turn: {reason}")
        return evidence

    if turn["is_open"] or turn["terminal_event"] != "task_complete" or not turn["start_observed"]:
        return blocked("native_parent_turn_unfinished")
    if any(t["is_open"] for t in root["turns"]):
        return blocked("unclosed_parent_turn")
    if root.get("pending_call_ids"):
        return blocked("pending_parent_tool_calls")
    final = str(turn.get("last_assistant_message") or turn.get("final_assistant") or "")
    evidence["last_assistant_message"] = final
    if turn.get("last_response_type") != "message" or not final.strip():
        return blocked("empty_or_nonfinal_parent_message")
    # Exec JSON lacks turn IDs in this release. Match the final started turn
    # ordinal, then require its own terminal event, never any earlier completed.
    started, terminal, thread = 0, None, None
    for event in events:
        kind = event.get("type")
        if kind == "thread.started":
            thread = event.get("thread_id")
        elif kind == "turn.started":
            started += 1
            terminal = None
        elif kind in {"turn.completed", "turn.failed", "error"}:
            terminal = kind
    if thread not in {None, root["id"]} or started != turn["index"] + 1 or terminal != "turn.completed":
        return blocked("missing_final_cli_turn_completed")
    pending, last = set(), None
    for row in api_rows:
        if not belongs_to_turn(row, root):
            continue
        if row.get("event") == "request_start":
            pending.add(row["start_ns"])
        if "end_ns" in row:
            pending.discard(row["start_ns"])
            if last is None or row["start_ns"] > last["start_ns"]:
                last = row
    if pending:
        return blocked("pending_parent_model_requests")
    if last is None:
        return blocked("missing_final_parent_response")
    response = api_response(last)
    evidence["response_status"] = response.get("status")
    if last.get("error") or response.get("status") not in {"completed", "incomplete"}:
        return blocked("failed_final_parent_response")
    output = response.get("output") or []
    if any(p.get("type") in {"function_call", "custom_tool_call"} for p in output):
        return blocked("final_parent_response_has_tool_calls")
    final_output = output[-1] if output else {}
    visible = "\n".join(p.get("text", "") for p in final_output.get("content") or [] if p.get("type") == "output_text")
    if final_output.get("type") != "message" or final_output.get("role") != "assistant" or visible != final:
        return blocked("final_parent_response_message_mismatch")
    if turn.get("final_item_id") and final_output.get("id") != turn["final_item_id"]:
        return blocked("final_parent_response_item_mismatch")
    marker = done_marker(final)
    evidence["done_marker"] = marker
    if continuation_preamble(final):
        evidence["continuation_eligible"] = response["status"] == "completed"
        return blocked("final_parent_message_continues_work")
    if final_output.get("phase") == "commentary" or turn.get("last_assistant_phase") == "commentary":
        evidence["continuation_eligible"] = response["status"] == "completed"
        return blocked("nonfinal_parent_commentary")
    if response["status"] == "incomplete" and not marker:
        return blocked("incomplete_final_parent_response")
    evidence.update(complete=True, completion_error=None, terminal_evidence=(
        "codex_final_parent_explicit_done" if marker else "codex_final_parent_assistant_stop"))
    return evidence
