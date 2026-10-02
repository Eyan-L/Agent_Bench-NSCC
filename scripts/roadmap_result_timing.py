"""Evidence-based OpenClaw child completion, delivery, and parent follow-up.

Delivery is a recorded context event, follow-up is an observed response, and
explicit access is a session-tool operation. None proves semantic adoption.
"""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY = {
    "completion": "runtime child terminal timestamp; censored endpoints are not completion",
    "delivery": "native subagent_announce provenance plus identity-bound envelope or unique batch label",
    "parent_followup": "first nonempty parent assistant response descending from a delivery event; not semantic adoption",
    "explicit_access": "identity-bound sessions_history result containing the child's final assistant response; truncated reads are explicitly marked partial",
    "explicit_reference": "parent text names the child session or unique full label; textual evidence only",
    "stream_windows": "request first-response-byte through request end; not per-token timestamps",
    "missing": "unknown; never replaced with zero or parent completion",
    "slack": "signed completion-to-delivery and delivery-to-followup intervals, reported separately",
    "result_ready": "child final assistant response matched to delivered result text; distinct from latest session endedAt after follow-ups",
}


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
    return out


def _ns(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1e9)
    except ValueError:
        return None


def _text(message: dict) -> str:
    content = message.get("content") or []
    if isinstance(content, str):
        return content
    return "\n".join(str(c.get("text") or "") for c in content if isinstance(c, dict))


def _payload(message: dict) -> bool:
    return bool(_text(message).strip()) or any(
        isinstance(c, dict) and c.get("type") == "toolCall"
        for c in (message.get("content") or []) if isinstance(message.get("content"), list)
    )


def _evidence(event: dict, source: str, quality: str) -> dict:
    return {"timestamp_ns": _ns(event.get("timestamp")), "event_id": event.get("id"),
            "parent_event_id": event.get("parentId"), "source": source, "quality": quality}


def _delta(start: int | None, end: int | None) -> float | None:
    return (end - start) / 1e9 if start is not None and end is not None else None


def _object(message: dict) -> dict:
    try:
        value = json.loads(_text(message))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _normalized(text: str) -> str:
    return " ".join(text.replace("\\n", "\n").split())


def _delivered_body(text: str, label: str) -> str:
    if text.startswith("[Subagent Context]"):
        label_match = re.search(r"(?m)^\d+\. " + re.escape(label) + r"\s*$", text)
        if not label_match:
            return ""
        text = text[label_match.end():]
    match = re.search(r"<prompt-data>\s*(.*?)\s*</prompt-data>", text, re.S)
    return match.group(1) if match else ""


def _followup(delivery: dict, events: list[dict]) -> dict | None:
    """Require transcript ancestry and a model start after the delivery record."""
    by_id = {str(e.get("id")): e for e in events if e.get("id")}
    delivered = delivery["timestamp_ns"]
    for event in events:
        message = event.get("message") or {}
        if message.get("role") != "assistant" or not _payload(message):
            continue
        if _text(message).strip() == "NO_REPLY":
            continue
        start = int(message["timestamp"] * 1e6) if isinstance(message.get("timestamp"), (int, float)) else None
        if start is None or delivered is None or start < delivered:
            continue
        cursor = str(event.get("parentId") or "")
        seen = set()
        while cursor and cursor not in seen:
            if cursor == delivery["event_id"]:
                result = _evidence(event, "parent_transcript", "descendant_assistant_model_start")
                result["timestamp_ns"] = start
                result["response_end_ns"] = _ns(event.get("timestamp"))
                result["control_only_no_reply"] = _text(message).strip() == "NO_REPLY"
                return result
            seen.add(cursor)
            cursor = str((by_id.get(cursor) or {}).get("parentId") or "")
    return None


def _requests(spans: list[dict], sessions: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    messages = []
    for sid, events in sessions.items():
        for event in events:
            m = event.get("message") or {}
            if m.get("role") == "assistant" and _payload(m) and isinstance(m.get("timestamp"), (int, float)):
                messages.append((sid, event, int(m["timestamp"] * 1e6), _ns(event.get("timestamp"))))
    output, used = [], set()
    counts = Counter()
    for span in spans:
        attrs = span.get("attrs") or {}
        if span.get("name") != "openclaw.model.call" or attrs.get("openclaw.model_call.observation_unit") != "request":
            continue
        counts["request_count"] += 1
        start, end = _ns(span.get("start_unix_nano")), _ns(span.get("end_unix_nano"))
        if start is None or end is None or end <= start:
            counts["invalid_timing_count"] += 1
            continue
        candidates = [(sid, e) for sid, e, ms, me in messages
                      if me is not None and -50e6 <= ms-start <= 150e6 and -20e6 <= me-end <= 100e6]
        if len(candidates) != 1:
            counts["ambiguous_or_unmatched_count"] += 1
            continue
        sid, event = candidates[0]
        key = (sid, event.get("id"))
        if key in used:
            counts["duplicate_match_count"] += 1
            continue
        used.add(key)
        first = attrs.get("openclaw.model_call.time_to_first_byte_ms")
        first_ns = start + int(float(first) * 1e6) if first is not None else None
        if first_ns is not None and not start <= first_ns <= end:
            first_ns = None
        output.append({"session_id": sid, "event_id": event.get("id"), "start_ns": start,
                       "end_ns": end, "first_byte_ns": first_ns,
                       "response_bytes": int(attrs.get("openclaw.model_call.response_bytes") or 0),
                       "quality": "unique_start_and_end_timestamp_match"})
    counts["matched_request_count"] = len(output)
    return output, dict(counts)


def analyze_task(run_dir: Path, observation: dict, manifest: dict,
                 *, sessions_dir: Path | None = None, spans_path: Path | None = None) -> dict:
    sessions_dir = sessions_dir or run_dir / "conversation/sessions"
    agents = (observation.get("agent_timeline") or {}).get("agents") or []
    parent = next((a for a in agents if a.get("role") == "parent"), {})
    parent_id = str(parent.get("session_id") or manifest.get("run_id") or "")
    events = _rows(sessions_dir / f"{parent_id}.jsonl")
    child_entries = observation.get("child_runtime_by_session") or {}
    sessions = {parent_id: events}
    labels = Counter(str(c.get("label") or "") for c in child_entries.values())
    span_rows = _rows(spans_path or run_dir / "spans.jsonl")
    exact_tools = {}
    for s in span_rows:
        a = s.get("attrs") or {}
        if s.get("name") == "openclaw.tool.call" and str(a.get("openclaw.sessionId") or "") == parent_id:
            call = a.get("openclaw.tool.callId")
            if call:
                exact_tools[str(call)] = s
    calls = {}
    for e in events:
        m = e.get("message") or {}
        if m.get("role") != "assistant":
            continue
        for c in m.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "toolCall":
                calls[str(c.get("id") or "")] = (e, c)

    children = []
    for session_key, child in child_entries.items():
        sid, label = str(child.get("session_id") or ""), str(child.get("label") or "")
        child_events = _rows(sessions_dir / f"{sid}.jsonl")
        sessions[sid] = child_events
        terminal = _ns(child.get("terminal_timestamp_ns")) if child.get("runtime_complete") else None
        finals = [e for e in child_events if (e.get("message") or {}).get("role") == "assistant"
                  and (e.get("message") or {}).get("stopReason") in {"stop", "end_turn"}
                  and _text(e.get("message") or {}).strip()
                  and _text(e.get("message") or {}).strip() not in {"NO_REPLY", "ANNOUNCE_SKIP"}
                  and not any(c.get("type") == "toolCall" for c in (e.get("message") or {}).get("content", []) if isinstance(c, dict))]
        final = finals[-1] if finals else None
        final_msg = (final or {}).get("message") or {}
        final_text, final_ms = _text(final_msg), final_msg.get("timestamp")
        notices, accesses, references = [], [], []
        spawn = None
        for e in events:
            m = e.get("message") or {}
            role, text = m.get("role"), _text(m)
            prov = m.get("provenance") or {}
            native = prov.get("kind") == "inter_session" and prov.get("sourceTool") == "subagent_announce"
            if role == "user" and native:
                header = text.split("Child result", 1)[0]
                identity = re.search(r"(?m)^session_key:\s*" + re.escape(session_key) + r"\s*$", header)
                batched = text.startswith("[Subagent Context]") and "Child completion results:" in text
                match_label = bool(label and labels[label] == 1 and re.search(r"(?m)^\d+\. " + re.escape(label) + r"\s*$", text))
                if ("[Internal task completion event]" in header and identity) or (batched and match_label):
                    n = _evidence(e, "parent_transcript", "native_envelope_session_key" if identity else "native_batch_unique_label")
                    n["kind"] = "individual_announce" if identity else "batch_settlement"
                    n["provenance"] = prov
                    n["_delivered_body"] = _delivered_body(text, label)
                    notices.append(n)
            if role == "toolResult":
                result = _object(m)
                if m.get("toolName") == "sessions_spawn" and result.get("childSessionKey") == session_key:
                    span = exact_tools.get(str(m.get("toolCallId") or ""))
                    spawn = _evidence(e, "parent_spawn_tool_result", "accepted_child_session_key")
                    spawn["accepted_ns"] = spawn["timestamp_ns"]
                    spawn["call_id"] = m.get("toolCallId")
                    spawn["run_id"] = result.get("runId")
                    spawn["call_start_ns"] = _ns((span or {}).get("start_unix_nano"))
                    spawn["call_start_quality"] = "tool_span" if span else "unknown"
                if m.get("toolName") == "sessions_history" and result.get("sessionKey") == session_key:
                    returned = result.get("messages") or []
                    final_matches = [(r, cf) for r in returned for cf in finals
                                     if isinstance(r, dict) and r.get("role") == "assistant"
                                     and (cf.get("message") or {}).get("timestamp") is not None
                                     and r.get("timestamp") == (cf.get("message") or {}).get("timestamp")
                                     and (_normalized(_text(r)) == _normalized(_text(cf["message"]))
                                          or (result.get("contentTruncated") is True and len(_normalized(_text(r))) >= 80
                                              and _normalized(_text(r))[:min(200,len(_normalized(_text(cf["message"]))))] == _normalized(_text(cf["message"]))[:min(200,len(_normalized(_text(cf["message"]))))]))]
                    if final_matches:
                        partial = not any(_normalized(_text(r)) == _normalized(_text(cf["message"])) for r, cf in final_matches)
                        a = _evidence(e, "parent_sessions_history_result", "final_message_timestamp_and_truncated_prefix" if partial else "exact_final_message_timestamp_and_text")
                        a["partial_result"] = partial
                        a["child_final_response_event_ids"] = sorted({str(cf.get("id")) for _, cf in final_matches})
                        a["call_id"] = m.get("toolCallId")
                        span = exact_tools.get(str(m.get("toolCallId") or ""))
                        a["call_start_ns"] = _ns((span or {}).get("start_unix_nano"))
                        accesses.append(a)
            if role == "assistant" and terminal is not None and (_ns(e.get("timestamp")) or 0) >= terminal:
                tokens = [session_key, sid] + ([label] if label and labels[label] == 1 and len(label) >= 8 else [])
                named = next((t for t in tokens if t and t in text), None)
                if named:
                    ref = _evidence(e, "parent_assistant_text", "exact_child_identifier_or_unique_full_label")
                    ref.update({"matched_text": named, "excerpt": text[:600]})
                    references.append(ref)
        notice = min(notices, key=lambda n: n["timestamp_ns"] or 0) if notices else None
        delivered_body = _normalized((notice or {}).get("_delivered_body", ""))
        result_matches = [cf for cf in finals if delivered_body and len(delivered_body) >= 80
                          and _normalized(_text(cf["message"]))[:min(200,len(delivered_body))] == delivered_body[:min(200,len(delivered_body))]
                          and (_ns(cf.get("timestamp")) or 0) <= (notice or {}).get("timestamp_ns", 0)]
        ready_event = result_matches[0] if len(result_matches) == 1 else None
        ready_ns = _ns((ready_event or {}).get("timestamp"))
        for n in notices:
            n.pop("_delivered_body", None)
        follow = _followup(notice, events) if notice else None
        access = min(accesses, key=lambda a: a["timestamp_ns"] or 0) if accesses else None
        reference = min(references, key=lambda a: a["timestamp_ns"] or 0) if references else None
        delivered = (notice or {}).get("timestamp_ns")
        responded = (follow or {}).get("timestamp_ns")
        delays = {"completion_to_delivery_seconds": _delta(terminal, delivered),
                  "result_ready_to_delivery_seconds": _delta(ready_ns, delivered),
                  "delivery_to_parent_followup_seconds": _delta(delivered, responded),
                  "completion_to_explicit_access_seconds": _delta(terminal, (access or {}).get("timestamp_ns")),
                  "completion_to_explicit_reference_seconds": _delta(terminal, (reference or {}).get("timestamp_ns"))}
        children.append({"session_key": session_key, "session_id": sid, "label": label,
                         "status": child.get("status"), "spawn": spawn,
                         "start_ns": _ns(child.get("start_timestamp_ns")), "completion_ns": terminal,
                         "last_observed_ns": _ns(child.get("last_observed_timestamp_ns")),
                         "final_response_recorded_ns": _ns((final or {}).get("timestamp")),
                         "result_ready_ns": ready_ns,
                         "result_ready_event_id": (ready_event or {}).get("id"),
                         "result_ready_quality": "matched_delivered_final_response_text" if ready_event else "unknown",
                         "child_final_response_events": [{"event_id": cf.get("id"), "timestamp_ns": _ns(cf.get("timestamp")), "run_id": (cf["message"].get("__openclaw") or {}).get("runId")} for cf in finals],
                         "completion_quality": child.get("terminal_timestamp_quality") if terminal else "unknown_or_censored",
                         "completion_source": child.get("terminal_timestamp_source"),
                         "deliveries": notices, "first_delivery": notice, "first_parent_followup": follow,
                         "explicit_final_result_accesses": accesses, "first_explicit_final_result_access": access,
                         "first_explicit_text_reference": reference, "delays": delays,
                         "timing_conflict": any(v is not None and v < 0 for v in delays.values()),
                         "session_end_after_first_delivery": terminal is not None and delivered is not None and terminal > delivered,
                         "semantic_adoption_ns": None, "semantic_adoption_quality": "not_observable_from_timestamps"})
    requests, request_quality = _requests(span_rows, sessions)
    return {"instance_id": manifest.get("instance_id"), "run_id": manifest.get("run_id"),
            "parent_session_id": parent_id,
            "execution_start_ns": _ns((manifest.get("phase_timing") or {}).get("agent_execution_start_ns")) or _ns(parent.get("start_timestamp_ns")),
            "execution_end_ns": _ns((manifest.get("phase_timing") or {}).get("agent_execution_end_ns")) or _ns(parent.get("end_timestamp_ns")),
            "children": children, "model_requests": requests, "model_request_attribution": request_quality}


def summarize(tasks: list[dict]) -> dict:
    children = [c for t in tasks for c in t["children"]]
    return {"tasks": len(tasks), "children": len(children),
            "children_with_completion": sum(c["completion_ns"] is not None for c in children),
            "children_with_matched_result_ready": sum(c["result_ready_ns"] is not None for c in children),
            "children_with_delivery": sum(c["first_delivery"] is not None for c in children),
            "children_with_parent_followup": sum(c["first_parent_followup"] is not None for c in children),
            "children_with_explicit_final_result_access": sum(c["first_explicit_final_result_access"] is not None for c in children),
            "children_with_explicit_text_reference": sum(c["first_explicit_text_reference"] is not None for c in children),
            "children_with_timing_conflict": sum(c["timing_conflict"] for c in children)}


def analyze_records(records, runtime_observability) -> list[dict]:
    """Share selected-run timing collection between analyze and compare."""
    tasks = []
    for record in records:
        # This reader understands OpenClaw durable sessions, not other harnesses.
        if record.manifest.get("agent_runtime_mode") in {"hermes", "codex"}:
            continue
        task = analyze_task(
            record.run_dir, runtime_observability(record)[0], record.manifest
        )
        task.update(arm=record.arm, seed=record.seed,
                    run_id=record.manifest.get("run_id") or record.run_dir.name,
                    job_id=record.job_id)
        tasks.append(task)
    return tasks


def render_markdown(export: dict) -> str:
    return (
        "\n## Child completion, result delivery and parent follow-up\n\n"
        "Child completion, result notification, the first subsequent parent response, "
        "and an explicit read of the final child result are separate observations. "
        "A response after delivery does not prove semantic adoption; missing evidence remains unknown.\n\n"
        f"- [Timeline and evidence viewer]({export['viewer']})\n"
        f"- [Per-child timing CSV]({export['csv']})\n"
        f"- [Detailed timing JSON]({export['json']})\n"
        f"- Coverage: `{export['summary']}`\n"
    )


def _task_title(task: dict) -> str:
    title = str(task["instance_id"])
    if task.get("arm"):
        title += f" — {task['arm']} / seed {task.get('seed', 'unknown')}"
    return title


def plot_tasks(tasks: list[dict], path: Path, *, dpi: int = 180, compact: bool = False) -> bool:
    try:
        import matplotlib
    except ImportError:
        print("WARNING: result-timing plots require matplotlib; JSON/CSV remain available.")
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not tasks or not any(t["children"] for t in tasks):
        return False
    height = (1.35*len(tasks)+.24*sum(len(t["children"]) for t in tasks)+.9) if compact else sum(1.45 + .3*len(t["children"]) for t in tasks)+1
    fig, axes = plt.subplots(len(tasks), 1, figsize=(13, height), squeeze=False)
    palette = {"main": "#2453A6", "child": "#139C91", "wait": "#E99C36", "after": "#9CA6B8"}
    for axrow, task in zip(axes, tasks):
        ax = axrow[0]
        origin = task["execution_start_ns"]
        if origin is None:
            ax.set_visible(False)
            continue
        sec = lambda ns: (ns-origin)/1e9
        end = task["execution_end_ns"]
        labels = ["Main"] + [f"S{i+1}" for i in range(len(task["children"]))]
        sid_to_y = {task["parent_session_id"]: 0}
        if end:
            ax.hlines(0, 0, sec(end), color=palette["main"], linewidth=8, alpha=.15)
        for i, c in enumerate(task["children"], 1):
            sid_to_y[c["session_id"]] = i
            start, done = c["start_ns"], c["completion_ns"]
            observed_end = done or c["last_observed_ns"]
            if start and observed_end and observed_end >= start:
                ax.hlines(i, sec(start), sec(observed_end), color=palette["child"], linewidth=8, alpha=.23,
                          linestyles="solid" if done else "dashed")
            spawn = (c.get("spawn") or {}).get("call_start_ns")
            if spawn is not None and start is not None:
                ax.plot([sec(spawn),sec(start)],[0,i],color="#BBC2CE",linewidth=.7,linestyle=":")
            ready = c.get("result_ready_ns")
            delivery = (c["first_delivery"] or {}).get("timestamp_ns")
            follow = (c["first_parent_followup"] or {}).get("timestamp_ns")
            access = (c["first_explicit_final_result_access"] or {}).get("timestamp_ns")
            if ready is not None and delivery is not None and delivery >= ready:
                ax.hlines(i,sec(ready),sec(delivery),color=palette["wait"],linewidth=5,linestyle="--")
                delay = c["delays"]["result_ready_to_delivery_seconds"]
                if delay is not None and delay>=10:
                    ax.text((sec(ready)+sec(delivery))/2,i-.2,f"{delay:.0f}s to notice",fontsize=10 if compact else 8,ha="center",color="#9C601C")
            if delivery is not None and follow is not None and follow >= delivery:
                ax.hlines(i,sec(delivery),sec(follow),color=palette["after"],linewidth=3,linestyle=":")
            for stamp,marker,color in [(done,"o",palette["child"]),(ready,"s",palette["child"]),(delivery,"v",palette["wait"]),(follow,"D",palette["main"]),(access,"*","#AD4F8F")]:
                if stamp is not None:
                    ax.scatter(sec(stamp),i,marker=marker,s=35,zorder=5,color=color,edgecolors="white",linewidths=.4)
        for r in task["model_requests"]:
            y=sid_to_y.get(r["session_id"])
            first=r.get("first_byte_ns")
            if y is not None and first is not None and r["response_bytes"]>0:
                ax.hlines(y,sec(first),sec(r["end_ns"]),color=palette["main"] if y==0 else palette["child"],linewidth=4,zorder=4)
        ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis();ax.set_ylim(len(labels)-.45,-.55)
        ax.set_title(_task_title(task).replace("-roadmap",""),loc="left",fontweight="bold",fontsize=11)
        ax.set_xlabel("Seconds since main agent execution start",fontsize=9 if compact else 8)
        ax.grid(axis="x",alpha=.14);ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=10 if compact else 8)
    legend=[Line2D([0],[0],color=palette["main"],lw=4,label="LLM response window"),
            Line2D([0],[0],color="#AABBD0",lw=7,alpha=.4,label="Session lifetime"),
            Line2D([0],[0],marker="s",color=palette["child"],linestyle="",label="Child result ready (circle: session ended)"),
            Line2D([0],[0],marker="v",color=palette["wait"],linestyle="",label="Result notice recorded"),
            Line2D([0],[0],marker="D",color=palette["main"],linestyle="",label="Main follow-up starts"),
            Line2D([0],[0],marker="*",color="#AD4F8F",linestyle="",label="Explicit final-result read")]
    fig.legend(handles=legend,loc="lower center",ncol=3,fontsize=9 if compact else 8,frameon=False,bbox_to_anchor=(.5,.007))
    fig.tight_layout(rect=(0,.09,1,1),h_pad=1.5)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,bbox_inches="tight");plt.close(fig)
    return True


def export_analysis(tasks: list[dict], bundle: Path, name: str, *, plots: bool = True, dpi: int = 180) -> dict:
    destination = bundle / "02_timeline_model_trace" / "result_timing"
    destination.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "policy": POLICY, "summary": summarize(tasks), "tasks": tasks}
    json_path=destination/"result-timing.json"
    json_path.write_text(json.dumps(result,indent=2)+"\n")
    csv_path=destination/"result-timing.csv"
    fields=["instance_id","arm","seed","run_id","job_id","child_session_id","label","spawn_call_ns","spawn_accepted_ns","child_start_ns","completion_ns","completion_quality","delivery_ns","delivery_kind","delivery_event_id","parent_followup_start_ns","parent_followup_event_id","explicit_final_result_access_ns","explicit_access_event_id","explicit_text_reference_ns","semantic_adoption_ns",*next(iter((c["delays"] for t in tasks for c in t["children"])),{}),"timing_conflict"]
    table=[]
    for t in tasks:
        for c in t["children"]:
            n=c["first_delivery"] or {};f=c["first_parent_followup"] or {};a=c["first_explicit_final_result_access"] or {};r=c["first_explicit_text_reference"] or {};s=c["spawn"] or {}
            table.append({"instance_id":t["instance_id"],"arm":t.get("arm"),"seed":t.get("seed"),"run_id":t.get("run_id"),"job_id":t.get("job_id"),"child_session_id":c["session_id"],"label":c["label"],"spawn_call_ns":s.get("call_start_ns"),"spawn_accepted_ns":s.get("accepted_ns"),"child_start_ns":c["start_ns"],"completion_ns":c["completion_ns"],"completion_quality":c["completion_quality"],"delivery_ns":n.get("timestamp_ns"),"delivery_kind":n.get("kind"),"delivery_event_id":n.get("event_id"),"parent_followup_start_ns":f.get("timestamp_ns"),"parent_followup_event_id":f.get("event_id"),"explicit_final_result_access_ns":a.get("timestamp_ns"),"explicit_access_event_id":a.get("event_id"),"explicit_text_reference_ns":r.get("timestamp_ns"),"semantic_adoption_ns":None,**c["delays"],"timing_conflict":c["timing_conflict"]})
    for row in table:
        for field in ["completion_ns", "delivery_ns", "parent_followup_start_ns", "explicit_final_result_access_ns"]:
            value = row[field]
            iso_field = field.removesuffix("_ns") + "_utc"
            row[iso_field] = datetime.fromtimestamp(value / 1e9, timezone.utc).isoformat(timespec="milliseconds") if value is not None else None
            if iso_field not in fields:
                fields.append(iso_field)
    for row, child in zip(table, [c for t in tasks for c in t["children"]]):
        row["result_ready_ns"] = child["result_ready_ns"]
        row["result_ready_event_id"] = child["result_ready_event_id"]
        row["explicit_access_partial"] = (child["first_explicit_final_result_access"] or {}).get("partial_result")
    fields.extend(["result_ready_ns", "result_ready_event_id", "explicit_access_partial"])
    with csv_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
    figures=[]
    figure_by_task = {}
    if plots:
        for i,t in enumerate(tasks):
            path=bundle/"05_figures"/f"{name}-result-timing-{i+1:02d}-{t['instance_id']}.png"
            if plot_tasks([t],path,dpi=dpi):
                figures.append(path)
                figure_by_task[i] = path
    def fmt(v):return "unknown" if v is None else f"{v:.3f}"
    sections=[]
    for i, t in enumerate(tasks):
        pic=figure_by_task.get(i)
        body=f'<img alt="Result timing" src="../../{pic.relative_to(bundle).as_posix()}" style="width:100%;max-width:1200px">' if pic else ''
        body+='<p>All event times below are seconds since main agent execution start. Result readiness is matched to the delivered child response; the latest session end may occur later after follow-up work.</p><table><tr><th>Child</th><th>Result ready at</th><th>Latest session ended at</th><th>Notice at</th><th>Main response starts at</th><th>Explicit read at</th><th>Ready → notice</th><th>Notice → main response</th><th>Evidence</th></tr>'
        for c in t['children']:
            d=c['delays'];e=html.escape(json.dumps(c,indent=2))
            times = [c["result_ready_ns"], c["completion_ns"], (c["first_delivery"] or {}).get("timestamp_ns"), (c["first_parent_followup"] or {}).get("timestamp_ns"), (c["first_explicit_final_result_access"] or {}).get("timestamp_ns")]
            cells = ''.join(f'<td>{fmt(_delta(t["execution_start_ns"], v))}</td>' for v in times)
            body+=f'<tr><td>{html.escape(c["label"])}</td>{cells}<td>{fmt(d["result_ready_to_delivery_seconds"])} s</td><td>{fmt(d["delivery_to_parent_followup_seconds"])} s</td><td><details><summary>Session IDs, event IDs, timestamps and quality</summary><pre>{e}</pre></details></td></tr>'
        body+='</table>'
        sections.append(f'<section><h2>{html.escape(_task_title(t))}</h2>{body}</section>')
    viewer=destination/'index.html'
    viewer.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Child result delivery and parent follow-up</title><style>body{font:15px Arial;margin:30px;color:#21324a}section{margin:40px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #d7dee8;text-align:left}pre{white-space:pre-wrap;max-width:850px}summary{cursor:pointer}</style><h1>Child result delivery and parent follow-up</h1><p>Faint lines show session lifetime; dark bars show LLM response windows. Completion, notification, first parent follow-up, and explicit final-result reads are separate observations. Timing alone does not establish semantic adoption. Unknown stays unknown; negative delays indicate conflicting or repeated-run evidence.</p><p><a href="result-timing.csv">Child timing CSV</a> · <a href="result-timing.json">Full evidence JSON</a></p>'+''.join(sections)+'</html>')
    return {"policy":POLICY,"summary":result["summary"],"json":str(json_path.relative_to(bundle)),"csv":str(csv_path.relative_to(bundle)),"viewer":str(viewer.relative_to(bundle)),"figure_paths":figures}
