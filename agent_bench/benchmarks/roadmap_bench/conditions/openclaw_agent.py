"""RoadmapBench framing for the shared Terminal-Bench OpenClaw runtime."""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

logger = logging.getLogger(__name__)


class RoadmapBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw lead engineer implementing a multi-target roadmap."""

    agent_id = "roadmap_bench"
    require_done_marker = True
    yield_watch_interval_seconds = 2.0
    # OpenClaw's native announce call can itself wait up to 120 seconds. Give
    # it one full timeout plus margin before treating a quiet requester as
    # abandoned after all children have completed.
    yield_resume_idle_timeout_seconds = 180.0
    _ACTIVE_CHILD_STATES = frozenset({"accepted", "queued", "pending", "running"})

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Lead the implementation of this multi-target software roadmap in "
            "the current repository. Inspect the codebase, identify target "
            "workstreams and dependencies, coordinate parallel work where "
            "appropriate, integrate the complete implementation, and validate "
            "all roadmap requirements. Declare DONE only when the integrated "
            "roadmap is complete.\n\n"
            f"Roadmap:\n{instance_data['instruction']}"
        )

    def _await_post_turn_lifecycle(
        self,
        payload: Optional[Dict],
        error: str,
        *,
        deadline: float,
    ) -> Tuple[Optional[Dict], str]:
        """Keep a yielded Roadmap requester alive for native child announces.

        OpenClaw owns continuation: child completion runs an internal follow-up
        turn against the requester session. We only watch its host-mounted
        durable session files, never poll model-visible session tools and never
        inject a synthetic completion message.
        """
        sessions_dir = self._live_sessions_dir()
        if sessions_dir is None:
            return payload, error

        snapshot = self._read_yield_snapshot(sessions_dir)
        if not snapshot or snapshot["yield_count"] == 0:
            return payload, error

        parent_session_id = snapshot["parent_session_id"]
        logger.info(
            "  Roadmap parent yielded; awaiting native announce/resume "
            "(session=%s pending_children=%d deadline_in=%.1fs)",
            parent_session_id,
            snapshot["pending_children"],
            max(0.0, deadline - time.monotonic()),
        )
        last_state = None
        last_progress_at = time.monotonic()
        while time.monotonic() < deadline:
            snapshot = self._read_yield_snapshot(sessions_dir)
            if snapshot:
                if snapshot["parent_session_id"] != parent_session_id:
                    message = (
                        "Roadmap requester session changed during yield/resume: "
                        f"{parent_session_id} -> {snapshot['parent_session_id']}"
                    )
                    self._last_completion_status = self.RUNTIME_ERROR
                    self._last_completion_error = message
                    return payload, message

                state = (
                    snapshot["yield_count"],
                    snapshot["pending_children"],
                    snapshot["message_count"],
                    snapshot.get("transcript_size_bytes", 0),
                    snapshot["post_yield_parent_activity"],
                    snapshot["post_yield_parent_terminal"],
                    snapshot.get("post_yield_terminal_reason", ""),
                    bool(snapshot["done_text"]),
                )
                if state != last_state:
                    last_progress_at = time.monotonic()
                    logger.info(
                        "  Roadmap yield lifecycle: session=%s yields=%d "
                        "pending_children=%d messages=%d resumed=%s "
                        "terminal=%s terminal_reason=%s done=%s",
                        parent_session_id,
                        snapshot["yield_count"],
                        snapshot["pending_children"],
                        snapshot["message_count"],
                        snapshot["post_yield_parent_activity"],
                        snapshot["post_yield_parent_terminal"],
                        snapshot.get("post_yield_terminal_reason", "") or "-",
                        bool(snapshot["done_text"]),
                    )
                    last_state = state

                if snapshot["done_text"]:
                    resumed_payload = self._payload_with_resumed_completion(
                        payload,
                        snapshot["done_text"],
                        yield_count=snapshot["yield_count"],
                        parent_session_id=parent_session_id,
                    )
                    status, completion_error = self._classify_completion(resumed_payload)
                    self._last_completion_status = status
                    self._last_completion_error = completion_error
                    return resumed_payload, completion_error

                if (
                    snapshot["pending_children"] == 0
                    and snapshot["post_yield_parent_terminal"]
                ):
                    terminal_reason = snapshot.get(
                        "post_yield_terminal_reason", "assistant_stop"
                    )
                    message = (
                        "Roadmap parent resumed after child completion but "
                        "returned without DONE "
                        f"(session={parent_session_id}, "
                        f"terminal_reason={terminal_reason})"
                    )
                    # Preserve an execution error already classified from the
                    # CLI payload (for example context_overflow or an LLM idle
                    # timeout). The watcher only decides that waiting is over.
                    if (
                        terminal_reason == "prompt_error"
                        and self._last_completion_status
                        in {self.CONTEXT_OVERFLOW, self.RUNTIME_ERROR}
                    ):
                        completion_error = (
                            self._last_completion_error or error or message
                        )
                        return payload, completion_error
                    self._last_completion_status = self.BLOCKED_INCOMPLETE
                    self._last_completion_error = message
                    return payload, message

                if snapshot["pending_children"] == 0:
                    idle_seconds = time.monotonic() - last_progress_at
                    if idle_seconds >= self.yield_resume_idle_timeout_seconds:
                        message = (
                            "Roadmap parent made no transcript progress for "
                            f"{idle_seconds:.1f}s after all children completed "
                            f"and did not declare DONE (session={parent_session_id})"
                        )
                        self._last_completion_status = self.BLOCKED_INCOMPLETE
                        self._last_completion_error = message
                        return payload, message

            time.sleep(self.yield_watch_interval_seconds)

        message = (
            "Roadmap parent remained yielded until the total agent timeout "
            f"(session={parent_session_id})"
        )
        self._last_exit_code = -1
        self._last_completion_status = self.BLOCKED_INCOMPLETE
        self._last_completion_error = message
        return payload, message

    def _live_sessions_dir(self) -> Optional[Path]:
        state_dir = self._env_info.get("state_dir") or self._env_info.get(
            "host_state_dir"
        )
        if not state_dir:
            return None
        return Path(state_dir) / "agents" / self.agent_id / "sessions"

    @classmethod
    def _read_yield_snapshot(cls, sessions_dir: Path) -> Optional[Dict]:
        index_path = sessions_dir / "sessions.json"
        try:
            index = json.loads(index_path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(index, dict):
            return None

        parent_key = f"agent:{cls.agent_id}:main"
        parent = index.get(parent_key) or {}
        parent_session_id = parent.get("sessionId")
        if not parent_session_id:
            return None

        pending_children = sum(
            1
            for value in index.values()
            if isinstance(value, dict)
            and value.get("spawnedBy") == parent_key
            and str(value.get("status") or "").lower() in cls._ACTIVE_CHILD_STATES
        )

        transcript_path = cls._resolve_parent_transcript_path(
            sessions_dir,
            parent,
        )
        if transcript_path is None:
            return None
        messages = []
        try:
            lines = transcript_path.read_text(errors="replace").splitlines()
            transcript_size_bytes = transcript_path.stat().st_size
        except OSError:
            return None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"message", "custom", "custom_message"}:
                messages.append(event)

        yield_indexes = [
            i
            for i, event in enumerate(messages)
            if event.get("type") == "custom_message"
            and event.get("customType") == "openclaw.sessions_yield"
        ]
        latest_yield = yield_indexes[-1] if yield_indexes else -1
        done_text = ""
        post_yield_parent_activity = False
        terminal_reason = ""
        for event in messages[latest_yield + 1:]:
            if (
                event.get("type") == "custom"
                and event.get("customType") == "openclaw:prompt-error"
            ):
                terminal_reason = "prompt_error"
                continue

            message = event.get("message") or {}
            role = message.get("role")
            # A later requester message means OpenClaw has started another
            # native resume attempt after a prior prompt error.
            if role == "user":
                terminal_reason = ""
            if role in {"assistant", "toolResult"}:
                post_yield_parent_activity = True
            if role != "assistant":
                continue

            # Activity is not completion. An assistant tool-call commonly has
            # stopReason="toolUse" and may be followed by toolResult + more
            # model work. Only a normal assistant stop proves that the resumed
            # parent turn itself has terminated.
            stop_reason = str(message.get("stopReason") or "").strip().lower()
            if stop_reason == "stop":
                terminal_reason = "assistant_stop"
            else:
                terminal_reason = ""

            parts = [
                str(item.get("text") or "")
                for item in message.get("content") or []
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "\n".join(part for part in parts if part).strip()
            if text and re.search(r"(?im)^\s*DONE(?:\s*$|\b)", text):
                done_text = text

        return {
            "parent_session_id": str(parent_session_id),
            "pending_children": pending_children,
            "yield_count": len(yield_indexes),
            "message_count": len(messages),
            "transcript_size_bytes": transcript_size_bytes,
            "post_yield_parent_activity": post_yield_parent_activity,
            "post_yield_parent_terminal": bool(terminal_reason),
            "post_yield_terminal_reason": terminal_reason,
            "done_text": done_text,
        }

    @staticmethod
    def _resolve_parent_transcript_path(
        sessions_dir: Path,
        parent_entry: Dict,
    ) -> Optional[Path]:
        """Resolve OpenClaw's physical transcript with containment checks."""
        sessions_root = sessions_dir.resolve()
        state_root = sessions_root.parents[2]
        raw_session_file = str(parent_entry.get("sessionFile") or "").strip()

        if raw_session_file:
            container_path = PurePosixPath(raw_session_file)
            if container_path.is_absolute():
                runtime_state_root = PurePosixPath("/root/.openclaw")
                try:
                    relative = container_path.relative_to(runtime_state_root)
                except ValueError:
                    return None
                candidate = (state_root / Path(*relative.parts)).resolve()
                containment_root = state_root
            else:
                candidate = (sessions_root / Path(*container_path.parts)).resolve()
                containment_root = sessions_root
        else:
            session_id = str(parent_entry.get("sessionId") or "").strip()
            if not session_id or Path(session_id).name != session_id:
                return None
            candidate = (sessions_root / f"{session_id}.jsonl").resolve()
            containment_root = sessions_root

        try:
            candidate.relative_to(containment_root)
        except ValueError:
            return None
        return candidate

    @classmethod
    def _payload_with_resumed_completion(
        cls,
        payload: Optional[Dict],
        done_text: str,
        *,
        yield_count: int,
        parent_session_id: str,
    ) -> Dict:
        resumed = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        if not isinstance(resumed.get("result"), dict):
            resumed["result"] = {}
        resumed["status"] = "ok"
        run = resumed["result"]
        if not isinstance(run.get("meta"), dict):
            run["meta"] = {}
        meta = run["meta"]
        meta.update({
            "livenessState": "completed",
            "replayInvalid": False,
            "stopReason": "stop",
            "finalAssistantVisibleText": done_text,
            "completion": {"stopReason": "stop", "finishReason": "stop"},
            "agentBenchYieldResume": {
                "yieldCount": yield_count,
                "parentSessionId": parent_session_id,
            },
        })
        meta.pop("error", None)
        if not isinstance(run.get("payloads"), list):
            run["payloads"] = []
        run["payloads"].append({"text": done_text, "mediaUrl": None})
        return resumed
