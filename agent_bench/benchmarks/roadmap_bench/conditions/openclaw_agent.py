"""RoadmapBench framing for the shared Terminal-Bench OpenClaw runtime."""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple

from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

logger = logging.getLogger(__name__)

_WORKSPACE_SCAFFOLD_FILES = frozenset({
    "AGENTS.md", "BOOTSTRAP.md", "HEARTBEAT.md", "IDENTITY.md",
    "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md",
})


class RoadmapBenchOpenClawCondition(OpenClawAgentCondition):
    """OpenClaw lead engineer implementing a multi-target roadmap."""

    agent_id = "roadmap_bench"
    require_done_marker = True
    yield_watch_interval_seconds = 2.0
    # A prompt error may be transient: OpenClaw can retry/fail over and append
    # another requester turn afterward. Only treat an unchanged prompt error
    # as final after one model idle-timeout window plus margin.
    prompt_error_grace_seconds = 600.0
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

    def _collect_artifacts(
        self,
        container_id: Optional[str],
        env_info: Dict,
        run_dir: Path,
        workspace_root: str,
    ) -> Optional[str]:
        """Capture tracked and agent-created untracked repository changes."""
        del container_id, run_dir, workspace_root
        workspace = Path(str(env_info.get("workspace_host_path") or ""))
        if not workspace.is_dir():
            logger.warning(
                "Roadmap patch extraction missing host workspace: %s",
                workspace,
            )
            return ""
        return self._collect_workspace_patch(workspace)

    @staticmethod
    def _collect_workspace_patch(workspace: Path) -> str:
        root_result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if root_result.returncode != 0:
            logger.warning(
                "Roadmap workspace is not a git repository: %s",
                root_result.stderr.strip(),
            )
            return ""

        repo_root = Path(root_result.stdout.strip()).resolve()
        try:
            workspace.resolve().relative_to(repo_root)
        except ValueError:
            logger.warning(
                "Roadmap workspace %s is outside git root %s",
                workspace,
                repo_root,
            )
            return ""

        excludes = [
            f":(exclude){name}" for name in sorted(_WORKSPACE_SCAFFOLD_FILES)
        ]
        tracked = subprocess.run(
            [
                "git", "-C", str(repo_root), "diff", "--binary", "HEAD",
                "--", ".", *excludes,
            ],
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            logger.warning(
                "Roadmap tracked patch extraction failed: %s",
                tracked.stderr.strip(),
            )
            return ""

        untracked = subprocess.run(
            [
                "git", "-C", str(repo_root), "ls-files", "--others",
                "--exclude-standard", "-z",
            ],
            capture_output=True,
        )
        if untracked.returncode != 0:
            logger.warning("Roadmap untracked file enumeration failed")
            return tracked.stdout

        patch_parts = [tracked.stdout]
        for raw_path in untracked.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = os.fsdecode(raw_path)
            if Path(relative).parts[0] in _WORKSPACE_SCAFFOLD_FILES:
                continue
            diff = subprocess.run(
                [
                    "git", "diff", "--no-index", "--binary", "--",
                    "/dev/null", relative,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            # git diff --no-index returns 1 when differences were found.
            if diff.returncode in (0, 1):
                patch_parts.append(diff.stdout)
            else:
                logger.warning(
                    "Roadmap untracked patch extraction failed for %s: %s",
                    relative,
                    diff.stderr.strip(),
                )

        return "".join(part for part in patch_parts if part)

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

        if getattr(self, "_last_completion_status", None) == self.TIMEOUT:
            return payload, getattr(self, "_last_completion_error", error)

        # The CLI payload can lag behind the durable requester transcript. In
        # particular, a no-yield requester can have appended a terminal DONE
        # while the payload still reports livenessState=working. Explicit
        # payload failures remain authoritative and must never be hidden by an
        # older transcript terminal event.
        payload_status, payload_error = self._classify_completion(payload)
        if payload_status in {self.RUNTIME_ERROR, self.CONTEXT_OVERFLOW}:
            self._last_completion_status = payload_status
            self._last_completion_error = payload_error
            return payload, payload_error

        snapshot = self._read_yield_snapshot(sessions_dir)
        if not snapshot:
            return payload, error

        reconciled = self._reconcile_terminal_snapshot(payload, snapshot)
        if reconciled is not None:
            return reconciled

        # Ordinary single-turn runs retain the base condition's behavior. A
        # child trajectory, yield, or retryable prompt error makes this a
        # multi-turn logical run even when the parent never called yield.
        should_watch = bool(
            snapshot.get("total_children")
            or snapshot.get("yield_count")
            or snapshot.get("post_yield_prompt_error")
        )
        if not should_watch:
            return payload, error

        parent_session_id = snapshot["parent_session_id"]
        logger.info(
            "  Roadmap trajectory awaiting native child/parent progress "
            "(session=%s yields=%d pending_children=%d deadline_in=%.1fs)",
            parent_session_id,
            snapshot["yield_count"],
            snapshot["pending_children"],
            max(0.0, deadline - time.monotonic()),
        )

        last_state = None
        prompt_error_signature = None
        prompt_error_since = None

        while time.monotonic() < deadline:
            snapshot = self._read_yield_snapshot(sessions_dir)
            if snapshot:
                if snapshot["parent_session_id"] != parent_session_id:
                    message = (
                        "Roadmap requester session changed during yield/resume: "
                        f"{parent_session_id} -> "
                        f"{snapshot['parent_session_id']}"
                    )
                    self._last_completion_status = self.RUNTIME_ERROR
                    self._last_completion_error = message
                    return payload, message

                state = (
                    snapshot["yield_count"],
                    snapshot["pending_children"],
                    snapshot.get("completed_children", 0),
                    snapshot["message_count"],
                    snapshot.get("transcript_size_bytes", 0),
                    snapshot.get("parent_active", False),
                    snapshot.get("parent_waiting", False),
                    snapshot.get("parent_resumed", False),
                    snapshot.get("parent_terminal", False),
                    bool(snapshot.get("post_yield_prompt_error")),
                    bool(snapshot["done_text"]),
                )

                if state != last_state:
                    logger.info(
                        "  Roadmap yield lifecycle: session=%s yields=%d "
                        "pending_children=%d completed_children=%d messages=%d "
                        "active=%s waiting=%s resumed=%s terminal=%s "
                        "prompt_error=%s done=%s",
                        parent_session_id,
                        snapshot["yield_count"],
                        snapshot["pending_children"],
                        snapshot.get("completed_children", 0),
                        snapshot["message_count"],
                        snapshot.get("parent_active", False),
                        snapshot.get("parent_waiting", False),
                        snapshot.get("parent_resumed", False),
                        snapshot.get("parent_terminal", False),
                        bool(snapshot.get("post_yield_prompt_error")),
                        bool(snapshot["done_text"]),
                    )
                    last_state = state

                reconciled = self._reconcile_terminal_snapshot(
                    payload, snapshot
                )
                if reconciled is not None:
                    return reconciled

                # A prompt error is only a terminal candidate. OpenClaw may
                # retry/fail over and append another requester/model turn.
                prompt_error = str(
                    snapshot.get("post_yield_prompt_error") or ""
                ).strip()

                if snapshot["pending_children"] == 0 and prompt_error:
                    signature = (
                        snapshot["yield_count"],
                        snapshot["message_count"],
                        snapshot.get("transcript_size_bytes", 0),
                        prompt_error,
                    )
                    now = time.monotonic()

                    if signature != prompt_error_signature:
                        prompt_error_signature = signature
                        prompt_error_since = now

                        logger.info(
                            "  Roadmap prompt-error candidate; waiting %.1fs "
                            "for native retry/failover "
                            "(session=%s error=%s)",
                            self.prompt_error_grace_seconds,
                            parent_session_id,
                            prompt_error,
                        )

                    elif (
                        prompt_error_since is not None
                        and now - prompt_error_since
                        >= self.prompt_error_grace_seconds
                    ):
                        status = (
                            self.CONTEXT_OVERFLOW
                            if self._is_context_overflow_text(prompt_error)
                            else self.RUNTIME_ERROR
                        )
                        self._last_completion_status = status
                        self._last_completion_error = prompt_error
                        return payload, prompt_error

                else:
                    # Any later requester/model/tool activity clears an old
                    # transient prompt-error candidate.
                    prompt_error_signature = None
                    prompt_error_since = None

            time.sleep(self.yield_watch_interval_seconds)

        message = (
            "Roadmap trajectory remained nonterminal until the total agent timeout "
            f"(session={parent_session_id})"
        )
        self._last_exit_code = -1
        self._last_completion_status = self.TIMEOUT
        self._last_completion_error = message
        return payload, message

    def _reconcile_terminal_snapshot(
        self,
        payload: Optional[Dict],
        snapshot: Dict,
    ) -> Optional[Tuple[Optional[Dict], str]]:
        """Apply a durable terminal requester event to the CLI result."""
        if snapshot.get("done_text"):
            reconciled_payload = self._payload_with_resumed_completion(
                payload,
                snapshot["done_text"],
                yield_count=int(snapshot.get("yield_count") or 0),
                parent_session_id=str(snapshot["parent_session_id"]),
            )
            self._attach_lifecycle_summary(reconciled_payload, snapshot)
            status, completion_error = self._classify_completion(
                reconciled_payload
            )
            self._last_completion_status = status
            self._last_completion_error = completion_error
            return reconciled_payload, completion_error

        # A terminal assistant stop is definitive regardless of whether the
        # trajectory yielded or whether children remain. A toolUse assistant
        # turn is intentionally not terminal.
        if snapshot.get("post_yield_parent_terminal"):
            message = (
                "Roadmap parent returned without DONE "
                f"(session={snapshot['parent_session_id']})"
            )
            self._last_completion_status = self.BLOCKED_INCOMPLETE
            self._last_completion_error = message
            return payload, message

        return None

    @staticmethod
    def _attach_lifecycle_summary(payload: Dict, snapshot: Dict) -> None:
        run = payload.setdefault("result", {})
        meta = run.setdefault("meta", {})
        pending = list(snapshot.get("pending_child_sessions") or [])
        meta["agentBenchLifecycle"] = {
            "parentState": "terminal",
            "yieldCount": int(snapshot.get("yield_count") or 0),
            "totalChildren": int(snapshot.get("total_children") or 0),
            "completedChildren": int(snapshot.get("completed_children") or 0),
            "pendingChildrenAtParentTerminal": len(pending),
            "abandonedChildren": pending,
        }

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

        child_entries = []
        for key, value in index.items():
            if not isinstance(value, dict) or value.get("spawnedBy") != parent_key:
                continue
            status = str(value.get("status") or "unknown").lower()
            child_entries.append({
                "session_key": str(key),
                "session_id": str(value.get("sessionId") or ""),
                "status": status,
            })
        pending_child_sessions = [
            child for child in child_entries
            if child["status"] in cls._ACTIVE_CHILD_STATES
        ]
        pending_children = len(pending_child_sessions)
        completed_children = sum(
            child["status"] in {
                "done", "completed", "failed", "error", "timeout",
                "timed_out", "cancelled", "canceled",
            }
            for child in child_entries
        )
        child_timeout = any(
            child["status"] in {"timeout", "timed_out"}
            for child in child_entries
        )

        transcript_path = cls._resolve_parent_transcript_path(
            sessions_dir,
            parent,
        )
        if transcript_path is None:
            return None

        messages = []
        try:
            lines = transcript_path.read_text(
                errors="replace"
            ).splitlines()
            transcript_size_bytes = transcript_path.stat().st_size
        except OSError:
            return None

        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") in {
                "message",
                "custom",
                "custom_message",
            }:
                messages.append(event)

        yield_indexes = [
            i
            for i, event in enumerate(messages)
            if event.get("type") == "custom_message"
            and event.get("customType")
            == "openclaw.sessions_yield"
        ]

        latest_yield = yield_indexes[-1] if yield_indexes else -1

        done_text = ""
        post_yield_parent_activity = False
        post_yield_parent_terminal = False
        post_yield_prompt_error = ""
        latest_event_kind = ""

        for event in messages[latest_yield + 1:]:
            if (
                event.get("type") == "custom"
                and event.get("customType")
                == "openclaw:prompt-error"
            ):
                # This is an attempt failure, not necessarily the end of
                # the logical resumed trajectory.
                data = event.get("data")

                if isinstance(data, dict):
                    raw_error = data.get("error")
                else:
                    raw_error = data

                post_yield_prompt_error = str(
                    raw_error or "OpenClaw prompt error"
                )
                post_yield_parent_terminal = False
                done_text = ""
                latest_event_kind = "prompt_error"
                continue

            if event.get("type") != "message":
                continue

            message = event.get("message") or {}
            role = message.get("role")

            # A later requester/model/tool message proves that any earlier
            # prompt error was transient.
            if role in {"user", "assistant", "toolResult"}:
                post_yield_prompt_error = ""

            if role == "user":
                post_yield_parent_terminal = False
                done_text = ""
                latest_event_kind = "resume"
                continue

            if role == "toolResult":
                post_yield_parent_activity = True
                post_yield_parent_terminal = False
                done_text = ""
                latest_event_kind = "tool_result"
                continue

            if role != "assistant":
                continue

            post_yield_parent_activity = True

            stop_reason = str(
                message.get("stopReason") or ""
            ).strip().lower()

            post_yield_parent_terminal = stop_reason == "stop"
            latest_event_kind = (
                "assistant_stop"
                if post_yield_parent_terminal
                else "assistant_active"
            )

            parts = [
                str(item.get("text") or "")
                for item in message.get("content") or []
                if isinstance(item, dict)
                and item.get("type") == "text"
            ]
            text = "\n".join(
                part for part in parts if part
            ).strip()

            if (
                post_yield_parent_terminal
                and text
                and re.search(
                    r"(?im)^\s*DONE(?:\s*$|\b)",
                    text,
                )
            ):
                done_text = text
            else:
                done_text = ""

        return {
            "parent_session_id": str(parent_session_id),
            "total_children": len(child_entries),
            "pending_children": pending_children,
            "pending_child_sessions": pending_child_sessions,
            "completed_children": completed_children,
            "child_states": child_entries,
            "child_timeout": child_timeout,
            "yield_count": len(yield_indexes),
            "message_count": len(messages),
            "transcript_size_bytes": transcript_size_bytes,
            "post_yield_parent_activity": (
                post_yield_parent_activity
            ),
            "post_yield_parent_terminal": (
                post_yield_parent_terminal
            ),
            "post_yield_terminal_reason": (
                "assistant_stop"
                if post_yield_parent_terminal
                else ""
            ),
            "post_yield_prompt_error": post_yield_prompt_error,
            "done_text": done_text,
            "parent_active": bool(
                post_yield_parent_activity
                and not post_yield_parent_terminal
                and not post_yield_prompt_error
            ),
            "parent_waiting": bool(
                yield_indexes and not post_yield_parent_activity
            ),
            "parent_resumed": bool(
                yield_indexes and post_yield_parent_activity
            ),
            "parent_terminal": post_yield_parent_terminal,
            "latest_parent_event": latest_event_kind,
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
