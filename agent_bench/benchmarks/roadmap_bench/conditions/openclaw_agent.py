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
from agent_bench.deployments.containers.base import ContainerHandle

logger = logging.getLogger(__name__)

_WORKSPACE_SCAFFOLD_FILES = frozenset({
    "AGENTS.md", "BOOTSTRAP.md", "HEARTBEAT.md", "IDENTITY.md",
    "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md",
})

# OpenClaw runtime/workspace metadata is not part of the candidate solution.
_WORKSPACE_SCAFFOLD_DIRS = frozenset({
    ".openclaw",
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
    _ACTIVE_PARENT_STATES = frozenset({
        "accepted", "queued", "pending", "processing", "running",
    })
    _TERMINAL_PARENT_STATES = frozenset({
        "abandoned", "cancelled", "canceled", "completed", "done", "error",
        "failed", "killed", "timed_out", "timeout",
    })
    # A turn-level stop can precede an OpenClaw retry/resume. Require the
    # durable session and transcript to remain terminal across multiple
    # watcher reads before accepting a non-DONE stop as logical completion.
    terminal_stop_confirmation_seconds = 10.0

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id: Optional[str],
        env_info: Optional[Dict] = None,
    ) -> Dict:
        runtime_env = env_info if env_info is not None else {}
        workspace = Path(str(runtime_env.get("workspace_host_path") or ""))
        baseline = self._capture_task_workspace_baseline(
            runtime_env, workspace_root
        )
        if not baseline:
            baseline = self._capture_host_workspace_baseline(workspace)
        if not baseline:
            raise RuntimeError(
                "Roadmap workspace Git baseline could not be captured before "
                "agent execution"
            )
        runtime_env["roadmap_workspace_baseline"] = baseline
        return super().run(
            instance_data,
            run_dir,
            seed,
            workspace_root,
            container_id,
            runtime_env,
        )

    def _capture_task_workspace_baseline(
        self, env_info: Dict, workspace_root: str
    ) -> Optional[Dict]:
        """Capture the initial Git reference in the task runtime when available."""
        task_container_id = str(env_info.get("task_container_id") or "")
        if self.container is None or not task_container_id:
            return None
        task_handle = ContainerHandle(
            id=task_container_id,
            name=str(env_info.get("task_container_name") or task_container_id),
            backend=str(env_info.get("backend") or self.container.backend_id),
        )
        excludes = [
            f":(exclude){name}" for name in sorted(_WORKSPACE_SCAFFOLD_FILES)
        ] + [
            f":(exclude){name}/**" for name in sorted(_WORKSPACE_SCAFFOLD_DIRS)
        ]

        def run(argv: list[str]):
            return self.container.exec(
                task_handle,
                argv,
                check=False,
                timeout=120,
            )

        git = ["git", "-c", f"safe.directory={workspace_root}"]
        try:
            root = run([
                *git, "-C", workspace_root, "rev-parse", "--show-toplevel",
            ])
            head = run([
                *git, "-C", workspace_root, "rev-parse", "--verify", "HEAD",
            ])
            tree = run([
                *git, "-C", workspace_root, "rev-parse", "HEAD^{tree}",
            ])
            status = run([
                *git, "-C", workspace_root, "status", "--porcelain=v1",
                "--untracked-files=all", "--", ".", *excludes,
            ])
        except Exception:  # noqa: BLE001 - host Git remains a valid fallback
            logger.exception(
                "Roadmap task-runtime baseline capture failed; trying host Git"
            )
            return None
        if any(result.returncode != 0 for result in (root, head, tree, status)):
            logger.warning(
                "Roadmap task-runtime baseline capture returned a Git error; "
                "trying host Git"
            )
            return None
        return {
            "git_head": head.stdout.strip(),
            "git_tree": tree.stdout.strip(),
            "clean": not bool(status.stdout.strip()),
            "status": status.stdout.splitlines(),
        }

    @staticmethod
    def _capture_host_workspace_baseline(workspace: Path) -> Optional[Dict]:
        """Capture the immutable Git reference before the agent can edit."""
        if not workspace.is_dir():
            return None
        excludes = [
            f":(exclude){name}" for name in sorted(_WORKSPACE_SCAFFOLD_FILES)
        ] + [
            f":(exclude){name}/**" for name in sorted(_WORKSPACE_SCAFFOLD_DIRS)
        ]

        def run(argv: list[str], *, text: bool = True):
            return subprocess.run(
                ["git", "-C", str(workspace), *argv],
                capture_output=True,
                text=text,
            )

        try:
            root = run(["rev-parse", "--show-toplevel"])
            head = run(["rev-parse", "--verify", "HEAD"])
            tree = run(["rev-parse", "HEAD^{tree}"])
            status = run([
                "status", "--porcelain=v1", "--untracked-files=all",
                "--", ".", *excludes,
            ])
        except FileNotFoundError:
            return None
        if any(result.returncode != 0 for result in (root, head, tree, status)):
            return None
        return {
            "git_head": head.stdout.strip(),
            "git_tree": tree.stdout.strip(),
            "clean": not bool(status.stdout.strip()),
            "status": status.stdout.splitlines(),
        }

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
        del container_id, run_dir
        baseline = env_info.get("roadmap_workspace_baseline") or {}
        baseline_head = str(baseline.get("git_head") or "")
        if not baseline_head:
            raise RuntimeError("Roadmap workspace baseline HEAD is missing")
        task_container_id = str(env_info.get("task_container_id") or "")
        task_container_name = str(
            env_info.get("task_container_name") or task_container_id
        )
        if self.container is not None and task_container_id:
            task_handle = ContainerHandle(
                id=task_container_id,
                name=task_container_name,
                backend=str(
                    env_info.get("backend") or self.container.backend_id
                ),
            )
            try:
                return self._collect_container_workspace_patch(
                    task_handle, workspace_root, baseline_head
                )
            except Exception:  # noqa: BLE001 - retain host fallback
                logger.exception(
                    "Roadmap task-runtime patch extraction failed; "
                    "trying host workspace"
                )

        workspace = Path(str(env_info.get("workspace_host_path") or ""))
        if not workspace.is_dir():
            logger.warning(
                "Roadmap patch extraction missing host workspace: %s",
                workspace,
            )
            return ""
        return self._collect_workspace_patch(workspace, baseline_head)

    def _collect_container_workspace_patch(
        self,
        task_handle: ContainerHandle,
        workspace_root: str,
        baseline_head: str = "HEAD",
    ) -> str:
        """Collect a patch inside the task runtime that owns the workspace."""
        assert self.container is not None

        def run(argv: list[str], *, workdir: Optional[str] = None):
            return self.container.exec(
                task_handle,
                argv,
                check=False,
                timeout=120,
                workdir=workdir,
            )

        root_result = run(
            [
                "git", "-c", f"safe.directory={workspace_root}",
                "-C", workspace_root, "rev-parse", "--show-toplevel",
            ]
        )
        if root_result.returncode != 0:
            logger.warning(
                "Roadmap task workspace is not a git repository: %s",
                root_result.stderr.strip(),
            )
            return ""
        repo_root = root_result.stdout.strip()

        excludes = [
            f":(exclude){name}" for name in sorted(_WORKSPACE_SCAFFOLD_FILES)
        ] + [
            f":(exclude){name}/**" for name in sorted(_WORKSPACE_SCAFFOLD_DIRS)
        ]
        tracked = run([
            "git", "-c", f"safe.directory={repo_root}",
            "-C", repo_root, "diff", "--binary", baseline_head,
            "--", ".", *excludes,
        ])
        if tracked.returncode != 0:
            logger.warning(
                "Roadmap tracked patch extraction failed in task runtime: %s",
                tracked.stderr.strip(),
            )
            return ""

        untracked = run([
            "git", "-c", f"safe.directory={repo_root}",
            "-C", repo_root, "ls-files", "--others",
            "--exclude-standard", "-z",
        ])
        if untracked.returncode != 0:
            logger.warning(
                "Roadmap untracked file enumeration failed in task runtime"
            )
            return tracked.stdout

        patch_parts = [tracked.stdout]
        for relative in untracked.stdout.split("\0"):
            if not relative:
                continue
            parts = Path(relative).parts
            if parts and (
                parts[0] in _WORKSPACE_SCAFFOLD_FILES
                or parts[0] in _WORKSPACE_SCAFFOLD_DIRS
            ):
                continue
            diff = run(
                [
                    "git", "-c", f"safe.directory={repo_root}",
                    "diff", "--no-index", "--binary", "--",
                    "/dev/null", relative,
                ],
                workdir=repo_root,
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

    @staticmethod
    def _collect_workspace_patch(
        workspace: Path, baseline_head: str = "HEAD"
    ) -> str:
        try:
            root_result = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.warning("Roadmap host patch extraction skipped: git not found")
            return ""
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
        ] + [
            f":(exclude){name}/**" for name in sorted(_WORKSPACE_SCAFFOLD_DIRS)
        ]
        tracked = subprocess.run(
            [
                "git", "-C", str(repo_root), "diff", "--binary",
                baseline_head,
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
            parts = Path(relative).parts
            if parts and (
                parts[0] in _WORKSPACE_SCAFFOLD_FILES
                or parts[0] in _WORKSPACE_SCAFFOLD_DIRS
            ):
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

        # Read the durable child/session state before handling any terminal
        # outcome. Abnormal parent termination must quiesce pending children
        # just like DONE / terminal-stop, otherwise they may keep writing the
        # shared workspace while the verifier starts.
        snapshot = self._read_yield_snapshot(sessions_dir)

        if getattr(self, "_last_completion_status", None) == self.TIMEOUT:
            payload = self._quiesce_abnormal_terminal_payload(
                payload, snapshot
            )
            return payload, getattr(self, "_last_completion_error", error)

        # The CLI payload can lag behind the durable requester transcript. In
        # particular, a no-yield requester can have appended a terminal DONE
        # while the payload still reports livenessState=working. Explicit
        # payload failures remain authoritative and must never be hidden by an
        # older transcript terminal event.
        payload_status, payload_error = self._classify_completion(payload)
        if payload_status in {self.RUNTIME_ERROR, self.CONTEXT_OVERFLOW}:
            payload = self._quiesce_abnormal_terminal_payload(
                payload, snapshot
            )
            self._last_completion_status = payload_status
            self._last_completion_error = payload_error
            return payload, payload_error

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
            or snapshot.get("parent_session_active")
            or snapshot.get("assistant_stop_candidate")
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
        last_snapshot = snapshot
        terminal_stop_signature = self._terminal_stop_signature(snapshot)
        terminal_stop_since = (
            time.monotonic() if terminal_stop_signature is not None else None
        )

        while time.monotonic() < deadline:
            snapshot = self._read_yield_snapshot(sessions_dir)
            if snapshot:
                last_snapshot = snapshot
                if snapshot["parent_session_id"] != parent_session_id:
                    message = (
                        "Roadmap requester session changed during yield/resume: "
                        f"{parent_session_id} -> "
                        f"{snapshot['parent_session_id']}"
                    )
                    self._last_completion_status = self.RUNTIME_ERROR
                    self._last_completion_error = message
                    payload = self._quiesce_abnormal_terminal_payload(
                        payload, snapshot
                    )
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
                    snapshot.get("parent_session_status", "unknown"),
                )

                if state != last_state:
                    logger.info(
                        "  Roadmap yield lifecycle: session=%s yields=%d "
                        "pending_children=%d completed_children=%d messages=%d "
                        "active=%s waiting=%s resumed=%s terminal=%s "
                        "parent_status=%s "
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
                        snapshot.get("parent_session_status", "unknown"),
                        bool(snapshot.get("post_yield_prompt_error")),
                        bool(snapshot["done_text"]),
                    )
                    last_state = state

                reconciled = self._reconcile_terminal_snapshot(payload, snapshot)
                if reconciled is not None:
                    return reconciled

                terminal_signature = self._terminal_stop_signature(snapshot)
                now = time.monotonic()
                if terminal_signature is None:
                    terminal_stop_signature = None
                    terminal_stop_since = None
                elif terminal_signature != terminal_stop_signature:
                    terminal_stop_signature = terminal_signature
                    terminal_stop_since = now
                elif (
                    terminal_stop_since is not None
                    and now - terminal_stop_since
                    >= self.terminal_stop_confirmation_seconds
                ):
                    reconciled = self._reconcile_terminal_snapshot(
                        payload, snapshot, confirm_non_done_stop=True
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
                        payload = self._quiesce_abnormal_terminal_payload(
                            payload, snapshot
                        )
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
        payload = self._quiesce_abnormal_terminal_payload(
            payload, last_snapshot
        )
        return payload, message

    def _quiesce_abnormal_terminal_payload(
        self,
        payload: Optional[Dict],
        snapshot: Optional[Dict],
    ) -> Optional[Dict]:
        """Quiesce agent writers before returning an abnormal terminal result."""
        if not snapshot:
            return payload

        reconciled_payload = (
            copy.deepcopy(payload) if isinstance(payload, dict) else {}
        )
        if not isinstance(reconciled_payload.get("result"), dict):
            reconciled_payload["result"] = {}

        snapshot, quiescence = self._quiesce_with_timing(snapshot)
        self._attach_lifecycle_summary(
            reconciled_payload, snapshot, quiescence
        )
        return reconciled_payload

    def _reconcile_terminal_snapshot(
        self,
        payload: Optional[Dict],
        snapshot: Dict,
        *,
        confirm_non_done_stop: bool = False,
    ) -> Optional[Tuple[Optional[Dict], str]]:
        """Apply a durable terminal requester event to the CLI result."""
        if snapshot.get("done_text"):
            snapshot = dict(snapshot, terminal_evidence="explicit_done")
            snapshot, quiescence = self._quiesce_with_timing(
                snapshot, logical_parent_terminal=True
            )
            reconciled_payload = self._payload_with_resumed_completion(
                payload,
                snapshot["done_text"],
                yield_count=int(snapshot.get("yield_count") or 0),
                parent_session_id=str(snapshot["parent_session_id"]),
            )
            self._attach_lifecycle_summary(
                reconciled_payload, snapshot, quiescence
            )
            status, completion_error = self._classify_completion(
                reconciled_payload
            )
            self._last_completion_status = status
            self._last_completion_error = completion_error
            return reconciled_payload, completion_error

        # A turn-level assistant stop is only a candidate. OpenClaw can append
        # a retry/resume after it, so require both a durable terminal session
        # status and stable confirmation by the lifecycle watcher.
        if (
            confirm_non_done_stop
            and snapshot.get("post_yield_parent_terminal")
            and snapshot.get("parent_session_terminal")
        ):
            snapshot = dict(
                snapshot,
                terminal_evidence="durable_session_terminal_assistant_stop",
            )
            snapshot, quiescence = self._quiesce_with_timing(
                snapshot, logical_parent_terminal=True
            )
            reconciled_payload = (
                copy.deepcopy(payload) if isinstance(payload, dict) else {}
            )
            if not isinstance(reconciled_payload.get("result"), dict):
                reconciled_payload["result"] = {}
            self._attach_lifecycle_summary(
                reconciled_payload, snapshot, quiescence
            )
            message = (
                "Roadmap parent returned without DONE "
                f"(session={snapshot['parent_session_id']})"
            )
            self._last_completion_status = self.BLOCKED_INCOMPLETE
            self._last_completion_error = message
            return reconciled_payload, message

        return None

    @staticmethod
    def _terminal_stop_signature(snapshot: Dict) -> Optional[Tuple]:
        """Return a stable identity for a confirmable non-DONE stop."""
        if (
            snapshot.get("done_text")
            or not snapshot.get("post_yield_parent_terminal")
            or not snapshot.get("parent_session_terminal")
        ):
            return None
        return (
            snapshot.get("parent_session_id"),
            snapshot.get("parent_session_status"),
            snapshot.get("message_count"),
            snapshot.get("transcript_size_bytes"),
            snapshot.get("latest_parent_event"),
        )

    @staticmethod
    def _terminal_activity_signature(snapshot: Dict) -> Tuple:
        """Return transcript activity that must not change across the freeze fence."""
        return (
            snapshot.get("parent_session_id"),
            snapshot.get("message_count"),
            snapshot.get("transcript_size_bytes"),
            snapshot.get("yield_count"),
            snapshot.get("latest_parent_event"),
        )

    def _quiesce_with_timing(
        self,
        snapshot: Dict,
        *,
        logical_parent_terminal: bool = False,
    ) -> Tuple[Dict, Dict]:
        """Fence all agent writers and timestamp only a validated quiescence."""
        timed_snapshot = dict(snapshot)
        if (
            logical_parent_terminal
            or snapshot.get("parent_terminal")
            or snapshot.get("parent_session_terminal")
        ):
            timed_snapshot["parent_terminal_observed_ns"] = time.time_ns()

        before_activity = self._terminal_activity_signature(snapshot)
        quiescence = self._quiesce_terminal_children(snapshot)

        if quiescence.get("workspaceQuiesced") is True and logical_parent_terminal:
            sessions_dir = self._live_sessions_dir()
            after_snapshot = (
                self._read_yield_snapshot(sessions_dir)
                if sessions_dir is not None else None
            )
            if after_snapshot is None:
                error = (
                    "Roadmap could not validate durable lifecycle after agent "
                    "writer quiescence"
                )
                quiescence = {
                    **quiescence,
                    "workspaceQuiesced": False,
                    "terminalFenceStable": False,
                    "lifecycleError": error,
                }
            elif self._terminal_activity_signature(after_snapshot) != before_activity:
                error = (
                    "Roadmap parent activity changed while establishing the "
                    "terminal quiescence fence"
                )
                quiescence = {
                    **quiescence,
                    "workspaceQuiesced": False,
                    "terminalFenceStable": False,
                    "lifecycleError": error,
                }
            else:
                quiescence = {**quiescence, "terminalFenceStable": True}

        if quiescence.get("workspaceQuiesced") is True:
            timed_snapshot["workspace_quiesced_ns"] = time.time_ns()
        return timed_snapshot, quiescence

    def _quiesce_terminal_children(self, snapshot: Dict) -> Dict:
        """Use the harness fence to stop every agent-side workspace writer."""
        pending = list(snapshot.get("pending_child_sessions") or [])
        harness = self._env_info.get("_harness")
        if harness is None or not hasattr(harness, "quiesce_orphan_children"):
            error = "Roadmap harness cannot quiesce terminal agent writers"
            logger.error(error)
            return {
                "workspaceQuiesced": False,
                "agentWriterQuiescenceAttempted": False,
                "orphanCancellationAttempted": False,
                "orphanCancellationError": None,
                "lifecycleError": error,
            }
        try:
            details = harness.quiesce_orphan_children(self._env_info)
            return {
                **(details if isinstance(details, dict) else {}),
                "workspaceQuiesced": True,
                "agentWriterQuiescenceAttempted": True,
                "orphanCancellationAttempted": bool(pending),
                "orphanCancellationError": None,
            }
        except Exception as exc:  # noqa: BLE001 - surface in lifecycle result
            error = f"Roadmap agent-writer quiescence failed: {exc}"
            logger.exception(error)
            return {
                "workspaceQuiesced": False,
                "agentWriterQuiescenceAttempted": True,
                "orphanCancellationAttempted": bool(pending),
                "orphanCancellationError": error if pending else None,
                "lifecycleError": error,
            }

    @staticmethod
    def _attach_lifecycle_summary(
        payload: Dict, snapshot: Dict, quiescence: Dict
    ) -> None:
        run = payload.setdefault("result", {})
        meta = run.setdefault("meta", {})
        pending = list(snapshot.get("pending_child_sessions") or [])
        child_states = list(snapshot.get("child_states") or [])
        lifecycle = {
            "parentState": "terminal",
            "parentSessionStatusAtTerminal": snapshot.get(
                "parent_session_status", "unknown"
            ),
            "terminalEvidence": snapshot.get("terminal_evidence", "unknown"),
            "parentTerminalObservedNs": snapshot.get("parent_terminal_observed_ns"),
            "parentTerminalTimestampQuality": (
                "watcher_observed"
                if snapshot.get("parent_terminal_observed_ns")
                else None
            ),
            "workspaceQuiescedNs": snapshot.get("workspace_quiesced_ns"),
            "workspaceQuiescedTimestampQuality": (
                "harness_observed"
                if snapshot.get("workspace_quiesced_ns")
                else None
            ),
            "yieldCount": int(snapshot.get("yield_count") or 0),
            "totalChildren": int(snapshot.get("total_children") or 0),
            "completedChildren": int(snapshot.get("completed_children") or 0),
            "pendingChildrenAtParentTerminal": len(pending),
            "abandonedChildren": pending,
            "orphanChildIds": [
                child.get("session_id") or child.get("session_key")
                for child in pending
            ],
            "childStatesAtParentTerminal": child_states,
            **quiescence,
        }
        if quiescence.get("orphanCancellationError"):
            lifecycle["lifecycleError"] = quiescence[
                "orphanCancellationError"
            ]
        meta["agentBenchLifecycle"] = lifecycle

    def _live_sessions_dir(self) -> Optional[Path]:
        state_dir = self._env_info.get("state_dir") or self._env_info.get(
            "host_state_dir"
        )
        if not state_dir:
            return None
        return Path(state_dir) / "agents" / self.agent_id / "sessions"

    @staticmethod
    def _is_heartbeat_request(text: str) -> bool:
        """Recognize OpenClaw heartbeat requester turns."""
        normalized = " ".join(str(text or "").lower().split())
        return (
            "read heartbeat.md" in normalized
            and "heartbeat_ok" in normalized
        )

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
        parent_session_status = str(
            parent.get("status") or "unknown"
        ).strip().lower()
        parent_session_active = (
            parent_session_status in cls._ACTIVE_PARENT_STATES
        )
        parent_session_terminal = (
            parent_session_status in cls._TERMINAL_PARENT_STATES
        )

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
        heartbeat_turn = False

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
            parts = [
                str(item.get("text") or "")
                for item in message.get("content") or []
                if isinstance(item, dict)
                and item.get("type") == "text"
            ]
            text = "\n".join(
                part for part in parts if part
            ).strip()

            # A later requester/model/tool message proves that any earlier
            # prompt error was transient.
            if role in {"user", "assistant", "toolResult"}:
                post_yield_prompt_error = ""

            if role == "user":
                if cls._is_heartbeat_request(text):
                    heartbeat_turn = True
                    latest_event_kind = "heartbeat_request"
                    continue

                heartbeat_turn = False
                post_yield_parent_terminal = False
                done_text = ""
                latest_event_kind = "resume"
                continue

            if role == "toolResult":
                if heartbeat_turn:
                    latest_event_kind = "heartbeat_tool_result"
                    continue
                post_yield_parent_activity = True
                post_yield_parent_terminal = False
                done_text = ""
                latest_event_kind = "tool_result"
                continue

            if role != "assistant":
                continue

            stop_reason = str(
                message.get("stopReason") or ""
            ).strip().lower()

            if heartbeat_turn:
                latest_event_kind = "heartbeat_response"
                if stop_reason == "stop":
                    heartbeat_turn = False
                continue

            post_yield_parent_activity = True
            post_yield_parent_terminal = stop_reason == "stop"
            latest_event_kind = (
                "assistant_stop"
                if post_yield_parent_terminal
                else "assistant_active"
            )

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

        assistant_stop_candidate = post_yield_parent_terminal
        logical_parent_terminal = bool(
            done_text
            or (
                assistant_stop_candidate
                and parent_session_terminal
            )
        )

        return {
            "parent_session_id": str(parent_session_id),
            "parent_session_status": parent_session_status,
            "parent_session_active": parent_session_active,
            "parent_session_terminal": parent_session_terminal,
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
                assistant_stop_candidate and parent_session_terminal
            ),
            "post_yield_terminal_reason": (
                "explicit_done"
                if done_text
                else (
                    "assistant_stop"
                    if assistant_stop_candidate and parent_session_terminal
                    else ""
                )
            ),
            "post_yield_prompt_error": post_yield_prompt_error,
            "done_text": done_text,
            "parent_active": bool(
                (
                    parent_session_active
                    or (
                        post_yield_parent_activity
                        and not logical_parent_terminal
                    )
                )
                and not post_yield_prompt_error
            ),
            "parent_waiting": bool(
                yield_indexes and not post_yield_parent_activity
            ),
            "parent_resumed": bool(
                yield_indexes and post_yield_parent_activity
            ),
            "parent_terminal": logical_parent_terminal,
            "assistant_stop_candidate": assistant_stop_candidate,
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
