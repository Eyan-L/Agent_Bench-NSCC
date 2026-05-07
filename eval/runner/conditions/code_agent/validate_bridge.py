"""Workspace-IPC bridge that lets the in-container agent self-validate against
the SWE-bench evaluator while the agent run is in progress.

Why this exists
---------------
The agent runs in the OC runtime sif (alpine/openclaw, no Python deps for the
target repo). SWE-bench scoring needs the per-instance .sif (the right Python
env, pinned versions). The agent therefore can't run pytest itself — it has to
ask the host. This bridge implements that ask via plain files in the workspace
(which is bind-mounted host↔container, so both sides see the same paths):

    <workspace>/_oc_validate.sh        — drop-in shell helper the agent invokes
    <workspace>/.oc_validate_req       — agent writes this to request validation
    <workspace>/.oc_validate_resp      — host writes the JSON verdict here

Lifecycle:
    bridge = ValidateBridge(harness, run_dir, workspace_root, instance_id,
                            instance_data, container_id, run_id, timeout)
    bridge.start()
    # ... agent run ...
    bridge.stop()

start() drops the helper script + a `.git/info/exclude` entry (so the helper
and sentinels don't pollute `git diff HEAD`). A daemon thread polls
`.oc_validate_req` ~4×/s; on each request it calls `harness.run_tests(...)`
and writes the result to `.oc_validate_resp`. stop() signals the thread,
removes the helper + sentinels, and joins.

Failure modes are best-effort: if anything goes wrong, the bridge logs and
keeps the agent run going (validation is convenience, not critical path).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_HELPER_NAME = "_oc_validate.sh"
_REQ_NAME = ".oc_validate_req"
_RESP_NAME = ".oc_validate_resp"
_BUSY_NAME = ".oc_validate_busy"

# Bash helper. Written into the workspace at bridge start. Touches a request
# file, polls for the response (with a generous timeout to cover singularity
# scoring on large repos), prints the JSON verdict, exits non-zero on FAIL/
# ERROR/TIMEOUT so the agent can react to `bash _oc_validate.sh && ...`.
_HELPER_TEMPLATE = """\
#!/usr/bin/env bash
# oc-eval validate helper — generated, do not edit. Cleaned up at run end.
set -u
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REQ="$HERE/{req}"
RESP="$HERE/{resp}"
BUSY="$HERE/{busy}"
TIMEOUT_SECONDS={timeout}

# Drop a stale resp from a previous call so we read the new one.
rm -f "$RESP" 2>/dev/null

# Signal the host: timestamp marks request order if multiple ever queue.
date -u +%s.%N > "$REQ"

# Poll up to TIMEOUT_SECONDS for the host to write a response.
deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
while (( $(date +%s) < deadline )); do
    if [[ -f "$RESP" ]]; then
        cat "$RESP"
        # Surface non-PASS as non-zero so `&&` and `||` work naturally.
        if grep -q '"status": *"PASS"' "$RESP" 2>/dev/null; then
            exit 0
        else
            exit 1
        fi
    fi
    sleep 0.5
done

echo '{{"status": "TIMEOUT", "error_message": "validate bridge did not respond in time"}}'
exit 2
"""


class ValidateBridge:
    """Daemon thread that handles agent-driven self-validation requests."""

    def __init__(
        self,
        harness,                     # any harness with .run_tests(...)
        run_dir: Path,
        workspace_root: str,
        instance_id: str,
        instance_data: Optional[Dict],
        container_id: Optional[str],
        run_id: Optional[str],
        timeout_seconds: int,
        poll_interval: float = 0.25,
    ) -> None:
        self.harness = harness
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace_root)
        self.instance_id = instance_id
        self.instance_data = instance_data
        self.container_id = container_id
        self.run_id = run_id
        self.timeout_seconds = int(timeout_seconds)
        self.poll_interval = poll_interval

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._call_count = 0
        self._verdicts_path = self.run_dir / "validate_verdicts.jsonl"

    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.workspace.is_dir():
            logger.warning(
                "validate bridge: workspace missing at %s; not starting",
                self.workspace,
            )
            return

        # Prevent the helper + sentinels from showing up in git diff HEAD.
        self._exclude_artifacts_from_git()

        # Drop the helper script. Generous polling timeout — singularity
        # scoring of a real SWE-bench instance can take 5-10 minutes.
        helper_path = self.workspace / _HELPER_NAME
        try:
            helper_path.write_text(_HELPER_TEMPLATE.format(
                req=_REQ_NAME, resp=_RESP_NAME, busy=_BUSY_NAME,
                timeout=max(self.timeout_seconds + 60, 600),
            ))
            os.chmod(helper_path, 0o755)
        except OSError as e:
            logger.warning("validate bridge: helper drop failed: %s", e)
            return

        self._thread = threading.Thread(
            target=self._run, name="oc-validate-bridge", daemon=True,
        )
        self._thread.start()
        logger.info(
            "validate bridge ready (instance=%s, helper=%s)",
            self.instance_id, helper_path.name,
        )

    def stop(self) -> int:
        """Signal the worker, clean up artifacts, return validation call count."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)

        for name in (_HELPER_NAME, _REQ_NAME, _RESP_NAME, _BUSY_NAME):
            try:
                (self.workspace / name).unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.debug("validate bridge: cleanup %s skipped: %s", name, e)

        return self._call_count

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        req_path = self.workspace / _REQ_NAME
        resp_path = self.workspace / _RESP_NAME
        busy_path = self.workspace / _BUSY_NAME

        while not self._stop.is_set():
            if req_path.is_file():
                # Claim the request immediately so concurrent invocations
                # (the agent shouldn't do this but the helper polls fast)
                # don't double-fire.
                try:
                    req_path.replace(busy_path)
                except (OSError, FileNotFoundError):
                    self._sleep()
                    continue

                self._call_count += 1
                t_start = time.time()
                verdict = self._score_now()
                duration = time.time() - t_start
                verdict.setdefault("instance_id", self.instance_id)
                verdict["validate_call_idx"] = self._call_count
                verdict["validate_wall_seconds"] = round(duration, 2)

                try:
                    resp_path.write_text(json.dumps(verdict, indent=2))
                except OSError as e:
                    logger.warning(
                        "validate bridge: resp write failed: %s", e,
                    )

                # Persist every verdict to the run dir so we can audit
                # how often the agent self-validated and what it saw.
                try:
                    with open(self._verdicts_path, "a") as fh:
                        fh.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                except OSError:
                    pass

                try:
                    busy_path.unlink()
                except OSError:
                    pass

                logger.info(
                    "validate bridge: call %d -> %s (%.1fs)",
                    self._call_count, verdict.get("status"), duration,
                )
            self._sleep()

    def _sleep(self) -> None:
        # Use Event.wait instead of time.sleep so stop() responds quickly.
        self._stop.wait(self.poll_interval)

    def _score_now(self) -> Dict:
        try:
            return self.harness.run_tests(
                container_id=self.container_id,
                instance_id=self.instance_id,
                workspace_root=str(self.workspace),
                timeout_seconds=self.timeout_seconds,
                patch=None,           # collect from the workspace
                run_id=self.run_id,
                instance_data=self.instance_data,
            )
        except Exception as e:  # noqa: BLE001 — bubble any error to the agent
            logger.exception("validate bridge: run_tests crashed")
            return {
                "status": "ERROR",
                "test_pass_rate": 0.0,
                "passed": 0, "failed": 0,
                "error_message": f"validate bridge run_tests raised: {e!r}",
            }

    # ------------------------------------------------------------------

    def _exclude_artifacts_from_git(self) -> None:
        """Append helper + sentinel names to `.git/info/exclude` so they are
        treated as untracked-and-ignored, hence absent from `git diff HEAD`
        and from `git add -A` at patch-collection time. Idempotent."""
        exc = self.workspace / ".git" / "info" / "exclude"
        if not exc.parent.is_dir():
            return  # not a git repo — nothing to do

        try:
            existing = exc.read_text() if exc.exists() else ""
        except OSError:
            existing = ""

        marker = "# oc-eval validate bridge"
        if marker in existing:
            return

        block = (
            f"\n{marker}\n"
            f"/{_HELPER_NAME}\n"
            f"/{_REQ_NAME}\n"
            f"/{_RESP_NAME}\n"
            f"/{_BUSY_NAME}\n"
        )
        try:
            with open(exc, "a") as fh:
                fh.write(block)
        except OSError as e:
            logger.debug("validate bridge: .git/info/exclude write skipped: %s", e)
