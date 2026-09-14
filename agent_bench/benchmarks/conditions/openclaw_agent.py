"""OpenClaw agent condition — base class for any benchmark.

Drives one run of the OpenClaw CLI against the per-app sandbox container.
Subclasses (in `applications/<app>/conditions/openclaw_agent.py`) plug in:

  - ``agent_id``        — the id registered in openclaw.json `agents.list[]`
                          and passed as ``openclaw agent --agent <id>``.
                          Also drives the session archive path
                          (``agents/<id>/sessions/``).
  - ``_build_message``  — task framing + the actual instruction text.
  - ``_collect_artifacts`` — return patch text (SWE-bench) or None (TB).

Runtime contract (unchanged from the original SWE-bench code path):
  - The OC container already has node + /opt/openclaw + openclaw.json mounted
    by the harness.
  - The harness pre-starts the per-run sandbox (task) container with the
    name OC computes from `containerPrefix + slugify(scopeKey)` so OC reuses
    it instead of spawning a new one.
  - The exec-bridge dispatcher is up so OC's sandbox execs land in the task
    container via bridge.py.

Returned dict matches the shape the driver expects:
    {step_trace_path, patch_path, total_prompt_tokens,
     total_completion_tokens, telemetry_path}
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from agent_bench.benchmarks.conditions.agent_otel import (
    otel_trace_offset,
    snapshot_otel_spans,
    wait_for_otel_flush,
)
from agent_bench.benchmarks.conditions.util_sampler import UtilSampler
from agent_bench.deployments.containers.base import Container, ContainerHandle

logger = logging.getLogger(__name__)

_GATEWAY_EXECUTION_MARKER = (
    "[agent-bench] OpenClaw gateway execution selected"
)
_ROADMAP_GATEWAY_EXECUTION_MARKER = (
    "[agent-bench] Roadmap gateway RPC ready; gateway execution selected"
)


def _otlp_attrs(items) -> Dict:
    """OTLP attribute list -> flat dict (first scalar value of each entry)."""
    out: Dict = {}
    for a in items or []:
        v = a.get("value", {})
        if not v:
            continue
        out[a["key"]] = next(iter(v.values()))
    return out


_OTEL_TRACE_FILE_DEFAULT = os.environ.get(
    "AB_TELEMETRY_TRACE_FILE",
    os.path.join(os.environ.get("AB_TELEMETRY_DATA_DIR", "/tmp/otel"), "otel-traces.jsonl"),
)
# Time to wait after the agent run before reading the trace file, so the
# diagnostics-otel plugin's batch processor has a chance to flush. The plugin
# is configured with flushIntervalMs=20 in openclaw.json; 1.5s is generous.
_OTEL_FLUSH_WAIT_SECONDS = float(os.environ.get("AB_TELEMETRY_FLUSH_WAIT_SECONDS", "1.5"))


class OpenClawAgentCondition:
    """Base class — instantiate a subclass that fixes agent_id + prompt."""

    # Subclass must set this. Used in `--agent <id>` CLI flag AND the
    # session-archive paths under state_dir/agents/<id>/.
    agent_id: ClassVar[str] = ""
    NORMAL_COMPLETION: ClassVar[str] = "normal_completion"
    CONTEXT_OVERFLOW: ClassVar[str] = "context_overflow"
    BLOCKED_INCOMPLETE: ClassVar[str] = "blocked_incomplete"
    RUNTIME_ERROR: ClassVar[str] = "runtime_error"
    TIMEOUT: ClassVar[str] = "timeout"
    require_done_marker: ClassVar[bool] = False

    def __init__(self, model_config, agent_config, container: Optional[Container] = None):
        if not self.agent_id:
            raise NotImplementedError(
                f"{type(self).__name__} must define a class-level `agent_id`"
            )
        self.model_config = model_config
        self.agent_config = agent_config
        # Injected by the driver. Required in singularity mode; in docker mode
        # the driver also passes a DockerContainer so this code path stays
        # uniform. None should not occur in practice for openclaw conditions,
        # but we defensively fall back to the legacy `docker exec` path so
        # callers that haven't been updated still work.
        self.container = container

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_message(self, instance_data: Dict) -> str:
        """Return the user message to send the agent. Subclass owns both the
        framing prose and which instance_data key to pull the task text from
        (problem_statement for SWE-bench, instruction for TB)."""
        raise NotImplementedError

    def _collect_artifacts(
        self,
        container_id: Optional[str],
        env_info: Dict,
        run_dir: Path,
        workspace_root: str,
    ) -> Optional[str]:
        """Optional: return artifact text (e.g. a git diff patch) to be
        written to ``candidate.patch``. Default = no artifact. SWE-bench
        overrides to collect a git diff; TB returns None (driver's
        ``skip_patch_gate`` handles the empty case for terminal_bench)."""
        return None

    def _await_post_turn_lifecycle(
        self,
        payload: Optional[Dict],
        error: str,
        *,
        deadline: float,
    ) -> Tuple[Optional[Dict], str]:
        """Benchmark hook for runtimes whose logical run spans several turns.

        The default is deliberately a no-op so ordinary OpenClaw benchmarks
        retain the existing single-CLI-turn behavior. RoadmapBench overrides
        this to keep a yielded requester alive while native subagent announce
        delivery resumes the same parent session.
        """
        return payload, error

    # ------------------------------------------------------------------

    def _resolve_agent_timeout_seconds(self, instance_data: Dict) -> int:
        default_timeout = int(self.agent_config.max_time_seconds)
        raw_timeout = instance_data.get("agent_timeout_sec")
        if raw_timeout is None:
            return default_timeout
        try:
            timeout = int(raw_timeout)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid agent_timeout_sec=%r for %s; using default %ds",
                raw_timeout,
                instance_data.get("instance_id", "<unknown>"),
                default_timeout,
            )
            return default_timeout
        if timeout <= 0:
            logger.warning(
                "Ignoring non-positive agent_timeout_sec=%r for %s; using default %ds",
                raw_timeout,
                instance_data.get("instance_id", "<unknown>"),
                default_timeout,
            )
            return default_timeout
        return timeout

    # ------------------------------------------------------------------

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id: Optional[str],
        env_info: Optional[Dict] = None,
    ) -> Dict:
        self._env_info = env_info or {}
        self._last_completion_status = self.RUNTIME_ERROR
        self._last_completion_error = "OpenClaw invocation did not complete"
        self._selected_agent_runtime_mode = None
        instance_id = instance_data["instance_id"]
        logger.info("Running OpenClaw agent (%s) for %s", self.agent_id, instance_id)

        # 1. Build the user message via subclass hook.
        message = self._build_message(instance_data)

        # Snapshot OTel trace file size BEFORE the agent runs. Under
        # max_workers=1 we can do a clean byte-slice; under concurrency,
        # multiple gateways write to the same file, so we ALSO filter the
        # captured envelopes by the resource attribute `oc.run_id` (set
        # in agent_bench/harnesses/openclaw/gateway_up.sh via OTEL_RESOURCE_ATTRIBUTES).
        otel_trace_file, pre_offset = otel_trace_offset()

        llm_container = (
            self._env_info.get("llm_container_name")
            or os.environ.get("OC_LLM_CONTAINER")
            or os.environ.get("VLLM_CONTAINER_NAME")
            or os.environ.get("LITELLM_CONTAINER_NAME")
        )
        client_container = self._env_info.get("container_name")
        tools_container = self._env_info.get("task_container_name")
        if tools_container == client_container:
            tools_container = None

        # In-loop verification runs entirely inside the sandbox container —
        # the `verify` skill (agent_bench/benchmarks/<app>/openclaw/skills/verify/) tells
        # the agent how to invoke the test command via its exec tool. The
        # harness's final `run_tests` (called by the driver after DONE) is the
        # authoritative score; this in-loop check is for iteration only.

        with UtilSampler(
            run_dir,
            llm_container=llm_container,
            client_container=client_container,
            tools_container=tools_container,
        ):
            step_start_ns = time.time_ns()
            timeout_seconds = self._resolve_agent_timeout_seconds(instance_data)
            lifecycle_deadline = time.monotonic() + timeout_seconds
            payload, err = self._invoke_agent(
                container_id=container_id,
                workspace_root=workspace_root,
                session_id=run_dir.name,
                message=message,
                timeout_seconds=timeout_seconds,
            )
            initial_runtime_mode = self._detect_agent_runtime_mode(payload)
            if initial_runtime_mode != "unknown":
                self._selected_agent_runtime_mode = initial_runtime_mode
            payload, err = self._await_post_turn_lifecycle(
                payload,
                err,
                deadline=lifecycle_deadline,
            )
            step_end_ns = time.time_ns()

        # Reuse the existing logical OpenClaw step boundary instead of
        # introducing another lifecycle clock. RoadmapBench returns from
        # _await_post_turn_lifecycle only after child/orphan quiescence.
        phase_timing = {
            "agent_execution_start_ns": step_start_ns,
            "agent_execution_end_ns": step_end_ns,
            "agent_execution_seconds": (step_end_ns - step_start_ns) / 1e9,
        }

        # diagnostics-otel uses BatchSpanProcessor under the hood. Spans only
        # flush on `sdk.shutdown()`, which runs when the gateway process exits.
        # Send SIGTERM to the in-container gateway so the final spans hit the
        # collector before we slice the trace file. We don't care about the
        # next agent run reusing this container — the harness rebuilds per run.
        if container_id is not None:
            try:
                handle = self._handle(container_id)
                if self.container is not None and handle is not None:
                    self.container.exec(
                        handle,
                        ["pkill", "-TERM", "-f", "openclaw-gateway"],
                        check=False, timeout=10,
                    )
                else:
                    subprocess.run(
                        ["docker", "exec", container_id,
                         "pkill", "-TERM", "-f", "openclaw-gateway"],
                        capture_output=True, timeout=10,
                    )
            except Exception as e:
                logger.warning("gateway SIGTERM failed: %s", e)
        wait_for_otel_flush()
        self._archive_gateway_log(run_dir)
        spans_path = snapshot_otel_spans(
            otel_trace_file, pre_offset, run_dir, expected_run_id=run_dir.name,
        )

        prompt_tokens, completion_tokens = self._parse_usage(payload)
        response_text = self._parse_text(payload)

        # 3. Log the run as a single LLM_INFER step. We note the subtype so
        #    downstream analysis can flag that this step aggregates OpenClaw's
        #    internal multi-step trajectory into one row rather than exposing
        #    individual LLM/tool calls.
        step = {
            "trajectory_id": run_dir.name,
            "attempt_id": 1,
            "step_id": 1,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": "LLM_INFER_OPENCLAW_AGENTIC_COARSE",
            "tool_name": "openclaw.agent",
            "command": f"openclaw agent --agent {self.agent_id} --session-id {run_dir.name}",
            "file_path": None,
            "start_ts": step_start_ns,
            "end_ts": step_end_ns,
            "duration_ms": (step_end_ns - step_start_ns) / 1e6,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "exit_code": None,
            "error_type": err or None,
            "agent_runtime_mode": self._detect_agent_runtime_mode(payload),
            "repo_id": instance_data.get("repo"),
            "instance_id": instance_id,
            "config_id": "openclaw_agent",
            "model_name": self.model_config.name,
            "response_preview": (response_text[:500] if response_text else None),
        }

        step_trace_path = run_dir / "steps.jsonl"
        with open(step_trace_path, "w") as f:
            f.write(json.dumps(step) + "\n")

        # Archive the full agent response for post-hoc inspection.
        if payload is not None:
            (run_dir / "openclaw_response.json").write_text(
                json.dumps(payload, indent=2)
            )

        # 3b. Always archive raw streams, successful parse or not.
        if getattr(self, "_last_stderr", None):
            (run_dir / "openclaw_stderr.log").write_text(self._last_stderr)
        if getattr(self, "_last_stdout", None):
            (run_dir / "openclaw_stdout.log").write_text(self._last_stdout)

        # 3c. Pull per-session conversation trace + cache trace out of the
        # container. session-id == run_dir.name; OpenClaw writes
        # <id>.jsonl + <id>.trajectory.jsonl under
        # /root/.openclaw/agents/<agent_id>/sessions/. cache-trace.jsonl
        # (enabled via diagnostics.cacheTrace) lands under .openclaw/logs/.
        # The raw response log is also archived here; it is the only artifact
        # that records one row per OpenAI-compatible LLM HTTP request.
        self._archive_session_artifacts(container_id, run_dir, run_dir.name)

        if spans_path is None:
            existing_spans_path = run_dir / "spans.jsonl"
            if existing_spans_path.exists():
                spans_path = existing_spans_path
            elif self._has_synthetic_span_sources(run_dir):
                spans_path = existing_spans_path
                spans_path.touch()
                logger.info("  created %s for synthesized spans", spans_path.name)

        # 3d. If the OC tool-call patches didn't fire (most builds), backfill
        # `openclaw.tool.call` spans from the archived session trajectory so
        # downstream analysis sees per-tool spans regardless.
        if spans_path is not None:
            try:
                self._maybe_synthesize_tool_call_spans(
                    spans_path, run_dir, run_dir.name
                )
            except Exception as exc:  # noqa: BLE001 - telemetry is non-critical
                logger.warning("tool span synthesis failed: %s", exc)

        # 3e. Prefer per-request model usage spans from the raw response log.
        # The older agentMeta.usage fallback is run-level aggregate data, so it
        # is intentionally used only when the per-request log is unavailable.
        if spans_path is not None:
            try:
                raw_usage_spans = self._maybe_synthesize_model_usage_spans_from_raw_log(
                    spans_path, run_dir, run_dir.name,
                )
                cache_usage_spans = 0
                if raw_usage_spans == 0:
                    cache_usage_spans = self._maybe_synthesize_model_usage_spans_from_cache_trace(
                        spans_path, run_dir, run_dir.name,
                    )
                if raw_usage_spans == 0 and cache_usage_spans == 0:
                    self._maybe_synthesize_model_usage_span(
                        spans_path, payload,
                        run_id=run_dir.name,
                        start_ns=step_start_ns,
                        end_ns=step_end_ns,
                    )
            except Exception as exc:  # noqa: BLE001 - telemetry is non-critical
                logger.warning("model usage span synthesis failed: %s", exc)

        measurement: Dict = {}
        try:
            from agent_bench.benchmarks.conditions.openclaw_measurement import (
                build_openclaw_measurement,
            )

            measurement = build_openclaw_measurement(
                run_dir,
                agent_id=self.agent_id,
                execution_start_ns=step_start_ns,
                execution_end_ns=step_end_ns,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the benchmark run
            logger.exception("OpenClaw measurement artifact generation failed")
            measurement = {"measurement_error": str(exc)}

        execution_time = (
            (measurement.get("runtime_observability") or {}).get(
                "execution_time"
            )
            if isinstance(measurement, dict)
            else None
        )
        if isinstance(execution_time, dict):
            for source, destination in (
                ("observed_operation_union_seconds", "observed_operation_seconds"),
                ("observed_model_union_seconds", "observed_model_seconds"),
                ("observed_tool_union_seconds", "observed_tool_seconds"),
                ("parent_yield_wait_seconds", "parent_yield_wait_seconds"),
                (
                    "unattributed_orchestration_seconds",
                    "unattributed_orchestration_seconds",
                ),
            ):
                value = execution_time.get(source)
                if isinstance(value, (int, float)):
                    phase_timing[destination] = float(value)

        agent_lifecycle = (
            (self._unwrap(payload).get("meta") or {}).get(
                "agentBenchLifecycle"
            )
            if payload else None
        )
        if self.agent_id == "roadmap_bench" and isinstance(agent_lifecycle, dict):
            parent_terminal_ns = agent_lifecycle.get("parentTerminalObservedNs")
            workspace_quiesced_ns = agent_lifecycle.get("workspaceQuiescedNs")
            if isinstance(parent_terminal_ns, int):
                phase_timing["parent_terminal_observed_ns"] = parent_terminal_ns
                phase_timing["parent_terminal_timestamp_quality"] = agent_lifecycle.get("parentTerminalTimestampQuality")
                phase_timing["parent_execution_seconds"] = max(
                    0.0, (parent_terminal_ns - step_start_ns) / 1e9
                )
            if isinstance(workspace_quiesced_ns, int):
                phase_timing["workspace_quiesced_ns"] = workspace_quiesced_ns
                phase_timing["workspace_quiesced_timestamp_quality"] = agent_lifecycle.get("workspaceQuiescedTimestampQuality")
                phase_timing["agent_execution_end_ns"] = workspace_quiesced_ns
                phase_timing["agent_execution_seconds"] = max(
                    0.0, (workspace_quiesced_ns - step_start_ns) / 1e9
                )
                if isinstance(parent_terminal_ns, int):
                    phase_timing["post_parent_quiescence_seconds"] = max(
                        0.0, (workspace_quiesced_ns - parent_terminal_ns) / 1e9
                    )
            try:
                from agent_bench.benchmarks.conditions.openclaw_measurement import (
                    append_trajectory_event,
                )

                append_trajectory_event(
                    run_dir,
                    "runtime_termination",
                    timestamp_ns=parent_terminal_ns,
                    status=self._last_completion_status,
                    parent_session_status=agent_lifecycle.get(
                        "parentSessionStatusAtTerminal"
                    ),
                    terminal_evidence=agent_lifecycle.get("terminalEvidence"),
                    done_marker=bool(agent_lifecycle.get("done_marker")),
                    stop_reason=agent_lifecycle.get("stopReason"),
                    pending_children=agent_lifecycle.get(
                        "pendingChildrenAtParentTerminal"
                    ),
                    workspace_quiesced=agent_lifecycle.get("workspaceQuiesced"),
                )
            except Exception:
                logger.exception("trajectory runtime termination event write failed")
        if (
            isinstance(agent_lifecycle, dict)
            and agent_lifecycle.get("pendingChildrenAtParentTerminal", 0)
        ):
            try:
                from agent_bench.benchmarks.conditions.openclaw_measurement import (
                    append_trajectory_event,
                )

                append_trajectory_event(
                    run_dir,
                    "orphan_cleanup",
                    status=(
                        "completed"
                        if agent_lifecycle.get("workspaceQuiesced") is True
                        else "error"
                    ),
                    lifecycle=agent_lifecycle,
                )
            except Exception:
                logger.exception("trajectory orphan cleanup event write failed")

        # 4. Collect benchmark-specific artifacts (patch for SWE-bench; nothing
        #    for TB). Always write candidate.patch — driver gates on its
        #    existence + size, and the empty-patch case is allowed for TB via
        #    skip_patch_gate at driver.py:187.
        candidate_freeze_start_ns = time.time_ns()
        patch_text = self._collect_artifacts(
            container_id, self._env_info, run_dir, workspace_root,
        ) or ""
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text(patch_text)
        candidate_freeze_end_ns = time.time_ns()
        phase_timing.update({
            "post_agent_before_freeze_seconds": max(
                0.0, (candidate_freeze_start_ns - step_end_ns) / 1e9
            ),
            "candidate_freeze_start_ns": candidate_freeze_start_ns,
            "candidate_freeze_end_ns": candidate_freeze_end_ns,
            "candidate_freeze_seconds": (
                candidate_freeze_end_ns - candidate_freeze_start_ns
            ) / 1e9,
        })
        if self.agent_id == "roadmap_bench":
            try:
                from agent_bench.benchmarks.conditions.openclaw_measurement import (
                    append_trajectory_event,
                )

                append_trajectory_event(
                    run_dir,
                    "candidate_freeze",
                    patch_path=str(patch_path),
                    patch_bytes=patch_path.stat().st_size,
                    workspace_baseline=self._env_info.get(
                        "roadmap_workspace_baseline"
                    ),
                )
            except Exception:
                logger.exception("trajectory candidate freeze event write failed")

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "telemetry_path": str(spans_path) if spans_path else None,
            "exit_code": getattr(self, "_last_exit_code", 0),
            "error_message": err or None,
            "agent_completion_status": self._last_completion_status,
            "agent_completion_error": self._last_completion_error or None,
            "agent_runtime_mode": self._detect_agent_runtime_mode(payload),
            "agent_lifecycle": agent_lifecycle,
            "agent_execution_seconds": phase_timing["agent_execution_seconds"],
            "phase_timing": phase_timing,
            **measurement,
        }

    # ------------------------------------------------------------------
    # OTel span snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_otel_spans(
        trace_file: str, pre_offset: int, run_dir: Path,
        expected_run_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Slice the OTel trace file from `pre_offset` to current EOF and
        write per-run span files into `run_dir`.

        When `expected_run_id` is set, envelopes are additionally filtered by
        the resource attribute `oc.run_id == expected_run_id`. Required
        when driver runs concurrent (multiple gateways write to one file);
        harmless when serial (each container's gateway tags its own runs).
        """
        try:
            cur_size = os.path.getsize(trace_file)
        except FileNotFoundError:
            return None
        if cur_size <= pre_offset:
            return None

        try:
            with open(trace_file, "rb") as fh:
                fh.seek(pre_offset)
                chunk = fh.read(cur_size - pre_offset)
        except OSError as e:
            logger.warning("OTel trace snapshot failed: %s", e)
            return None

        raw_lines: List[str] = []
        spans: List[Dict] = []
        for line in chunk.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                env = json.loads(line)
            except json.JSONDecodeError:
                continue
            rs_list = env.get("resourceSpans") or []
            if not rs_list:
                continue
            kept_envelopes = []
            for rs in rs_list:
                res_attrs = _otlp_attrs(
                    rs.get("resource", {}).get("attributes")
                )
                if expected_run_id is not None:
                    if res_attrs.get("oc.run_id") != expected_run_id:
                        continue
                kept_envelopes.append(rs)
                for ss in rs.get("scopeSpans", []) or []:
                    for sp in ss.get("spans", []) or []:
                        start = int(sp.get("startTimeUnixNano", 0))
                        end = int(sp.get("endTimeUnixNano", 0))
                        from datetime import datetime, timezone
                        start_iso = (
                            datetime.fromtimestamp(start / 1e9, tz=timezone.utc).isoformat()
                            if start else None
                        )
                        end_iso = (
                            datetime.fromtimestamp(end / 1e9, tz=timezone.utc).isoformat()
                            if end else None
                        )
                        attrs = _otlp_attrs(sp.get("attributes"))
                        attrs.setdefault("openclaw.span.source", "diagnostics-otel")
                        spans.append({
                            "name": sp.get("name"),
                            "start_unix_nano": start,
                            "end_unix_nano": end,
                            "start_iso": start_iso,
                            "end_iso": end_iso,
                            "duration_ms": (end - start) / 1e6 if end and start else 0,
                            "attrs": attrs,
                            "resource": res_attrs,
                        })
            if not kept_envelopes:
                continue
            raw_lines.append(json.dumps({"resourceSpans": kept_envelopes}, ensure_ascii=False))

        raw_path = run_dir / "spans_raw.jsonl"
        try:
            raw_path.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""))
        except OSError as e:
            logger.warning("OTel raw write failed: %s", e)
            return None

        spans_path = run_dir / "spans.jsonl"
        with spans_path.open("a") as fh:
            for s in spans:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(
            "  captured %d OTel span(s) -> %s", len(spans), spans_path.name
        )
        return spans_path

    # ------------------------------------------------------------------
    # OpenClaw CLI invocation
    # ------------------------------------------------------------------

    def _watch_durable_invocation_terminal(self) -> bool:
        """Whether this benchmark has a durable logical-session lifecycle."""
        return False

    def _durable_invocation_terminal_evidence(
        self,
    ) -> Optional[Tuple[Tuple, str, str]]:
        """Return (stable identity, completion status, error), if terminal."""
        return None

    def _payload_for_durable_invocation_terminal(
        self,
        status: str,
        diagnostic: str,
    ) -> Optional[Dict]:
        """Build a result when a durable session outlives its blocked CLI."""
        return None

    def _communicate_with_durable_watch(
        self,
        proc: subprocess.Popen,
        timeout_seconds: float,
    ) -> Tuple[str, str, Optional[Tuple[str, str]]]:
        """Wait for a CLI while allowing stable durable terminal state to end it."""
        deadline = time.monotonic() + timeout_seconds
        candidate = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_seconds)
            try:
                stdout, stderr = proc.communicate(
                    timeout=min(2.0, remaining)
                )
                return stdout or "", stderr or "", None
            except subprocess.TimeoutExpired:
                evidence = self._durable_invocation_terminal_evidence()
                if evidence is None:
                    candidate = None
                    continue
                signature, status, error = evidence
                if candidate != signature:
                    candidate = signature
                    continue

                logger.warning(
                    "  interrupting openclaw CLI after stable durable "
                    "terminal evidence: %s",
                    error,
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    stdout, stderr = proc.communicate()
                return stdout or "", stderr or "", (status, error)

    def _invoke_agent(
        self,
        container_id: Optional[str],
        workspace_root: str,
        session_id: str,
        message: str,
        timeout_seconds: int,
    ) -> Tuple[Optional[Dict], str]:
        """Run `openclaw agent ... --json`.

        Two modes:
          - container_id given → in-container via Gateway (sources
            /tmp/oc_gateway_env so OPENCLAW_GATEWAY_URL/_TOKEN reach the CLI).
          - container_id is None → singularity / host-local mode. Gateway +
            state dir come from env_info (set by the harness).

        Returns (parsed_json_or_none, error_message).
        """
        if container_id is None:
            ei = getattr(self, "_env_info", {}) or {}
            backend = ei.get("backend", "singularity")
            state_dir = ei.get("state_dir") or ei.get("host_state_dir")
            gateway_url = ei.get("gateway_url")
            gateway_token = ei.get("gateway_token")
            if not (state_dir and gateway_url and gateway_token):
                return None, (
                    f"{backend} mode but env_info missing state_dir / "
                    "gateway_url / gateway_token (harness setup failed?)"
                )
            argv = [
                "openclaw", "agent",
                "--agent", self.agent_id,
                "--session-id", session_id,
                "--message", message,
                "--timeout", str(int(timeout_seconds)),
                "--json",
            ]
            env = os.environ.copy()
            env["OPENCLAW_STATE_DIR"] = state_dir
            env["OPENCLAW_GATEWAY_URL"] = gateway_url
            env["OPENCLAW_GATEWAY_TOKEN"] = gateway_token
            if backend == "singularity":
                # Run openclaw inside the OC runtime sif. The harness already
                # validated that sif/overlay exist. The wrap helper is a pure
                # argv/env builder — benchmark-agnostic.
                from agent_bench.benchmarks.conditions.gateway_helpers import wrap_singularity_argv
                binds = [
                    (workspace_root, workspace_root),
                    (state_dir, state_dir),
                ]
                plugin_cache = ei.get("plugin_cache_dir")
                if plugin_cache:
                    binds.append((plugin_cache, f"{state_dir}/plugin-runtime-deps"))
                argv, env = wrap_singularity_argv(
                    argv, env, binds,
                    sif=ei["singularity_sif"],
                    overlay=ei.get("singularity_overlay"),
                )
            run_kwargs = {"cwd": workspace_root, "env": env}
        else:
            # Tee stderr live to a state_dir-side log so we can tail it from
            # the host while the agent is running — `subprocess.run` with
            # capture_output only flushes when the child exits, which makes
            # diagnosis of hangs impossible. /root/.openclaw is bind-mounted
            # to <state_dir> on the host, so /root/.openclaw/logs/agent.live.stderr
            # appears immediately at <state_dir>/logs/agent.live.stderr.
            # Stdout is the agent's --json payload; we still capture it
            # exclusively to parse, no need to split.
            inner = (
                "mkdir -p /root/.openclaw/logs && "
                "source /tmp/oc_gateway_env && "
                f"echo {shlex.quote(_GATEWAY_EXECUTION_MARKER)} >&2 && "
                "exec openclaw agent "
                f"--agent {shlex.quote(self.agent_id)} "
                f"--session-id {shlex.quote(session_id)} "
                f"--message {shlex.quote(message)} "
                f"--timeout {int(timeout_seconds)} "
                "--json "
                "2> >(tee /root/.openclaw/logs/agent.live.stderr >&2)"
            )
            argv = None  # signalled below: use self.container.exec when available
            run_kwargs = {}

        subprocess_timeout = timeout_seconds + 120
        # Log what we're about to launch — line is visible in run.log
        # immediately, so when the user sees the driver "stuck waiting for
        # agent" they can tell exactly which command is blocking.
        logger.info(
            "  openclaw agent dispatch: backend=%s container_id=%s "
            "agent=%s session=%s timeout=%ds wait_budget=%ds",
            (self.container.backend_id if self.container else "subprocess"),
            (container_id[:12] if container_id else None),
            self.agent_id, session_id, int(timeout_seconds), subprocess_timeout,
        )
        if container_id is None:
            logger.info(
                "    host-mode argv=%r (env keys: OPENCLAW_GATEWAY_URL=%s, "
                "OPENCLAW_GATEWAY_TOKEN=<%d chars>, OPENCLAW_STATE_DIR=%s)",
                argv[:6] + ["...truncated..."] if argv and len(argv) > 6 else argv,
                env.get("OPENCLAW_GATEWAY_URL"),
                len(env.get("OPENCLAW_GATEWAY_TOKEN", "")),
                env.get("OPENCLAW_STATE_DIR"),
            )
        else:
            logger.info(
                "    in-container inner=%r (token comes from /tmp/oc_gateway_env)",
                inner[:200] + ("..." if len(inner) > 200 else ""),
            )
        dispatch_t0 = time.time()
        durable_interrupt = None
        try:
            if self._watch_durable_invocation_terminal():
                if argv is None:
                    handle = self._handle(container_id)
                    if self.container is None or handle is None:
                        command = [
                            "docker", "exec", container_id, "bash", "-lc", inner
                        ]
                        running = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            start_new_session=True,
                            **run_kwargs,
                        )
                    else:
                        running = self.container.popen_exec(
                            handle, ["bash", "-lc", inner]
                        )
                else:
                    running = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                        **run_kwargs,
                    )
                stdout, stderr, durable_interrupt = (
                    self._communicate_with_durable_watch(
                        running, subprocess_timeout
                    )
                )
                proc = subprocess.CompletedProcess(
                    running.args, running.returncode, stdout, stderr
                )
            if argv is None:
                # In-container path. Route through the Container abstraction
                # so docker and singularity backends share one call site.
                handle = self._handle(container_id)
                if self._watch_durable_invocation_terminal():
                    pass
                elif self.container is None or handle is None:
                    # Legacy fallback — driver hasn't been updated to inject
                    # self.container, but we still have a docker container id.
                    argv = ["docker", "exec", container_id, "bash", "-lc", inner]
                    proc = subprocess.run(
                        argv,
                        capture_output=True, text=True,
                        timeout=subprocess_timeout,
                        **run_kwargs,
                    )
                else:
                    proc = self.container.exec(
                        handle,
                        ["bash", "-lc", inner],
                        check=False,
                        timeout=subprocess_timeout,
                    )
            elif not self._watch_durable_invocation_terminal():
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=subprocess_timeout,
                    **run_kwargs,
                )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - dispatch_t0
            logger.warning(
                "  openclaw agent SUBPROCESS TIMEOUT after %.1fs "
                "(budget was %ds). The child is still alive in the container; "
                "check <state_dir>/logs/agent.live.stderr for what it was doing.",
                elapsed, subprocess_timeout,
            )
            self._last_exit_code = -1
            self._last_completion_status = self.TIMEOUT
            self._last_completion_error = (
                "openclaw agent wall-clock timeout (subprocess)"
            )
            return None, "openclaw agent wall-clock timeout (subprocess)"

        elapsed = time.time() - dispatch_t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        self._last_exit_code = proc.returncode
        logger.info(
            "  openclaw agent returned: rc=%s elapsed=%.1fs "
            "stdout=%d bytes stderr=%d bytes",
            proc.returncode, elapsed, len(stdout), len(stderr),
        )

        self._last_stdout = stdout
        self._last_stderr = stderr

        if durable_interrupt is not None:
            status, diagnostic = durable_interrupt
            payload = self._payload_for_durable_invocation_terminal(
                status, diagnostic
            )
            if payload is not None and status == self.NORMAL_COMPLETION:
                self._last_exit_code = 0
                self._last_completion_status = status
                self._last_completion_error = ""
                return payload, ""
            self._last_completion_status = status
            self._last_completion_error = diagnostic
            return None, diagnostic

        payload = self._extract_last_json(stderr)
        if payload is None:
            payload = self._extract_last_json(stdout)
        if payload is None:
            snippet = (stderr or stdout)[-300:]
            error = (
                f"openclaw exit {proc.returncode}: " if proc.returncode != 0 else ""
            ) + (
                "no parseable JSON in openclaw output "
                f"(last 300 chars: {snippet})"
            )
            self._last_completion_status = self.RUNTIME_ERROR
            self._last_completion_error = error
            return None, error

        status, error = self._classify_completion(payload)
        if proc.returncode != 0 and status not in {
            self.CONTEXT_OVERFLOW, self.TIMEOUT
        }:
            status = self.RUNTIME_ERROR
            if not error:
                error = f"openclaw exit {proc.returncode}"
        self._last_completion_status = status
        self._last_completion_error = error
        return payload, error

    @staticmethod
    def _extract_last_json(text: str) -> Optional[Dict]:
        """Find the final agent-result JSON in OpenClaw's output."""
        candidates = []
        decoder = json.JSONDecoder()
        i = 0
        n = len(text)
        while True:
            pos = text.find("{", i)
            if pos < 0:
                break
            try:
                obj, end = decoder.raw_decode(text, pos)
                candidates.append(obj)
                i = end
            except json.JSONDecodeError:
                i = pos + 1

        if not candidates:
            return None

        def _is_final(obj):
            if not isinstance(obj, dict):
                return False
            if "payloads" in obj:
                return True
            meta = obj.get("meta") or {}
            if isinstance(meta, dict) and "agentMeta" in meta:
                return True
            return False

        for obj in reversed(candidates):
            if _is_final(obj):
                return obj
        return candidates[-1]

    def _detect_agent_runtime_mode(self, payload: Optional[Dict]) -> str:
        stderr = str(getattr(self, "_last_stderr", "") or "")
        if (
            "Gateway agent failed; falling back to embedded" in stderr
            or "falling back to embedded" in stderr.lower()
        ):
            return "embedded_fallback"
        if (
            _GATEWAY_EXECUTION_MARKER in stderr
            or _ROADMAP_GATEWAY_EXECUTION_MARKER in stderr
        ):
            return "gateway"
        selected = getattr(self, "_selected_agent_runtime_mode", None)
        if selected in {"gateway", "embedded_fallback"}:
            return str(selected)
        # A successful CLI response is not evidence of which execution path
        # produced it, and a non-zero rc does not erase an already selected
        # Gateway path. Only explicit path evidence is authoritative here.
        return "unknown"

    def _archive_gateway_log(self, run_dir: Path) -> None:
        raw_path = self._env_info.get("gateway_log_path")
        if not raw_path:
            return
        source = Path(str(raw_path))
        try:
            if source.is_file():
                shutil.copyfile(source, run_dir / "openclaw_gateway.log")
        except OSError as exc:
            logger.warning("gateway log archive failed for %s: %s", source, exc)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(payload: Optional[Dict]) -> Dict:
        """Gateway responses wrap the run in a `result` key; --local responses
        inline payloads/meta at the top level. Accept both."""
        if not payload:
            return {}
        if isinstance(payload.get("result"), dict):
            return payload["result"]
        return payload

    @classmethod
    def _classify_completion(cls, payload: Optional[Dict]) -> Tuple[str, str]:
        """Classify OpenClaw's semantic result independently of CLI rc.

        OpenClaw can return rc=0 and top-level ``status=ok`` after its tool
        loop was blocked by a context overflow. The authoritative lifecycle
        fields are under ``result.meta`` (or top-level ``meta`` in local
        mode), so inspect those before accepting a successful stop.
        """
        run = cls._unwrap(payload)
        if not run:
            return cls.RUNTIME_ERROR, "OpenClaw returned no result payload"

        meta = run.get("meta") or {}
        if not isinstance(meta, dict):
            return cls.RUNTIME_ERROR, "OpenClaw result meta is malformed"

        # Emitted only by a benchmark durable-session watcher after stable
        # logical completion when the command-line RPC itself remains stuck.
        if meta.get("agentBenchDurableRuntimeTermination") is True:
            return cls.NORMAL_COMPLETION, ""

        raw_error = meta.get("error")
        if isinstance(raw_error, dict) and raw_error:
            kind = str(raw_error.get("kind") or "runtime_error").strip().lower()
            message = str(raw_error.get("message") or kind)
            if kind == cls.CONTEXT_OVERFLOW or cls._is_context_overflow_text(message):
                return cls.CONTEXT_OVERFLOW, message
            return cls.RUNTIME_ERROR, f"{kind}: {message}"
        if raw_error:
            message = str(raw_error)
            if cls._is_context_overflow_text(message):
                return cls.CONTEXT_OVERFLOW, message
            return cls.RUNTIME_ERROR, message

        payload_error = cls._payload_error_text(payload)
        if payload_error:
            if cls._is_context_overflow_text(payload_error):
                return cls.CONTEXT_OVERFLOW, payload_error
            return cls.RUNTIME_ERROR, payload_error

        liveness = str(meta.get("livenessState") or "").strip().lower()
        if liveness in {
            "working", "blocked", "incomplete", "aborted", "cancelled", "canceled"
        }:
            return (
                cls.BLOCKED_INCOMPLETE,
                f"OpenClaw trajectory ended with livenessState={liveness}",
            )

        if meta.get("replayInvalid"):
            return (
                cls.BLOCKED_INCOMPLETE,
                "OpenClaw trajectory replay is invalid",
            )

        completion = meta.get("completion") or {}
        stop_reason = str(
            meta.get("stopReason")
            or (completion.get("stopReason") if isinstance(completion, dict) else "")
            or (completion.get("finishReason") if isinstance(completion, dict) else "")
            or ""
        ).strip().lower()
        final_text = str(meta.get("finalAssistantVisibleText") or cls._parse_text(payload))
        if stop_reason in {"stop", "completed", "complete", "done"} and final_text.strip():
            if cls.require_done_marker and not re.search(
                r"(?im)^\s*DONE(?:\s*$|\b)", final_text
            ):
                return (
                    cls.BLOCKED_INCOMPLETE,
                    "RoadmapBench final response did not explicitly declare DONE",
                )
            return cls.NORMAL_COMPLETION, ""

        return (
            cls.BLOCKED_INCOMPLETE,
            "OpenClaw returned without an explicit normal completion",
        )

    @staticmethod
    def _is_context_overflow_text(text: str) -> bool:
        normalized = " ".join(str(text).lower().replace("_", "-").split())
        return any(marker in normalized for marker in (
            "context-overflow",
            "context overflow",
            "context length exceeded",
            "maximum context length",
            "context size exceeds",
        ))

    @classmethod
    def _payload_error_text(cls, payload: Optional[Dict]) -> str:
        """Extract explicit error payloads without scanning normal prose."""
        if not isinstance(payload, dict):
            return ""
        run = cls._unwrap(payload)
        candidates = [payload.get("error"), run.get("error")]
        for item in run.get("payloads", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            item_text = str(item.get("text") or "")
            if item.get("isError") or item.get("error") or item_type == "error":
                candidates.append(item.get("error") or item_text or item)
            elif cls._is_context_overflow_text(item_text):
                candidates.append(item_text)

        for candidate in candidates:
            if not candidate:
                continue
            if isinstance(candidate, str):
                return candidate
            if isinstance(candidate, dict):
                return str(
                    candidate.get("message")
                    or candidate.get("text")
                    or candidate.get("kind")
                    or candidate
                )
            return str(candidate)
        return ""

    @classmethod
    def _parse_usage(cls, payload: Optional[Dict]) -> Tuple[int, int]:
        run = cls._unwrap(payload)
        if not run:
            return 0, 0
        meta = run.get("meta", {}) or {}
        agent = meta.get("agentMeta", {}) or {}
        usage = agent.get("usage", {}) or {}
        return int(usage.get("input", 0) or 0), int(usage.get("output", 0) or 0)

    @classmethod
    def _parse_text(cls, payload: Optional[Dict]) -> str:
        run = cls._unwrap(payload)
        if not run:
            return ""
        parts: List[str] = []
        for p in run.get("payloads", []) or []:
            t = p.get("text")
            if t:
                parts.append(t)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Synthetic openclaw.model.usage span (works around silent OCSB patch)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_synthetic_span_sources(run_dir: Path) -> bool:
        conversation_dir = run_dir / "conversation"
        return (
            (conversation_dir / "llm-raw-responses.jsonl").is_file()
            or (conversation_dir / "cache-trace.jsonl").is_file()
            or (conversation_dir / "sessions").is_dir()
        )

    @staticmethod
    def _maybe_synthesize_model_usage_span(
        spans_path: Path,
        payload: Optional[Dict],
        run_id: str,
        start_ns: int,
        end_ns: int,
    ) -> None:
        if payload is None:
            return
        meta = (payload.get("result") or {}).get("meta") or payload.get("meta") or {}
        am = meta.get("agentMeta") or {}
        usage = am.get("usage") or {}
        if not any(usage.get(k) for k in ("input", "output", "cacheRead", "cacheWrite", "total")):
            return

        try:
            existing = [json.loads(l) for l in open(spans_path) if l.strip()]
        except (OSError, json.JSONDecodeError):
            existing = []
        if any(
            s.get("name") == "openclaw.model.usage"
            and (s.get("attrs") or {}).get("openclaw.channel") not in (None, "heartbeat")
            for s in existing
        ):
            return

        duration_ms = float(meta.get("durationMs") or ((end_ns - start_ns) / 1e6))
        from datetime import datetime, timezone
        attrs = {
            "openclaw.channel": "synth.agent_call",
            "openclaw.provider": am.get("provider") or "unknown",
            "openclaw.model": am.get("model") or "unknown",
            "openclaw.sessionKey": am.get("sessionKey") or "",
            "openclaw.sessionId": am.get("sessionId") or "",
            "openclaw.tokens.input": int(usage.get("input") or 0),
            "openclaw.tokens.output": int(usage.get("output") or 0),
            "openclaw.tokens.cache_read": int(usage.get("cacheRead") or 0),
            "openclaw.tokens.cache_write": int(usage.get("cacheWrite") or 0),
            "openclaw.tokens.total": int(usage.get("total") or 0),
            "agent.canonical_name": "agent.model.usage",
            "agent.channel": "synth.agent_call",
            "agent.provider": am.get("provider") or "unknown",
            "agent.model": am.get("model") or "unknown",
            "agent.session_id": am.get("sessionId") or "",
            "agent.run_id": run_id,
            "agent.llm.request_index": 1,
            "agent.tokens.input": int(usage.get("input") or 0),
            "agent.tokens.output": int(usage.get("output") or 0),
            "agent.tokens.cache_read": int(usage.get("cacheRead") or 0),
            "agent.tokens.cache_write": int(usage.get("cacheWrite") or 0),
            "agent.tokens.total": int(usage.get("total") or 0),
        }
        span = {
            "name": "openclaw.model.usage",
            "canonical_name": "agent.model.usage",
            "start_unix_nano": start_ns,
            "end_unix_nano": end_ns,
            "start_iso": datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat(),
            "end_iso": datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "attrs": attrs,
            "resource": {"oc.run_id": run_id, "service.name": "agent-bench-openclaw"},
            "synthesized": True,
        }
        with open(spans_path, "a") as fh:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")
        logger.info("  synthesized model.usage span (input=%d output=%d)",
                    attrs["openclaw.tokens.input"], attrs["openclaw.tokens.output"])

    @staticmethod
    def _maybe_synthesize_model_usage_spans_from_raw_log(
        spans_path: Path,
        run_dir: Path,
        run_id: str,
    ) -> int:
        """Backfill one ``openclaw.model.usage`` span per LLM HTTP request.

        The OpenClaw agent result only exposes ``agentMeta.usage`` after the
        whole agent run completes. That aggregate is useful as a fallback, but
        it necessarily creates one long span whose duration is the full agent
        wall time. The patched pi-ai OpenAI provider writes
        ``conversation/llm-raw-responses.jsonl`` with one line per streaming
        chat-completions request, including request start time, response
        latency, and API usage. This method turns those rows into per-request
        spans so ``spans.jsonl`` can represent every LLM call.
        """
        raw_path = run_dir / "conversation" / "llm-raw-responses.jsonl"
        if not raw_path.is_file():
            return 0

        try:
            existing = [json.loads(l) for l in open(spans_path) if l.strip()]
        except (OSError, json.JSONDecodeError):
            existing = []
        if any(
            s.get("name") == "openclaw.model.usage"
            and (s.get("attrs") or {}).get("openclaw.channel") == "synth.raw_response"
            for s in existing
        ):
            return 0

        from datetime import datetime, timezone

        def _parse_start_ns(value) -> int:
            if isinstance(value, (int, float)):
                # Raw logger currently writes ISO strings, but accept epoch
                # seconds/milliseconds/nanoseconds for compatibility with
                # ad-hoc logs.
                f = float(value)
                if f > 1e15:
                    return int(f)
                return int(f * (1e6 if f > 1e12 else 1e9))
            if isinstance(value, str):
                try:
                    return int(datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp() * 1e9)
                except ValueError:
                    return 0
            return 0

        def _as_int(value) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        spans_out: List[Dict] = []
        try:
            lines = raw_path.read_text(errors="replace").splitlines()
        except OSError as e:
            logger.warning("raw model usage read failed: %s", e)
            return 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            metrics = row.get("metrics") or {}
            response = row.get("response") or {}
            usage = response.get("usage") or {}
            start_ns = _parse_start_ns(row.get("ts"))
            try:
                duration_ms = float(metrics.get("total_ms") or 0)
            except (TypeError, ValueError):
                duration_ms = 0.0
            end_ns = start_ns + int(duration_ms * 1e6) if start_ns else 0
            start_iso = (
                datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
                if start_ns else None
            )
            end_iso = (
                datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat()
                if end_ns else None
            )

            prompt_tokens = _as_int(metrics.get("prompt_tokens") or usage.get("prompt_tokens"))
            completion_tokens = _as_int(
                metrics.get("completion_tokens") or usage.get("completion_tokens")
            )
            cached_tokens = _as_int(metrics.get("cached_tokens"))
            if not cached_tokens:
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = _as_int(details.get("cached_tokens"))
            total_tokens = _as_int(metrics.get("total_tokens") or usage.get("total_tokens"))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            attrs = {
                "openclaw.channel": "synth.raw_response",
                "openclaw.span.source": "raw_response_log",
                "openclaw.provider": row.get("provider") or "unknown",
                "openclaw.model": row.get("model") or response.get("model") or "unknown",
                "openclaw.api": row.get("api") or "unknown",
                "openclaw.sessionId": row.get("sessionId") or "",
                "openclaw.runId": row.get("runId") or run_id,
                "openclaw.llm.request_index": len(spans_out) + 1,
                "openclaw.http.status_code": row.get("httpStatus"),
                "openclaw.tokens.input": prompt_tokens,
                "openclaw.tokens.output": completion_tokens,
                "openclaw.tokens.cache_read": cached_tokens,
                "openclaw.tokens.cache_write": 0,
                "openclaw.tokens.total": total_tokens,
                "openclaw.ttft_ms": metrics.get("ttft_ms"),
                "openclaw.tpot_ms": metrics.get("tbt_avg_ms"),
                "openclaw.headers_ms": metrics.get("headers_ms"),
                "openclaw.chunks.count": metrics.get("chunks_count"),
                "agent.canonical_name": "agent.model.usage",
                "agent.channel": "synth.raw_response",
                "agent.provider": row.get("provider") or "unknown",
                "agent.model": row.get("model") or response.get("model") or "unknown",
                "agent.api": row.get("api") or "unknown",
                "agent.session_id": row.get("sessionId") or "",
                "agent.run_id": row.get("runId") or run_id,
                "agent.llm.request_index": len(spans_out) + 1,
                "agent.http.status_code": row.get("httpStatus"),
                "agent.tokens.input": prompt_tokens,
                "agent.tokens.output": completion_tokens,
                "agent.tokens.cache_read": cached_tokens,
                "agent.tokens.cache_write": 0,
                "agent.tokens.total": total_tokens,
                "agent.ttft_ms": metrics.get("ttft_ms"),
                "agent.tpot_ms": metrics.get("tbt_avg_ms"),
                "agent.headers_ms": metrics.get("headers_ms"),
                "agent.chunks.count": metrics.get("chunks_count"),
            }
            spans_out.append({
                "name": "openclaw.model.usage",
                "canonical_name": "agent.model.usage",
                "start_unix_nano": start_ns,
                "end_unix_nano": end_ns,
                "start_iso": start_iso,
                "end_iso": end_iso,
                "duration_ms": duration_ms,
                "attrs": attrs,
                "resource": {
                    "oc.run_id": run_id,
                    "service.name": "agent-bench-openclaw",
                },
                "synthesized": True,
            })

        if not spans_out:
            return 0

        try:
            with open(spans_path, "a") as fh:
                for s in spans_out:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("raw model usage span write failed: %s", e)
            return 0

        logger.info("  synthesized %d per-request model.usage span(s)", len(spans_out))
        return len(spans_out)

    @staticmethod
    def _maybe_synthesize_model_usage_spans_from_cache_trace(
        spans_path: Path,
        run_dir: Path,
        run_id: str,
    ) -> int:
        """Backfill one ``openclaw.model.usage`` span per assistant response.

        Some OpenClaw builds do not emit ``llm-raw-responses.jsonl`` but do
        archive ``conversation/cache-trace.jsonl``. Its final ``session:after``
        snapshot contains every assistant message with per-response usage. This
        is less precise than the raw HTTP log, but still preserves one usage
        span per LLM serving call instead of collapsing the whole run into one
        aggregate span.
        """
        trace_path = run_dir / "conversation" / "cache-trace.jsonl"
        if not trace_path.is_file():
            return 0

        try:
            existing = [json.loads(l) for l in open(spans_path) if l.strip()]
        except (OSError, json.JSONDecodeError):
            existing = []
        if any(
            s.get("name") == "openclaw.model.usage"
            and (s.get("attrs") or {}).get("openclaw.channel") == "synth.cache_trace"
            for s in existing
        ):
            return 0

        from datetime import datetime, timezone

        def _parse_time_ns(value) -> int:
            if isinstance(value, (int, float)):
                f = float(value)
                if f > 1e15:
                    return int(f)
                if f > 1e12:
                    return int(f * 1e6)
                return int(f * 1e9)
            if isinstance(value, str):
                try:
                    return int(datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp() * 1e9)
                except ValueError:
                    return 0
            return 0

        def _as_int(value) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        try:
            rows = [
                json.loads(l)
                for l in trace_path.read_text(errors="replace").splitlines()
                if l.strip()
            ]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("cache trace model usage read failed: %s", e)
            return 0

        final_snapshot = None
        context_rows: List[Dict] = []
        for row in rows:
            if row.get("stage") == "stream:context":
                context_rows.append(row)
            if row.get("stage") == "session:after":
                final_snapshot = row
        if final_snapshot is None:
            return 0

        assistant_messages = [
            m for m in final_snapshot.get("messages", []) or []
            if (
                isinstance(m, dict)
                and m.get("role") == "assistant"
                and isinstance(m.get("usage"), dict)
            )
        ]
        if not assistant_messages:
            return 0

        tool_start_ns = sorted(
            int(s.get("start_unix_nano") or 0)
            for s in existing
            if s.get("name") in ("agent.tool.call", "openclaw.tool.call")
            and int(s.get("start_unix_nano") or 0) > 0
        )
        session_after_ns = _parse_time_ns(final_snapshot.get("ts"))
        spans_out: List[Dict] = []
        for idx, message in enumerate(assistant_messages, start=1):
            usage = message.get("usage") or {}
            context = context_rows[idx - 1] if idx <= len(context_rows) else {}
            start_ns = _parse_time_ns(context.get("ts"))
            end_ns = 0
            latency_source = "unknown"
            if start_ns:
                for candidate_ns in tool_start_ns:
                    if candidate_ns > start_ns:
                        end_ns = candidate_ns
                        latency_source = "next_tool_start"
                        break
            if not end_ns and start_ns and session_after_ns and session_after_ns > start_ns:
                end_ns = session_after_ns
                latency_source = "session_after"
            if not end_ns:
                end_ns = _parse_time_ns(message.get("timestamp"))
                latency_source = "message_timestamp"
            if not start_ns and end_ns:
                start_ns = end_ns
            if start_ns and not end_ns:
                end_ns = start_ns
            if start_ns and end_ns and end_ns < start_ns:
                start_ns = end_ns
            duration_ms = (end_ns - start_ns) / 1e6 if start_ns and end_ns else 0

            input_tokens = _as_int(usage.get("input"))
            output_tokens = _as_int(usage.get("output"))
            cache_read_tokens = _as_int(usage.get("cacheRead"))
            cache_write_tokens = _as_int(usage.get("cacheWrite"))
            total_tokens = _as_int(usage.get("totalTokens") or usage.get("total"))
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens

            attrs = {
                "openclaw.channel": "synth.cache_trace",
                "openclaw.span.source": "cache_trace",
                "openclaw.provider": (
                    message.get("provider") or final_snapshot.get("provider") or "unknown"
                ),
                "openclaw.model": (
                    message.get("model") or final_snapshot.get("modelId") or "unknown"
                ),
                "openclaw.api": (
                    message.get("api") or final_snapshot.get("modelApi") or "unknown"
                ),
                "openclaw.sessionKey": final_snapshot.get("sessionKey") or "",
                "openclaw.sessionId": final_snapshot.get("sessionId") or "",
                "openclaw.runId": final_snapshot.get("runId") or run_id,
                "openclaw.llm.request_index": idx,
                "openclaw.llm.response_id": message.get("responseId") or "",
                "openclaw.llm.stop_reason": message.get("stopReason") or "",
                "openclaw.latency.source": latency_source,
                "openclaw.tokens.input": input_tokens,
                "openclaw.tokens.output": output_tokens,
                "openclaw.tokens.cache_read": cache_read_tokens,
                "openclaw.tokens.cache_write": cache_write_tokens,
                "openclaw.tokens.total": total_tokens,
            }
            spans_out.append({
                "name": "openclaw.model.usage",
                "start_unix_nano": start_ns,
                "end_unix_nano": end_ns,
                "start_iso": (
                    datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
                    if start_ns else None
                ),
                "end_iso": (
                    datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat()
                    if end_ns else None
                ),
                "duration_ms": duration_ms,
                "attrs": attrs,
                "resource": {
                    "oc.run_id": run_id,
                    "service.name": "agent-bench-openclaw",
                },
                "synthesized": True,
            })

        try:
            with open(spans_path, "a") as fh:
                for s in spans_out:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("cache trace model usage span write failed: %s", e)
            return 0

        logger.info("  synthesized %d cache-trace model.usage span(s)", len(spans_out))
        return len(spans_out)

    # ------------------------------------------------------------------
    # Synthetic openclaw.tool.call spans (replay session trajectory)
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_synthesize_tool_call_spans(
        spans_path: Path,
        run_dir: Path,
        run_id: str,
    ) -> int:
        sessions_dir = run_dir / "conversation" / "sessions"
        if not sessions_dir.is_dir():
            return 0

        try:
            existing = [json.loads(l) for l in open(spans_path) if l.strip()]
        except (OSError, json.JSONDecodeError):
            existing = []
        if any(s.get("name") == "openclaw.tool.call" for s in existing):
            return 0

        from datetime import datetime, timezone

        arg_limit = int(os.environ.get("OC_TOOL_SPAN_ARGUMENT_LIMIT", "20000"))
        result_limit = int(os.environ.get("OC_TOOL_SPAN_RESULT_LIMIT", "20000"))
        details_limit = int(os.environ.get("OC_TOOL_SPAN_DETAILS_LIMIT", "20000"))

        def _json_preview(value, limit: int) -> Tuple[str, int, bool]:
            if value is None:
                return "", 0, False
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                text = str(value)
            original_len = len(text)
            if limit >= 0 and original_len > limit:
                return text[:limit], original_len, True
            return text, original_len, False

        def _content_text(content) -> str:
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return "" if content is None else str(content)
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)

        def _to_unix_ns(t) -> int:
            if t is None:
                return 0
            if isinstance(t, int):
                if t > 10**15:
                    return t
                if t > 10**12:
                    return t * 1_000_000
                return t * 1_000_000_000
            if isinstance(t, float):
                value = float(t)
                if value > 1e15:
                    return int(value)
                if value > 1e12:
                    return int(value * 1e6)
                return int(value * 1e9)
            if isinstance(t, str):
                try:
                    iso = t.replace("Z", "+00:00")
                    return int(datetime.fromisoformat(iso).timestamp() * 1e9)
                except ValueError:
                    return 0
            return 0

        pending: Dict[str, Dict] = {}
        spans_out: List[Dict] = []
        session_id = ""
        session_key = ""

        raw_files = sorted(
            p for p in sessions_dir.glob("*.jsonl")
            if not p.name.endswith(".trajectory.jsonl")
        )
        traj_files = sorted(sessions_dir.glob("*.trajectory.jsonl"))
        files = raw_files or traj_files

        for path in files:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if evt.get("type") == "session":
                    session_id = evt.get("id") or session_id

                if evt.get("type") != "message":
                    continue
                msg = evt.get("message") or {}
                role = msg.get("role")

                if role == "assistant":
                    sk = msg.get("sessionKey") or ""
                    if sk:
                        session_key = sk
                    content = msg.get("content") or []
                    if not isinstance(content, list):
                        continue
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "toolCall":
                            cid = c.get("id")
                            if not cid:
                                continue
                            pending[cid] = {
                                "name": c.get("name") or "unknown",
                                "arguments": c.get("arguments"),
                                "timestamp": msg.get("timestamp") or evt.get("timestamp"),
                            }
                    continue

                if role == "toolResult":
                    cid = msg.get("toolCallId")
                    if not cid:
                        continue
                    info = pending.pop(cid, None)
                    tool_name = (
                        (info or {}).get("name")
                        or msg.get("toolName")
                        or "unknown"
                    )
                    details = msg.get("details") or {}
                    duration_ms = float(details.get("durationMs") or 0)
                    end_ns = _to_unix_ns(
                        msg.get("timestamp") or evt.get("timestamp")
                    )
                    start_ns = 0
                    if end_ns and duration_ms > 0:
                        start_ns = end_ns - int(duration_ms * 1e6)
                    elif end_ns:
                        start_ns = end_ns
                    else:
                        start_ns = _to_unix_ns((info or {}).get("timestamp"))
                        end_ns = start_ns
                    if start_ns and end_ns and start_ns > end_ns:
                        start_ns = end_ns
                    duration_ms = (end_ns - start_ns) / 1e6 if start_ns and end_ns else 0.0
                    is_error = bool(
                        msg.get("isError")
                        or details.get("status") == "failed"
                    )
                    args_json, args_chars, args_truncated = _json_preview(
                        (info or {}).get("arguments"), arg_limit,
                    )
                    result_text = _content_text(msg.get("content"))
                    result_chars = len(result_text)
                    result_preview = result_text[:result_limit]
                    result_truncated = result_limit >= 0 and result_chars > result_limit
                    details_json, details_chars, details_truncated = _json_preview(
                        details, details_limit,
                    )
                    spans_out.append({
                        "name": "openclaw.tool.call",
                        "canonical_name": "agent.tool.call",
                        "start_unix_nano": start_ns,
                        "end_unix_nano": end_ns,
                        "start_iso": (
                            datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
                            if start_ns else None
                        ),
                        "end_iso": (
                            datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat()
                            if end_ns else None
                        ),
                        "duration_ms": duration_ms,
                        "attrs": {
                            "openclaw.channel": "synth.tool_call",
                            "openclaw.span.source": "session_trajectory",
                            "openclaw.tool.timing": (
                                "result_timestamp_minus_duration"
                                if details.get("durationMs") is not None
                                else "result_timestamp"
                            ),
                            "openclaw.tool.name": tool_name,
                            "openclaw.tool.callId": cid,
                            "openclaw.tool.isError": is_error,
                            "openclaw.tool.arguments_json": args_json,
                            "openclaw.tool.arguments_chars": args_chars,
                            "openclaw.tool.arguments_truncated": args_truncated,
                            "openclaw.tool.result_preview": result_preview,
                            "openclaw.tool.result_chars": result_chars,
                            "openclaw.tool.result_truncated": result_truncated,
                            "openclaw.tool.details_json": details_json,
                            "openclaw.tool.details_chars": details_chars,
                            "openclaw.tool.details_truncated": details_truncated,
                            "openclaw.runId": run_id,
                            "openclaw.sessionKey": session_key,
                            "openclaw.sessionId": session_id,
                            "agent.canonical_name": "agent.tool.call",
                            "agent.channel": "synth.tool_call",
                            "agent.tool.name": tool_name,
                            "agent.tool.call_id": cid,
                            "agent.tool.is_error": is_error,
                            "agent.tool.arguments_json": args_json,
                            "agent.tool.arguments_chars": args_chars,
                            "agent.tool.arguments_truncated": args_truncated,
                            "agent.tool.result_preview": result_preview,
                            "agent.tool.result_chars": result_chars,
                            "agent.tool.result_truncated": result_truncated,
                            "agent.run_id": run_id,
                            "agent.session_id": session_id,
                        },
                        "resource": {
                            "oc.run_id": run_id,
                            "service.name": "agent-bench-openclaw",
                        },
                        "synthesized": True,
                    })

        if not spans_out:
            return 0

        try:
            with open(spans_path, "a") as fh:
                for s in spans_out:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("synthesized tool.call write failed: %s", e)
            return 0

        logger.info("  synthesized %d tool.call span(s)", len(spans_out))
        return len(spans_out)

    # ------------------------------------------------------------------
    # Session / cache trace archival — path scoped by `agent_id`
    # ------------------------------------------------------------------

    def _archive_session_artifacts(
        self, container_id: Optional[str], run_dir: Path, session_id: str,
    ) -> None:
        """Copy the per-session trajectory + cache trace out of the container.

        OpenClaw assigns its own UUID for session files (the --session-id we
        pass is just a session *key* hint); rather than guess the file name,
        we copy the entire sessions/ dir for this agent. Each container is
        fresh so the dir contains exactly this run's artifacts.
        """
        out_dir = run_dir / "conversation"
        out_dir.mkdir(exist_ok=True)

        sessions_subpath = f"agents/{self.agent_id}/sessions"
        cache_subpath = "logs/cache-trace.jsonl"
        exec_bridge_subpath = "logs/openclaw_exec_bridge.jsonl"
        # Raw per-LLM-call response stream (pi-ai patch writes here when
        # OPENCLAW_RAW_RESPONSE_LOG points at it). Each line = one HTTP request.
        raw_resp_subpath = "logs/llm-raw-responses.jsonl"

        ei = getattr(self, "_env_info", {}) or {}
        host_state = ei.get("state_dir") or ei.get("host_state_dir")
        if host_state or container_id is None:
            state = Path(host_state or os.environ.get(
                "OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")
            ))
            for src, dst in [
                (state / sessions_subpath, out_dir / "sessions"),
                (state / cache_subpath, out_dir / "cache-trace.jsonl"),
                (state / raw_resp_subpath, out_dir / "llm-raw-responses.jsonl"),
            ]:
                if not src.exists():
                    continue
                try:
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    logger.debug("host session archive skipped %s: %s", src, e)
            exec_bridge_src = state / exec_bridge_subpath
            if exec_bridge_src.exists():
                try:
                    shutil.copy2(exec_bridge_src, run_dir / "openclaw_exec_bridge.jsonl")
                except Exception as e:
                    logger.debug("host exec-bridge log skipped %s: %s", exec_bridge_src, e)
            return

        handle = self._handle(container_id)
        for src, dst in [
            (f"/root/.openclaw/{sessions_subpath}", "sessions"),
            (f"/root/.openclaw/{cache_subpath}", "cache-trace.jsonl"),
            (f"/root/.openclaw/{raw_resp_subpath}", "llm-raw-responses.jsonl"),
        ]:
            try:
                if self.container is not None and handle is not None:
                    self.container.copy_out(handle, src, out_dir / dst)
                else:
                    r = subprocess.run(
                        ["docker", "cp", f"{container_id}:{src}", str(out_dir / dst)],
                        capture_output=True, text=True,
                    )
                    if r.returncode != 0:
                        logger.debug(
                            "session archive skipped %s: %s",
                            src, r.stderr.strip(),
                        )
            except Exception as e:
                logger.debug("session archive skipped %s: %s", src, e)
        exec_bridge_src = f"/root/.openclaw/{exec_bridge_subpath}"
        exec_bridge_dst = run_dir / "openclaw_exec_bridge.jsonl"
        try:
            if self.container is not None and handle is not None:
                self.container.copy_out(handle, exec_bridge_src, exec_bridge_dst)
            else:
                r = subprocess.run(
                    ["docker", "cp", f"{container_id}:{exec_bridge_src}", str(exec_bridge_dst)],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    logger.debug(
                        "exec-bridge log skipped %s: %s",
                        exec_bridge_src, r.stderr.strip(),
                    )
        except Exception as e:
            logger.debug("exec-bridge log skipped %s: %s", exec_bridge_src, e)

    # ------------------------------------------------------------------
    # In-container exec helper — routes through self.container when present.
    # ------------------------------------------------------------------

    def _handle(self, container_id: Optional[str]) -> Optional[ContainerHandle]:
        """Build a ContainerHandle for the in-container ops.

        In singularity mode `container_id` is the instance name (not a docker
        id); the Container.exec implementation addresses instances by name,
        so this is fine.
        """
        if container_id is None or self.container is None:
            return None
        return ContainerHandle(
            id=container_id, name=container_id,
            backend=self.container.backend_id,
        )

    def _exec(self, container_id: str, argv: List[str], check: bool = True, timeout: int = 60):
        handle = self._handle(container_id)
        if self.container is not None and handle is not None:
            return self.container.exec(
                handle, argv, check=check, timeout=timeout,
            )
        # Legacy fallback when no container is injected.
        cmd = ["docker", "exec", container_id] + argv
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({argv[0]}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r
