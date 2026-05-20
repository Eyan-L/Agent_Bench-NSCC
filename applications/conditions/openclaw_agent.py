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
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from applications.conditions.util_sampler import UtilSampler
from orchestration.container.base import Container, ContainerHandle

logger = logging.getLogger(__name__)


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
    "OC_OTEL_TRACE_FILE",
    os.path.join(os.environ.get("OC_OTEL_DATA_DIR", "/tmp/otel"), "otel-traces.jsonl"),
)
# Time to wait after the agent run before reading the trace file, so the
# diagnostics-otel plugin's batch processor has a chance to flush. The plugin
# is configured with flushIntervalMs=20 in openclaw.json; 1.5s is generous.
_OTEL_FLUSH_WAIT_SECONDS = float(os.environ.get("OC_OTEL_FLUSH_WAIT_SECONDS", "1.5"))


class OpenClawAgentCondition:
    """Base class — instantiate a subclass that fixes agent_id + prompt."""

    # Subclass must set this. Used in `--agent <id>` CLI flag AND the
    # session-archive paths under state_dir/agents/<id>/.
    agent_id: ClassVar[str] = ""

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
        instance_id = instance_data["instance_id"]
        logger.info("Running OpenClaw agent (%s) for %s", self.agent_id, instance_id)

        # 1. Build the user message via subclass hook.
        message = self._build_message(instance_data)

        # Snapshot OTel trace file size BEFORE the agent runs. Under
        # max_workers=1 we can do a clean byte-slice; under concurrency,
        # multiple gateways write to the same file, so we ALSO filter the
        # captured envelopes by the resource attribute `oc.run_id` (set
        # in harnesses/openclaw/gateway_up.sh via OTEL_RESOURCE_ATTRIBUTES).
        otel_trace_file = _OTEL_TRACE_FILE_DEFAULT
        try:
            pre_offset = os.path.getsize(otel_trace_file)
        except FileNotFoundError:
            pre_offset = 0

        gw_pid = self._env_info.get("gateway_pid")
        try:
            gw_pid = int(gw_pid) if gw_pid else None
        except (TypeError, ValueError):
            gw_pid = None

        # In-loop verification runs entirely inside the sandbox container —
        # the `verify` skill (applications/<app>/openclaw/skills/verify/) tells
        # the agent how to invoke the test command via its exec tool. The
        # harness's final `run_tests` (called by the driver after DONE) is the
        # authoritative score; this in-loop check is for iteration only.

        with UtilSampler(run_dir, gateway_pid=gw_pid):
            step_start_ns = time.time_ns()
            payload, err = self._invoke_agent(
                container_id=container_id,
                workspace_root=workspace_root,
                session_id=run_dir.name,
                message=message,
                timeout_seconds=self.agent_config.max_time_seconds,
            )
            step_end_ns = time.time_ns()

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
        time.sleep(_OTEL_FLUSH_WAIT_SECONDS)
        spans_path = self._snapshot_otel_spans(
            otel_trace_file, pre_offset, run_dir, expected_run_id=run_dir.name,
        )

        prompt_tokens, completion_tokens = self._parse_usage(payload)
        response_text = self._parse_text(payload)

        # The OpenClaw OCSB patch is *supposed* to emit `openclaw.model.usage`
        # spans via the gateway's diagnostic-events bus, but on this build the
        # patch is silent for direct CLI→gateway agent calls. Synthesize an
        # equivalent span from agentMeta.usage so downstream analysis sees a
        # uniform shape regardless of which OpenClaw version produced the
        # trajectory.
        if spans_path is not None:
            self._maybe_synthesize_model_usage_span(
                spans_path, payload,
                run_id=run_dir.name,
                start_ns=step_start_ns,
                end_ns=step_end_ns,
            )

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
        self._archive_session_artifacts(container_id, run_dir, run_dir.name)

        # 3d. If the OC tool-call patches didn't fire (most builds), backfill
        # `openclaw.tool.call` spans from the archived session trajectory so
        # downstream analysis sees per-tool spans regardless.
        if spans_path is not None:
            self._maybe_synthesize_tool_call_spans(spans_path, run_dir, run_dir.name)

        # 4. Collect benchmark-specific artifacts (patch for SWE-bench; nothing
        #    for TB). Always write candidate.patch — driver gates on its
        #    existence + size, and the empty-patch case is allowed for TB via
        #    skip_patch_gate at driver.py:187.
        patch_text = self._collect_artifacts(
            container_id, self._env_info, run_dir, workspace_root,
        ) or ""
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text(patch_text)

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "telemetry_path": str(spans_path) if spans_path else None,
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
                        spans.append({
                            "name": sp.get("name"),
                            "start_unix_nano": start,
                            "end_unix_nano": end,
                            "start_iso": start_iso,
                            "end_iso": end_iso,
                            "duration_ms": (end - start) / 1e6 if end and start else 0,
                            "attrs": _otlp_attrs(sp.get("attributes")),
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
        with spans_path.open("w") as fh:
            for s in spans:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(
            "  captured %d OTel span(s) -> %s", len(spans), spans_path.name
        )
        return spans_path

    # ------------------------------------------------------------------
    # OpenClaw CLI invocation
    # ------------------------------------------------------------------

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
                from applications.conditions.gateway_helpers import wrap_singularity_argv
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
        try:
            if argv is None:
                # In-container path. Route through the Container abstraction
                # so docker and singularity backends share one call site.
                handle = self._handle(container_id)
                if self.container is None or handle is None:
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
            else:
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
            return None, "openclaw agent wall-clock timeout (subprocess)"

        elapsed = time.time() - dispatch_t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        logger.info(
            "  openclaw agent returned: rc=%s elapsed=%.1fs "
            "stdout=%d bytes stderr=%d bytes",
            proc.returncode, elapsed, len(stdout), len(stderr),
        )

        if proc.returncode != 0:
            self._last_stdout = stdout
            self._last_stderr = stderr
            tail = stderr[-500:] or stdout[-500:]
            return None, f"openclaw exit {proc.returncode}: {tail}"

        self._last_stdout = stdout
        self._last_stderr = stderr

        payload = self._extract_last_json(stderr)
        if payload is None:
            payload = self._extract_last_json(stdout)
        if payload is None:
            snippet = (stderr or stdout)[-300:]
            return None, f"no parseable JSON in openclaw output (last 300 chars: {snippet})"

        return payload, ""

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
        }
        span = {
            "name": "openclaw.model.usage",
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
        if any(
            s.get("name") == "openclaw.tool.call"
            and (s.get("attrs") or {}).get("openclaw.channel") != "synth.tool_call"
            for s in existing
        ):
            return 0

        from datetime import datetime, timezone

        def _to_unix_ns(t) -> int:
            if t is None:
                return 0
            if isinstance(t, (int, float)):
                return int(float(t) * 1e6)
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
                            pending[cid] = {"name": c.get("name") or "unknown"}
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
                    start_ns = (
                        end_ns - int(duration_ms * 1e6) if end_ns else 0
                    )
                    is_error = bool(
                        msg.get("isError")
                        or details.get("status") == "failed"
                    )
                    spans_out.append({
                        "name": "openclaw.tool.call",
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
                            "openclaw.tool.name": tool_name,
                            "openclaw.tool.callId": cid,
                            "openclaw.tool.isError": is_error,
                            "openclaw.runId": run_id,
                            "openclaw.sessionKey": session_key,
                            "openclaw.sessionId": session_id,
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
        # Raw per-LLM-call response stream (pi-ai patch writes here when
        # OPENCLAW_RAW_RESPONSE_LOG points at it). Each line = one HTTP request.
        raw_resp_subpath = "logs/llm-raw-responses.jsonl"

        if container_id is None:
            ei = getattr(self, "_env_info", {}) or {}
            state = Path(ei.get("state_dir") or ei.get("host_state_dir") or os.environ.get(
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
