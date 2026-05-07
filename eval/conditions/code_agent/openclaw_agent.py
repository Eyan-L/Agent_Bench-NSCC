"""OpenClaw agent condition — uses the real OpenClaw CLI inside the container.

Runtime contract:
    - The container already has:
        * node at /usr/local/bin/node
        * OpenClaw package at /opt/openclaw
        * openclaw.json mounted at /root/.openclaw/openclaw.json (ro)
        * agents/ mounted at /root/.openclaw/agents (ro)
    - Agent id `swebench` is registered in the config with workspace=/testbed
      and tools.profile=coding / exec.host=node / exec.security=full.

What this condition does for one run:
    1. Make `openclaw` callable: symlink /opt/openclaw/openclaw.mjs into PATH.
    2. Invoke `openclaw agent --local --agent swebench --session-id <run>
       --message <issue> --json`.
    3. Parse the response JSON — extract assistant text, token usage, and
       durationMs — and log a single LLM_INFER step with those numbers. This
       is coarse: we cannot decompose OpenClaw's internal multi-step
       trajectory into individual tool/LLM steps from the CLI output alone.
       Phase 0/1 accept this; finer-grained tracing requires hooking
       OpenClaw internals.
    4. `git diff HEAD` inside the container — agent's edits live in /testbed.
    5. Write steps.jsonl + candidate.patch next to the manifest.

Returned dict matches the shape the driver expects.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

_UTIL_SAMPLER_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "util_sampler.py"

logger = logging.getLogger(__name__)


class _UtilSampler:
    """Spawn scripts/util_sampler.py as a sidecar around the OC request.

    Toggle off via OC_UTIL_SAMPLE=0; tune cadence via OC_UTIL_SAMPLE_INTERVAL
    (seconds, default 1.0). Output lands at <run_dir>/util.jsonl.

    Per-group CPU split ({llm_engine, client, tools, other}) is enabled when
    any of llm_pid / client_pid / gateway_pid is provided. llm_pid defaults
    to the value of $OC_LLM_PID (the PBS driver exports this from $VLLM_PID
    once vLLM is up). client_pid defaults to the calling process's pid (the
    eval driver). gateway_pid is the OC gateway whose descendants are tools.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        llm_pid: Optional[int] = None,
        client_pid: Optional[int] = None,
        gateway_pid: Optional[int] = None,
    ):
        self.run_dir = run_dir
        self.proc: Optional[subprocess.Popen] = None
        self.enabled = (
            os.environ.get("OC_UTIL_SAMPLE", "1") != "0"
            and _UTIL_SAMPLER_SCRIPT.exists()
        )
        try:
            self.interval = float(os.environ.get("OC_UTIL_SAMPLE_INTERVAL", "1.0"))
        except ValueError:
            self.interval = 1.0

        if llm_pid is None:
            try:
                llm_pid = int(os.environ.get("OC_LLM_PID") or 0) or None
            except ValueError:
                llm_pid = None
        if client_pid is None:
            client_pid = os.getpid()
        self.llm_pid = llm_pid
        self.client_pid = client_pid
        self.gateway_pid = gateway_pid

    def __enter__(self) -> "_UtilSampler":
        if not self.enabled:
            return self
        out_path = self.run_dir / "util.jsonl"
        argv = [
            sys.executable, str(_UTIL_SAMPLER_SCRIPT),
            "--output", str(out_path),
            "--interval", str(self.interval),
        ]
        if self.llm_pid:
            argv += ["--llm-pid", str(self.llm_pid)]
        if self.client_pid:
            argv += ["--client-pid", str(self.client_pid)]
        if self.gateway_pid:
            argv += ["--gateway-pid", str(self.gateway_pid)]
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            logger.info(
                "  util sampler pid=%s -> %s (interval=%.2fs, "
                "llm=%s client=%s gateway=%s)",
                self.proc.pid, out_path.name, self.interval,
                self.llm_pid, self.client_pid, self.gateway_pid,
            )
        except Exception as e:
            logger.warning("util sampler launch failed: %s", e)
            self.proc = None
        return self

    def __exit__(self, *exc) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception as e:
            logger.warning("util sampler stop failed: %s", e)


# ------------------------------------------------------------------
# Prompt — kept short; the real instructions live in AGENTS.md.
# ------------------------------------------------------------------

def _build_message(issue_text: str) -> str:
    return (
        "You are fixing a bug. The repository is in the current working "
        "directory. Read the issue, modify the source files to fix it, then "
        "reply with the single word DONE.\n\n"
        f"Issue:\n{issue_text}"
    )


class OpenClawAgentCondition:
    """Run the `swebench` OpenClaw agent against a task via docker exec."""

    def __init__(self, model_config, agent_config):
        self.model_config = model_config
        self.agent_config = agent_config

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
        logger.info("Running OpenClaw agent for %s", instance_id)

        # 1. Invoke the agent. --local = embedded mode, no gateway.
        #    session-id is the run_dir basename so trajectories stay isolated
        #    per run even if we reuse the container in the future.
        message = _build_message(instance_data["problem_statement"])

        # Snapshot OTel trace file size BEFORE the agent runs. Under
        # max_workers=1 we can do a clean byte-slice; under concurrency,
        # multiple gateways write to the same file, so we ALSO filter the
        # captured envelopes by the resource attribute `oc.run_id` (set
        # in frontend/openclaw/gateway_up.sh via OTEL_RESOURCE_ATTRIBUTES).
        # The byte slice is just a heuristic to bound the read window.
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

        # Self-validation bridge: lets the agent invoke `bash _oc_validate.sh`
        # inside its container to score the current workspace and iterate
        # until tests pass. No-op if the harness ref isn't available.
        from eval.conditions._validate_bridge import ValidateBridge
        bridge: Optional[ValidateBridge] = None
        harness = self._env_info.get("_harness")
        if harness is not None and container_id is None:
            # ValidateBridge talks to harness.run_tests via shared workspace
            # files — it works for the host-bound (singularity / cluster)
            # path. For docker mode the workspace is bind-mounted at /testbed
            # but not at the host path, so reuse only when container_id is None
            # (HostHarness / SingularityHarness).
            bridge = ValidateBridge(
                harness=harness,
                run_dir=run_dir,
                workspace_root=workspace_root,
                instance_id=instance_id,
                instance_data=instance_data,
                container_id=container_id,
                run_id=run_dir.name,
                timeout_seconds=getattr(
                    self.agent_config, "validate_timeout_seconds", 600,
                ),
            )
            bridge.start()

        with _UtilSampler(run_dir, gateway_pid=gw_pid):
            step_start_ns = time.time_ns()
            try:
                payload, err = self._invoke_agent(
                    container_id=container_id,
                    workspace_root=workspace_root,
                    session_id=run_dir.name,
                    message=message,
                    timeout_seconds=self.agent_config.max_time_seconds,
                )
            finally:
                if bridge is not None:
                    n = bridge.stop()
                    logger.info("validate bridge: %d call(s) during run", n)
            step_end_ns = time.time_ns()

        # diagnostics-otel uses BatchSpanProcessor under the hood. Spans
        # only flush on `sdk.shutdown()`, which runs when the gateway
        # process exits. Send SIGTERM to the in-container gateway so the
        # final spans hit the collector before we slice the trace file.
        # We don't care about the next agent run reusing this container —
        # the harness rebuilds per run.
        if container_id is not None:
            try:
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
        # patch is silent for direct CLI→gateway agent calls (gateway-side
        # `agentCommandFromIngress` patch path doesn't fire in this code
        # path). Synthesize an equivalent span from agentMeta.usage so
        # downstream analysis sees a uniform shape regardless of which
        # OpenClaw version produced the trajectory.
        if spans_path is not None:
            self._maybe_synthesize_model_usage_span(
                spans_path, payload,
                run_id=run_dir.name,
                start_ns=step_start_ns,
                end_ns=step_end_ns,
            )

        # 3. Log the run as a single LLM_INFER step.
        #    We note the subtype so downstream analysis can flag that this
        #    step aggregates OpenClaw's internal multi-step trajectory into
        #    one row rather than exposing individual LLM/tool calls.
        step = {
            "trajectory_id": run_dir.name,
            "attempt_id": 1,
            "step_id": 1,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": "LLM_INFER_OPENCLAW_AGENTIC_COARSE",
            "tool_name": "openclaw.agent",
            "command": f"openclaw agent --agent swebench --session-id {run_dir.name}",
            "file_path": None,
            "start_ts": step_start_ns,
            "end_ts": step_end_ns,
            "duration_ms": (step_end_ns - step_start_ns) / 1e6,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "exit_code": None,
            "error_type": err or None,
            "repo_id": instance_data["repo"],
            "instance_id": instance_id,
            "config_id": "openclaw_agent",
            "model_name": self.model_config.name,
            # Preserve a truncated response for debugging; full JSON is in
            # the _raw_response.json side file.
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
        # <id>.jsonl (raw turns) + <id>.trajectory.jsonl (structured trace)
        # under /root/.openclaw/agents/swebench/sessions/. cache-trace.jsonl
        # (enabled via diagnostics.cacheTrace) lands under .openclaw/logs/.
        self._archive_session_artifacts(container_id, run_dir, run_dir.name)

        # 3d. If the OC tool-call patches didn't fire (most builds), backfill
        # `openclaw.tool.call` spans from the archived session trajectory so
        # downstream analysis sees per-tool spans regardless.
        if spans_path is not None:
            self._maybe_synthesize_tool_call_spans(spans_path, run_dir, run_dir.name)

        # 4. Collect the patch from the container. The agent edited files in
        #    /testbed directly (workspace=/testbed, tools.exec.host=node),
        #    so `git diff HEAD` gives us what to score.
        patch_text = self._collect_patch(container_id, workspace_root)
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

        Produces two files (best-effort — never raises):
            spans_raw.jsonl  — raw OTLP envelopes for this run
            spans.jsonl      — flattened, one span per line:
                {name, duration_ms, attrs, resource}
        Returns the path of `spans.jsonl` or None if nothing was captured.
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

        # Decode + filter envelopes. spans_raw.jsonl keeps only resourceSpans
        # envelopes (the file exporter mixes spans/metrics/logs in one file).
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
    # OpenClaw invocation
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
          - container_id is None → host mode (--local), cwd=workspace_root.
            No gateway, no docker. OTel still emits if collector is up.

        Returns (parsed_json_or_none, error_message).
        """
        if container_id is None:
            # Host or singularity mode: gateway + state dir come from the
            # harness via env_info (set in run()). Drop --local so OTel + the
            # patched gateway-side dispatcher are active, matching docker mode.
            ei = getattr(self, "_env_info", {}) or {}
            backend = ei.get("backend", "host")
            state_dir = ei.get("host_state_dir")
            gateway_url = ei.get("gateway_url")
            gateway_token = ei.get("gateway_token")
            if not (state_dir and gateway_url and gateway_token):
                return None, (
                    f"{backend} mode but env_info missing host_state_dir / "
                    "gateway_url / gateway_token (harness setup failed?)"
                )
            argv = [
                "openclaw", "agent",
                "--agent", "swebench",
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
                # validated that sif/overlay exist.
                from eval.benchmarks.swebench.harness import wrap_singularity_argv
                binds = [
                    (workspace_root, workspace_root),
                    (state_dir, state_dir),
                ]
                # Plugin cache bind mirrors the gateway's so the agent process
                # sees the same pre-installed plugin runtime deps (acpx,
                # browser, …). Without this, OpenClaw would try to npm-install
                # them again per run — fatal on HPC compute nodes (no internet).
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
            inner = (
                "source /tmp/oc_gateway_env && "
                "openclaw agent "
                f"--agent swebench "
                f"--session-id {shlex.quote(session_id)} "
                f"--message {shlex.quote(message)} "
                f"--timeout {int(timeout_seconds)} "
                "--json"
            )
            argv = ["docker", "exec", container_id, "bash", "-lc", inner]
            run_kwargs = {}

        # Give OpenClaw a margin above its own timeout so we get its error
        # output instead of a docker-level kill.
        subprocess_timeout = timeout_seconds + 120

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=subprocess_timeout,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired:
            return None, "openclaw agent wall-clock timeout (subprocess)"

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            self._last_stdout = stdout
            self._last_stderr = stderr
            tail = stderr[-500:] or stdout[-500:]
            return None, f"openclaw exit {proc.returncode}: {tail}"

        # OpenClaw --json writes the JSON payload to STDERR (plugin load lines
        # and agent logs go there too; the final JSON object is last).
        # stdout is typically empty. Try stderr first, fall back to stdout.
        # Archive raw streams on self so run() can persist them. OpenClaw
        # emits per-tool events to stderr; keeping the raw log is the only
        # way to reconstruct a fine-grained trajectory post-hoc.
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
        """Find the final agent-result JSON in OpenClaw's output.

        OpenClaw --json can emit:
        (a) a single top-level JSON object (plain JSON mode)
        (b) JSONL — many events, one per line, the final one being the result
        (c) mixed log lines + the result JSON at the very end

        Strategy: enumerate every balanced top-level {...} in the text,
        parse each, and return the LAST one that "looks like" a final
        agent result (has either "payloads" or "meta.agentMeta").

        Fall back to the last parseable object if nothing looks like a
        final result.
        """
        # Use json.JSONDecoder.raw_decode at every `{` position. raw_decode
        # is string-aware (won't be tripped by `{`/`}` inside string literals,
        # which happen when the agent's stderr embeds Python code blocks).
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

        # Prefer the last object that looks like a final agent result.
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

        # Nothing looked final — return the last parseable object for diagnostics.
        return candidates[-1]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(payload: Optional[Dict]) -> Dict:
        """Gateway responses wrap the run in a `result` key:
            {runId, status, summary, result: {payloads, meta: {agentMeta: ...}}}
        --local responses inline payloads/meta at the top level. Accept both.
        """
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
    # Patch collection
    # ------------------------------------------------------------------

    # OpenClaw writes these files into the workspace at agent startup. They
    # are not part of the SWE-bench repo and must be stripped before diffing
    # or the resulting patch will be rejected by run_evaluation.
    _OC_SCAFFOLD_FILES = (
        "AGENTS.md SOUL.md IDENTITY.md USER.md TOOLS.md HEARTBEAT.md "
        "BOOT.md MEMORY.md fix.patch"
    )
    _OC_SCAFFOLD_DIRS = ".openclaw memory sessions"

    def _collect_patch(self, container_id: Optional[str], workspace_root: str) -> str:
        ws = workspace_root
        if container_id is None:
            # Host mode: run git directly in the workspace dir.
            for f in self._OC_SCAFFOLD_FILES.split():
                try:
                    os.remove(os.path.join(ws, f))
                except FileNotFoundError:
                    pass
            for d in self._OC_SCAFFOLD_DIRS.split():
                shutil.rmtree(os.path.join(ws, d), ignore_errors=True)
            subprocess.run(["git", "-C", ws, "add", "-A"], capture_output=True)
            r = subprocess.run(
                ["git", "-C", ws, "diff", "--cached"],
                capture_output=True, text=True,
            )
            return r.stdout if r.returncode == 0 else ""

        # Docker mode.
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OC_SCAFFOLD_FILES} && "
             f"rm -rf {self._OC_SCAFFOLD_DIRS}"],
            check=False,
        )
        self._exec(container_id, ["git", "-C", ws, "add", "-A"], check=False)
        r = self._exec(container_id, ["git", "-C", ws, "diff", "--cached"], check=False)
        return r.stdout if r.returncode == 0 else ""

    # ------------------------------------------------------------------
    # Synthetic model.usage span (works around silent OCSB patch)
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_synthesize_model_usage_span(
        spans_path: Path,
        payload: Optional[Dict],
        run_id: str,
        start_ns: int,
        end_ns: int,
    ) -> None:
        """Append a synthesized openclaw.model.usage span if the captured
        OTel stream didn't already include one for this run.

        Attribute names match what diagnostics-otel would have produced via
        `recordModelUsage` (see openclaw/dist/extensions/diagnostics-otel/
        index.js around line 54247) so downstream consumers don't need a
        separate code path.
        """
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
        # OC core also emits a periodic `openclaw.model.usage` span on the
        # `heartbeat` channel — that's a session aggregate, not a per-agent
        # call, so it must NOT shadow the synthesized per-call span. Dedupe
        # by (name, channel) instead of name alone.
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
    # Synthetic openclaw.tool.call spans (works around unapplied OC
    # selection/diagnostics-otel patches by replaying the archived
    # session trajectory).
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_synthesize_tool_call_spans(
        spans_path: Path,
        run_dir: Path,
        run_id: str,
    ) -> int:
        """Synthesize one openclaw.tool.call span per (toolCall, toolResult)
        pair found in the archived session trajectory.

        Schema mirrors openclaw-diagnostics-otel-tool-call.patch. Tagged
        with `openclaw.channel: "synth.tool_call"` so synth and patch spans
        are distinguishable in downstream analysis.

        No-op if real (non-synth) tool.call spans already exist — i.e. the
        OC patches did fire and the OTel stream already covered them.
        """
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
                # OC writes message timestamps as unix milliseconds.
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

        # Prefer raw <UUID>.jsonl (top-level toolCall/toolResult messages);
        # fall back to *.trajectory.jsonl if that's all we have. Skip the
        # already-aggregated trajectory variant when the raw file exists to
        # avoid double-counting.
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
    # Session / cache trace archival
    # ------------------------------------------------------------------

    def _archive_session_artifacts(
        self, container_id: Optional[str], run_dir: Path, session_id: str,
    ) -> None:
        """Copy the per-session trajectory + cache trace out of the container.

        OpenClaw assigns its own UUID for session files (the --session-id we
        pass is just a session *key* hint); rather than guess the file name,
        we copy the entire sessions/ dir for this agent. Each container is
        fresh so the dir contains exactly this run's artifacts.

        Pulled into run_dir/conversation/:
          sessions/                         — raw + trajectory jsonl (UUID-named)
          cache-trace.jsonl                 — if diagnostics.cacheTrace.enabled
        """
        out_dir = run_dir / "conversation"
        out_dir.mkdir(exist_ok=True)

        if container_id is None:
            # Host mode: per-run state dir lives in env_info (set by HostHarness).
            ei = getattr(self, "_env_info", {}) or {}
            state = Path(ei.get("host_state_dir") or os.environ.get(
                "OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")
            ))
            for src, dst in [
                (state / "agents/swebench/sessions", out_dir / "sessions"),
                (state / "logs/cache-trace.jsonl", out_dir / "cache-trace.jsonl"),
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

        for src, dst in [
            ("/root/.openclaw/agents/swebench/sessions", "sessions"),
            ("/root/.openclaw/logs/cache-trace.jsonl", "cache-trace.jsonl"),
        ]:
            r = subprocess.run(
                ["docker", "cp", f"{container_id}:{src}", str(out_dir / dst)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                logger.debug("session archive skipped %s: %s", src, r.stderr.strip())

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _exec(container_id: str, argv: List[str], check: bool = True, timeout: int = 60):
        cmd = ["docker", "exec", container_id] + argv
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker exec failed ({argv[0]}): rc={r.returncode}\n"
                f"stderr: {r.stderr[-500:]}"
            )
        return r