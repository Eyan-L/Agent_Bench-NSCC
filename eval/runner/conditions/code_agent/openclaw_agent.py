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
import subprocess
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

logger = logging.getLogger(__name__)


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
        container_id: str,
    ) -> Dict:
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
        # in eval/frontend/openclaw/gateway_up.sh via OTEL_RESOURCE_ATTRIBUTES).
        # The byte slice is just a heuristic to bound the read window.
        otel_trace_file = _OTEL_TRACE_FILE_DEFAULT
        try:
            pre_offset = os.path.getsize(otel_trace_file)
        except FileNotFoundError:
            pre_offset = 0

        step_start_ns = time.time_ns()
        payload, err = self._invoke_agent(
            container_id=container_id,
            session_id=run_dir.name,
            message=message,
            timeout_seconds=self.agent_config.max_time_seconds,
        )
        step_end_ns = time.time_ns()

        # diagnostics-otel uses BatchSpanProcessor under the hood. Spans
        # only flush on `sdk.shutdown()`, which runs when the gateway
        # process exits. Send SIGTERM to the in-container gateway so the
        # final spans hit the collector before we slice the trace file.
        # We don't care about the next agent run reusing this container —
        # the harness rebuilds per run.
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
                        spans.append({
                            "name": sp.get("name"),
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
        container_id: str,
        session_id: str,
        message: str,
        timeout_seconds: int,
    ) -> Tuple[Optional[Dict], str]:
        """Run `openclaw agent ... --json` against the in-container Gateway.

        Gateway endpoint + token are written into /tmp/oc_gateway_env by
        eval/frontend/openclaw/gateway_up.sh during prepare_environment. We source
        that file inside the container so OPENCLAW_GATEWAY_URL/_TOKEN are
        visible to the CLI. We do NOT pass --local — gateway mode is what
        avoids the "[tools] exec failed: gateway closed" fallback and what
        makes the patched server-side dispatcher emit model.usage spans.

        Returns (parsed_json_or_none, error_message).
        """
        # shlex.quote `message` defends against agent prompts containing
        # quotes, backticks, $, etc. that would otherwise break out of the
        # `bash -c` wrapper. timeout_seconds and session_id are int/uuid-ish
        # so they don't need quoting but we run them through quote anyway.
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

        # Give OpenClaw a margin above its own timeout so we get its error
        # output instead of a docker-level kill.
        subprocess_timeout = timeout_seconds + 120

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=subprocess_timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "openclaw agent wall-clock timeout (subprocess)"

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
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
        candidates = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1

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

    def _collect_patch(self, container_id: str, workspace_root: str) -> str:
        ws = workspace_root
        # Strip OpenClaw scaffolding.
        self._exec(
            container_id,
            ["bash", "-c",
             f"cd {ws} && rm -f {self._OC_SCAFFOLD_FILES} && "
             f"rm -rf {self._OC_SCAFFOLD_DIRS}"],
            check=False,
        )
        # Stage + diff only the real changes.
        self._exec(
            container_id,
            ["git", "-C", ws, "add", "-A"],
            check=False,
        )
        r = self._exec(
            container_id,
            ["git", "-C", ws, "diff", "--cached"],
            check=False,
        )
        return r.stdout if r.returncode == 0 else ""

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