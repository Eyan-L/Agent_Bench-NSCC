"""ReAct-style condition for Terminal-Bench 2 tasks.

The TB tasks need an agent that can execute commands inside a running task
container (writing files, installing packages, spawning the gRPC server,
etc.) — OpenClaw integration is more involved because the task image
(python:3.13-slim) doesn't ship node/openclaw, so OC's tool routing into
the task container needs a derivative .sif. That's a separate piece of work.

For the MVP we use a plain multi-turn loop with the OpenAI-compatible
endpoint already in use by ar_static_context. The agent emits one or more
<bash>...</bash> blocks per turn; we run each in the task container via
`singularity exec instance://<instance>` and feed stdout/stderr back. The
agent stops by emitting `DONE` (or hitting the step limit).

Output shape matches what the driver expects:
    {step_trace_path, patch_path, total_prompt_tokens,
     total_completion_tokens, telemetry_path}

`patch_path` is written as an empty marker file — TB has no patch artifact.
The driver's empty-patch gate is bypassed for benchmark=terminal_bench
instances (see agent_bench/runtime/driver.py).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent_bench.benchmarks.conditions._llm_client import call_llm
from agent_bench.benchmarks.conditions.util_sampler import UtilSampler

logger = logging.getLogger(__name__)


def _emit_span(name: str, start_ns: int, end_ns: int,
               attrs: Dict, resource: Dict) -> Dict:
    """Build one flattened span dict matching the shape that
    agent_bench/benchmarks/conditions/openclaw_agent.py:_snapshot_otel_spans
    writes — so spans.jsonl is interchangeable across conditions and the
    downstream analyzers don't have to branch on producer.
    """
    return {
        "name": name,
        "start_unix_nano": int(start_ns),
        "end_unix_nano": int(end_ns),
        "start_iso": datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat(),
        "end_iso": datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat(),
        "duration_ms": (end_ns - start_ns) / 1e6,
        "attrs": attrs,
        "resource": resource,
    }


_BASH_BLOCK_RE = re.compile(r"<bash>\s*(.*?)\s*</bash>", re.DOTALL | re.IGNORECASE)
_DONE_RE = re.compile(r"\bDONE\b")

# Defaults; condition __init__ overrides from agent_config.
_DEFAULT_MAX_STEPS = 20
_DEFAULT_PER_CMD_TIMEOUT_SECONDS = 120
_DEFAULT_CONTEXT_TAIL_BYTES = 4000


_SYSTEM_PROMPT = """\
You are an autonomous shell agent solving a Terminal-Bench task. You are
running on a host that has shell access to a Linux container where the task
must be completed.

You can execute one or more shell commands per turn by wrapping them in
<bash>...</bash> blocks. Each block runs as `bash -lc <content>` inside the
task container. You will receive the combined stdout+stderr (truncated) and
exit code as the assistant's next user message.

Rules:
- Write files with heredocs: cat > /app/foo.py <<'EOF' ... EOF
- Background processes: append `&` to keep them running across turns.
- pip install must specify exact versions when the task asks for them.
- Do not emit DONE while a required server is not yet running.
- If a command fails, read the stderr and try a different approach.
- One <bash> block per turn is preferred; multiple is allowed.

Validation before DONE:
- Before emitting DONE, verify the verifier will pass by running:
      <bash>bash /tests/test.sh && cat /logs/verifier/reward.txt</bash>
  The reward file holds `1` for pass, `0` for fail. Only emit DONE on `1`.
- If reward is `0`, read the verifier output, fix the issue, and re-run.
- If you've tried several iterations and the verifier still fails, emit DONE
  anyway so the run is recorded — repeated identical failures waste budget.
"""


class TerminalBenchReactCondition:
    """Driver-compatible condition for benchmark=terminal_bench instances."""

    def __init__(self, model_config, agent_config=None):
        self.model_config = model_config
        self.agent_config = agent_config
        self.max_steps = (
            agent_config.max_steps if agent_config else _DEFAULT_MAX_STEPS
        )
        # Per-bash-command timeout; total wall budget is enforced by the
        # driver via wall_clock_budget_seconds and run_tests timeout.
        self.per_cmd_timeout = _DEFAULT_PER_CMD_TIMEOUT_SECONDS

    # ------------------------------------------------------------------

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id: Optional[str],          # None — TB harness uses instance
        env_info: Optional[Dict] = None,
    ) -> Dict:
        env_info = env_info or {}
        instance_name = env_info.get("instance_name") or env_info.get("container_name")
        if not instance_name:
            raise RuntimeError(
                "TerminalBenchReactCondition needs env_info.instance_name "
                "(set by TerminalBench{Docker,Singularity}Harness.prepare_environment)"
            )
        # docker vs singularity exec is chosen by the harness via env_info.
        # Default to docker so a future harness that forgets to set the key
        # still works on a docker host.
        self._container_engine = env_info.get("container_engine", "docker")
        # For docker mode, the task container id is more stable across exec
        # calls than the name (no slug collisions); prefer it when present.
        self._task_target = (
            env_info.get("task_container_id")
            or env_info.get("task_container_name")
            or instance_name
        )

        instance_id = instance_data["instance_id"]
        instruction = instance_data["instruction"]
        logger.info("TB ReAct: instance=%s task=%s", instance_name, instance_id)

        steps: List[Dict] = []
        spans: List[Dict] = []
        # Resource attrs match what OC's gateway tags spans with, so the
        # filter on `oc.run_id` in cross-run analysis works the same way.
        resource_attrs = {
            "oc.run_id": run_dir.name,
            "service.name": "tb_react",
            "tb.instance_id": instance_id,
            "tb.condition_id": "tb_react",
            "tb.model_name": getattr(self.model_config, "name", None) or "",
        }
        run_start_ns = time.time_ns()

        total_prompt_tokens = 0
        total_completion_tokens = 0
        done = False

        # Conversation history rolls forward between turns. We reuse call_llm
        # which is single-shot, so we collapse the running history into one
        # `prompt` per turn. That's lossy on multi-message structure but
        # adequate for a 20-step ReAct loop.
        history: List[Tuple[str, str]] = []  # [(role, content)]

        # TB has no OC gateway; tool processes are children of the driver
        # (singularity exec subprocesses). Pass the driver pid as the
        # "tools origin" so the sampler attributes those children to the
        # tools group; the driver process itself stays under client.
        import os as _os
        sampler_cm = UtilSampler(run_dir, gateway_pid=_os.getpid())
        sampler_cm.__enter__()
        try:
          for step_idx in range(self.max_steps):
            prompt = self._compose_prompt(instruction, history)
            step_start_ns = time.time_ns()
            text, pt, ct, err = call_llm(
                self.model_config, prompt, seed=seed,
                timeout_seconds=600,
            )
            step_end_ns = time.time_ns()
            total_prompt_tokens += pt
            total_completion_tokens += ct

            steps.append(self._make_step(
                trajectory_id=run_dir.name,
                step_id=step_idx * 2 + 1,
                step_type="LLM_INFER",
                tool_name=None,
                command=None,
                start_ns=step_start_ns,
                end_ns=step_end_ns,
                instance_id=instance_id,
                prompt_tokens=pt,
                completion_tokens=ct,
                err=err,
                response_preview=text[:500] if text else None,
            ))
            spans.append(_emit_span(
                name="tb.llm.infer",
                start_ns=step_start_ns, end_ns=step_end_ns,
                attrs={
                    "tb.step_idx": step_idx,
                    "tb.prompt_tokens": pt,
                    "tb.completion_tokens": ct,
                    "tb.error": err or "",
                },
                resource=resource_attrs,
            ))

            if err:
                history.append(("assistant", f"[llm error: {err}]"))
                break

            history.append(("assistant", text))

            commands = _BASH_BLOCK_RE.findall(text or "")
            if not commands:
                if _DONE_RE.search(text or ""):
                    done = True
                    break
                # No commands AND no DONE — nudge the model.
                history.append((
                    "user",
                    "You did not emit any <bash> block and did not signal DONE. "
                    "Either emit a <bash>...</bash> block to make progress, or "
                    "emit DONE if the task is complete.",
                ))
                continue

            # Run each command, accumulate the tool_call results into history.
            tool_outputs = []
            for cmd_idx, cmd in enumerate(commands):
                t_start = time.time_ns()
                result = self._exec_in_container(self._task_target, cmd)
                t_end = time.time_ns()

                steps.append(self._make_step(
                    trajectory_id=run_dir.name,
                    step_id=step_idx * 2 + 2,
                    step_type="TOOL_EXEC",
                    tool_name="container_bash",
                    command=cmd[:500],
                    start_ns=t_start,
                    end_ns=t_end,
                    instance_id=instance_id,
                    exit_code=result["rc"],
                    err=("timeout" if result["timed_out"] else None),
                ))
                spans.append(_emit_span(
                    name="tb.tool.exec",
                    start_ns=t_start, end_ns=t_end,
                    attrs={
                        "tb.step_idx": step_idx,
                        "tb.cmd_idx": cmd_idx,
                        "tb.command": cmd[:500],
                        "tb.exit_code": result["rc"],
                        "tb.timed_out": result["timed_out"],
                    },
                    resource=resource_attrs,
                ))

                tool_outputs.append(self._format_tool_output(cmd_idx, cmd, result))

            history.append(("user", "\n\n".join(tool_outputs)))

            if _DONE_RE.search(text or ""):
                done = True
                break
        finally:
            sampler_cm.__exit__(None, None, None)

        # Emit empty patch marker so the driver's exists+nonempty gate stays
        # uniform across benchmarks. The driver branches on benchmark to skip
        # the gate; this file is for archival consistency.
        patch_path = run_dir / "candidate.patch"
        patch_path.write_text("# terminal_bench: no patch — agent state in container\n")

        step_trace_path = run_dir / "steps.jsonl"
        with step_trace_path.open("w") as fh:
            for step in steps:
                fh.write(json.dumps(step) + "\n")

        # Save the full transcript for post-hoc inspection.
        (run_dir / "tb_transcript.json").write_text(json.dumps([
            {"role": role, "content": content} for role, content in history
        ], indent=2))

        if not done:
            logger.warning("TB ReAct hit max_steps=%d without DONE", self.max_steps)

        # Session span wraps the whole trajectory; written first so spans.jsonl
        # opens with the parent (mirrors how OC orders its run-level span).
        run_end_ns = time.time_ns()
        session_span = _emit_span(
            name="tb.session",
            start_ns=run_start_ns, end_ns=run_end_ns,
            attrs={
                "tb.steps_run": len(steps),
                "tb.done": done,
                "tb.total_prompt_tokens": total_prompt_tokens,
                "tb.total_completion_tokens": total_completion_tokens,
            },
            resource=resource_attrs,
        )
        spans_path: Optional[Path] = None
        try:
            spans_path = run_dir / "spans.jsonl"
            with spans_path.open("w") as fh:
                fh.write(json.dumps(session_span, ensure_ascii=False) + "\n")
                for span in spans:
                    fh.write(json.dumps(span, ensure_ascii=False) + "\n")
            logger.info(
                "  wrote %d span(s) -> %s", len(spans) + 1, spans_path,
            )
        except OSError as e:
            logger.warning("spans.jsonl write failed: %s", e)
            spans_path = None

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "telemetry_path": str(spans_path) if spans_path else None,
        }

    # ------------------------------------------------------------------

    def _compose_prompt(self, instruction: str, history: List[Tuple[str, str]]) -> str:
        parts = [_SYSTEM_PROMPT, "", "Task:", instruction.strip(), ""]
        if history:
            parts.append("Conversation so far:")
            for role, content in history:
                snippet = content
                if len(snippet) > _DEFAULT_CONTEXT_TAIL_BYTES:
                    snippet = (
                        snippet[: _DEFAULT_CONTEXT_TAIL_BYTES // 2]
                        + f"\n... (truncated {len(snippet) - _DEFAULT_CONTEXT_TAIL_BYTES} chars)\n"
                        + snippet[-_DEFAULT_CONTEXT_TAIL_BYTES // 2 :]
                    )
                parts.append(f"[{role}]\n{snippet}")
        parts.append(
            "\nRespond now. Use <bash>...</bash> blocks to act, or emit DONE "
            "if the task is complete."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------

    # Wrapper so `cmd &` doesn't pin our subprocess pipes for the full
    # timeout. Foreground stdout/stderr go to a tmp file; after the
    # foreground command exits we cat the file and exit bash. Any
    # backgrounded child keeps writing into the (now-unlinked) tmp file
    # instead of our pipes, so subprocess.run's read returns immediately.
    #
    # Newline-separated (not ;) so user commands using heredocs (where
    # the terminator must be alone on its line) and trailing `&` (which
    # already terminates the statement) parse cleanly.
    _BG_SAFE_WRAPPER = (
        '__oc_o=$(mktemp)\n'
        '__oc_e=$(mktemp)\n'
        'exec 3>&1 4>&2\n'
        '{\n'
        '%s\n'
        '} >"$__oc_o" 2>"$__oc_e" </dev/null\n'
        '__oc_rc=$?\n'
        'exec 1>&3 2>&4 3>&- 4>&-\n'
        'cat "$__oc_o"\n'
        'cat "$__oc_e" >&2\n'
        'rm -f "$__oc_o" "$__oc_e"\n'
        'exit "$__oc_rc"\n'
    )

    def _exec_in_container(self, target: str, cmd: str) -> Dict:
        """Run `bash -lc <cmd>` inside the TB task container.

        `target` is a singularity instance name (singularity backend) or a
        docker container id/name (docker backend). The backend is set by the
        harness via env_info["container_engine"]; we branch here so the
        condition itself is backend-agnostic.
        """
        wrapped = self._BG_SAFE_WRAPPER % cmd
        if self._container_engine == "singularity":
            argv = [
                "singularity", "exec", f"instance://{target}",
                "bash", "-lc", wrapped,
            ]
        else:
            argv = [
                "docker", "exec", "-i", target,
                "bash", "-lc", wrapped,
            ]
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=self.per_cmd_timeout,
            )
        except subprocess.TimeoutExpired as e:
            # text=True doesn't apply to the partial buffers stashed on the
            # exception — they come back as bytes. Decode defensively.
            def _as_str(x):
                if x is None:
                    return ""
                return x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else x
            return {
                "rc": -1,
                "stdout": _as_str(e.stdout),
                "stderr": _as_str(e.stderr) + f"\n[harness] timeout after {self.per_cmd_timeout}s",
                "timed_out": True,
            }
        return {
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "timed_out": False,
        }

    @staticmethod
    def _format_tool_output(idx: int, cmd: str, result: Dict) -> str:
        # Cap stdout/stderr so a verbose pip install doesn't drown the
        # next prompt. The agent rarely needs full output to decide next step.
        cap = 1500

        def _tail(s: str) -> str:
            if len(s) > cap:
                return s[-cap:]
            return s

        stdout = _tail(result["stdout"])
        stderr = _tail(result["stderr"])
        rc = result["rc"]
        timed = " [TIMED OUT]" if result["timed_out"] else ""
        return (
            f"<tool_output index={idx} rc={rc}{timed}>\n"
            f"$ {cmd[:200]}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}\n"
            f"</tool_output>"
        )

    @staticmethod
    def _make_step(**kwargs) -> Dict:
        """Build a step record matching the SWE-bench step schema (so existing
        analysis code keeps working). Most fields default to None."""
        defaults = {
            "trajectory_id": None,
            "attempt_id": 1,
            "step_id": 0,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": None,
            "tool_name": None,
            "command": None,
            "file_path": None,
            "start_ts": None,
            "end_ts": None,
            "duration_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "exit_code": None,
            "error_type": None,
            "repo_id": None,
            "instance_id": None,
            "config_id": "tb_react",
            "model_name": None,
            "response_preview": None,
        }
        # Friendly shorthand for ts conversion.
        s_ns = kwargs.pop("start_ns", None)
        e_ns = kwargs.pop("end_ns", None)
        err = kwargs.pop("err", None)
        if s_ns is not None:
            defaults["start_ts"] = s_ns
        if e_ns is not None:
            defaults["end_ts"] = e_ns
            if s_ns is not None:
                defaults["duration_ms"] = (e_ns - s_ns) / 1e6
        if err:
            defaults["error_type"] = err
        defaults.update(kwargs)
        return defaults
