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
instances (see eval/runner/driver.py).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.runner.conditions.code_agent._llm_client import call_llm

logger = logging.getLogger(__name__)


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
- When the task is complete and the verifier should pass, emit a final
  message containing the literal token DONE on its own line.
- Do not emit DONE while a required server is not yet running.
- If a command fails, read the stderr and try a different approach.
- One <bash> block per turn is preferred; multiple is allowed.
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
                "(set by TerminalBenchHarness.prepare_environment)"
            )

        instance_id = instance_data["instance_id"]
        instruction = instance_data["instruction"]
        logger.info("TB ReAct: instance=%s task=%s", instance_name, instance_id)

        steps: List[Dict] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        done = False

        # Conversation history rolls forward between turns. We reuse call_llm
        # which is single-shot, so we collapse the running history into one
        # `prompt` per turn. That's lossy on multi-message structure but
        # adequate for a 20-step ReAct loop.
        history: List[Tuple[str, str]] = []  # [(role, content)]

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
                result = self._exec_in_container(instance_name, cmd)
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

                tool_outputs.append(self._format_tool_output(cmd_idx, cmd, result))

            history.append(("user", "\n\n".join(tool_outputs)))

            if _DONE_RE.search(text or ""):
                done = True
                break

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

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "telemetry_path": None,
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

    def _exec_in_container(self, instance_name: str, cmd: str) -> Dict:
        """Run `bash -lc <cmd>` inside the TB singularity instance."""
        argv = [
            "singularity", "exec", f"instance://{instance_name}",
            "bash", "-lc", cmd,
        ]
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=self.per_cmd_timeout,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "rc": -1,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + f"\n[harness] timeout after {self.per_cmd_timeout}s",
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
