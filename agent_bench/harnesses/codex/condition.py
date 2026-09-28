"""Generic Codex CLI invocation; benchmark adapters supply prompts and freeze."""
from __future__ import annotations
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from agent_bench.benchmarks.conditions.util_sampler import UtilSampler
from agent_bench.deployments.containers import ContainerHandle
from agent_bench.harnesses.codex.runtime import command
from agent_bench.harnesses.codex.telemetry import api_response, jsonl, summarize


def archive_state(source: Path, destination: Path):
    """Retain durable threads and ledgers, excluding CLI caches and helpers."""
    source = Path(source)

    def ignored(directory, names):
        if Path(directory) == source:
            return [name for name in names if not (
                name in {"sessions", "archived_sessions", "prompt.txt", "final.txt", "installation_id"}
                or ".sqlite" in name
            )]
        return [name for name in names if name in {"auth.json", "config.toml"} or name.endswith(".sock")]

    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True, ignore=ignored)


class CodexAgentCondition:
    def __init__(self, model_config, agent_config, container):
        self.model_config, self.agent_config, self.container = model_config, agent_config, container

    def _build_message(self, instance_data):
        return instance_data["instruction"]

    def _finalize_runtime(self, env_info, run_dir, workspace_root):
        harness = env_info.get("_harness")
        if harness is None or not hasattr(harness, "quiesce_agent_writers"):
            raise RuntimeError("Codex writer fencing requires its harness")
        return harness.quiesce_agent_writers(env_info), ""

    def run(self, instance_data, run_dir, seed, workspace_root, container_id, env_info=None):
        env_info = env_info if env_info is not None else {}
        run_dir.mkdir(parents=True, exist_ok=True)
        task = ContainerHandle(container_id, env_info.get("task_container_name", container_id), self.container.backend_id)
        Path(env_info["codex_state_dir"]).joinpath("prompt.txt").write_text(self._build_message(instance_data))
        start, timed_out, errors = time.time_ns(), False, []
        with UtilSampler(run_dir, client_container=task.name, tools_container=task.name):
            proc = self.container.popen_exec(task, command(workspace_root), workdir=workspace_root, env=env_info["agent_env"])
            def capture(pipe, filename, structured=False):
                try:
                    with (run_dir / filename).open("w") as raw:
                        events = (run_dir / "codex_events.jsonl").open("w") if structured else None
                        try:
                            for line in iter(pipe.readline, ""):
                                raw.write(line)
                                raw.flush()
                                if events:
                                    try:
                                        event = json.loads(line)
                                    except ValueError:
                                        continue
                                    events.write(json.dumps({"timestamp_ns": time.time_ns(), "event": event}) + "\n")
                                    events.flush()
                        finally:
                            if events:
                                events.close()
                except Exception as exc:
                    errors.append(str(exc))
            readers = [threading.Thread(target=capture, args=(proc.stdout, "codex_stdout.jsonl", True), daemon=True),
                       threading.Thread(target=capture, args=(proc.stderr, "codex_stderr.log"), daemon=True)]
            for reader in readers:
                reader.start()
            try:
                proc.wait(timeout=self.agent_config.max_time_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        parent_end = time.time_ns()
        try:
            lifecycle, patch = self._finalize_runtime(env_info, run_dir, workspace_root)
        except Exception as exc:
            self._partial_execution_result = {"agent_runtime_mode": "codex_cli", "agent_execution_outcome": "runtime_error",
                "agent_completion_error": str(exc), "failure_phase": "candidate_freeze", "agent_lifecycle": {"workspaceQuiesced": False}}
            raise
        end = lifecycle["workspaceQuiescedNs"]
        for reader in readers:
            reader.join(timeout=10)
        if any(reader.is_alive() for reader in readers):
            errors.append("Codex output stream remained open after fencing")
        state = Path(env_info["state_dir"])
        archive_state(Path(env_info["codex_state_dir"]), run_dir / "codex_state")
        if (state / "codex_api.jsonl").is_file():
            shutil.copyfile(state / "codex_api.jsonl", run_dir / "codex_api.jsonl")
        adapter_id = env_info.get("adapter_container_id")
        if adapter_id:
            log_path = Path(env_info["adapter_log_path"]) if env_info.get("adapter_log_path") else None
            adapter_handle = ContainerHandle(adapter_id, adapter_id, self.container.backend_id, log_path=log_path)
            (run_dir / "codex_adapter.log").write_text(self.container.logs(adapter_handle))
        sessions, obs = summarize(run_dir)
        events = list(jsonl(run_dir / "codex_stdout.jsonl"))
        completed = any(r.get("type") == "turn.completed" for r in events)
        roots = [s for s in sessions if not s["parent_thread_id"]]
        final = roots[0]["final_assistant"] if len(roots) == 1 else None
        root_complete = len(roots) == 1 and roots[0].get("terminal_event") == "task_complete"
        api_rows = [r for r in jsonl(run_dir / "codex_api.jsonl")
                    if "end_ns" in r and roots and r.get("thread_id") == roots[0]["id"]]
        last_response = api_response(api_rows[-1]) if api_rows else {}
        truncated = last_response.get("status") == "incomplete" and str(final or "").strip() != "DONE"
        if timed_out:
            outcome = "timeout"
        elif proc.returncode != 0 or errors:
            outcome = "runtime_error"
        elif completed and root_complete and final and not truncated and obs["child_runtime_unknown_count"] == 0:
            outcome = "normal_completion"
        else:
            outcome = "blocked_incomplete"
        lifecycle.update(strictDone=bool(final and final.strip() == "DONE"), done_marker=bool(final and final.strip() == "DONE"),
                         terminalEvidence="codex_native_turn_completed" if completed else "missing_turn_completed",
                         runtimeVersion=env_info.get("runtime_version"), workspaceQuiescedNs=end, parentTerminalObservedNs=parent_end)
        prompt = sum(int(s["usage"].get("input_tokens") or 0) for s in sessions)
        output = sum(int(s["usage"].get("output_tokens") or 0) for s in sessions)
        tokens = run_dir / "token_accounting.json"
        tokens.write_text(json.dumps({"source": "codex_native_per_thread_usage", "prompt_tokens": prompt, "completion_tokens": output, "session_count": len(sessions)}, indent=2))
        patch_path, steps = run_dir / "candidate.patch", run_dir / "steps.jsonl"
        patch_path.write_text(patch)
        steps.write_text(json.dumps({"step": 0, "tool_name": "codex.exec", "start_ts": start, "end_ts": parent_end, "exit_code": proc.returncode, "duration_ms": (parent_end-start)/1e6}) + "\n")
        phase = {"agent_execution_start_ns": start, "parent_terminal_observed_ns": parent_end, "agent_execution_end_ns": end,
                 "workspace_quiesced_ns": end, "agent_execution_seconds": (end-start)/1e9, "parent_execution_seconds": (parent_end-start)/1e9}
        return {"harness": "codex", "exit_code": -1 if timed_out else proc.returncode, "patch_path": str(patch_path), "step_trace_path": str(steps),
                "telemetry_path": str(run_dir / "spans.jsonl"), "trajectory_path": str(run_dir / "trajectory.jsonl"),
                "runtime_observability_path": str(run_dir / "runtime_observability.json"), "runtime_observability": obs,
                "agent_runtime_mode": "codex_cli", "agent_execution_outcome": outcome, "agent_completion_status": outcome,
                "agent_completion_error": None if outcome == "normal_completion" else f"Codex {outcome}; exit={proc.returncode}",
                "agent_lifecycle": lifecycle, "agent_execution_seconds": phase["agent_execution_seconds"], "phase_timing": phase,
                "total_prompt_tokens": prompt, "total_completion_tokens": output, "token_accounting_path": str(tokens)}
