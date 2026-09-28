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
from agent_bench.harnesses.codex.telemetry import jsonl, summarize
from agent_bench.harnesses.codex.completion import parent_completion


def archive_state(source: Path, destination: Path):
    """Retain durable threads and ledgers, excluding CLI caches and helpers."""
    source = Path(source)

    def ignored(directory, names):
        if Path(directory) == source:
            return [name for name in names if not (
                name in {"sessions", "archived_sessions", "prompt.txt", "continuation.txt", "final.txt", "installation_id"}
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

    def _freeze_candidate(self, env_info, run_dir, workspace_root, lifecycle, patch):
        (run_dir / "candidate.patch").write_text(patch)

    def run(self, instance_data, run_dir, seed, workspace_root, container_id, env_info=None):
        self._partial_execution_result = None
        env_info = env_info if env_info is not None else {}
        run_dir.mkdir(parents=True, exist_ok=True)
        task = ContainerHandle(container_id, env_info.get("task_container_name", container_id), self.container.backend_id)
        Path(env_info["codex_state_dir"]).joinpath("prompt.txt").write_text(self._build_message(instance_data))
        start, timed_out, errors = time.time_ns(), False, []
        deadline = time.monotonic() + self.agent_config.max_time_seconds
        invocations, readers, proc = [], [], None
        with UtilSampler(run_dir, client_container=task.name, tools_container=task.name):
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
                                    events.write(json.dumps({"timestamp_ns": time.time_ns(), "invocation": 0, "event": event}) + "\n")
                                    events.flush()
                        finally:
                            if events:
                                events.close()
                except Exception as exc:
                    errors.append(str(exc))
            invocation_start = time.time_ns()
            try:
                proc = self.container.popen_exec(task, command(workspace_root), workdir=workspace_root, env=env_info["agent_env"])
            except Exception as exc:
                errors.append(f"Codex invocation launch: {exc}")
            if proc is not None:
                readers = [threading.Thread(target=capture, args=(proc.stdout, "codex_stdout.jsonl", True), daemon=True),
                           threading.Thread(target=capture, args=(proc.stderr, "codex_stderr.log"), daemon=True)]
                for reader in readers:
                    reader.start()
                try:
                    proc.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                invocation_end = time.time_ns()
                invocations.append({"step": 0, "tool_name": "codex.exec",
                    "start_ts": invocation_start, "end_ts": invocation_end, "exit_code": proc.returncode,
                    "duration_ms": (invocation_end - invocation_start) / 1e6})
                for reader in readers:
                    reader.join(timeout=10)
                if any(reader.is_alive() for reader in readers):
                    errors.append("Codex output stream remained open after its invocation")
        parent_end = time.time_ns()
        exit_code = proc.returncode if proc is not None else 1
        artifact_errors = []
        lifecycle, patch = {"workspaceQuiesced": False}, ""
        try:
            lifecycle, patch = self._finalize_runtime(env_info, run_dir, workspace_root)
        except Exception as exc:
            lifecycle["lifecycleError"] = str(exc)
            artifact_errors.append(("writer_fencing", exc))
        end = lifecycle.get("workspaceQuiescedNs") or time.time_ns()
        for reader in readers:
            reader.join(timeout=10)
        if any(reader.is_alive() for reader in readers):
            errors.append("Codex output stream remained open after fencing")
        patch_path, steps = run_dir / "candidate.patch", run_dir / "steps.jsonl"
        phase = {"agent_execution_start_ns": start, "parent_terminal_observed_ns": parent_end, "agent_execution_end_ns": end,
                 "workspace_quiesced_ns": lifecycle.get("workspaceQuiescedNs"), "agent_execution_seconds": (end-start)/1e9,
                 "parent_execution_seconds": (parent_end-start)/1e9}
        # Save independent execution evidence before any workspace tree copy.
        # Each collector runs even if another one fails.
        def preserve(stage, action):
            try:
                return action()
            except Exception as exc:
                artifact_errors.append((stage, exc))
                return None

        preserve("phase_timing", lambda: (run_dir / "phase_timing.json").write_text(json.dumps(phase, indent=2)))
        preserve("step_trace", lambda: steps.write_text("".join(json.dumps(row) + "\n" for row in invocations)))
        state = Path(env_info["state_dir"])
        if (state / "codex_api.jsonl").is_file():
            preserve("api_archive", lambda: shutil.copyfile(state / "codex_api.jsonl", run_dir / "codex_api.jsonl"))
        preserve("session_archive", lambda: archive_state(Path(env_info["codex_state_dir"]), run_dir / "codex_state"))
        adapter_id = env_info.get("adapter_container_id")
        if adapter_id:
            log_path = Path(env_info["adapter_log_path"]) if env_info.get("adapter_log_path") else None
            adapter_handle = ContainerHandle(adapter_id, adapter_id, self.container.backend_id, log_path=log_path)
            preserve("adapter_log", lambda: (run_dir / "codex_adapter.log").write_text(self.container.logs(adapter_handle)))
        summary = preserve("native_telemetry", lambda: summarize(run_dir))
        sessions, obs = summary if summary is not None else ([], {})
        events = list(jsonl(run_dir / "codex_stdout.jsonl"))
        evidence = parent_completion(sessions, events, jsonl(run_dir / "codex_api.jsonl"))
        unfinished = obs.get("unfinished_child_session_ids") or []
        if timed_out:
            outcome = "timeout"
        elif exit_code != 0 or errors or not lifecycle.get("workspaceQuiesced"):
            outcome = "runtime_error"
        elif evidence["complete"] and not unfinished:
            outcome = "normal_completion"
        else:
            outcome = "blocked_incomplete"
        completion_error = None if outcome == "normal_completion" else (
            "; ".join(errors) or lifecycle.get("lifecycleError") or evidence.get("completion_error")
            or (f"Codex unfinished child work: {', '.join(unfinished)}" if unfinished else f"Codex {outcome}; exit={exit_code}"))
        lifecycle.update(strictDone=bool(evidence["complete"] and evidence["done_marker"]), done_marker=evidence["done_marker"],
                         terminalEvidence=evidence["terminal_evidence"], parentCompletionEvidence=evidence,
                         totalChildren=obs.get("child_session_count", 0), completedChildren=obs.get("child_execution_complete_count", 0),
                         unfinishedChildren=unfinished, childStatesAtFence=obs.get("child_lifecycle_by_session", {}),
                         childTimingUnknownCount=obs.get("child_runtime_unknown_count"),
                         continuationCount=0, cliInvocationCount=len(invocations),
                         runtimeVersion=env_info.get("runtime_version"), parentTerminalObservedNs=parent_end)
        prompt = sum(int(s["usage"].get("input_tokens") or 0) for s in sessions)
        output = sum(int(s["usage"].get("output_tokens") or 0) for s in sessions)
        tokens = run_dir / "token_accounting.json"
        preserve("token_accounting", lambda: tokens.write_text(json.dumps({"source": "codex_native_per_thread_usage",
            "prompt_tokens": prompt, "completion_tokens": output, "session_count": len(sessions)}, indent=2)))
        if lifecycle.get("workspaceQuiesced"):
            phase["candidate_freeze_start_ns"] = time.time_ns()
            before_freeze_errors = len(artifact_errors)
            preserve("candidate_freeze", lambda: self._freeze_candidate(env_info, run_dir, workspace_root, lifecycle, patch))
            freeze_end = time.time_ns()
            phase["candidate_freeze_error_ns" if len(artifact_errors) > before_freeze_errors else "candidate_freeze_end_ns"] = freeze_end
            phase["candidate_freeze_seconds"] = (freeze_end - phase["candidate_freeze_start_ns"]) / 1e9
            preserve("phase_timing", lambda: (run_dir / "phase_timing.json").write_text(json.dumps(phase, indent=2)))
        result = {"harness": "codex", "exit_code": -1 if timed_out else exit_code, "patch_path": str(patch_path), "step_trace_path": str(steps),
                "telemetry_path": str(run_dir / "spans.jsonl"), "trajectory_path": str(run_dir / "trajectory.jsonl"),
                "runtime_observability_path": str(run_dir / "runtime_observability.json"), "runtime_observability": obs,
                "agent_runtime_mode": "codex_cli", "agent_execution_outcome": outcome, "agent_completion_status": outcome,
                "agent_completion_error": completion_error,
                "agent_lifecycle": lifecycle, "agent_execution_seconds": phase["agent_execution_seconds"], "phase_timing": phase,
                "total_prompt_tokens": prompt, "total_completion_tokens": output, "token_accounting_path": str(tokens)}
        self._partial_execution_result = result
        if artifact_errors:
            result["failure_phase"] = artifact_errors[0][0]
            diagnostics = [{"stage": stage, "error": str(exc)} for stage, exc in artifact_errors]
            preserve("artifact_diagnostics", lambda: (run_dir / "artifact_errors.json").write_text(json.dumps(diagnostics, indent=2)))
            raise RuntimeError("; ".join(f"{d['stage']}: {d['error']}" for d in diagnostics)) from artifact_errors[0][1]
        return result
