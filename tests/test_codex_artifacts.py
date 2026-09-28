"""Frozen links and independent evidence survive artifact collection failures."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from agent_bench.benchmarks.roadmap_bench.conditions.codex_agent import RoadmapBenchCodexCondition
from agent_bench.runtime.driver import _preserve_partial_agent_result


@pytest.fixture
def roadmapped_run(tmp_path):
    state, workspace, run = tmp_path / "state", tmp_path / "workspace", tmp_path / "run"
    source = state / "codex"
    sessions = source / "sessions/2026/09/28"
    sessions.mkdir(parents=True)
    workspace.mkdir()
    (workspace / "target").write_text("implementation")
    (workspace / "valid-link").symlink_to("target")
    (workspace / "dangling-link").symlink_to("missing")
    (workspace / "container-link").symlink_to("/app/target")
    (workspace / "cycle").symlink_to(".", target_is_directory=True)
    (workspace / ".git").mkdir()
    base = datetime.datetime(2026, 9, 28, tzinfo=datetime.timezone.utc)
    final = {"type": "message", "id": "answer", "role": "assistant", "phase": "final_answer",
             "content": [{"type": "output_text", "text": "DONE"}],
             "internal_chat_message_metadata_passthrough": {"turn_id": "final-turn"}}
    rows = [{"timestamp": base.isoformat(), "type": "session_meta", "payload": {"id": "parent", "source": "exec"}}]
    for seconds, kind, payload in [(0, "event_msg", {"type": "task_started", "turn_id": "final-turn"}),
                                  (1, "response_item", final), (2, "event_msg", {"type": "task_complete", "turn_id": "final-turn"})]:
        rows.append({"timestamp": (base + datetime.timedelta(seconds=seconds)).isoformat(), "type": kind, "payload": payload})
    (sessions / "rollout-parent.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (state / "codex_api.jsonl").write_text(json.dumps({"start_ns": 1, "end_ns": 2, "thread_id": "parent", "turn_id": "final-turn",
        "response": {"status": "completed", "output": [final]}}) + "\n")

    class Container:
        backend_id = "singularity"

        def popen_exec(self, *_args, **_kwargs):
            events = [{"type": "thread.started", "thread_id": "parent"}, {"type": "turn.started"}, {"type": "turn.completed"}]
            script = "print(" + repr("\n".join(json.dumps(e) for e in events)) + ")"
            return subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def logs(self, _handle):
            return "adapter log evidence"

    class Harness:
        def quiesce_agent_writers(self, env):
            env["roadmap_workspace_quiesced"] = True
            return {"workspaceQuiesced": True, "workspaceQuiescedNs": time.time_ns()}

    condition = RoadmapBenchCodexCondition(None, SimpleNamespace(max_time_seconds=10), Container())
    condition._capture_task_workspace_baseline = lambda *_args: {"git_head": "baseline"}
    condition._collect_container_workspace_patch = lambda *_args, **_kwargs: "candidate patch evidence\n"
    env = {"workspace_host_path": str(workspace), "state_dir": str(state), "codex_state_dir": str(source),
           "task_container_id": "task", "task_container_name": "task", "adapter_container_id": "adapter",
           "agent_env": {}, "_harness": Harness()}
    return condition, env, run, workspace


def execute(case):
    condition, env, run, _workspace = case
    return condition.run({"instruction": "implement"}, run, 42, "/app", "task", env)


def test_frozen_candidate_preserves_links_without_dereferencing(roadmapped_run):
    result = execute(roadmapped_run)
    _condition, _env, run, workspace = roadmapped_run
    candidate = run / "candidate_workspace"
    for name in ["valid-link", "dangling-link", "container-link", "cycle"]:
        assert (candidate / name).is_symlink()
        assert (candidate / name).readlink() == (workspace / name).readlink()
    assert (candidate / "target").read_text() == "implementation"
    assert not (candidate / ".git").exists()
    assert json.loads((run / "candidate_metadata.json").read_text())["candidate_snapshot_complete"] is True
    assert result["agent_execution_outcome"] == "normal_completion"


@pytest.mark.parametrize("failed_destination", ["candidate_workspace", "codex_state"])
def test_copy_failure_still_saves_other_evidence(roadmapped_run, monkeypatch, failed_destination):
    condition, _env, run, _workspace = roadmapped_run
    original = shutil.copytree

    def failing_copy(source, destination, *args, **kwargs):
        if Path(destination).name == failed_destination:
            raise OSError("injected copy failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", failing_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        execute(roadmapped_run)
    assert (run / "codex_api.jsonl").is_file()
    assert (run / "codex_stdout.jsonl").is_file()
    assert (run / "codex_adapter.log").read_text() == "adapter log evidence"
    assert (run / "candidate.patch").read_text() == "candidate patch evidence\n"
    phase = json.loads((run / "phase_timing.json").read_text())
    assert phase["workspace_quiesced_ns"] == condition._partial_execution_result["agent_lifecycle"]["workspaceQuiescedNs"]
    assert (run / "steps.jsonl").is_file() and (run / "token_accounting.json").is_file()
    assert json.loads((run / "artifact_errors.json").read_text())[0]["stage"] == (
        "candidate_freeze" if failed_destination == "candidate_workspace" else "session_archive")
    if failed_destination == "candidate_workspace":
        assert list((run / "codex_state").rglob("rollout-*.jsonl"))
        assert (run / "trajectory.jsonl").is_file() and (run / "spans.jsonl").is_file()
        assert condition._partial_execution_result["agent_execution_outcome"] == "normal_completion"
        assert json.loads((run / "candidate_metadata.json").read_text())["candidate_snapshot_complete"] is False
    else:
        assert (run / "candidate_workspace/target").is_file()
    manifest = SimpleNamespace()
    _preserve_partial_agent_result(manifest, condition._partial_execution_result,
        failure_phase=condition._partial_execution_result["failure_phase"], failure_message="copy failed")
    assert manifest.benchmark_valid is False and manifest.verifier_skipped is True and manifest.reward is None
    assert manifest.phase_timing["agent_execution_seconds"] == phase["agent_execution_seconds"]


def test_fence_failure_preserves_evidence_and_refuses_candidate(roadmapped_run):
    condition, env, run, _workspace = roadmapped_run

    def fail(_env):
        raise RuntimeError("writers still active")

    env["_harness"].quiesce_agent_writers = fail
    with pytest.raises(RuntimeError, match="writers still active"):
        execute(roadmapped_run)
    partial = condition._partial_execution_result
    assert partial["agent_execution_outcome"] == "runtime_error"
    assert partial["agent_lifecycle"]["workspaceQuiesced"] is False
    assert partial["phase_timing"]["workspace_quiesced_ns"] is None
    assert (run / "codex_api.jsonl").is_file()
    assert list((run / "codex_state").rglob("rollout-*.jsonl"))
    assert (run / "phase_timing.json").is_file()
    assert not (run / "candidate_workspace").exists() and not (run / "candidate.patch").exists()


def test_launch_failure_preserves_independent_evidence_without_normal_completion(roadmapped_run, monkeypatch):
    condition, _env, run, _workspace = roadmapped_run
    def launch(*_args, **_kwargs):
        raise OSError("injected launch failure")

    monkeypatch.setattr(condition.container, "popen_exec", launch)
    result = execute(roadmapped_run)
    assert result["agent_execution_outcome"] == "runtime_error"
    assert "injected" in result["agent_completion_error"]
    assert result["agent_lifecycle"]["cliInvocationCount"] == 0
    for name in ["phase_timing.json", "codex_api.jsonl", "codex_sessions.json", "candidate.patch", "steps.jsonl"]:
        assert (run / name).is_file()
