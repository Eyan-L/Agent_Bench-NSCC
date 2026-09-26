"""Focused checks for the NSCC Roadmap Hermes candidate and SIF handoff."""
import json
import sqlite3
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from agent_bench.benchmarks.roadmap_bench.conditions.hermes_agent import (
    RoadmapBenchHermesCondition,
)
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.benchmarks.roadmap_bench.hermes.app import RoadmapBenchHermesHarness
from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition
from agent_bench.benchmarks.terminal_bench.app import _seed_workspace_helpers
from agent_bench.deployments.containers.base import ContainerHandle, ContainerSpec
from agent_bench.deployments.pbs.stages import (
    _agent_runtime_defaults,
    _hermes_runtime_commands,
)
from scripts.compare_roadmap_runs_impl import inspect_trajectory
from agent_bench.runtime.driver import _merge_agent_completion_into_verdict


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_roadmap_hermes_freezes_candidate_and_keeps_sessions(tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.name", "Test")
    _git(workspace, "config", "user.email", "test@example.invalid")
    (workspace / "tracked.py").write_text("old\n")
    _git(workspace, "add", "tracked.py")
    _git(workspace, "commit", "-qm", "baseline")
    baseline = RoadmapBenchOpenClawCondition._capture_host_workspace_baseline(workspace)
    assert baseline and baseline["clean"] is True
    (workspace / "tracked.py").write_text("new\n")
    (workspace / "created.py").write_text("created\n")
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.yaml").write_text("api_key: secret\n")
    (state / "exec-bridge.token").write_text("secret")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Harness:
        def exec_in_container(self, *_args, **_kwargs):
            (state / "sessions.jsonl").write_text('{"role":"assistant"}\n')
            return subprocess.CompletedProcess([], 0, "", "")

        def quiesce_orphan_children(self, env):
            env["roadmap_workspace_quiesced"] = True
            return {"workspaceQuiesced": True, "taskRuntimeRestarted": True}

    condition = RoadmapBenchHermesCondition.__new__(RoadmapBenchHermesCondition)
    condition._roadmap_run_dir = run_dir
    condition._roadmap_baseline = baseline
    patch = condition._collect_artifacts(
        {"_harness": Harness(), "state_dir": str(state), "workspace_host_path": str(workspace)},
        "agent", "/app",
    )
    assert "+new" in patch and "created.py" in patch
    assert (run_dir / "candidate_workspace" / "created.py").read_text() == "created\n"
    assert (run_dir / "hermes_state" / "sessions.jsonl").is_file()
    assert not (run_dir / "hermes_state" / "config.yaml").exists()
    assert not (run_dir / "hermes_state" / "exec-bridge.token").exists()
    assert json.loads((run_dir / "candidate_metadata.json").read_text())["baseline"]["git_head"] == baseline["git_head"]


def test_roadmap_hermes_uses_task_runtime_git_without_host_git(tmp_path, monkeypatch):
    workspace = tmp_path / "app"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.name", "Test")
    _git(workspace, "config", "user.email", "test@example.invalid")
    (workspace / "tracked.py").write_text("old\n")
    _git(workspace, "add", "tracked.py")
    _git(workspace, "commit", "-qm", "baseline")

    class Container:
        backend_id = "singularity"

        def exec(self, handle, argv, **kwargs):
            assert handle.name == "task"
            return subprocess.run(
                argv, cwd=kwargs.get("workdir"), capture_output=True, text=True,
                check=False,
            )

    class Harness:
        container = Container()

        def exec_in_container(self, *_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, "", "")

        def quiesce_orphan_children(self, env):
            env["roadmap_workspace_quiesced"] = True
            return {"workspaceQuiesced": True, "taskRuntimeRestarted": True}

    def host_git_unavailable(*_args):
        raise AssertionError("host Git should not be needed")

    monkeypatch.setattr(
        RoadmapBenchOpenClawCondition,
        "_capture_host_workspace_baseline",
        staticmethod(host_git_unavailable),
    )
    monkeypatch.setattr(
        RoadmapBenchOpenClawCondition,
        "_collect_workspace_patch",
        staticmethod(host_git_unavailable),
    )
    state = tmp_path / "state"
    state.mkdir()
    env = {
        "_harness": Harness(),
        "task_container_id": "task",
        "task_container_name": "task",
        "backend": "singularity",
        "workspace_host_path": str(workspace),
        "state_dir": str(state),
    }
    condition = RoadmapBenchHermesCondition.__new__(RoadmapBenchHermesCondition)
    baseline = condition._capture_workspace_baseline(env, str(workspace))
    assert baseline and baseline["git_head"] == _git(workspace, "rev-parse", "HEAD")
    (workspace / "tracked.py").write_text("new\n")
    (workspace / "created.py").write_text("created\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    condition._roadmap_run_dir = run_dir
    condition._roadmap_baseline = baseline
    patch = condition._collect_artifacts(env, "agent", str(workspace))
    assert "+new" in patch and "created.py" in patch
    assert (run_dir / "candidate_workspace" / "created.py").read_text() == "created\n"
    assert json.loads((run_dir / "candidate_metadata.json").read_text())["baseline"]["git_head"] == baseline["git_head"]


def test_roadmap_hermes_quiesces_agent_and_restarts_task():
    class Container:
        backend_id = "singularity"

        def __init__(self):
            self.active = {"agent", "task"}

        def stop(self, handle):
            self.active.discard(handle.name)

        def is_running(self, handle):
            return handle.name in self.active

        def start(self, spec):
            self.active.add(spec.name)
            return ContainerHandle(spec.name, spec.name, self.backend_id)

    harness = RoadmapBenchHermesHarness.__new__(RoadmapBenchHermesHarness)
    harness.container = Container()
    harness._task_specs_lock = threading.Lock()
    harness._task_specs = {"task": ContainerSpec(name="task", image="task.sif")}
    env = {"container_id": "agent", "task_container_id": "task", "task_container_name": "task"}
    result = harness.quiesce_orphan_children(env)
    assert result["workspaceQuiesced"] is True
    assert env["roadmap_workspace_quiesced"] is True
    assert harness.container.active == {"task"}


def test_roadmap_hermes_prepares_singularity_task_and_agent(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AB_REPO_ROOT", str(root))
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(root / "profiles/harnesses/hermes_roadmapbench.yaml"))
    monkeypatch.setenv("AB_MODEL_CONFIG", str(root / "profiles/models/qwen36_35b_a3b.json"))
    monkeypatch.setenv("AB_SERVING_CONFIG", str(root / "profiles/serving/sglang/nscc_tp4.yaml"))
    monkeypatch.setenv("AB_HARNESS_WORKSPACE_BASE", str(tmp_path / "workspaces"))
    runtime = tmp_path / "hermes.sif"
    runtime.touch()
    monkeypatch.setenv("AB_AGENT_RUNTIME_SIF", str(runtime))
    task_images = tmp_path / "task-images"
    task_images.mkdir()
    (task_images / "roadmap__task__latest.sif").touch()
    monkeypatch.setenv("AB_DEPLOYMENT_TASK_IMAGE_DIR", str(task_images))

    class Container:
        backend_id = "singularity"
        host_address = "127.0.0.1"

        def __init__(self):
            self.started = []

        def check_available(self):
            pass

        def check_image(self, _image):
            pass

        def copy_from_image(self, _image, _workspace, app_dir):
            (app_dir / "source.py").write_text("original\n")

        def create_network(self, _network):
            pass

        def remove_network(self, _network):
            pass

        def stop(self, _handle):
            pass

        def start(self, spec):
            self.started.append(spec)
            return ContainerHandle(spec.name, spec.name, self.backend_id)

    class Dispatcher:
        def start(self, **_kwargs):
            pass

        def agent_mounts_and_env(self):
            return [], {"HERMES_EXEC_BRIDGE": "/usr/local/bin/hm-exec-bridge"}

        def env_info(self):
            return {"bridge_pid": None, "bridge_socket": None}

        def stop(self):
            pass

    import agent_bench.benchmarks.roadmap_bench.hermes.app as app
    monkeypatch.setattr(app, "HermesDispatcher", Dispatcher)
    container = Container()
    harness = RoadmapBenchHermesHarness(container=container)
    env = harness.prepare_environment(
        "task-id", "repo", "commit", run_id="run-id",
        instance_data={
            "benchmark": "terminal_bench", "benchmark_variant": "roadmapbench",
            "docker_image": "roadmap/task:latest", "workspace": "/app",
            "instruction": "Implement roadmap",
        },
    )
    assert len(container.started) == 2
    task_spec, agent_spec = container.started
    assert Path(task_spec.image).name == "roadmap__task__latest.sif"
    assert any(m.container == "/tests" for m in task_spec.mounts)
    assert agent_spec.image == str(runtime)
    assert any(m.container == "/opt/data" for m in agent_spec.mounts)
    assert env["task_container_name"] == task_spec.name
    assert env["container_name"] == agent_spec.name
    assert (Path(env["workspace_host_path"]) / "AGENTS.md").is_file()


def test_roadmap_hermes_requires_uploaded_sif(tmp_path):
    defaults = _agent_runtime_defaults(
        "hermes", str(tmp_path), "/scratch/user", roadmapbench=True
    )
    assert defaults["AB_AGENT_RUNTIME_SIF"].endswith("hermes_roadmap_v2026.5.16.sif")
    assert defaults["AB_AGENT_RUNTIME_DOCKER"] == ""
    commands = _hermes_runtime_commands(
        SimpleNamespace(log_dir=tmp_path), defaults, tmp_path, roadmapbench=True
    )
    assert len(commands) == 1
    assert "upload" in commands[0].argv[-1]
    assert "pull" not in commands[0].label


def test_roadmap_hermes_renders_local_model_config(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AB_REPO_ROOT", str(root))
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(root / "profiles/harnesses/hermes_roadmapbench.yaml"))
    monkeypatch.setenv("AB_MODEL_CONFIG", str(root / "profiles/models/qwen36_35b_a3b.json"))
    monkeypatch.setenv("AB_SERVING_CONFIG", str(root / "profiles/serving/sglang/nscc_tp4.yaml"))
    monkeypatch.setenv("AB_SERVING_BASE_URL", "http://127.0.0.1:30000/v1")
    harness = RoadmapBenchHermesHarness.__new__(RoadmapBenchHermesHarness)
    harness.state_base = tmp_path
    harness.container = SimpleNamespace(host_address="127.0.0.1")
    state = harness._build_hermes_state("run-id", "/app", "roadmap/task:latest")
    config = (state / "config.yaml").read_text()
    assert 'default: "Qwen/Qwen3.6-35B-A3B"' in config
    assert 'base_url: "http://127.0.0.1:30000/v1"' in config
    assert 'cwd: "/app"' in config
    assert "max_turns: 600" in config
    assert "max_iterations: 600" in config
    assert "child_timeout_seconds: 7200" in config
    assert "max_concurrent_children: 16" in config
    assert "    - delegation" in config
    assert RoadmapBenchHermesCondition.toolsets.endswith(",delegation")
    assert RoadmapBenchHermesCondition.ignore_rules is False
    workspace = tmp_path / "app"
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(root / "profiles/harnesses/hermes_roadmapbench.yaml"))
    _seed_workspace_helpers(workspace, {"instruction": "build roadmap"}, "/app")
    assert "delegate_task" in (workspace / "AGENTS.md").read_text()
    assert "roadmap" in (workspace / "BOOTSTRAP.md").read_text().lower()
    for name in ("IDENTITY.md", "TOOLS.md"):
        expected = (
            root / "agent_bench/benchmarks/roadmap_bench/openclaw/agents" / name
        ).read_text()
        assert (workspace / name).read_text() == expected


def test_roadmap_hermes_uses_openclaw_task_message():
    instance = {"instruction": "Implement the roadmap"}
    hermes_message = RoadmapBenchHermesCondition._build_message(None, instance)
    openclaw_message = RoadmapBenchOpenClawCondition._build_message(None, instance)
    assert hermes_message == openclaw_message


def test_roadmap_hermes_classifies_final_parent_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RoadmapBenchHermesCondition, "_capture_workspace_baseline",
        lambda *_args: {"git_head": "baseline"},
    )
    monkeypatch.setattr(
        RoadmapBenchHermesCondition, "_write_runtime_observability",
        staticmethod(lambda _run_dir, _sessions: None),
    )
    monkeypatch.setattr(
        RoadmapBenchHermesCondition, "_append_hermes_timeline",
        staticmethod(lambda _run_dir, _trajectory, _sessions: None),
    )
    response = ""
    end_reason = None
    monkeypatch.setattr(
        RoadmapBenchHermesCondition, "_read_session_ledger",
        staticmethod(lambda _run_dir: [{
            "id": "parent", "source": "cli", "parent_session_id": None,
            "end_reason": end_reason, "input_tokens": 0, "output_tokens": 0,
        }]),
    )

    def fake_run(_self, _instance, run_dir, *_args):
        (run_dir / "steps.jsonl").write_text(json.dumps({
            "start_ts": 1_000_000_000, "end_ts": 2_000_000_000,
        }) + "\n")
        if response:
            (run_dir / "hermes_stdout.log").write_text(response)
        return {"exit_code": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0}

    monkeypatch.setattr(HermesAgentCondition, "run", fake_run)
    for index, (
        response, final_text, finish_reason, tool_calls, end_reason,
        expected, expected_strict_done, expected_marker,
    ) in enumerate((
        ("DONE\n", "DONE\n", "stop", None, None, "normal_completion", True, True),
        ("Summary\nDONE", "Summary\nDONE", "stop", None, None, "normal_completion", True, True),
        ("DONE\nexternal warning", "DONE", "stop", None, None, "normal_completion", True, True),
        ("DONE", "Still working", "stop", None, None, "normal_completion", False, False),
        ("DONE", "DONE", "length", None, None, "normal_completion", True, True),
        ("DONE", "DONE", "stop", '[{"id":"call-1"}]', None, "blocked_incomplete", False, True),
        ("DONE", "DONE", "stop", None, "error", "runtime_error", False, False),
        ("DONE", "DONE", "stop", None, "timed_out", "timeout", False, False),
        ("DONE", "DONE", "stop", None, "context_overflow", "context_overflow", False, False),
        ("DONE", "I mentioned DONE earlier", "stop", None, None, "normal_completion", False, False),
        ("DONE", "DONEness", "stop", None, None, "normal_completion", False, False),
        ("Let me continue", "Let me continue", "stop", None, None, "normal_completion", False, False),
        ("", "", "stop", None, None, "blocked_incomplete", False, False),
        ("", "   ", "stop", None, None, "blocked_incomplete", False, False),
        ("unfinished", "unfinished", "length", None, None, "blocked_incomplete", False, False),
        ("unfinished", "unfinished", None, None, None, "blocked_incomplete", False, False),
    )):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        state = run_dir / "hermes_state"
        state.mkdir()
        with sqlite3.connect(state / "state.db") as db:
            db.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
                "role TEXT, content TEXT, tool_calls TEXT, finish_reason TEXT)"
            )
            db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (1, "parent", "assistant", final_text, tool_calls, finish_reason),
            )
        condition = RoadmapBenchHermesCondition.__new__(RoadmapBenchHermesCondition)
        result = condition.run(
            {"instance_id": "task"}, run_dir, 42, "/app", "agent", {}
        )
        verdict = _merge_agent_completion_into_verdict(
            {"status": "FAIL", "reward": 0.25}, result
        )
        assert result["agent_execution_outcome"] == expected
        assert verdict["agent_execution_outcome"] == expected
        assert verdict["reward"] == 0.25
        assert result["agent_lifecycle"]["strictDone"] is expected_strict_done
        assert result["agent_lifecycle"]["done_marker"] is expected_marker


def test_roadmap_hermes_done_uses_final_parent_continuation(tmp_path):
    state = tmp_path / "hermes_state"
    state.mkdir()
    with sqlite3.connect(state / "state.db") as db:
        db.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, tool_calls TEXT, finish_reason TEXT)"
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "parent", "assistant", "DONE", None, "stop"),
                (2, "child", "assistant", "DONE", None, "stop"),
                (3, "continuation", "assistant", "   ", None, "stop"),
            ],
        )
    sessions = [
        {"id": "parent", "source": "cli", "parent_session_id": None,
         "started_at": 1.0, "ended_at": 2.0, "end_reason": "compression"},
        {"id": "child", "source": "delegation", "parent_session_id": "parent",
         "started_at": 1.2, "ended_at": 1.9, "end_reason": "done"},
        {"id": "continuation", "source": "cli", "parent_session_id": "parent",
         "started_at": 2.1, "ended_at": None, "end_reason": None},
    ]
    evidence = RoadmapBenchHermesCondition._read_terminal_assistant_evidence(
        tmp_path, sessions
    )
    assert evidence["session_id"] == "continuation"
    assert evidence["done_marker"] is False
    assert evidence["completion_error"]
    with sqlite3.connect(state / "state.db") as db:
        db.execute(
            "UPDATE messages SET content = 'Still working' WHERE session_id = 'continuation'"
        )
    evidence = RoadmapBenchHermesCondition._read_terminal_assistant_evidence(
        tmp_path, sessions
    )
    assert evidence["completion_error"] is None
    assert evidence["done_marker"] is False
    assert evidence["terminal_evidence"] == "hermes_state_final_assistant_stop"
    with sqlite3.connect(state / "state.db") as db:
        db.execute(
            "UPDATE messages SET content = 'DONE' WHERE session_id = 'continuation'"
        )
    evidence = RoadmapBenchHermesCondition._read_terminal_assistant_evidence(
        tmp_path, sessions
    )
    assert evidence["completion_error"] is None
    assert evidence["done_marker"] is True


def test_roadmap_hermes_events_feed_shared_trajectory_report(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    RoadmapBenchHermesCondition._event(trajectory, "agent_start")
    (tmp_path / "hermes_events.jsonl").write_text(
        json.dumps({
            "type": "model.usage", "start_ns": 100, "end_ns": 200,
            "duration_ms": 0.0001, "session_id": "s1",
            "input_tokens": 10, "output_tokens": 5,
        }) + "\n"
    )
    RoadmapBenchHermesCondition._append_hermes_timeline(tmp_path, trajectory)
    summary = inspect_trajectory(trajectory)
    assert summary["event_types"]["model_request"] == 1
    assert summary["session_keys"] == ["s1"]


def test_roadmap_hermes_session_ledger_tracks_child_lineage(tmp_path):
    state = tmp_path / "hermes_state"
    state.mkdir()
    with sqlite3.connect(state / "state.db") as db:
        db.execute(
            "CREATE TABLE sessions (id TEXT, source TEXT, parent_session_id TEXT, "
            "model TEXT, started_at REAL, ended_at REAL, end_reason TEXT, "
            "input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, "
            "cache_write_tokens INTEGER, reasoning_tokens INTEGER, api_call_count INTEGER, "
            "message_count INTEGER, tool_call_count INTEGER)"
        )
        db.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("parent", "cli", None, "qwen", 1.0, 4.0, "compression", 100, 30, 0, 0, 0, 2, 5, 1),
                ("child", "cli", "parent", "qwen", 2.0, 3.0, "compression", 40, 15, 0, 0, 0, 1, 3, 1),
                ("child-tip", "cli", "child", "qwen", 3.1, 3.5, "done", 5, 2, 0, 0, 0, 1, 2, 0),
                ("continuation", "cli", "parent", "qwen", 4.1, 5.0, "done", 20, 5, 0, 0, 0, 1, 2, 0),
            ],
        )
    sessions = RoadmapBenchHermesCondition._read_session_ledger(tmp_path)
    assert sum(item["input_tokens"] for item in sessions) == 165
    assert sessions[1]["parent_session_id"] == "parent"
    assert (tmp_path / "hermes_sessions.json").is_file()
    obs_path = RoadmapBenchHermesCondition._write_runtime_observability(tmp_path, sessions)
    observation = json.loads(obs_path.read_text())
    assert observation["materialized_child_session_count"] == 1
    assert observation["max_concurrent_children"] == 1
    assert set(observation["child_runtime_by_session"]) == {"child"}
    assert observation["child_runtime_by_session"]["child"]["continuation_session_ids"] == ["child-tip"]
    assert observation["child_runtime_by_session"]["child"]["runtime_seconds"] == 1.5
    trajectory = tmp_path / "trajectory.jsonl"
    RoadmapBenchHermesCondition._event(trajectory, "agent_start")
    RoadmapBenchHermesCondition._append_hermes_timeline(tmp_path, trajectory, sessions)
    summary = inspect_trajectory(trajectory)
    assert summary["event_types"]["session_start"] == 4
    assert summary["event_types"]["session_end"] == 4
    assert summary["session_keys"] == ["child", "child-tip", "continuation", "parent"]
