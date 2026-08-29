from __future__ import annotations

import subprocess
import json
from pathlib import Path

from agent_bench.benchmarks.terminal_bench import app as tb_app
from agent_bench.benchmarks.terminal_bench import loader as tb_loader
from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.benchmarks.terminal_bench.conditions.openclaw_agent import (
    TerminalBenchOpenClawCondition,
)
from agent_bench.deployments.containers.base import ContainerHandle
from agent_bench.deployments.containers.singularity import SingularityContainer
from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition
from agent_bench.runtime.driver import _merge_agent_completion_into_verdict


class _FakeContainer:
    backend_id = "singularity"
    host_address = "127.0.0.1"

    def __init__(self) -> None:
        self.specs = []
        self.stop_calls = []
        self.created_networks = []
        self.removed_networks = []
        self.exec_calls = []

    def check_available(self) -> None:
        pass

    def check_image(self, image: str) -> None:
        assert Path(image).is_file()

    def create_network(self, name: str) -> None:
        self.created_networks.append(name)

    def remove_network(self, name: str) -> None:
        self.removed_networks.append(name)

    def copy_from_image(self, image: str, src: str, dst: Path, **_kw) -> None:
        (dst / "starter.txt").write_text("starter")

    def start(self, spec):
        self.specs.append(spec)
        return ContainerHandle(
            id=f"{spec.name}-handle", name=spec.name, backend=self.backend_id
        )

    def wait_log_pattern(self, *_args, **_kw) -> None:
        pass

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    def stop(self, handle, **_kw) -> None:
        self.stop_calls.append(handle)


class _FakeDispatcher:
    pid = 42
    last_start_kwargs = None

    def start(self, **kwargs) -> None:
        type(self).last_start_kwargs = kwargs

    def stop(self) -> None:
        pass

    @staticmethod
    def stop_external(**_kw) -> None:
        pass

    def agent_mounts_and_env(self):
        return [], {"OPENCLAW_EXEC_BRIDGE_SOCKET": "/tmp/bridge.sock"}

    def env_info(self):
        return {"bridge_socket": "/tmp/bridge.sock", "bridge_pid": self.pid}


def test_terminal_bench_openclaw_singularity_returns_runtime_handle(tmp_path, monkeypatch):
    _FakeDispatcher.last_start_kwargs = None
    runtime_sif = tmp_path / "openclaw.sif"
    runtime_sif.touch()
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    docker_image = "alexgshaw/multi-source-data-merger:20251031"
    task_sif = task_dir / "alexgshaw__multi-source-data-merger__20251031.sif"
    task_sif.touch()
    monkeypatch.setenv("AB_AGENT_RUNTIME_SIF", str(runtime_sif))
    monkeypatch.setenv("AB_DEPLOYMENT_TASK_IMAGE_DIR", str(task_dir))
    monkeypatch.setattr(tb_app, "OpenClawRemoteDispatcher", _FakeDispatcher)

    container = _FakeContainer()
    harness = tb_app.TerminalBenchOpenClawHarness(
        container=container, workspace_base=tmp_path / "workspaces"
    )

    def build_state(**_kw):
        state = harness.state_base / "run-1"
        state.mkdir(parents=True)
        (state / "openclaw.json").write_text("{}")
        return state

    monkeypatch.setattr(harness, "_build_state_dir", build_state)
    env_info = harness.prepare_environment(
        "multi-source-data-merger", "unused", "unused", run_id="run-1",
        instance_data={
            "benchmark": "terminal_bench", "docker_image": docker_image,
            "workspace": "/app", "instruction": "do the task",
        },
    )

    assert env_info["backend"] == "singularity"
    assert env_info["container_id"] == "oc-run-1-handle"
    assert env_info["oc_container_id"] == "oc-run-1-handle"
    assert env_info["task_container_id"] != env_info["container_id"]
    assert container.specs[0].image == str(task_sif)
    assert container.specs[1].image == str(runtime_sif)
    assert container.specs[1].env["HOME"] == "/root"
    assert container.specs[1].env["OPENCLAW_BYPASS_ENSURE_SANDBOX"] == "1"
    assert container.specs[1].env["OPENCLAW_DISABLE_BONJOUR"] == "1"

    # NSCC runtime hardening.
    assert tb_app.TerminalBenchOpenClawHarness._GATEWAY_READY_TIMEOUT_SECONDS == 600
    assert container.specs[1].env["OPENCLAW_PLUGIN_STAGE_DIR"] == (
        "/tmp/oc-plugin-deps/run-1"
    )
    assert container.specs[1].env["NPM_CONFIG_FETCH_RETRIES"] == "1"
    assert container.specs[1].env["NPM_CONFIG_FETCH_TIMEOUT"] == "15000"
    assert container.specs[1].env["NODE_OPTIONS"] == "--dns-result-order=ipv4first"

    # Task/runtime separation and solve-time verifier isolation.
    # Singularity needs an existing writable bind target because the SIF root
    # filesystem is immutable. The mounted /tests directory must still be
    # empty while the agent is solving; verifier files are injected only
    # afterward by run_tests().
    test_mounts = [
        m for m in container.specs[0].mounts
        if m.container == "/tests"
    ]
    assert len(test_mounts) == 1
    assert Path(test_mounts[0].host).is_dir()
    assert list(Path(test_mounts[0].host).iterdir()) == []

    assert container.specs[0].labels["openclaw.sessionKey"].startswith("agent:")

    # Exec bridge must target the Terminal-Bench task instance.
    assert _FakeDispatcher.last_start_kwargs is not None
    assert _FakeDispatcher.last_start_kwargs["backend"] == "singularity"
    assert (
        _FakeDispatcher.last_start_kwargs["target"]
        == env_info["task_container_name"]
    )

    # Retry preflight must attempt cleanup of deterministic stale instances.
    stopped_names = [h.name for h in container.stop_calls]
    assert "oc-run-1" in stopped_names
    assert env_info["task_container_name"] in stopped_names
    assert "oc-eval-net-run-1" in container.removed_networks
    assert "oc-eval-net-run-1" in container.created_networks

    # Gateway and CLI use separate node-local plugin staging directories.
    gateway_env_writes = [
        args[1]
        for args, _kwargs in container.exec_calls
        if len(args) >= 2 and isinstance(args[1], list)
    ]
    assert any(
        "/tmp/oc-plugin-deps/run-1-cli" in str(argv)
        for argv in gateway_env_writes
    )


def test_driver_routes_terminal_bench_openclaw_to_generic_harness():
    driver = (Path(__file__).resolve().parents[1] / "agent_bench/runtime/driver.py").read_text()
    assert 'if harness_choice == "openclaw":' in driver
    assert "TerminalBenchOpenClawHarness as tb_cls" in driver
    assert "tb_harness = tb_cls(benchmark_data_root, container=tb_container)" in driver
    assert 'instance_data.get("benchmark_variant") == "roadmapbench"' in driver
    assert "RoadmapBenchOpenClawCondition as _Cls" in driver


def test_terminal_bench_prompt_variant_is_selected_explicitly(tmp_path, monkeypatch):
    profile = tmp_path / "harness.yaml"
    profile.write_text(
        "terminal_bench:\n"
        "  agent_prompt_file: AGENTS.md.dr_matched\n"
        "  workspace_bootstrap_file: BOOTSTRAP.md.dr_matched\n"
    )
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(profile))

    assert tb_app._terminal_bench_prompt_filename(
        "agent_prompt_file", "AGENTS.md"
    ) == "AGENTS.md.dr_matched"
    assert tb_app._terminal_bench_prompt_filename(
        "workspace_bootstrap_file", "BOOTSTRAP.md"
    ) == "BOOTSTRAP.md.dr_matched"


def test_roadmapbench_uses_own_assets_and_condition(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    profile = tmp_path / "harness.yaml"
    profile.write_text(
        "terminal_bench:\n"
        "  agent_assets_dir: "
        "agent_bench/benchmarks/roadmap_bench/openclaw/agents\n"
        "  skills_dir: "
        "agent_bench/benchmarks/roadmap_bench/openclaw/skills\n"
    )
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(profile))
    monkeypatch.setenv("AB_REPO_ROOT", str(repo_root))

    assets = tb_app._terminal_bench_agent_assets_dir(repo_root=repo_root)
    skills = tb_app._terminal_bench_profile_dir(
        "skills_dir", repo_root=repo_root
    )
    assert assets.name == "agents"
    assert (assets / "AGENTS.md").is_file()
    assert "lead engineer" in (assets / "AGENTS.md").read_text()
    assert "Terminal-Bench" not in (assets / "AGENTS.md").read_text()
    assert skills is not None
    assert (skills / "verify" / "SKILL.md").is_file()
    assert "SWE-bench" not in (skills / "verify" / "SKILL.md").read_text()

    condition = object.__new__(RoadmapBenchOpenClawCondition)
    message = condition._build_message({"instruction": "Implement targets."})
    assert condition.agent_id == "roadmap_bench"
    assert "multi-target software roadmap" in message
    assert "Carry out this terminal task" not in message


def test_roadmapbench_materializes_all_prompt_assets_in_resolved_workspace(
    tmp_path, monkeypatch
):
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("AGENTS.md", "IDENTITY.md", "TOOLS.md", "BOOTSTRAP.md"):
        (assets / name).write_text(f"{name}: work in /app\n")
    profile = tmp_path / "harness.yaml"
    profile.write_text(
        "terminal_bench:\n"
        f"  agent_assets_dir: {assets}\n"
        "  workspace_asset_files:\n"
        "  - AGENTS.md\n"
        "  - IDENTITY.md\n"
        "  - TOOLS.md\n"
        "  - BOOTSTRAP.md\n"
    )
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(profile))
    workspace_bind = tmp_path / "workspace-bind"

    tb_app._seed_workspace_helpers(
        workspace_bind, {}, task_workspace="/workspace/project"
    )

    assert {p.name for p in workspace_bind.iterdir()} == {
        "AGENTS.md", "IDENTITY.md", "TOOLS.md", "BOOTSTRAP.md"
    }
    for path in workspace_bind.iterdir():
        assert "/workspace/project" in path.read_text()
        assert "/app" not in path.read_text()


def test_terminal_bench_workspace_seed_defaults_to_bootstrap_only(
    tmp_path, monkeypatch
):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "BOOTSTRAP.md").write_text("bootstrap")
    (assets / "AGENTS.md").write_text("must not overwrite task prompt")
    profile = tmp_path / "harness.yaml"
    profile.write_text(
        "terminal_bench:\n"
        f"  agent_assets_dir: {assets}\n"
    )
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(profile))
    workspace_bind = tmp_path / "workspace-bind"

    tb_app._seed_workspace_helpers(workspace_bind, {})

    assert [p.name for p in workspace_bind.iterdir()] == ["BOOTSTRAP.md"]


def test_openclaw_completion_classification():
    normal = {
        "status": "ok",
        "result": {"meta": {
            "stopReason": "stop",
            "finalAssistantVisibleText": "DONE",
        }},
    }
    overflow = {
        "status": "ok",
        "result": {"meta": {
            "livenessState": "blocked",
            "error": {"kind": "context_overflow", "message": "too large"},
        }},
    }
    blocked = {"result": {"meta": {"livenessState": "blocked"}}}
    runtime_error = {
        "result": {"meta": {"error": {"kind": "gateway_error", "message": "down"}}}
    }
    working = {
        "result": {"meta": {
            "livenessState": "working",
            "stopReason": "stop",
            "finalAssistantVisibleText": "DONE",
        }}
    }
    replay_invalid = {
        "result": {"meta": {
            "replayInvalid": True,
            "stopReason": "stop",
            "finalAssistantVisibleText": "DONE",
        }}
    }
    overflow_payload = {
        "result": {
            "meta": {},
            "payloads": [{
                "text": "Maximum context length exceeded during tool execution",
            }],
        }
    }

    assert OpenClawAgentCondition._classify_completion(normal) == (
        "normal_completion", ""
    )
    assert OpenClawAgentCondition._classify_completion(overflow) == (
        "context_overflow", "too large"
    )
    assert OpenClawAgentCondition._classify_completion(blocked)[0] == (
        "blocked_incomplete"
    )
    assert OpenClawAgentCondition._classify_completion(runtime_error)[0] == (
        "runtime_error"
    )
    assert OpenClawAgentCondition._classify_completion(working)[0] == (
        "blocked_incomplete"
    )
    assert OpenClawAgentCondition._classify_completion(replay_invalid)[0] == (
        "blocked_incomplete"
    )
    assert OpenClawAgentCondition._classify_completion(overflow_payload)[0] == (
        "context_overflow"
    )


def test_roadmapbench_requires_explicit_done():
    stopped_without_done = {
        "result": {"meta": {
            "stopReason": "stop",
            "finalAssistantVisibleText": "Implementation and tests are complete.",
        }}
    }
    stopped_with_done = {
        "result": {"meta": {
            "stopReason": "stop",
            "finalAssistantVisibleText": "DONE\n\nImplementation complete.",
        }}
    }

    assert RoadmapBenchOpenClawCondition._classify_completion(
        stopped_without_done
    )[0] == "blocked_incomplete"
    assert RoadmapBenchOpenClawCondition._classify_completion(
        stopped_with_done
    ) == ("normal_completion", "")


def _write_roadmap_session_state(
    sessions_dir: Path,
    events: list,
    *,
    child_statuses=("running",),
    parent_session_id="parent-session",
):
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "agent:roadmap_bench:main": {
            "sessionId": parent_session_id,
            "status": "done",
        }
    }
    for i, status in enumerate(child_statuses):
        index[f"agent:roadmap_bench:subagent:{i}"] = {
            "sessionId": f"child-{i}",
            "spawnedBy": "agent:roadmap_bench:main",
            "status": status,
        }
    (sessions_dir / "sessions.json").write_text(json.dumps(index))
    (sessions_dir / f"{parent_session_id}.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )


def _yield_event(message="waiting"):
    return {
        "type": "custom_message",
        "customType": "openclaw.sessions_yield",
        "content": message,
    }


def _assistant_event(text):
    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stopReason": "stop",
        },
    }


def _prompt_error_event(error="LLM idle timeout (120s): no response from model"):
    return {
        "type": "custom",
        "customType": "openclaw:prompt-error",
        "data": {"error": error},
    }


def test_roadmap_yield_with_running_child_does_not_complete(tmp_path):
    sessions_dir = tmp_path / "sessions"
    _write_roadmap_session_state(sessions_dir, [_yield_event()])

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot["parent_session_id"] == "parent-session"
    assert snapshot["yield_count"] == 1
    assert snapshot["pending_children"] == 1
    assert snapshot["done_text"] == ""


def test_roadmap_watcher_uses_session_file_when_it_differs_from_session_id(
    tmp_path,
):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    sessions_dir.mkdir(parents=True)
    actual_name = "physical-transcript.jsonl"
    index = {
        "agent:roadmap_bench:main": {
            "sessionId": "logical-session-id",
            "sessionFile": (
                "/root/.openclaw/agents/roadmap_bench/sessions/" + actual_name
            ),
            "status": "done",
        },
        "agent:roadmap_bench:subagent:1": {
            "sessionId": "child-1",
            "spawnedBy": "agent:roadmap_bench:main",
            "status": "running",
        },
    }
    (sessions_dir / "sessions.json").write_text(json.dumps(index))
    (sessions_dir / actual_name).write_text(json.dumps(_yield_event()) + "\n")

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot["parent_session_id"] == "logical-session-id"
    assert snapshot["yield_count"] == 1
    assert snapshot["pending_children"] == 1


def test_roadmap_session_file_resolution_is_contained(tmp_path):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    sessions_dir.mkdir(parents=True)

    relative = RoadmapBenchOpenClawCondition._resolve_parent_transcript_path(
        sessions_dir,
        {"sessionId": "logical", "sessionFile": "physical.jsonl"},
    )
    escaped_relative = RoadmapBenchOpenClawCondition._resolve_parent_transcript_path(
        sessions_dir,
        {"sessionId": "logical", "sessionFile": "../../outside.jsonl"},
    )
    escaped_absolute = RoadmapBenchOpenClawCondition._resolve_parent_transcript_path(
        sessions_dir,
        {"sessionId": "logical", "sessionFile": "/tmp/outside.jsonl"},
    )

    assert relative == (sessions_dir / "physical.jsonl").resolve()
    assert escaped_relative is None
    assert escaped_absolute is None


def test_openclaw_runtime_mode_detects_gateway_and_embedded_fallback():
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._last_stderr = ""
    condition._last_exit_code = 0
    assert condition._detect_agent_runtime_mode({"result": {"meta": {}}}) == (
        "gateway"
    )

    condition._last_stderr = (
        "Gateway agent failed; falling back to embedded: gateway closed (1000)"
    )
    assert condition._detect_agent_runtime_mode(None) == "embedded_fallback"

    condition._last_stderr = ""
    embedded_payload = {
        "result": {"meta": {"executionTrace": {"runner": "embedded"}}}
    }
    assert condition._detect_agent_runtime_mode(embedded_payload) == (
        "gateway"
    )
    condition._last_exit_code = 1
    assert condition._detect_agent_runtime_mode(None) == "unknown"


def test_roadmap_child_completion_resumes_same_parent_and_finishes(tmp_path):
    sessions_dir = tmp_path / "sessions"
    _write_roadmap_session_state(
        sessions_dir,
        [_yield_event(), {"type": "message", "message": {
            "role": "user", "content": [{"type": "text", "text": "child completed"}],
        }}, _assistant_event("DONE\nIntegrated and validated.")],
        child_statuses=("done",),
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)
    payload = RoadmapBenchOpenClawCondition._payload_with_resumed_completion(
        {"result": {"meta": {"livenessState": "working"}}},
        snapshot["done_text"],
        yield_count=snapshot["yield_count"],
        parent_session_id=snapshot["parent_session_id"],
    )

    assert snapshot["parent_session_id"] == "parent-session"
    assert snapshot["pending_children"] == 0
    assert RoadmapBenchOpenClawCondition._classify_completion(payload) == (
        "normal_completion", ""
    )


def test_roadmap_multiple_yield_resume_cycles_continue(tmp_path):
    sessions_dir = tmp_path / "sessions"
    events = [
        _yield_event("wave one"),
        {"type": "message", "message": {"role": "user", "content": [
            {"type": "text", "text": "first child completed"}
        ]}},
        _yield_event("wave two"),
    ]
    _write_roadmap_session_state(
        sessions_dir, events, child_statuses=("done", "running")
    )
    waiting = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)
    assert waiting["yield_count"] == 2
    assert waiting["pending_children"] == 1
    assert waiting["post_yield_parent_activity"] is False
    assert waiting["post_yield_parent_terminal"] is False
    assert waiting["done_text"] == ""

    events.append(_assistant_event("DONE\nBoth waves integrated."))
    _write_roadmap_session_state(
        sessions_dir, events, child_statuses=("done", "done")
    )
    finished = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)
    assert finished["yield_count"] == 2
    assert finished["pending_children"] == 0
    assert finished["post_yield_parent_activity"] is True
    assert finished["post_yield_parent_terminal"] is True
    assert finished["done_text"].startswith("DONE")


def test_roadmap_child_done_without_announce_activity_keeps_waiting(tmp_path):
    sessions_dir = tmp_path / "sessions"
    _write_roadmap_session_state(
        sessions_dir, [_yield_event()], child_statuses=("done",)
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot["pending_children"] == 0
    assert snapshot["post_yield_parent_activity"] is False
    assert snapshot["post_yield_parent_terminal"] is False
    assert snapshot["done_text"] == ""


def test_roadmap_resumed_parent_tool_loop_is_not_terminal(tmp_path):
    sessions_dir = tmp_path / "sessions"
    tool_use = {
        "type": "message",
        "message": {
            "role": "assistant",
            "stopReason": "toolUse",
            "content": [{
                "type": "toolCall",
                "name": "exec",
                "arguments": {"command": "pytest"},
            }],
        },
    }
    tool_result = {
        "type": "message",
        "message": {
            "role": "toolResult",
            "content": [{"type": "text", "text": "891 passed"}],
        },
    }

    _write_roadmap_session_state(
        sessions_dir,
        [_yield_event(), tool_use, tool_result],
        child_statuses=("done",),
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot["pending_children"] == 0
    assert snapshot["post_yield_parent_activity"] is True
    assert snapshot["post_yield_parent_terminal"] is False
    assert snapshot["done_text"] == ""


def test_roadmap_final_prompt_error_is_terminal(tmp_path):
    sessions_dir = tmp_path / "sessions"
    events = [
        _yield_event(),
        _prompt_error_event(),
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "retry"}],
            },
        },
        _prompt_error_event(),
    ]
    _write_roadmap_session_state(
        sessions_dir, events, child_statuses=("done",)
    )

    snapshot = RoadmapBenchOpenClawCondition._read_yield_snapshot(sessions_dir)

    assert snapshot["pending_children"] == 0
    assert snapshot["post_yield_parent_terminal"] is True
    assert snapshot["post_yield_terminal_reason"] == "prompt_error"


def test_roadmap_prompt_error_preserves_runtime_outcome(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    _write_roadmap_session_state(
        sessions_dir,
        [_yield_event(), _prompt_error_event()],
        child_statuses=("done",),
    )
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"state_dir": str(tmp_path)}
    condition._last_exit_code = 0
    condition._last_completion_status = condition.RUNTIME_ERROR
    condition._last_completion_error = "LLM idle timeout"
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.monotonic",
        lambda: 0.0,
    )

    _payload, error = condition._await_post_turn_lifecycle(
        {"result": {"meta": {"error": "LLM idle timeout"}}},
        "LLM idle timeout",
        deadline=10.0,
    )

    assert error == "LLM idle timeout"
    assert condition._last_completion_status == "runtime_error"


def test_roadmap_all_children_done_and_transcript_idle_is_blocked(
    tmp_path, monkeypatch
):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    _write_roadmap_session_state(
        sessions_dir, [_yield_event()], child_statuses=("done",)
    )
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"state_dir": str(tmp_path)}
    condition._last_exit_code = 0
    condition._last_completion_status = condition.BLOCKED_INCOMPLETE
    condition._last_completion_error = "initial yielded turn"
    condition.yield_resume_idle_timeout_seconds = 4.0
    clock = [0.0]
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    _payload, error = condition._await_post_turn_lifecycle(
        {"result": {"meta": {"livenessState": "working"}}},
        "initial yielded turn",
        deadline=100.0,
    )

    assert "no transcript progress for 4.0s" in error
    assert condition._last_exit_code == 0
    assert condition._last_completion_status == "blocked_incomplete"



def test_roadmap_child_done_and_parent_active_without_done_is_blocked(
    tmp_path, monkeypatch
):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    _write_roadmap_session_state(
        sessions_dir,
        [_yield_event(), _assistant_event("Integrating child result.")],
        child_statuses=("done",),
    )
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"state_dir": str(tmp_path)}
    condition._last_exit_code = 0
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.monotonic",
        lambda: 0.0,
    )

    _payload, error = condition._await_post_turn_lifecycle(
        {"result": {"meta": {"livenessState": "working"}}},
        "initial yielded turn",
        deadline=10.0,
    )

    assert "returned without DONE" in error
    assert condition._last_exit_code == 0
    assert condition._last_completion_status == "blocked_incomplete"


def test_roadmap_yield_wait_ends_only_at_timeout(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    _write_roadmap_session_state(sessions_dir, [_yield_event()])
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"state_dir": str(tmp_path)}
    condition._last_exit_code = 0
    ticks = iter((0.0, 0.0, 1.0, 2.0, 3.0))
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.monotonic",
        lambda: next(ticks),
    )
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.sleep",
        lambda _seconds: None,
    )

    _payload, error = condition._await_post_turn_lifecycle(
        {"result": {"meta": {"livenessState": "working"}}},
        "initial yielded turn",
        deadline=2.0,
    )

    assert "total agent timeout" in error
    assert condition._last_exit_code == -1
    assert condition._last_completion_status == "blocked_incomplete"


def test_roadmap_yield_watcher_survives_reyield_and_returns_final_done(
    tmp_path, monkeypatch
):
    sessions_dir = tmp_path / "agents" / "roadmap_bench" / "sessions"
    _write_roadmap_session_state(sessions_dir, [_yield_event()])
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"state_dir": str(tmp_path)}
    condition._last_exit_code = 0
    snapshots = iter((
        {
            "parent_session_id": "parent-session", "pending_children": 2,
            "yield_count": 1, "message_count": 4,
            "post_yield_parent_activity": False,
            "post_yield_parent_terminal": False,
            "done_text": "",
        },
        {
            "parent_session_id": "parent-session", "pending_children": 1,
            "yield_count": 2, "message_count": 8,
            "post_yield_parent_activity": False,
            "post_yield_parent_terminal": False,
            "done_text": "",
        },
        {
            "parent_session_id": "parent-session", "pending_children": 0,
            "yield_count": 2, "message_count": 12,
            "post_yield_parent_activity": True,
            "post_yield_parent_terminal": True,
            "done_text": "DONE\nIntegrated and validated.",
        },
    ))
    monkeypatch.setattr(condition, "_read_yield_snapshot", lambda _path: next(snapshots))
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.monotonic",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent.time.sleep",
        lambda _seconds: None,
    )

    payload, error = condition._await_post_turn_lifecycle(
        {"result": {"meta": {"livenessState": "working"}, "payloads": None}},
        "initial yielded turn",
        deadline=10.0,
    )

    assert error == ""
    assert condition._last_exit_code == 0
    assert condition._last_completion_status == "normal_completion"
    assert payload["result"]["meta"]["agentBenchYieldResume"] == {
        "yieldCount": 2,
        "parentSessionId": "parent-session",
    }


def test_roadmapbench_abnormal_completion_preserves_partial_reward():
    verdict = {"status": "FAIL", "reward": 0.7, "test_pass_rate": 0.7}
    result = {
        "exit_code": 0,
        "agent_completion_status": "context_overflow",
        "agent_completion_error": "Context overflow",
    }

    merged = _merge_agent_completion_into_verdict(verdict, result)

    assert merged["status"] == "FAIL"
    assert merged["verifier_status"] == "FAIL"
    assert merged["agent_completion_status"] == "context_overflow"
    assert merged["agent_execution_outcome"] == "context_overflow"
    assert merged["reward"] == 0.7
    assert merged["test_pass_rate"] == 0.7
    assert merged["agent_completion_error"] == "Context overflow"
    assert "error_message" not in merged


def test_terminal_bench_condition_and_forced_capability_remain(monkeypatch):
    condition = object.__new__(TerminalBenchOpenClawCondition)
    monkeypatch.delenv("AB_TB_FORCE_ONE_SUBAGENT", raising=False)
    baseline = condition._build_message({"instruction": "Do terminal work."})
    assert "Carry out this terminal task" in baseline
    assert "sessions_spawn exactly once" not in baseline

    monkeypatch.setenv("AB_TB_FORCE_ONE_SUBAGENT", "1")
    forced = condition._build_message({"instruction": "Do terminal work."})
    assert "sessions_spawn exactly once" in forced


def test_terminal_bench_fractional_reward_is_preserved():
    assert tb_app._terminal_bench_reward("1.0", timed_out=False) == ("PASS", 1.0)
    assert tb_app._terminal_bench_reward("0.5556", timed_out=False) == (
        "FAIL", 0.5556
    )
    assert tb_app._terminal_bench_reward("", timed_out=True) == ("TIMEOUT", 0.0)


def test_terminal_bench_loader_accepts_roadmapbench_v1_metadata(tmp_path):
    task = tmp_path / "fal-1.3.0-roadmap"
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Implement five roadmap targets.\n")
    (task / "task.toml").write_text(
        'version = "1"\n'
        '[metadata]\nname = "fal-1.3.0-roadmap"\n'
        '[verifier]\ntimeout_sec = 1800\n'
        '[agent]\ntimeout_sec = 7200\n'
        '[environment]\n'
        'docker_image = "znpt/roadmapbench-fal-1.3.0-roadmap"\n'
    )

    row = tb_loader.load_task("fal-1.3.0-roadmap", tasks_root=tmp_path)

    assert row["task_name"] == "fal-1.3.0-roadmap"
    assert row["repo"] == "fal-1.3.0-roadmap"
    assert row["benchmark_variant"] == "roadmapbench"
    assert row["agent_timeout_sec"] == 7200
    assert row["verifier_timeout_sec"] == 1800


def test_terminal_bench_loader_keeps_terminal_variant(tmp_path):
    task = tmp_path / "terminal-task"
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Do terminal work.\n")
    (task / "task.toml").write_text(
        '[task]\nname = "terminal-task"\n'
        '[environment]\n'
        'docker_image = "example/terminal-task:latest"\n'
    )

    row = tb_loader.load_task("terminal-task", tasks_root=tmp_path)

    assert row["benchmark_variant"] == "terminal_bench"


def test_singularity_copy_into_directory_streams_tar(tmp_path, monkeypatch):
    source = tmp_path / "tests"
    source.mkdir()
    (source / "test.sh").write_text("#!/bin/sh\n")
    calls = []

    class _Stream:
        def close(self):
            pass

    class _Proc:
        returncode = 0
        stdout = _Stream()

        def communicate(self, timeout=None):
            return b"", b""

        def kill(self):
            pass

    def popen(cmd, **kw):
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    container = object.__new__(SingularityContainer)
    SingularityContainer.copy_into(
        container, ContainerHandle("task", "task", "singularity"), source, "/tests"
    )

    assert calls[0][:3] == ["tar", "-C", str(source)]
    assert calls[1][:3] == ["singularity", "exec", "instance://task"]
    assert "mkdir -p '/tests'" in calls[1][-1]
