"""Focused contracts for the independent RoadmapBench/Hermes path."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bench.benchmarks.roadmap_bench.hermes.app import RoadmapBenchHermesHarness
from agent_bench.benchmarks.roadmap_bench.hermes.condition import RoadmapBenchHermesCondition
from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition
from agent_bench.deployments.containers import ContainerHandle
from agent_bench.deployments.pbs.stages import (
    _runtime_image_commands,
    _terminal_bench_image_commands,
)


class FakeContainer:
    backend_id = "singularity"
    host_address = "127.0.0.1"

    def __init__(self):
        self.specs = []
        self.stops = []
        self.copies = []
        self.log_dir = None

    def check_available(self):
        pass

    def check_image(self, image):
        assert Path(image).is_file()

    def create_network(self, name):
        pass

    def remove_network(self, name):
        pass

    def copy_from_image(self, image, src, dst):
        (dst / "file.txt").write_text("baseline\n")

    def start(self, spec):
        self.specs.append(spec)
        return ContainerHandle(spec.name, spec.name, self.backend_id)

    def stop(self, handle):
        self.stops.append(handle.name)

    def copy_into(self, handle, src, dst):
        self.copies.append((str(src), dst))

    def exec(self, handle, argv, *, check=False, timeout=None, workdir=None, env=None, **_kw):
        if argv[:1] == ["bash"]:
            (self.log_dir / "reward.txt").write_text("0.6")
        return subprocess.CompletedProcess(argv, 0, "", "")


class FakeDispatcher:
    @staticmethod
    def stop_external(**_kw):
        pass

    def start(self, **_kw):
        pass

    def stop(self):
        pass

    def agent_mounts_and_env(self):
        return [], {"HERMES_EXEC_BRIDGE": "/usr/local/bin/hm-exec-bridge"}

    def env_info(self):
        return {"bridge_pid": 123, "bridge_socket": "/tmp/hm-test.sock"}


def test_roadmap_hermes_singularity_runtime_and_private_tests(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = "znpt/roadmapbench-demo:1"
    task_sif = image_dir / "znpt__roadmapbench-demo__1.sif"
    task_sif.write_bytes(b"task")
    runtime_sif = tmp_path / "hermes.sif"
    runtime_sif.write_bytes(b"runtime")
    tests = tmp_path / "private-tests"
    tests.mkdir()
    (tests / "test.sh").write_text("true\n")
    monkeypatch.setenv("AB_AGENT_RUNTIME_SIF", str(runtime_sif))
    monkeypatch.setenv("AB_DEPLOYMENT_TASK_IMAGE_DIR", str(image_dir))
    monkeypatch.delenv("AB_RUN_TAG", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(RoadmapBenchHermesHarness, "dispatcher_cls", FakeDispatcher)
    monkeypatch.setattr(RoadmapBenchHermesHarness, "_build_state_dir", lambda self, **_kw: tmp_path / "state")
    monkeypatch.setattr(RoadmapBenchHermesHarness, "_agent_env", lambda self, **_kw: {})
    container = FakeContainer()
    harness = RoadmapBenchHermesHarness(container=container, workspace_base=tmp_path / "workspaces")
    row = {
        "benchmark": "terminal_bench", "benchmark_variant": "roadmapbench",
        "docker_image": image, "instruction": "Implement roadmap", "workspace": "/app",
        "tests_host_path": str(tests),
    }
    info = harness.prepare_environment("demo", "repo", "base", run_id="run1", instance_data=row)
    assert info["container_id"] == info["hm_container_id"]
    assert info["container_id"] != info["task_container_id"]
    assert info["backend"] == "singularity"
    assert container.specs[0].image == str(task_sif)
    assert container.specs[1].image == str(runtime_sif)
    assert container.specs[0].mounts[-1].container == "/tests"
    assert not list((tmp_path / "workspaces" / "run1" / "tests").iterdir())
    assert "HERMES_EXEC_BRIDGE" in info["agent_env"]
    assert "/tests" not in [m.container for m in container.specs[1].mounts]
    container.log_dir = tmp_path / "workspaces" / "run1" / "logs" / "verifier"
    harness.quiesce_for_scoring(info)
    assert info["workspace_quiesced"] is True
    assert info["container_id"] in container.stops
    assert info["task_container_name"] in container.stops
    verdict = harness.run_tests(None, "demo", "/app", 60, run_id="run1", instance_data=row)
    assert verdict["status"] == "FAIL"
    assert verdict["reward"] == pytest.approx(0.6)
    assert container.copies == [(str(tests), "/tests")]
    cleanup_kwargs = dict(info)
    cleanup_kwargs.pop("container_id")
    harness.cleanup(info["container_id"], **cleanup_kwargs)
    assert info["task_container_name"] not in harness._task_specs


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def test_roadmap_hermes_candidate_includes_commits_and_untracked(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "file.txt").write_text("initial\n")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-qm", "initial")

    class GitContainer:
        def exec(self, _handle, argv, **kwargs):
            return subprocess.run(argv, cwd=kwargs.get("workdir"), capture_output=True, text=True)

    class Harness:
        state_dir_in_agent = "/opt/data"
        container = GitContainer()

        def quiesce_for_scoring(self, info):
            info["workspace_quiesced"] = True

    info = {
        "_harness": Harness(), "task_container_id": "task", "task_container_name": "task",
        "task_workspace_root": str(repo), "backend": "singularity",
    }
    condition = RoadmapBenchHermesCondition(SimpleNamespace(name="model"), SimpleNamespace())
    condition._run_dir = tmp_path
    condition._baseline = condition._capture_baseline(info)
    (repo / "file.txt").write_text("committed\n")
    (repo / "new.txt").write_text("committed new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "solution")
    (repo / "file.txt").write_text("final\n")
    (repo / "later.txt").write_text("untracked\n")
    (repo / "AGENTS.md").write_text("scaffold\n")
    (repo / ".openclaw").mkdir()
    (repo / ".openclaw" / "state.json").write_text("metadata\n")
    monkeypatch.setattr(HermesAgentCondition, "_collect_sessions", lambda *_a, **_kw: None)
    patch = condition._collect_artifacts(info, "agent", "/app")
    assert "committed new" in patch
    assert "-initial" in patch and "+final" in patch
    assert "untracked" in patch
    assert "AGENTS.md" not in patch
    assert ".openclaw" not in patch
    assert info["workspace_quiesced"] is True


def test_roadmap_hermes_pbs_preflights_without_pull(tmp_path):
    resolved = SimpleNamespace(
        spec=SimpleNamespace(benchmark_name="terminal_bench", harness_name="hermes"),
        harness={"harness": "hermes", "roadmapbench": True},
        benchmark={"instances": ["demo-roadmap"]},
    )
    spec = SimpleNamespace(container_runtime="singularity", log_dir=tmp_path)
    env = {
        "AB_AGENT_RUNTIME_SIF": str(tmp_path / "hermes.sif"),
        "AB_BENCHMARK_TASKS_ROOT": str(tmp_path / "tasks"),
        "AB_DEPLOYMENT_TASK_IMAGE_DIR": str(tmp_path / "images"),
    }
    runtime = _runtime_image_commands(resolved, spec, env, tmp_path)
    tasks = _terminal_bench_image_commands(resolved, spec, env)
    assert len(runtime) == len(tasks) == 1
    assert "preflight" in " ".join(runtime[0].argv)
    assert "preflight" in " ".join(tasks[0].argv)
    assert not any("pull" in part or "build" in part for cmd in (*runtime, *tasks) for part in cmd.argv)


def test_roadmap_hermes_cli_delegation_is_isolated(monkeypatch):
    calls = []

    def record(_self, _container_id, argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(
        "agent_bench.benchmarks.terminal_bench.hermes.app.TerminalBenchHermesHarness.exec_in_container",
        record,
    )
    roadmap = RoadmapBenchHermesHarness.__new__(RoadmapBenchHermesHarness)
    roadmap.exec_in_container(
        "agent", ["hermes", "--toolsets", "terminal,file,code_execution", "-z", "roadmap"],
    )
    assert calls[-1][2] == "terminal,file,code_execution,delegation"
    roadmap.exec_in_container("agent", ["hermes", "sessions", "export", "sessions.jsonl"])
    assert calls[-1] == ["hermes", "sessions", "export", "sessions.jsonl"]


def test_roadmap_hermes_delegation_budget_is_rendered(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.yaml").write_text("agent:\n  max_turns: 600\n")
    monkeypatch.setattr(
        "agent_bench.benchmarks.terminal_bench.hermes.app.TerminalBenchHermesHarness._build_state_dir",
        lambda _self, **_kw: state,
    )
    roadmap = RoadmapBenchHermesHarness.__new__(RoadmapBenchHermesHarness)
    roadmap._load_harness_config = lambda: {
        "hermes": {"delegation": {
            "max_concurrent_children": 16,
            "max_iterations": 600,
            "child_timeout_seconds": 7200,
        }}
    }
    assert roadmap._build_state_dir() == state
    rendered = (state / "config.yaml").read_text()
    assert "max_concurrent_children: 16" in rendered
    assert "max_iterations: 600" in rendered
    assert "child_timeout_seconds: 7200" in rendered
