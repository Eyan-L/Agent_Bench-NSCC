import json
import subprocess
from pathlib import Path

import pytest

from agent_bench.benchmarks.roadmap_bench.conditions.openclaw_agent import (
    RoadmapBenchOpenClawCondition,
)
from agent_bench.benchmarks.terminal_bench.app import TerminalBenchOpenClawHarness


class _PrimeFailContainer:
    backend_id = "singularity"
    host_address = "127.0.0.1"

    def __init__(self) -> None:
        self.removed_networks = []

    def stop(self, _handle) -> None:
        return None

    def remove_network(self, name: str) -> None:
        self.removed_networks.append(name)

    def create_network(self, _name: str) -> None:
        return None

    def copy_from_image(self, _image: str, _src: str, _dst: Path) -> None:
        raise RuntimeError("workspace copy stalled")


class _VerifierTimeoutContainer:
    backend_id = "singularity"

    def __init__(self) -> None:
        self.exec_count = 0

    def exec(self, _handle, argv, **kwargs):
        self.exec_count += 1
        if self.exec_count == 1:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise subprocess.TimeoutExpired(
            argv,
            kwargs.get("timeout", 1),
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    def copy_into(self, _handle, _src: Path, _dst: str) -> None:
        return None


class _QuiesceHarness:
    def quiesce_orphan_children(self, _env_info):
        return {
            "workspaceQuiesced": True,
            "quiescenceMethod": "stop-agent-and-restart-task-runtime",
            "openClawRuntimeStopped": True,
            "execBridgeStopped": True,
            "taskRuntimeRestarted": True,
        }


def test_openclaw_workspace_prime_failure_is_fail_closed(tmp_path):
    container = _PrimeFailContainer()
    harness = TerminalBenchOpenClawHarness.__new__(TerminalBenchOpenClawHarness)
    harness.container = container
    harness.workspace_base = tmp_path / "workspaces"
    harness.workspace_base.mkdir()
    harness._resolve_task_image = lambda _image: "/tmp/task.sif"

    with pytest.raises(RuntimeError, match="refusing to start the agent"):
        harness.prepare_environment(
            instance_id="task-roadmap",
            repo_id="",
            base_commit="",
            run_id="prime-fail-run",
            instance_data={
                "benchmark": "terminal_bench",
                "docker_image": "roadmap/task:latest",
                "instruction": "implement roadmap",
                "workspace": "/app",
            },
        )

    assert not (harness.workspace_base / "prime-fail-run").exists()
    assert container.removed_networks[-1] == "oc-eval-net-prime-fail-r"


def test_verifier_timeout_bytes_are_json_serializable(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    harness = TerminalBenchOpenClawHarness.__new__(TerminalBenchOpenClawHarness)
    harness.container = _VerifierTimeoutContainer()
    harness.workspace_base = tmp_path / "workspaces"
    harness.workspace_base.mkdir()

    result = harness.run_tests(
        container_id=None,
        instance_id="task-roadmap",
        workspace_root="/app",
        timeout_seconds=5,
        run_id="timeout-run",
        instance_data={
            "workspace": "/app",
            "tests_host_path": str(tests_dir),
            "verifier_entrypoint": "/tests/test.sh",
            "verifier_timeout_sec": 5,
        },
    )

    json.dumps(result)
    assert result["status"] == "TIMEOUT"
    assert result["verifier_status"] == "TIMEOUT"
    assert result["reward"] is None
    assert result["test_pass_rate"] is None
    assert result["verifier_timed_out"] is True
    assert result["verifier_timeout_seconds"] == 5
    assert result["verifier_stdout_tail"] == "partial stdout"
    assert "partial stderr" in result["verifier_stderr_tail"]
    assert "verifier timeout after 5s" in result["verifier_stderr_tail"]


def test_abnormal_no_child_exit_records_quiescence_metadata():
    condition = object.__new__(RoadmapBenchOpenClawCondition)
    condition._env_info = {"_harness": _QuiesceHarness()}
    snapshot = {
        "parent_session_status": "error",
        "yield_count": 0,
        "total_children": 0,
        "completed_children": 0,
        "pending_children": 0,
        "pending_child_sessions": [],
        "child_states": [],
    }

    payload = condition._quiesce_abnormal_terminal_payload(
        {"result": {"meta": {}}},
        snapshot,
    )

    lifecycle = payload["result"]["meta"]["agentBenchLifecycle"]
    assert lifecycle["workspaceQuiesced"] is True
    assert lifecycle["quiescenceMethod"] == "stop-agent-and-restart-task-runtime"
    assert lifecycle["agentWriterQuiescenceAttempted"] is True
    assert lifecycle["orphanCancellationAttempted"] is False
    assert lifecycle["pendingChildrenAtParentTerminal"] == 0


def test_roadmapbench_merge_preserves_verifier_timeout_status():
    from agent_bench.runtime.driver import _merge_agent_completion_into_verdict

    merged = _merge_agent_completion_into_verdict(
        {
            "status": "TIMEOUT",
            "verifier_status": "TIMEOUT",
            "reward": None,
            "reward_raw": None,
            "test_pass_rate": None,
            "verifier_timed_out": True,
        },
        {
            "exit_code": 0,
            "agent_completion_status": "normal_completion",
        },
    )

    assert merged["status"] == "TIMEOUT"
    assert merged["verifier_status"] == "TIMEOUT"
    assert merged["reward"] is None
    assert merged["test_pass_rate"] is None
    assert merged["agent_execution_outcome"] == "normal_completion"
