from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from agent_bench.benchmarks.terminal_bench.claude_code import app as cc_app
from agent_bench.benchmarks.terminal_bench.conditions import claude_code_agent as cc_condition
from agent_bench.deployments.containers import ContainerHandle
from agent_bench.deployments.pbs.stages import _agent_runtime_defaults


class _FakeContainer:
    backend_id = "singularity"
    host_address = "127.0.0.1"

    def __init__(self):
        self.specs = []
        self.running = set()

    def check_available(self):
        pass

    def check_image(self, image):
        assert Path(image).is_file()

    def create_network(self, _name):
        pass

    def remove_network(self, _name):
        pass

    def copy_from_image(self, _image, src, dst, **_kwargs):
        dst.mkdir(parents=True, exist_ok=True)
        if src == "/opt/claude-code":
            (dst / "bin").mkdir()
            (dst / "bin" / "claude").write_text("#!/bin/sh\n")
        else:
            (dst / "starter.txt").write_text("starter\n")

    def start(self, spec):
        self.specs.append(spec)
        handle = ContainerHandle(spec.name, spec.name, self.backend_id)
        self.running.add(spec.name)
        return handle

    def stop(self, handle):
        self.running.discard(handle.name)

    def is_running(self, handle):
        return handle.name in self.running

    def exec(self, _handle, argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, "", "")

    def logs(self, _handle):
        return "adapter log"


def test_claude_code_singularity_harness_uses_task_runtime_and_adapter(
    tmp_path, monkeypatch
):
    runtime_sif = tmp_path / "claude.sif"
    runtime_sif.write_bytes(b"runtime")
    task_dir = tmp_path / "task-images"
    task_dir.mkdir()
    task_sif = task_dir / "alexgshaw__multi-source-data-merger__20251031.sif"
    task_sif.write_bytes(b"task")
    monkeypatch.setenv("AB_AGENT_RUNTIME_SIF", str(runtime_sif))
    monkeypatch.setenv("AB_DEPLOYMENT_TASK_IMAGE_DIR", str(task_dir))
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AB_REPO_ROOT", str(repo_root))
    monkeypatch.setenv(
        "AB_HARNESS_CONFIG",
        str(
            repo_root
            / "profiles/harnesses/claude_code_terminal_bench_subagent_smoke.yaml"
        ),
    )
    monkeypatch.setattr(
        cc_app,
        "load_active_model_and_serving",
        lambda **_kwargs: (
            SimpleNamespace(
                name="Qwen/Qwen3.6-35B-A3B",
                endpoint="http://127.0.0.1:30000/v1",
                api_key="EMPTY",
                max_tokens=32768,
            ),
            SimpleNamespace(),
        ),
    )

    container = _FakeContainer()
    harness = cc_app.TerminalBenchClaudeCodeHarness(
        container=container,
        workspace_base=tmp_path / "workspaces",
    )
    env_info = harness.prepare_environment(
        "multi-source-data-merger",
        "unused",
        "unused",
        run_id="run-1",
        instance_data={
            "benchmark": "terminal_bench",
            "docker_image": "alexgshaw/multi-source-data-merger:20251031",
            "workspace": "/app",
            "instruction": "merge data",
        },
    )

    assert env_info["container_id"] == env_info["task_container_id"]
    assert env_info["adapter_container_id"] != env_info["container_id"]
    assert env_info["backend"] == "singularity"
    assert container.specs[0].image == str(task_sif)
    assert container.specs[1].image == str(runtime_sif)
    assert container.specs[1].entrypoint == "/opt/litellm/bin/litellm"
    assert env_info["agent_env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == (
        "Qwen/Qwen3.6-35B-A3B"
    )
    assert env_info["agent_env"]["ANTHROPIC_BASE_URL"].startswith(
        "http://127.0.0.1:"
    )
    tests_mount = next(
        mount for mount in container.specs[0].mounts if mount.container == "/tests"
    )
    assert list(Path(tests_mount.host).iterdir()) == []
    adapter_config = json.loads(
        (Path(env_info["state_dir"]) / "litellm.json").read_text()
    )
    params = adapter_config["model_list"][0]["litellm_params"]
    assert params["model"] == "openai/Qwen/Qwen3.6-35B-A3B"
    assert params["api_base"] == "http://127.0.0.1:30000/v1"
    workspace = Path(env_info["workspace_host_path"])
    claude_prompt = (workspace / "CLAUDE.md").read_text()
    assert "lead engineer responsible" in claude_prompt
    assert "native `Agent` tool" in claude_prompt
    assert "multi-target software roadmap" not in claude_prompt

    harness.cleanup(**env_info)


def test_claude_code_condition_archives_native_subagent_evidence(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "state"
    child_dir = state_dir / "projects" / "task" / "subagents"
    child_dir.mkdir(parents=True)
    tool_use = {
        "type": "assistant",
        "message": {
            "id": "msg-1",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Agent",
                    "input": {"description": "inspect source schemas"},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
    }
    (state_dir / "session.jsonl").write_text(json.dumps(tool_use) + "\n")
    (child_dir / "agent-child-1.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"id": "child-1"}}) + "\n"
    )
    result_event = {
        "type": "result",
        "subtype": "success",
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }
    stdout = "\n".join((json.dumps(tool_use), json.dumps(result_event))) + "\n"

    class _ConditionContainer:
        backend_id = "singularity"

        def exec(self, _handle, _argv, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout, "")

    class _Sampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(cc_condition, "UtilSampler", _Sampler)
    monkeypatch.setattr(cc_condition, "otel_trace_offset", lambda: (None, 0))
    monkeypatch.setattr(cc_condition, "wait_for_otel_flush", lambda: None)
    monkeypatch.setattr(cc_condition, "snapshot_otel_spans", lambda *_a, **_k: None)
    condition = cc_condition.TerminalBenchClaudeCodeCondition(
        SimpleNamespace(name="Qwen/Qwen3.6-35B-A3B"),
        SimpleNamespace(max_steps=200, max_time_seconds=1800),
        _ConditionContainer(),
    )
    run_dir = tmp_path / "run"
    output = condition.run(
        {"instance_id": "task", "instruction": "do it"},
        run_dir,
        42,
        "/app",
        "task-runtime",
        {
            "container_name": "task-runtime",
            "task_container_name": "task-runtime",
            "claude_state_dir": str(state_dir),
            "agent_env": {},
        },
    )

    observability = json.loads((run_dir / "runtime_observability.json").read_text())
    assert observability["subagent_tool_call_count"] == 1
    assert observability["materialized_subagent_count"] == 1
    assert output["total_prompt_tokens"] == 50
    assert output["total_completion_tokens"] == 20
    assert (run_dir / "conversation" / "claude-code" / "session.jsonl").is_file()
    assert (run_dir / "trajectory.jsonl").stat().st_size > 0


def test_claude_code_runtime_defaults_require_uploaded_sif(tmp_path):
    defaults = _agent_runtime_defaults("claude_code", str(tmp_path), "/scratch/u")
    assert defaults["AB_AGENT_RUNTIME_VERSION"] == "2.1.278"
    assert defaults["AB_AGENT_RUNTIME_SIF"].endswith(
        "/sif_images/claude_code_2.1.278_litellm_1.102.0.sif"
    )
