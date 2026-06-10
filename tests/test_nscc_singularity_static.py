"""Static checks for NSCC singularity-capable harness wiring."""

from pathlib import Path

from agent_bench.core.spec import load_resolved_experiment
from agent_bench.deployments.pbs.job_spec import build_pbs_job_spec
from agent_bench.deployments.pbs.stages import prepare_environment


ROOT = Path(__file__).resolve().parents[1]


def test_openhands_uses_container_backend_for_nscc():
    app = (ROOT / "agent_bench/benchmarks/swebench/openhands/app.py").read_text()
    condition = (ROOT / "agent_bench/benchmarks/swebench/conditions/openhands_agent.py").read_text()
    dispatcher = (ROOT / "agent_bench/harnesses/openhands/dispatcher.py").read_text()

    assert "ContainerSpec" in app
    assert "self.container.start" in app
    assert "self.container.exec" in app
    assert "AB_AGENT_RUNTIME_SIF" in app
    assert "AB_SCORING_IMAGE_DIR" in app
    assert "ContainerHandle" in condition
    assert "self.container.exec" in condition
    assert '["docker", "exec", container_id, "bash", "-lc", inner]' in condition
    assert "agent_mounts_and_env" in dispatcher
    assert "_workspace_by_container" in app


def test_hermes_uses_container_backend_for_nscc():
    app = (ROOT / "agent_bench/benchmarks/swebench/hermes/app.py").read_text()
    condition = (ROOT / "agent_bench/benchmarks/swebench/conditions/hermes_agent.py").read_text()
    dispatcher = (ROOT / "agent_bench/harnesses/hermes/dispatcher.py").read_text()

    assert "ContainerSpec" in app
    assert "self.container.start" in app
    assert "self.container.exec" in app
    assert "AB_AGENT_RUNTIME_SIF" in app
    assert "AB_SCORING_IMAGE_DIR" in app
    assert "exec_in_container" in condition
    assert '["docker", "exec", container_id, *argv]' in condition
    assert "agent_mounts_and_env" in dispatcher
    assert "_workspace_by_container" in app


def test_swebench_task_containers_use_private_home():
    for rel in (
        "agent_bench/benchmarks/swebench/openclaw/app.py",
        "agent_bench/benchmarks/swebench/openhands/app.py",
        "agent_bench/benchmarks/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert '"HOME": "/root"' in app


def test_swebench_harnesses_can_use_remote_scoring_server():
    for rel in (
        "agent_bench/benchmarks/swebench/openclaw/app.py",
        "agent_bench/benchmarks/swebench/openhands/app.py",
        "agent_bench/benchmarks/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert "score_via_scoring_server" in app
    helper = (ROOT / "agent_bench/benchmarks/swebench/scoring.py").read_text()
    assert "AB_SCORING_SERVICE_URL" in helper
    assert "/v1/submit" in helper
    assert "/v1/submission/" in helper


def test_swebench_harnesses_can_score_with_singularity_sifs():
    helper = (ROOT / "agent_bench/benchmarks/swebench/scoring.py").read_text()
    assert "def score_via_singularity" in helper
    assert "AB_SCORING_IMAGE_DIR" in helper
    assert '["singularity", "exec"' in helper
    assert "get_eval_report" in helper

    for rel in (
        "agent_bench/benchmarks/swebench/openclaw/app.py",
        "agent_bench/benchmarks/swebench/openhands/app.py",
        "agent_bench/benchmarks/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert "score_via_singularity" in app
        assert app.index("score_via_singularity(") < app.index("score_via_scoring_server(")


def test_openclaw_uses_runtime_serving_endpoint():
    app = (ROOT / "agent_bench/benchmarks/swebench/openclaw/app.py").read_text()

    assert 'os.environ.get("OPENAI_BASE_URL")' in app
    assert 'os.environ.get("AB_SERVING_BASE_URL")' in app
    assert "endpoint_for_container(" in app


def test_openclaw_docker_harnesses_use_bridge_mounts_and_env():
    for rel in (
        "agent_bench/benchmarks/terminal_bench/app.py",
        "agent_bench/benchmarks/deepresearch_bench/app.py",
        "agent_bench/benchmarks/longbench/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert "dispatcher.agent_mounts_and_env()" in app
        assert "*bridge_args" in app
        assert "*dispatcher.agent_container_args()" not in app


def test_terminal_bench_bridge_dispatcher_uses_backend_attrs():
    legacy_app = (ROOT / "agent_bench/benchmarks/terminal_bench/app.py").read_text()
    agent_base = (
        ROOT / "agent_bench/benchmarks/terminal_bench/agent_harness_base.py"
    ).read_text()

    assert "backend=self.backend_id" in legacy_app
    assert "backend=self.container.backend_id" in agent_base
    assert 'backend="docker"' not in legacy_app
    assert 'backend="docker"' not in agent_base


def test_openclaw_forces_chat_completions_transport():
    stages = (ROOT / "agent_bench/deployments/pbs/stages.py").read_text()
    configure = (ROOT / "agent_bench/harnesses/openclaw/configure.sh").read_text()

    for rel in (
        "agent_bench/benchmarks/swebench/openclaw/app.py",
        "agent_bench/benchmarks/terminal_bench/app.py",
        "agent_bench/benchmarks/deepresearch_bench/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert '_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"' in app
        assert 'vllm_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API' in app
        assert 'model["api"] = _OPENAI_CHAT_COMPLETIONS_API' in app

    assert 'vllm_provider["api"] = "openai-completions"' in stages
    assert '"api": "openai-completions"' in stages
    assert "models.providers.vllm.api openai-completions" in configure


def test_pbs_prepares_runtime_sif_by_harness_choice():
    stages = (ROOT / "agent_bench/deployments/pbs/stages.py").read_text()

    assert "AB_AGENT_RUNTIME_SIF" in stages
    assert "AB_AGENT_RUNTIME_SIF" in stages
    assert "AB_AGENT_RUNTIME_BASE_DOCKER" in stages
    assert "AB_AGENT_RUNTIME_BASE_DOCKER" in stages
    assert "build_oh_sif.sh" in stages
    assert "build_hm_sif.sh" in stages
    assert 'harness == "openhands"' in stages
    assert 'harness == "hermes"' in stages
    assert 'out["AB_AGENT_RUNTIME_IMAGE"] = env.get("AB_AGENT_RUNTIME_IMAGE") or out["AB_AGENT_RUNTIME_SIF"]' in stages
    assert 'out["AB_AGENT_RUNTIME_IMAGE"] = env.get("AB_AGENT_RUNTIME_IMAGE") or out["AB_AGENT_RUNTIME_SIF"]' in stages
    assert "pull OpenHands" in stages
    assert "pull Hermes" in stages


def test_nscc_runtime_sif_builders_exist():
    build_oh = ROOT / "agent_bench/deployments/pbs/build_oh_sif.sh"
    build_hm = ROOT / "agent_bench/deployments/pbs/build_hm_sif.sh"

    oh = build_oh.read_text()
    hm = build_hm.read_text()

    assert "singularity build --sandbox" in oh
    assert "openhands-sdk==" in oh
    assert "/opt/openhands_driver.py" in oh
    assert "singularity build --sandbox" in hm
    assert "apply_patch.py apply" in hm
    assert "HERMES_PATCH_MODE=baked" in hm
    patcher = (ROOT / "agent_bench/harnesses/hermes/apply_patch.py").read_text()
    assert "apply_unified_patch_native" in patcher


def test_singularity_env_uses_parent_env_for_commas():
    import pytest

    from agent_bench.deployments.containers.singularity import _set_singularity_env

    env = {}
    _set_singularity_env(env, "OTEL_RESOURCE_ATTRIBUTES", "oc.run_id=a,hm.run_id=a")
    assert env["SINGULARITYENV_OTEL_RESOURCE_ATTRIBUTES"] == "oc.run_id=a,hm.run_id=a"
    with pytest.raises(ValueError):
        _set_singularity_env(env, "hm.run_id", "a")


def test_hermes_singularity_exec_runs_from_workspace():
    app = (ROOT / "agent_bench/benchmarks/swebench/hermes/app.py").read_text()
    condition = (ROOT / "agent_bench/benchmarks/swebench/conditions/hermes_agent.py").read_text()

    assert "otel_env(" in app
    assert 'service_name="agent-bench-hermes"' in app
    assert 'extra_attrs={"hm.run_id": run_id}' in app
    assert "workdir: Optional[str] = None" in app
    assert "workdir=workspace_root" in condition


def test_singularity_subprocess_env_drops_home_override(monkeypatch):
    from agent_bench.deployments.containers.singularity import _subprocess_env

    monkeypatch.setenv("SINGULARITYENV_HOME", "/root")
    monkeypatch.setenv("SINGULARITYENV_NPM_CONFIG_REGISTRY", "https://registry.npmmirror.com")

    env = _subprocess_env({"SINGULARITYENV_HOME": "/other"})

    assert "SINGULARITYENV_HOME" not in env
    assert env["SINGULARITYENV_NPM_CONFIG_REGISTRY"] == "https://registry.npmmirror.com"


def test_pbs_groups_logs_under_experiment_dir(tmp_path: Path):
    resolved = load_resolved_experiment(ROOT / "experiments/nscc_vllm.toml", root=ROOT)
    env = {
        "AB_REPO_ROOT": str(ROOT),
        "USER": "alice",
        "SCRATCH": str(tmp_path / "scratch"),
        "AB_RUN_DIR": str(tmp_path / "run"),
        "AB_LOG_DIR": str(tmp_path / "run/logs"),
    }
    spec = build_pbs_job_spec(resolved, env=env)

    prepare_environment(resolved, spec, env)

    assert env["AB_EXPERIMENT_DIR"] == str(tmp_path / "run")
    assert env["AB_LOG_DIR"] == str(tmp_path / "run/logs")
    assert env["AB_LOG_DIR"] == str(tmp_path / "run/logs")


def test_runner_maps_app_output_under_experiment_dir(monkeypatch):
    from agent_bench.core.config import load_composed_config

    monkeypatch.setenv("AB_EXPERIMENT_DIR", "/tmp/agent-bench-exp")
    cfg = load_composed_config(
        ROOT / "profiles/benchmarks/swebench/phase0.json",
        ROOT / "profiles/harnesses/openclaw_swebench.yaml",
        ROOT / "profiles/models/qwen36_27b.json",
        ROOT / "profiles/serving/vllm/default.yaml",
    )

    assert cfg.output_dir == Path("/tmp/agent-bench-exp/runs/phase0")


def test_openclaw_raw_response_log_synthesizes_per_request_model_usage(tmp_path):
    import json

    from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

    run_dir = tmp_path / "run-1"
    raw_dir = run_dir / "conversation"
    raw_dir.mkdir(parents=True)
    spans_path = run_dir / "spans.jsonl"
    spans_path.write_text("")
    rows = [
        {
            "ts": "2026-06-01T00:00:00.000Z",
            "agentId": "agent",
            "sessionId": "sess-1",
            "runId": "run-1",
            "model": "model-a",
            "provider": "provider-a",
            "api": "openai-completions",
            "httpStatus": 200,
            "response": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
            "metrics": {"total_ms": 1200, "ttft_ms": 100, "tbt_avg_ms": 20},
        },
        {
            "ts": "2026-06-01T00:00:02.000Z",
            "agentId": "agent",
            "sessionId": "sess-1",
            "runId": "run-1",
            "model": "model-a",
            "provider": "provider-a",
            "api": "openai-completions",
            "httpStatus": 200,
            "response": {
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                }
            },
            "metrics": {"total_ms": 900, "ttft_ms": 80, "tbt_avg_ms": 15},
        },
    ]
    raw_path = raw_dir / "llm-raw-responses.jsonl"
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    count = OpenClawAgentCondition._maybe_synthesize_model_usage_spans_from_raw_log(
        spans_path, run_dir, "run-1"
    )

    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    assert count == 2
    assert len(spans) == 2
    assert [span["attrs"]["openclaw.llm.request_index"] for span in spans] == [1, 2]
    assert [span["duration_ms"] for span in spans] == [1200.0, 900.0]
    assert [span["attrs"]["openclaw.tokens.input"] for span in spans] == [10, 20]
    assert [span["attrs"]["openclaw.tokens.output"] for span in spans] == [5, 7]
    assert all(span["attrs"]["openclaw.channel"] == "synth.raw_response" for span in spans)


def test_openclaw_cache_trace_synthesizes_per_request_model_usage(tmp_path):
    import json

    from agent_bench.benchmarks.conditions.openclaw_agent import OpenClawAgentCondition

    run_dir = tmp_path / "run-1"
    trace_dir = run_dir / "conversation"
    trace_dir.mkdir(parents=True)
    spans_path = run_dir / "spans.jsonl"
    existing_tool_spans = [
        {
            "name": "agent.tool.call",
            "start_unix_nano": 1780272001000000000,
            "end_unix_nano": 1780272001500000000,
            "attrs": {},
        },
        {
            "name": "agent.tool.call",
            "start_unix_nano": 1780272002600000000,
            "end_unix_nano": 1780272002800000000,
            "attrs": {},
        },
    ]
    spans_path.write_text("".join(json.dumps(row) + "\n" for row in existing_tool_spans))
    rows = [
        {
            "stage": "stream:context",
            "ts": "2026-06-01T00:00:00.000Z",
            "runId": "oc-run-1",
            "sessionId": "sess-1",
            "sessionKey": "agent:test:main",
            "provider": "provider-a",
            "modelId": "model-a",
            "modelApi": "openai-completions",
        },
        {
            "stage": "stream:context",
            "ts": "2026-06-01T00:00:02.000Z",
            "runId": "oc-run-1",
            "sessionId": "sess-1",
            "sessionKey": "agent:test:main",
            "provider": "provider-a",
            "modelId": "model-a",
            "modelApi": "openai-completions",
        },
        {
            "stage": "session:after",
            "runId": "oc-run-1",
            "sessionId": "sess-1",
            "sessionKey": "agent:test:main",
            "provider": "provider-a",
            "modelId": "model-a",
            "modelApi": "openai-completions",
            "messages": [
                {
                    "role": "assistant",
                    "api": "openai-completions",
                    "provider": "provider-a",
                    "model": "model-a",
                    "usage": {
                        "input": 10,
                        "output": 5,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 15,
                    },
                    "stopReason": "toolUse",
                    "timestamp": 1780272001000,
                    "responseId": "resp-1",
                },
                {"role": "toolResult", "content": []},
                {
                    "role": "assistant",
                    "api": "openai-completions",
                    "provider": "provider-a",
                    "model": "model-a",
                    "usage": {
                        "input": 4,
                        "output": 7,
                        "cacheRead": 20,
                        "cacheWrite": 0,
                        "totalTokens": 31,
                    },
                    "stopReason": "stop",
                    "timestamp": 1780272002600,
                    "responseId": "resp-2",
                },
            ],
        },
    ]
    trace_path = trace_dir / "cache-trace.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    count = OpenClawAgentCondition._maybe_synthesize_model_usage_spans_from_cache_trace(
        spans_path, run_dir, "run-1"
    )

    spans = [
        json.loads(line)
        for line in spans_path.read_text().splitlines()
        if json.loads(line).get("name") == "openclaw.model.usage"
    ]
    assert count == 2
    assert len(spans) == 2
    assert [span["attrs"]["openclaw.llm.request_index"] for span in spans] == [1, 2]
    assert [span["duration_ms"] for span in spans] == [1000.0, 600.0]
    assert [span["attrs"]["openclaw.latency.source"] for span in spans] == [
        "next_tool_start",
        "next_tool_start",
    ]
    assert [span["attrs"]["openclaw.tokens.input"] for span in spans] == [10, 4]
    assert [span["attrs"]["openclaw.tokens.output"] for span in spans] == [5, 7]
    assert [span["attrs"]["openclaw.tokens.cache_read"] for span in spans] == [0, 20]
    assert all(span["attrs"]["openclaw.channel"] == "synth.cache_trace" for span in spans)
