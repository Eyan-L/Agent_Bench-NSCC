"""Static integration checks for the Hermes SWE-bench harness."""

import json
from pathlib import Path
from types import SimpleNamespace

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition
from agent_bench.core.profile_io import load_profile


ROOT = Path(__file__).resolve().parents[2]


def test_hermes_patch_manifest_and_backend_contract():
    manifest = ROOT / "agent_bench/harnesses/hermes/patch/MANIFEST.json"
    patch = ROOT / "agent_bench/harnesses/hermes/patch/hermes-external-exec-bridge.patch"
    telemetry_patch = ROOT / "agent_bench/harnesses/hermes/patch/hermes-agent-bench-telemetry.patch"
    assert manifest.is_file()
    assert patch.is_file()
    assert telemetry_patch.is_file()
    text = patch.read_text()
    assert "tools/environments/docker.py" in text
    assert "AB_HARNESS_HERMES_PATCH_EXEC_BRIDGE_BEGIN" in text
    assert "AB_HARNESS_HERMES_PATCH_EXEC_BRIDGE_RUN_BEGIN" in text
    assert "HERMES_EXEC_BRIDGE" in text
    telemetry_text = telemetry_patch.read_text()
    assert "AB_HARNESS_HERMES_PATCH_TELEMETRY_BEGIN" in telemetry_text
    assert "AB_AGENT_RUNTIME_EVENTS_FILE" in telemetry_text
    assert "_hm_ab_coerce_json_string_args" in telemetry_text
    assert "emit_model_usage_delta" in telemetry_text
    assert '"request_index": usage_events_written' in telemetry_text


def test_hermes_docker_smoke_checks_patch_status():
    smoke = (ROOT / "agent_bench/harnesses/hermes/docker/smoke.sh").read_text()
    assert "HERMES_PATCH_MODE=baked" in smoke
    assert "/opt/hermes-tools/apply_patch.py status" in smoke
    assert "DockerEnvironment" in smoke
    assert "fake-hm-bridge.py" in smoke


def test_hermes_app_uses_external_task_container_and_bridge():
    app = (ROOT / "agent_bench/benchmarks/swebench/hermes/app.py").read_text()
    assert "task_container_name = _oc_sandbox_name" in app
    assert "HermesDispatcher()" in app
    assert "target=task_container_name" in app
    assert "ContainerSpec" in app
    assert "self.container.start" in app
    assert '"TERMINAL_ENV": "docker"' in app
    assert "TERMINAL_DOCKER_IMAGE" in app
    assert 'os.environ.get("OPENAI_API_KEY")' in app
    assert 'os.environ.get("OPENAI_BASE_URL")' in app
    assert '"HERMES_INFERENCE_PROVIDER": provider' in app
    assert "AB_AGENT_RUNTIME_EVENTS_FILE" in app
    assert 'f"{_STATE_DIR_IN_HERMES}/events.jsonl"' in app
    assert "MINIMAX_API_KEY" in app
    assert "otel_env(" in app
    assert 'service_name="agent-bench-hermes"' in app
    assert 'extra_attrs={"hm.run_id": run_id}' in app
    assert "/var/run/docker.sock" not in app


def test_hermes_driver_and_condition_are_registered():
    driver = (ROOT / "agent_bench/runtime/driver.py").read_text()
    assert 'condition_type == "hermes_agent"' in driver
    assert 'harness_choice == "hermes"' in driver
    assert "SingularityContainer()" in driver
    assert "SWEBenchHermesHarness(benchmark_data_root, container=container)" in driver
    condition = (ROOT / "agent_bench/benchmarks/swebench/conditions/hermes_agent.py").read_text()
    assert '_resolve_provider(self.model_config)' in condition
    assert '"--provider", provider' in condition
    cfg = load_profile(ROOT / "profiles/harnesses/hermes_swebench.yaml")
    assert cfg["harness"] == "hermes"
    assert cfg["conditions"][0]["condition_type"] == "hermes_agent"


def test_hermes_condition_writes_agent_namespace_spans(tmp_path):
    condition = HermesAgentCondition(
        model_config=SimpleNamespace(name="MiniMax-M2.7"),
        agent_config=SimpleNamespace(),
    )
    spans_path = tmp_path / "spans.jsonl"

    condition._write_spans(
        spans_path,
        run_id="run-1",
        session_id="run-1",
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        model_name="MiniMax-M2.7",
        provider="minimax",
        exit_code=1,
    )

    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    raw = json.loads((tmp_path / "spans_raw.jsonl").read_text())
    assert raw["resourceSpans"]
    assert [span["canonical_name"] for span in spans] == [
        "agent.model.usage",
        "agent.tool.call",
    ]

    model_attrs = spans[0]["attrs"]
    assert model_attrs["agent.provider"] == "minimax"
    assert model_attrs["agent.model"] == "MiniMax-M2.7"
    assert model_attrs["agent.tokens.total"] == 0
    assert model_attrs["agent.channel"] == "synth.agent_call"
    assert model_attrs["agent.llm.request_index"] == 1

    tool_attrs = spans[1]["attrs"]
    assert tool_attrs["agent.tool.name"] == "hermes.agent"
    assert tool_attrs["agent.tool.call_id"] == "run-1:hermes-agent"
    assert tool_attrs["agent.tool.is_error"] is True
    assert tool_attrs["agent.run_id"] == "run-1"
    assert tool_attrs["agent.session_id"] == "run-1"


def test_hermes_condition_turns_agent_events_into_usage_and_tool_spans(tmp_path):
    condition = HermesAgentCondition(
        model_config=SimpleNamespace(name="fallback-model"),
        agent_config=SimpleNamespace(),
    )
    spans_path = tmp_path / "spans.jsonl"
    events = [
        {
            "type": "model.usage",
            "start_ns": 1_000_000_000,
            "end_ns": 2_000_000_000,
            "provider": "minimax",
            "model": "MiniMax-M2.7",
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_tokens": 3,
            "cache_write_tokens": 2,
            "total_tokens": 23,
            "api_calls": 2,
        },
        {
            "type": "tool.call",
            "start_ns": 1_200_000_000,
            "end_ns": 1_500_000_000,
            "duration_ms": 300.0,
            "run_id": "run-1",
            "session_id": "run-1",
            "tool_name": "terminal",
            "tool_call_id": "call-1",
            "is_error": False,
            "result_chars": 42,
        },
    ]
    usage = condition._aggregate_usage(events)

    condition._write_spans(
        spans_path,
        run_id="run-1",
        session_id="run-1",
        start_ns=900_000_000,
        end_ns=2_100_000_000,
        model_name="fallback-model",
        provider="fallback-provider",
        exit_code=0,
        events=events,
        usage=usage,
    )

    spans = [json.loads(line) for line in spans_path.read_text().splitlines()]
    raw = json.loads((tmp_path / "spans_raw.jsonl").read_text())
    assert raw["resourceSpans"]
    assert [span["canonical_name"] for span in spans] == [
        "agent.model.usage",
        "agent.tool.call",
    ]
    assert spans[0]["attrs"]["agent.provider"] == "minimax"
    assert spans[0]["attrs"]["agent.model"] == "MiniMax-M2.7"
    assert spans[0]["attrs"]["agent.tokens.input"] == 11
    assert spans[0]["attrs"]["agent.tokens.output"] == 7
    assert spans[0]["attrs"]["agent.tokens.cache_read"] == 3
    assert spans[0]["attrs"]["agent.tokens.cache_write"] == 2
    assert spans[0]["attrs"]["agent.tokens.total"] == 23
    assert spans[0]["attrs"]["agent.channel"] == "hermes.events"
    assert spans[0]["attrs"]["agent.llm.request_index"] == 1
    assert spans[1]["attrs"]["agent.tool.name"] == "terminal"
    assert spans[1]["attrs"]["agent.tool.call_id"] == "call-1"
    assert spans[1]["attrs"]["agent.tool.result_chars"] == 42


def test_hermes_condition_reads_standard_events_filename(tmp_path):
    condition = HermesAgentCondition(
        model_config=SimpleNamespace(name="fallback-model"),
        agent_config=SimpleNamespace(),
    )
    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    state_dir.mkdir()
    run_dir.mkdir()
    (state_dir / "events.jsonl").write_text(
        json.dumps({"type": "model.usage", "input_tokens": 2}) + "\n"
    )

    events = condition._read_hermes_events(
        {"state_dir": str(state_dir)},
        run_dir,
    )

    assert events == [{"type": "model.usage", "input_tokens": 2}]
    assert (run_dir / "hermes_events.jsonl").is_file()
    assert not (run_dir / "hermes_agent_events.jsonl").exists()


def test_hermes_config_template_selects_docker_backend():
    template = (ROOT / "agent_bench/harnesses/hermes/state/config.yaml.template").read_text()
    assert 'backend: "docker"' in template
    assert 'provider: "$PROVIDER"' in template
    assert 'cwd: "$TERMINAL_CWD"' in template
    assert "docker_image" in template
    assert "terminal" in template
    assert "platform_toolsets" in template
