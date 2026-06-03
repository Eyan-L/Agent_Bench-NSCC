from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


from agent_bench.benchmarks.request_bench import loader
import agent_bench.benchmarks.request_bench.app as request_app
from agent_bench.benchmarks.request_bench.app import RequestBenchHarness, build_answer_verdict
from agent_bench.core.execution import build_execution_plan, materialize_benchmark_data
from agent_bench.core.profile_io import load_profile
from agent_bench.core.spec import load_resolved_experiment
import agent_bench.runtime.driver as driver

ROOT = Path(__file__).resolve().parents[1]


def test_request_bench_profile_materializes_instance(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "profiles/benchmarks/request_bench/llm_comparison.json")
    out = tmp_path / "instances.json"

    loader.materialize_profile(profile, out)

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["benchmark"] == "request_bench"
    assert row["instance_id"] == "llm-comparison-table"
    assert "comparison table" in row["prompt"].lower()
    assert row["answer_file"] == "/workspace/answer.md"


def test_request_bench_core_materialization_sets_env() -> None:
    resolved = load_resolved_experiment(
        ROOT / "experiments/client_only_request_bench_openclaw.toml",
        root=ROOT,
    )
    env: dict[str, str] = {}

    path = materialize_benchmark_data(resolved, env=env)

    assert path.name == "request_bench_llm_comparison_instances.json"
    assert env["AB_BENCHMARK_INSTANCE_DATA"] == str(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows[0]["benchmark"] == "request_bench"


def test_request_bench_answer_verdict_passes_for_non_empty_answer(tmp_path: Path) -> None:
    answer = tmp_path / "workspace" / "answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("| Model | Use case |\n| --- | --- |\n", encoding="utf-8")

    verdict = build_answer_verdict(
        instance_id="llm-comparison-table",
        answer_path=answer,
        output_dir=tmp_path / "verifier",
    )

    assert verdict["status"] == "PASS"
    assert verdict["answer_text"].startswith("| Model")
    assert Path(verdict["answer_path"]).read_text(encoding="utf-8").startswith("| Model")


def test_request_bench_answer_verdict_fails_for_empty_answer(tmp_path: Path) -> None:
    answer = tmp_path / "workspace" / "answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("\n", encoding="utf-8")

    verdict = build_answer_verdict(
        instance_id="llm-comparison-table",
        answer_path=answer,
        output_dir=tmp_path / "verifier",
    )

    assert verdict["status"] == "FAIL"
    assert verdict["failed"] == 1
    assert verdict["error_message"] == "empty answer.md"


def test_request_bench_experiments_resolve_and_dry_run_plan() -> None:
    for experiment, harness in [
        ("client_only_request_bench_openclaw", "openclaw"),
        ("client_only_request_bench_hermes", "hermes"),
    ]:
        resolved = load_resolved_experiment(ROOT / "experiments" / f"{experiment}.toml", root=ROOT)
        plan = build_execution_plan(resolved, env={"KIT_ROOT": str(ROOT), "AB_RUN_STAMP": "test-stamp"})

        assert resolved.spec.benchmark_name == "request_bench"
        assert resolved.spec.harness_name == harness
        assert plan.stage_enabled("materialize_benchmark") is True
        assert plan.stage_enabled("run_benchmark") is True



def test_request_bench_run_sweep_does_not_require_benchmark_data_root(tmp_path: Path, monkeypatch) -> None:
    profile = load_profile(ROOT / "profiles/benchmarks/request_bench/llm_comparison.json")
    instance_data = tmp_path / "instances.json"
    loader.materialize_profile(profile, instance_data)
    constructed: list[Path | None] = []

    class FakeRequestBenchHarness:
        def __init__(self, benchmark_data_root=None):
            constructed.append(benchmark_data_root)

    import agent_bench.benchmarks.request_bench.app as request_app

    monkeypatch.setattr(request_app, "RequestBenchHarness", FakeRequestBenchHarness)
    monkeypatch.setattr(
        driver,
        "execute_run",
        lambda run_config, harness, instance, retry_errors: SimpleNamespace(status="PASS"),
    )
    monkeypatch.setenv("AB_RUN_STAMP", "request-bench-test")
    monkeypatch.delenv("AB_DEPLOYMENT_CONTAINER_BACKEND", raising=False)

    driver.run_sweep(
        benchmark_path=ROOT / "profiles/benchmarks/request_bench/llm_comparison.json",
        harness_path=ROOT / "profiles/harnesses/openclaw_request_bench.yaml",
        model_path=ROOT / "profiles/models/qwen36_35b_a3b.json",
        serving_path=ROOT / "profiles/serving/litellm/qwen36_35b_a3b.yaml",
        benchmark_data_root=None,
        instance_data_path=instance_data,
        retry_errors=True,
        max_workers=1,
    )

    assert constructed == [None]



def test_request_bench_state_dir_uses_rebased_active_model_signature(tmp_path: Path, monkeypatch) -> None:
    harness = RequestBenchHarness.__new__(RequestBenchHarness)
    harness.state_base = tmp_path / "states"
    harness.state_base.mkdir()
    monkeypatch.setenv("AB_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(ROOT / "profiles/harnesses/openclaw_request_bench.yaml"))
    monkeypatch.setenv("AB_MODEL_CONFIG", str(ROOT / "profiles/models/qwen36_35b_a3b.json"))
    monkeypatch.setenv("AB_SERVING_CONFIG", str(ROOT / "profiles/serving/litellm/qwen36_35b_a3b.yaml"))

    state_dir = harness._build_state_dir(
        run_id="request-state-dir-test",
        workspace_root="/workspace",
        sandbox_image="python:3.13-slim-bookworm",
        sandbox_binds=[],
        container_prefix="oc-sbx-test-",
        agent_id="request_writer",
    )

    cfg = json.loads((state_dir / "openclaw.json").read_text(encoding="utf-8"))
    assert cfg["agents"]["defaults"]["workspace"] == "/workspace"
    assert cfg["models"]["providers"]["vllm"]["baseUrl"].startswith("http://vllm-host:")



def test_request_bench_run_tests_accepts_driver_run_dir(tmp_path: Path) -> None:
    harness = RequestBenchHarness.__new__(RequestBenchHarness)
    run_id = "request-run-tests"
    harness.workspace_base = tmp_path / "workspaces"
    answer = harness.workspace_base / run_id / "workspace" / "answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("request answer\n", encoding="utf-8")
    run_dir = tmp_path / "run-dir"

    verdict = harness.run_tests(
        container_id="unused",
        instance_id="llm-comparison-table",
        workspace_root="/workspace",
        timeout_seconds=10,
        run_id=run_id,
        instance_data={"answer_file": "/workspace/answer.md"},
        run_dir=run_dir,
    )

    assert verdict["status"] == "PASS"
    assert Path(verdict["answer_path"]) == run_dir / "request_bench" / "answer.md"
    assert Path(verdict["answer_path"]).read_text(encoding="utf-8") == "request answer\n"



def test_request_bench_workspace_seeds_bootstrap(tmp_path: Path) -> None:
    request_app._write_prompt_workspace(
        tmp_path,
        {
            "instance_id": "llm-comparison-table",
            "description": "Smoke task",
            "prompt": "Write a comparison table.",
        },
    )

    assert (tmp_path / "PROMPT.md").is_file()
    assert (tmp_path / "AGENTS.md").is_file()
    bootstrap = (tmp_path / "BOOTSTRAP.md").read_text(encoding="utf-8")
    assert "/workspace/PROMPT.md" in bootstrap
    assert "/workspace/answer.md" in bootstrap
