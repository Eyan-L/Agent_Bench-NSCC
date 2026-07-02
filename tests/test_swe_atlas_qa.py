from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_bench.benchmarks.swe_atlas_qa import loader
from agent_bench.benchmarks.swe_atlas_qa.scoring import score_answer
from agent_bench.core.execution import materialize_benchmark_data
from agent_bench.core.spec import load_resolved_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_swe_atlas_qa_loader_normalization() -> None:
    raw_row = {
        "instance_id": "test_id_1",
        "question": "What is the capital of France?",
        "reference_answer": "Paris",
        "repository_url": "https://github.com/test/repo",
        "repository_base_commit": "abcdef123456",
        "docker_image": "test/image:latest",
        "rubric": {
            "positive_verifiers": ["The answer must mention Paris."]
        }
    }
    normalized = loader.normalize_instance(raw_row)
    assert normalized["benchmark"] == "swe_atlas_qa"
    assert normalized["instance_id"] == "test_id_1"
    assert normalized["repo"] == "https://github.com/test/repo"
    assert normalized["base_commit"] == "abcdef123456"
    assert normalized["problem_statement"] == "What is the capital of France?"


def test_swe_atlas_qa_materialization_sets_env() -> None:
    resolved = load_resolved_experiment(
        ROOT / "experiments/swe_atlas_qa_smoke_openclaw.toml",
        root=ROOT,
    )
    env: dict[str, str] = {}
    path = materialize_benchmark_data(resolved, env=env)

    assert path.name == "swe_atlas_qa_smoke_instances.json"
    assert env["AB_BENCHMARK_INSTANCE_DATA"] == str(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows[0]["benchmark"] == "swe_atlas_qa"
    assert rows[0]["instance_id"] == "6905333b74f22949d97ba9e3"


def test_score_answer_logic_with_rubric(monkeypatch) -> None:
    # Mock load_active_model_and_serving to avoid network dependencies
    class FakeModelConfig:
        endpoint = "http://localhost:8000/v1"
        api_key = "test_key"
        name = "test_model"

    monkeypatch.setattr(
        "agent_bench.benchmarks.swe_atlas_qa.scoring.load_active_model_and_serving",
        lambda: (FakeModelConfig(), None),
    )

    # Mock grade_verifier to simulate YES response
    monkeypatch.setattr(
        "agent_bench.benchmarks.swe_atlas_qa.scoring.grade_verifier",
        lambda *args, **kwargs: True,
    )

    instance_data = {
        "question": "Question?",
        "reference_answer": "Ref",
        "rubric": {
            "positive_verifiers": ["Must contain Paris"]
        }
    }

    verdict = score_answer(instance_data, "Paris is the capital.")
    assert verdict["status"] == "PASS"
    assert verdict["passed"] == 1
    assert verdict["failed"] == 0


def test_swe_atlas_qa_run_tests_with_tags(tmp_path: Path, monkeypatch) -> None:
    from agent_bench.benchmarks.swe_atlas_qa.openclaw.app import SWEAtlasQAHarness
    from types import SimpleNamespace

    harness = SWEAtlasQAHarness.__new__(SWEAtlasQAHarness)
    harness.workspace_base = tmp_path / "workspaces"
    harness.agent_id = "swe_atlas_qa"
    raw_content = (
        "Some reasoning text here.\n"
        "<<FINAL_ANSWER>>\n"
        "This is the actual final response graded by the rubric.\n"
        "<<FINAL_ANSWER>>\n"
        "Some post-answer details."
    )
    harness.container = SimpleNamespace(
        backend_id="docker",
        exec=lambda handle, argv, **kwargs: SimpleNamespace(returncode=0, stdout=raw_content)
    )

    # Mock scoring.score_answer to check the parsed answer
    parsed_answers = []
    def fake_score_answer(instance_data, answer_text):
        parsed_answers.append(answer_text)
        return {"status": "PASS", "test_pass_rate": 1.0, "passed": 1, "failed": 0}

    monkeypatch.setattr(
        "agent_bench.benchmarks.swe_atlas_qa.scoring.score_answer",
        fake_score_answer,
    )

    verdict = harness.run_tests(
        container_id="unused",
        instance_id="swe_atlas_qa_smoke_1",
        workspace_root="/app",
        timeout_seconds=10,
        run_id="smoke-run",
        instance_data={"rubric": {}},
        run_dir=tmp_path / "run-dir",
    )

    assert verdict["status"] == "PASS"
    assert len(parsed_answers) == 1
    assert parsed_answers[0] == "This is the actual final response graded by the rubric."

