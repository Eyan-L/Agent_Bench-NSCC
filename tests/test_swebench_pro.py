from __future__ import annotations

import json
import os
from pathlib import Path

from agent_bench.benchmarks.swebench_pro.common import (
    get_dockerhub_image_uri,
    normalize_instance,
)
from agent_bench.benchmarks.swebench_pro.hermes.app import SWEBenchProHermesHarness
from agent_bench.benchmarks.swebench_pro.loader import attach_official_resources
from agent_bench.benchmarks.swebench_pro.openclaw.app import SWEBenchProHarness
from agent_bench.benchmarks.swebench_pro.openhands.app import SWEBenchProOpenHandsHarness
from agent_bench.benchmarks.swebench_pro.scoring import (
    assemble_workspace_files,
    create_entryscript,
    parse_output_verdict,
)
from agent_bench.core.spec import load_resolved_experiment
from agent_bench.runtime.driver import _attach_swebench_pro_runtime_resources


ROOT = Path(__file__).resolve().parents[1]


class _FakeContainer:
    backend_id = "docker"
    host_address = "127.0.0.1"

    def check_available(self) -> None:
        pass

    def check_image(self, image: str) -> None:
        pass


def test_swebench_pro_image_uri_matches_public_helper() -> None:
    assert get_dockerhub_image_uri(
        "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
        repo_name="NodeBB/NodeBB",
    ) == (
        "jefzda/sweap-images:"
        "nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5"
    )
    assert get_dockerhub_image_uri(
        "instance_element-hq__element-web-b007ea81b2ccd001b00f332bee65070aa7fc00f9-vnan",
        repo_name="element-hq/element-web",
    ).startswith("jefzda/sweap-images:element-hq.element-element-hq__element-web-b007")


def test_swebench_pro_normalizes_test_fields() -> None:
    inst = normalize_instance({
        "instance_id": "instance_demo__repo-abc-vnan",
        "repo": "demo/repo",
        "base_commit": "abc",
        "FAIL_TO_PASS": "['test_a']",
        "PASS_TO_PASS": ["test_b"],
        "selected_test_files_to_run": "['tests/a.py']",
    })

    assert inst["benchmark"] == "swebench_pro"
    assert inst["fail_to_pass"] == ["test_a"]
    assert inst["pass_to_pass"] == ["test_b"]
    assert inst["selected_test_files_to_run"] == ["tests/a.py"]
    assert inst["container_image"].startswith("jefzda/sweap-images:")


def test_swebench_pro_verdict_requires_all_required_tests(tmp_path: Path) -> None:
    inst = normalize_instance({
        "instance_id": "instance_demo__repo-abc",
        "repo": "demo/repo",
        "base_commit": "abc",
        "fail_to_pass": ["test_a"],
        "pass_to_pass": ["test_b"],
    })
    output = {
        "tests": [
            {"name": "test_a", "status": "PASSED"},
            {"name": "test_b", "status": "FAILED"},
        ]
    }
    verdict = parse_output_verdict(
        output,
        inst,
        evaluation_dir=tmp_path,
        output_path=tmp_path / "output.json",
    )

    assert verdict["status"] == "FAIL"
    assert verdict["passed"] == 1
    assert verdict["failed"] == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report[inst["instance_id"]]["tests_status"]["PASS_TO_PASS"]["failure"] == ["test_b"]


def test_swebench_pro_materialized_rows_embed_official_resources(tmp_path: Path) -> None:
    instance_id = "instance_demo__repo-abc"
    resource_root = tmp_path / "resources"
    run_dir = resource_root / "run_scripts" / instance_id
    base_dir = resource_root / "dockerfiles" / "base_dockerfile" / instance_id
    instance_dir = resource_root / "dockerfiles" / "instance_dockerfile" / instance_id
    run_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)
    instance_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("echo running\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")
    (base_dir / "Dockerfile").write_text("FROM scratch\nENV BASE_FLAG=1\n", encoding="utf-8")
    (instance_dir / "Dockerfile").write_text("ENV INSTANCE_FLAG=2\n", encoding="utf-8")

    row = normalize_instance({
        "instance_id": instance_id,
        "repo": "demo/repo",
        "base_commit": "abc",
    })
    embedded = attach_official_resources([row], resource_root=resource_root)[0]
    assert embedded["run_script"] == "echo running\n"
    assert embedded["parsing_script"] == "print('parse')\n"
    assert "BASE_FLAG=1" in embedded["base_dockerfile"]
    assert "INSTANCE_FLAG=2" in embedded["instance_dockerfile"]

    files = assemble_workspace_files(
        embedded,
        "diff --git a/file b/file\n",
        benchmark_data_root=tmp_path / "empty_benchmark_data_root",
    )
    assert files["run_script.sh"] == "echo running\n"
    assert files["parser.py"] == "print('parse')\n"
    assert "export BASE_FLAG=1" in files["entryscript.sh"]
    assert "export INSTANCE_FLAG=2" in files["entryscript.sh"]


def test_swebench_pro_entryscript_preserves_before_repo_set_cmd_order(tmp_path: Path) -> None:
    inst = normalize_instance({
        "instance_id": "instance_demo__repo-abc",
        "repo": "demo/repo",
        "base_commit": "base123",
        "before_repo_set_cmd": "\n".join([
            "git reset --hard base123",
            "git clean -fd",
            "git checkout base123",
            "git checkout gold456 -- tests/test_demo.py",
            "python scripts/prepare_hidden_tests.py",
        ]),
        "selected_test_files_to_run": ["tests/test_demo.py"],
    })

    script = create_entryscript(inst, tmp_path)

    reset_idx = script.index("git reset --hard base123")
    clean_idx = script.index("git clean -fd")
    base_checkout_idx = script.index("git checkout base123")
    apply_idx = script.index("git apply -v /workspace/patch.diff")
    test_checkout_idx = script.index("git checkout gold456 -- tests/test_demo.py")
    prepare_idx = script.index("python scripts/prepare_hidden_tests.py")
    run_idx = script.index("bash /workspace/run_script.sh tests/test_demo.py")

    assert script.count("git reset --hard base123") == 1
    assert script.count("git checkout base123") == 1
    assert reset_idx < clean_idx < base_checkout_idx < apply_idx
    assert apply_idx < test_checkout_idx < prepare_idx < run_idx


def test_swebench_pro_runtime_loader_backfills_explicit_instance_data(tmp_path: Path) -> None:
    instance_id = "instance_demo__repo-abc"
    resource_root = tmp_path / "resources"
    run_dir = resource_root / "run_scripts" / instance_id
    run_dir.mkdir(parents=True)
    (resource_root / "dockerfiles").mkdir()
    (run_dir / "run_script.sh").write_text("echo runtime\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('runtime parse')\n", encoding="utf-8")
    row = normalize_instance({
        "instance_id": instance_id,
        "repo": "demo/repo",
        "base_commit": "abc",
    })

    previous = os.environ.get("AB_SWEBENCH_PRO_ROOT")
    os.environ["AB_SWEBENCH_PRO_ROOT"] = str(resource_root)
    try:
        loaded = _attach_swebench_pro_runtime_resources([row])[0]
    finally:
        if previous is None:
            os.environ.pop("AB_SWEBENCH_PRO_ROOT", None)
        else:
            os.environ["AB_SWEBENCH_PRO_ROOT"] = previous

    assert loaded["run_script"] == "echo runtime\n"
    assert loaded["parsing_script"] == "print('runtime parse')\n"


def test_swebench_pro_experiment_profiles_resolve() -> None:
    for name in (
        "swebench_pro_smoke_openclaw.toml",
        "swebench_pro_smoke_openhands.toml",
        "swebench_pro_smoke_hermes.toml",
        "swebench_pro_subset10_openclaw.toml",
        "swebench_pro_subset10_openhands.toml",
        "swebench_pro_subset10_hermes.toml",
        "swebench_pro_subset10_openclaw_sglang.toml",
        "swebench_pro_subset10_openhands_sglang.toml",
        "swebench_pro_subset10_hermes_sglang.toml",
    ):
        resolved = load_resolved_experiment(ROOT / "experiments" / name, root=ROOT)
        assert resolved.spec.benchmark_name == "swebench_pro"
        assert resolved.benchmark["application"] == "swebench_pro"


def test_swebench_pro_subset_profiles_are_nested_and_complete() -> None:
    previous: set[str] = set()
    for size in (5, 10, 20, 50):
        path = ROOT / "profiles" / "benchmarks" / "swebench_pro" / f"subset{size}.json"
        profile = json.loads(path.read_text())
        instances = profile["instances"]
        repo_mapping = profile["repo_mapping"]

        assert profile["application"] == "swebench_pro"
        assert profile["dataset"] == "ScaleAI/SWE-bench_Pro"
        assert profile["split"] == "test"
        assert profile["container_image"] == "swebench_pro/UNUSED:latest"
        assert profile["selection"]["source_dataset_rows"] == 731
        assert len(instances) == size
        assert len(set(instances)) == size
        assert set(instances) == set(repo_mapping)
        assert previous <= set(instances)
        for instance_id in instances:
            item = repo_mapping[instance_id]
            assert item["repo_id"]
            assert item["base_commit"]
        previous = set(instances)


def test_swebench_pro_harnesses_do_not_require_swebench_package(tmp_path: Path) -> None:
    for cls in (
        SWEBenchProHarness,
        SWEBenchProOpenHandsHarness,
        SWEBenchProHermesHarness,
    ):
        harness = cls(tmp_path, container=_FakeContainer())
        assert harness.requires_swebench_package is False
        assert harness.task_workspace == "/app"
        assert harness.agent_id == "swebench_pro"


def test_swebench_pro_hermes_harness_builds_container_handle(tmp_path: Path) -> None:
    harness = SWEBenchProHermesHarness(tmp_path, container=_FakeContainer())

    handle = harness._handle("task-container")

    assert handle is not None
    assert handle.id == "task-container"
    assert handle.name == "task-container"
    assert handle.backend == "docker"


def test_swebench_pro_harnesses_default_workspace_to_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    previous_workspace = os.environ.get("AB_HARNESS_WORKSPACE_BASE")
    previous_agent_run = os.environ.get("AGENT_BENCH_RUN_DIR")
    previous_ab_run = os.environ.get("AB_RUN_DIR")
    os.environ.pop("AB_HARNESS_WORKSPACE_BASE", None)
    os.environ.pop("AB_RUN_DIR", None)
    os.environ["AGENT_BENCH_RUN_DIR"] = str(run_dir)
    try:
        for cls in (
            SWEBenchProHarness,
            SWEBenchProOpenHandsHarness,
            SWEBenchProHermesHarness,
        ):
            harness = cls(tmp_path, container=_FakeContainer())
            assert harness.workspace_base == run_dir / "workspaces"
    finally:
        if previous_workspace is None:
            os.environ.pop("AB_HARNESS_WORKSPACE_BASE", None)
        else:
            os.environ["AB_HARNESS_WORKSPACE_BASE"] = previous_workspace
        if previous_agent_run is None:
            os.environ.pop("AGENT_BENCH_RUN_DIR", None)
        else:
            os.environ["AGENT_BENCH_RUN_DIR"] = previous_agent_run
        if previous_ab_run is None:
            os.environ.pop("AB_RUN_DIR", None)
        else:
            os.environ["AB_RUN_DIR"] = previous_ab_run


def test_swebench_pro_openclaw_does_not_mask_runtime_app_dir(tmp_path: Path) -> None:
    harness = SWEBenchProHarness(tmp_path, container=_FakeContainer())

    assert harness._openclaw_runtime_workspace_mount(tmp_path) is None
    assert harness._openclaw_runtime_workdir() == "/root"
