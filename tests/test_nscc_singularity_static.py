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
    assert "OH_RUNTIME_SIF" in app
    assert "OC_SCORING_SIF_DIR" in app
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
    assert "HM_RUNTIME_SIF" in app
    assert "OC_SCORING_SIF_DIR" in app
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
    assert "OC_SCORING_URL" in helper
    assert "/v1/submit" in helper
    assert "/v1/submission/" in helper


def test_swebench_harnesses_can_score_with_singularity_sifs():
    helper = (ROOT / "agent_bench/benchmarks/swebench/scoring.py").read_text()
    assert "def score_via_singularity" in helper
    assert "OC_SCORING_SIF_DIR" in helper
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
    assert 'os.environ.get("MODEL_ENDPOINT")' in app
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

    assert "OH_RUNTIME_SIF" in stages
    assert "HM_RUNTIME_SIF" in stages
    assert "OH_RUNTIME_BASE_DOCKER" in stages
    assert "HM_RUNTIME_BASE_DOCKER" in stages
    assert "build_oh_sif.sh" in stages
    assert "build_hm_sif.sh" in stages
    assert 'harness == "openhands"' in stages
    assert 'harness == "hermes"' in stages
    assert 'out["OH_IMAGE"] = env.get("OH_IMAGE") or out["OH_RUNTIME_SIF"]' in stages
    assert 'out["HM_IMAGE"] = env.get("HM_IMAGE") or out["HM_RUNTIME_SIF"]' in stages
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

    assert '"OTEL_RESOURCE_ATTRIBUTES": f"agent.run_id={run_id}"' in app
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
        "KIT_ROOT": str(ROOT),
        "USER": "alice",
        "SCRATCH": str(tmp_path / "scratch"),
        "AGENT_BENCH_RUN_DIR": str(tmp_path / "run"),
        "AGENT_BENCH_LOG_DIR": str(tmp_path / "run/logs"),
    }
    spec = build_pbs_job_spec(resolved, env=env)

    prepare_environment(resolved, spec, env)

    assert env["AB_EXPERIMENT_DIR"] == str(tmp_path / "run")
    assert env["OC_LOG_DIR"] == str(tmp_path / "run/logs")
    assert env["AGENT_BENCH_LOG_DIR"] == str(tmp_path / "run/logs")


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
