"""Static checks for NSCC singularity-capable harness wiring."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_openhands_uses_container_backend_for_nscc():
    app = (ROOT / "applications/swebench/openhands/app.py").read_text()
    condition = (ROOT / "applications/swebench/conditions/openhands_agent.py").read_text()
    dispatcher = (ROOT / "orchestration/dispatcher/openhands.py").read_text()

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
    app = (ROOT / "applications/swebench/hermes/app.py").read_text()
    condition = (ROOT / "applications/swebench/conditions/hermes_agent.py").read_text()
    dispatcher = (ROOT / "orchestration/dispatcher/hermes.py").read_text()

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
        "applications/swebench/openclaw/app.py",
        "applications/swebench/openhands/app.py",
        "applications/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert '"HOME": "/root"' in app


def test_swebench_harnesses_can_use_remote_scoring_server():
    for rel in (
        "applications/swebench/openclaw/app.py",
        "applications/swebench/openhands/app.py",
        "applications/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert "score_via_scoring_server" in app
    helper = (ROOT / "applications/swebench/scoring.py").read_text()
    assert "OC_SCORING_URL" in helper
    assert "/v1/submit" in helper
    assert "/v1/submission/" in helper


def test_swebench_harnesses_can_score_with_singularity_sifs():
    helper = (ROOT / "applications/swebench/scoring.py").read_text()
    assert "def score_via_singularity" in helper
    assert "OC_SCORING_SIF_DIR" in helper
    assert '["singularity", "exec"' in helper
    assert "get_eval_report" in helper

    for rel in (
        "applications/swebench/openclaw/app.py",
        "applications/swebench/openhands/app.py",
        "applications/swebench/hermes/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert "score_via_singularity" in app
        assert app.index("score_via_singularity(") < app.index("score_via_scoring_server(")


def test_openclaw_uses_runtime_serving_endpoint():
    app = (ROOT / "applications/swebench/openclaw/app.py").read_text()

    assert 'os.environ.get("OPENAI_BASE_URL")' in app
    assert 'os.environ.get("MODEL_ENDPOINT")' in app
    assert "endpoint_for_container(" in app


def test_openclaw_forces_chat_completions_transport():
    pbs = (ROOT / "scripts/execution_pbs/run_benchmark.pbs").read_text()
    configure = (ROOT / "harnesses/openclaw/configure.sh").read_text()

    for rel in (
        "applications/swebench/openclaw/app.py",
        "applications/terminal_bench/app.py",
        "applications/deepresearch_bench/app.py",
    ):
        app = (ROOT / rel).read_text()
        assert '_OPENAI_CHAT_COMPLETIONS_API = "openai-completions"' in app
        assert 'vllm_provider["api"] = _OPENAI_CHAT_COMPLETIONS_API' in app
        assert 'model["api"] = _OPENAI_CHAT_COMPLETIONS_API' in app

    assert 'vllm_provider["api"] = "openai-completions"' in pbs
    assert '"api": "openai-completions"' in pbs
    assert "models.providers.vllm.api openai-completions" in configure


def test_pbs_prepares_runtime_sif_by_harness_choice():
    pbs = (ROOT / "scripts/execution_pbs/run_benchmark.pbs").read_text()

    assert "HARNESS_CHOICE=" in pbs
    assert pbs.index("module load python/3.12.1-gcc11") < pbs.index("load_model_env.py")
    assert "OH_RUNTIME_SIF" in pbs
    assert "HM_RUNTIME_SIF" in pbs
    assert "OH_RUNTIME_BASE_DOCKER" in pbs
    assert "HM_RUNTIME_BASE_DOCKER" in pbs
    assert "build_oh_sif.sh" in pbs
    assert "build_hm_sif.sh" in pbs
    assert 'openhands)' in pbs
    assert 'hermes)' in pbs
    assert 'export OH_IMAGE="$OH_RUNTIME_SIF"' in pbs
    assert 'export HM_IMAGE="$HM_RUNTIME_SIF"' in pbs
    assert "FATAL: failed to pull $label runtime" in pbs
    assert "FATAL: OpenHands runtime sif missing" in pbs
    assert "FATAL: Hermes runtime sif missing" in pbs


def test_nscc_runtime_sif_builders_exist():
    build_oh = ROOT / "scripts/execution_pbs/build_oh_sif.sh"
    build_hm = ROOT / "scripts/execution_pbs/build_hm_sif.sh"

    oh = build_oh.read_text()
    hm = build_hm.read_text()

    assert "singularity build --sandbox" in oh
    assert "openhands-sdk==" in oh
    assert "/opt/openhands_driver.py" in oh
    assert "singularity build --sandbox" in hm
    assert "apply_patch.py apply" in hm
    assert "HERMES_PATCH_MODE=baked" in hm
    patcher = (ROOT / "harnesses/hermes/apply_patch.py").read_text()
    assert "apply_unified_patch_native" in patcher


def test_singularity_env_uses_parent_env_for_commas():
    import pytest

    from orchestration.container.singularity import _set_singularity_env

    env = {}
    _set_singularity_env(env, "OTEL_RESOURCE_ATTRIBUTES", "oc.run_id=a,hm.run_id=a")
    assert env["SINGULARITYENV_OTEL_RESOURCE_ATTRIBUTES"] == "oc.run_id=a,hm.run_id=a"
    with pytest.raises(ValueError):
        _set_singularity_env(env, "hm.run_id", "a")


def test_hermes_singularity_exec_runs_from_workspace():
    app = (ROOT / "applications/swebench/hermes/app.py").read_text()
    condition = (ROOT / "applications/swebench/conditions/hermes_agent.py").read_text()

    assert '"OTEL_RESOURCE_ATTRIBUTES": f"agent.run_id={run_id}"' in app
    assert "workdir: Optional[str] = None" in app
    assert "workdir=workspace_root" in condition


def test_singularity_subprocess_env_drops_home_override(monkeypatch):
    from orchestration.container.singularity import _subprocess_env

    monkeypatch.setenv("SINGULARITYENV_HOME", "/root")
    monkeypatch.setenv("SINGULARITYENV_NPM_CONFIG_REGISTRY", "https://registry.npmmirror.com")

    env = _subprocess_env({"SINGULARITYENV_HOME": "/other"})

    assert "SINGULARITYENV_HOME" not in env
    assert env["SINGULARITYENV_NPM_CONFIG_REGISTRY"] == "https://registry.npmmirror.com"


def test_pbs_groups_logs_under_experiment_dir():
    pbs = (ROOT / "scripts/execution_pbs/run_benchmark.pbs").read_text()

    assert "$KIT_ROOT/data/experiments/$AB_RUN_STAMP" in pbs
    assert "PBS_JOBID" in pbs
    assert 'export AB_EXPERIMENT_DIR="$EXPERIMENT_DIR"' in pbs
    assert 'LOG_DIR="${OC_LOG_DIR:-$EXPERIMENT_DIR/logs}"' in pbs
    assert '$EXPERIMENT_DIR/runs' in pbs


def test_runner_maps_app_output_under_experiment_dir(monkeypatch):
    from orchestration.runner.config import load_composed_config

    monkeypatch.setenv("AB_EXPERIMENT_DIR", "/tmp/agent-bench-exp")
    cfg = load_composed_config(
        ROOT / "config/application/swebench/phase0.json",
        ROOT / "config/harness/openclaw_swebench.json",
        ROOT / "config/model/qwen36_27b.json",
        ROOT / "config/serving/vllm/default.json",
    )

    assert cfg.output_dir == Path("/tmp/agent-bench-exp/runs/phase0")
