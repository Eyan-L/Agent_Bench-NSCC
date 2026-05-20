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


def test_pbs_prepares_runtime_sif_by_harness_choice():
    pbs = (ROOT / "scripts/execution_pbs/run_benchmark.pbs").read_text()

    assert "HARNESS_CHOICE=" in pbs
    assert "OH_RUNTIME_SIF" in pbs
    assert "HM_RUNTIME_SIF" in pbs
    assert 'openhands)' in pbs
    assert 'hermes)' in pbs
    assert 'export OH_IMAGE="$OH_RUNTIME_SIF"' in pbs
    assert 'export HM_IMAGE="$HM_RUNTIME_SIF"' in pbs


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
        ROOT / "config/harness/openclaw_27b.json",
        ROOT / "config/backend/vllm/tp1.json",
    )

    assert cfg.output_dir == Path("/tmp/agent-bench-exp/runs/phase0")
