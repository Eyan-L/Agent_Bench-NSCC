from __future__ import annotations

import subprocess
from pathlib import Path

from agent_bench.benchmarks.terminal_bench import app as tb_app
from agent_bench.deployments.containers.base import ContainerHandle
from agent_bench.deployments.containers.singularity import SingularityContainer


class _FakeContainer:
    backend_id = "singularity"
    host_address = "127.0.0.1"

    def __init__(self) -> None:
        self.specs = []
        self.stop_calls = []
        self.created_networks = []
        self.removed_networks = []
        self.exec_calls = []

    def check_available(self) -> None:
        pass

    def check_image(self, image: str) -> None:
        assert Path(image).is_file()

    def create_network(self, name: str) -> None:
        self.created_networks.append(name)

    def remove_network(self, name: str) -> None:
        self.removed_networks.append(name)

    def copy_from_image(self, image: str, src: str, dst: Path, **_kw) -> None:
        (dst / "starter.txt").write_text("starter")

    def start(self, spec):
        self.specs.append(spec)
        return ContainerHandle(
            id=f"{spec.name}-handle", name=spec.name, backend=self.backend_id
        )

    def wait_log_pattern(self, *_args, **_kw) -> None:
        pass

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    def stop(self, handle, **_kw) -> None:
        self.stop_calls.append(handle)


class _FakeDispatcher:
    pid = 42
    last_start_kwargs = None

    def start(self, **kwargs) -> None:
        type(self).last_start_kwargs = kwargs

    def stop(self) -> None:
        pass

    @staticmethod
    def stop_external(**_kw) -> None:
        pass

    def agent_mounts_and_env(self):
        return [], {"OPENCLAW_EXEC_BRIDGE_SOCKET": "/tmp/bridge.sock"}

    def env_info(self):
        return {"bridge_socket": "/tmp/bridge.sock", "bridge_pid": self.pid}


def test_terminal_bench_openclaw_singularity_returns_runtime_handle(tmp_path, monkeypatch):
    _FakeDispatcher.last_start_kwargs = None
    runtime_sif = tmp_path / "openclaw.sif"
    runtime_sif.touch()
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    docker_image = "alexgshaw/multi-source-data-merger:20251031"
    task_sif = task_dir / "alexgshaw__multi-source-data-merger__20251031.sif"
    task_sif.touch()
    monkeypatch.setenv("AB_AGENT_RUNTIME_SIF", str(runtime_sif))
    monkeypatch.setenv("AB_DEPLOYMENT_TASK_IMAGE_DIR", str(task_dir))
    monkeypatch.setattr(tb_app, "OpenClawRemoteDispatcher", _FakeDispatcher)

    container = _FakeContainer()
    harness = tb_app.TerminalBenchOpenClawHarness(
        container=container, workspace_base=tmp_path / "workspaces"
    )

    def build_state(**_kw):
        state = harness.state_base / "run-1"
        state.mkdir(parents=True)
        (state / "openclaw.json").write_text("{}")
        return state

    monkeypatch.setattr(harness, "_build_state_dir", build_state)
    env_info = harness.prepare_environment(
        "multi-source-data-merger", "unused", "unused", run_id="run-1",
        instance_data={
            "benchmark": "terminal_bench", "docker_image": docker_image,
            "workspace": "/app", "instruction": "do the task",
        },
    )

    assert env_info["backend"] == "singularity"
    assert env_info["container_id"] == "oc-run-1-handle"
    assert env_info["oc_container_id"] == "oc-run-1-handle"
    assert env_info["task_container_id"] != env_info["container_id"]
    assert container.specs[0].image == str(task_sif)
    assert container.specs[1].image == str(runtime_sif)
    assert container.specs[1].env["HOME"] == "/root"
    assert container.specs[1].env["OPENCLAW_BYPASS_ENSURE_SANDBOX"] == "1"
    assert container.specs[1].env["OPENCLAW_DISABLE_BONJOUR"] == "1"

    # NSCC runtime hardening.
    assert tb_app.TerminalBenchOpenClawHarness._GATEWAY_READY_TIMEOUT_SECONDS == 600
    assert container.specs[1].env["OPENCLAW_PLUGIN_STAGE_DIR"] == (
        "/tmp/oc-plugin-deps/run-1"
    )
    assert container.specs[1].env["NPM_CONFIG_FETCH_RETRIES"] == "1"
    assert container.specs[1].env["NPM_CONFIG_FETCH_TIMEOUT"] == "15000"
    assert container.specs[1].env["NODE_OPTIONS"] == "--dns-result-order=ipv4first"

    # Task/runtime separation and solve-time verifier isolation.
    # Singularity needs an existing writable bind target because the SIF root
    # filesystem is immutable. The mounted /tests directory must still be
    # empty while the agent is solving; verifier files are injected only
    # afterward by run_tests().
    test_mounts = [
        m for m in container.specs[0].mounts
        if m.container == "/tests"
    ]
    assert len(test_mounts) == 1
    assert Path(test_mounts[0].host).is_dir()
    assert list(Path(test_mounts[0].host).iterdir()) == []

    assert container.specs[0].labels["openclaw.sessionKey"].startswith("agent:")

    # Exec bridge must target the Terminal-Bench task instance.
    assert _FakeDispatcher.last_start_kwargs is not None
    assert _FakeDispatcher.last_start_kwargs["backend"] == "singularity"
    assert (
        _FakeDispatcher.last_start_kwargs["target"]
        == env_info["task_container_name"]
    )

    # Retry preflight must attempt cleanup of deterministic stale instances.
    stopped_names = [h.name for h in container.stop_calls]
    assert "oc-run-1" in stopped_names
    assert env_info["task_container_name"] in stopped_names
    assert "oc-eval-net-run-1" in container.removed_networks
    assert "oc-eval-net-run-1" in container.created_networks

    # Gateway and CLI use separate node-local plugin staging directories.
    gateway_env_writes = [
        args[1]
        for args, _kwargs in container.exec_calls
        if len(args) >= 2 and isinstance(args[1], list)
    ]
    assert any(
        "/tmp/oc-plugin-deps/run-1-cli" in str(argv)
        for argv in gateway_env_writes
    )


def test_driver_routes_terminal_bench_openclaw_to_generic_harness():
    driver = (Path(__file__).resolve().parents[1] / "agent_bench/runtime/driver.py").read_text()
    assert 'if harness_choice == "openclaw":' in driver
    assert "TerminalBenchOpenClawHarness as tb_cls" in driver
    assert "tb_harness = tb_cls(benchmark_data_root, container=tb_container)" in driver


def test_terminal_bench_prompt_variant_is_selected_explicitly(tmp_path, monkeypatch):
    profile = tmp_path / "harness.yaml"
    profile.write_text(
        "terminal_bench:\n"
        "  agent_prompt_file: AGENTS.md.dr_matched\n"
        "  workspace_bootstrap_file: BOOTSTRAP.md.dr_matched\n"
    )
    monkeypatch.setenv("AB_HARNESS_CONFIG", str(profile))

    assert tb_app._terminal_bench_prompt_filename(
        "agent_prompt_file", "AGENTS.md"
    ) == "AGENTS.md.dr_matched"
    assert tb_app._terminal_bench_prompt_filename(
        "workspace_bootstrap_file", "BOOTSTRAP.md"
    ) == "BOOTSTRAP.md.dr_matched"


def test_singularity_copy_into_directory_streams_tar(tmp_path, monkeypatch):
    source = tmp_path / "tests"
    source.mkdir()
    (source / "test.sh").write_text("#!/bin/sh\n")
    calls = []

    class _Stream:
        def close(self):
            pass

    class _Proc:
        returncode = 0
        stdout = _Stream()

        def communicate(self, timeout=None):
            return b"", b""

        def kill(self):
            pass

    def popen(cmd, **kw):
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    container = object.__new__(SingularityContainer)
    SingularityContainer.copy_into(
        container, ContainerHandle("task", "task", "singularity"), source, "/tests"
    )

    assert calls[0][:3] == ["tar", "-C", str(source)]
    assert calls[1][:3] == ["singularity", "exec", "instance://task"]
    assert "mkdir -p '/tests'" in calls[1][-1]
