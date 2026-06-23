from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "agent_bench" / "runtime" / "exec_bridge" / "server.py"


def test_openclaw_dispatcher_legacy_docker_args_include_bridge_wiring() -> None:
    from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher

    dispatcher = OpenClawRemoteDispatcher()
    dispatcher.token = "test-token"

    args = dispatcher.agent_container_args()

    assert "-v" in args
    assert any("/usr/local/bin/oc-exec-bridge:ro" in arg for arg in args)
    assert "-e" in args
    assert "OPENCLAW_EXEC_BRIDGE=/usr/local/bin/oc-exec-bridge" in args
    assert "OPENCLAW_EXEC_BRIDGE_SOCKET=/root/.openclaw/exec-bridge.sock" in args
    assert "OPENCLAW_EXEC_BRIDGE_TOKEN=test-token" in args


def test_exec_bridge_server_stays_alive_after_socket_bind(tmp_path: Path) -> None:
    token_file = tmp_path / "exec-bridge.token"
    socket_path = tmp_path / "exec-bridge.sock"
    token_file.write_text("test-token", encoding="utf-8")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "--socket",
            str(socket_path),
            "--token-file",
            str(token_file),
            "--backend",
            "docker",
            "--target",
            "missing-container-for-startup-test",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not socket_path.exists():
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        assert socket_path.exists(), _read_output(proc)
        time.sleep(0.2)
        assert proc.poll() is None, _read_output(proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_openclaw_dispatcher_uses_relative_socket_for_long_state_dir(tmp_path: Path) -> None:
    from agent_bench.harnesses.openclaw.dispatcher import OpenClawRemoteDispatcher

    state_dir = tmp_path / ("a" * 70) / ("b" * 70)
    state_dir.mkdir(parents=True)
    assert len(str(state_dir / "exec-bridge.sock")) > 108

    dispatcher = OpenClawRemoteDispatcher()
    try:
        dispatcher.start(
            state_dir=state_dir,
            backend="docker",
            target="missing-container-for-startup-test",
        )
        assert (state_dir / "exec-bridge.sock").exists()
    finally:
        dispatcher.stop()


def _read_output(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        return "process is still running"
    if proc.stdout is None:
        return ""
    try:
        return proc.stdout.read()
    except Exception:
        return ""
