from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_bench.deployments.containers.docker import DockerContainer


def test_docker_container_traced_subprocess_writes_span(tmp_path: Path) -> None:
    spans_path = tmp_path / "spans.jsonl"
    container = DockerContainer()
    container.set_trace_path(spans_path)

    result = container._run_traced(
        "agent_bench.container.exec",
        [sys.executable, "-c", "import sys; print('ok'); print('err', file=sys.stderr)"],
        attrs={"agent_bench.container.name": "demo"},
    )

    assert result.returncode == 0
    rows = [json.loads(line) for line in spans_path.read_text().splitlines()]
    assert len(rows) == 1
    span = rows[0]
    attrs = span["attrs"]
    assert span["name"] == "agent_bench.container.exec"
    assert attrs["agent_bench.container.name"] == "demo"
    assert attrs["agent_bench.returncode"] == 0
    assert "ok" in attrs["agent_bench.stdout_preview"]
    assert "err" in attrs["agent_bench.stderr_preview"]
    assert sys.executable in attrs["agent_bench.command_json"]
