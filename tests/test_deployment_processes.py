from __future__ import annotations

from agent_bench.deployments import processes
from agent_bench.deployments.processes import http_ready, tail_text


def test_http_ready_supports_authorization_headers(monkeypatch) -> None:
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _fake_urlopen(req, timeout):
        captured["authorization"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(processes.request, "urlopen", _fake_urlopen)

    assert http_ready(
        "http://127.0.0.1:8000/v1/models",
        headers={"Authorization": "Bearer sk-local"},
    )
    assert captured == {"authorization": "Bearer sk-local", "timeout": 2}


def test_tail_text_returns_last_lines(tmp_path) -> None:
    path = tmp_path / "engine.log"
    path.write_text("\n".join(str(i) for i in range(10)), encoding="utf-8")

    assert tail_text(path, lines=3) == "7\n8\n9"
