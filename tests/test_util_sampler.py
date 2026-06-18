import json

from agent_bench.telemetry import util_sampler


class _FakeUrlopenResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_read_docker_stats_ignores_malformed_cadvisor_payload(monkeypatch):
    payload = {
        "/docker/container-id": {
            "stats": [
                {
                    "timestamp": "2026-06-18T00:00:00Z",
                    "cpu": {"usage": {"total": 100}},
                    "memory": {"usage": 1024},
                },
                {
                    "timestamp": "2026-06-18T00:00:01Z",
                    "memory": {"usage": 2048},
                },
            ],
            "spec": {"memory": {"limit": 4096}},
        }
    }

    monkeypatch.setattr(
        util_sampler.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeUrlopenResponse(payload),
    )

    assert util_sampler._read_docker_stats("worker") is None


def test_read_container_groups_preserves_unlimited_memory_limit(monkeypatch):
    stats_by_name = {
        "limited": {
            "name": "limited",
            "pct": 10.0,
            "mem_used_mb": 128,
            "mem_total_mb": 256,
            "mem_pct": 50.0,
            "n_proc": None,
        },
        "unlimited": {
            "name": "unlimited",
            "pct": 15.0,
            "mem_used_mb": 512,
            "mem_total_mb": None,
            "mem_pct": None,
            "n_proc": None,
        },
    }

    monkeypatch.setattr(
        util_sampler,
        "_read_docker_stats",
        lambda name, port=8088: stats_by_name[name],
    )

    groups = util_sampler._read_container_groups({
        "tools": ["limited", "unlimited"],
    })

    assert groups["tools"]["mem_used_mb"] == 640
    assert groups["tools"]["mem_total_mb"] is None
    assert groups["tools"]["mem_pct"] is None
