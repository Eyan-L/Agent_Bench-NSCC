import json

from agent_bench.benchmarks.conditions.openclaw_measurement import (
    build_openclaw_measurement,
)


def test_post_compaction_loop_guard_is_machine_readable(tmp_path):
    (tmp_path / "openclaw_gateway.log").write_text(
        "2026-09-01T05:45:00Z [post-compaction-guard] "
        "compaction_loop_persisted: tool=read count=3\n"
    )

    result = build_openclaw_measurement(
        tmp_path,
        agent_id="roadmap_bench",
    )

    assert result["termination_reason"] == "compaction_loop_persisted"
    assert result["loop_detector"] == "compaction_loop_persisted"
    assert result["loop_tool"] == "read"
    assert result["loop_count"] == 3

    observability = result["runtime_observability"]
    assert observability["loop_detection_count"] == 1
    assert observability["loop_detections"] == [
        {
            "timestamp": "2026-09-01T05:45:00+00:00",
            "detector": "compaction_loop_persisted",
            "tool_name": "read",
            "count": 3,
        }
    ]

    trajectory = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    loop_events = [
        event for event in trajectory if event["event_type"] == "loop_detected"
    ]
    assert len(loop_events) == 1
    assert loop_events[0]["detector"] == "compaction_loop_persisted"
    assert loop_events[0]["tool_name"] == "read"
    assert loop_events[0]["count"] == 3


def test_normal_run_has_no_loop_termination_metadata(tmp_path):
    result = build_openclaw_measurement(
        tmp_path,
        agent_id="roadmap_bench",
    )

    assert "termination_reason" not in result
    assert "loop_detector" not in result
    assert "loop_tool" not in result
    assert "loop_count" not in result
    assert result["runtime_observability"]["loop_detection_count"] == 0
    assert result["runtime_observability"]["loop_detections"] == []
