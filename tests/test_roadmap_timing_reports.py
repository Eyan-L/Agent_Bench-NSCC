"""Verify real report entrypoints share timing evidence and selected-run policy."""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_roadmap_runs as analyze
import compare_roadmap_runs as compare
import roadmap_result_timing as timing


def _event(eid, parent, seconds, role, text, **extra):
    return {
        "id": eid, "parentId": parent,
        "timestamp": datetime.fromtimestamp(seconds, timezone.utc).isoformat(),
        "message": {"role": role, "timestamp": seconds * 1000,
                    "content": [{"type": "text", "text": text}], **extra},
    }


def _run(root, job, *, child=True, outcome="normal_completion", mode="gateway"):
    run = root / "fixture" / f"2026-10-02_{job}" / "runs" / "fixture" / "main"
    sessions = run / "conversation" / "sessions"
    sessions.mkdir(parents=True)
    manifest = {
        "run_id": "main", "instance_id": "fixture-roadmap", "seed": 42,
        "status": "PASS", "reward": 1, "benchmark_valid": True,
        "agent_execution_outcome": outcome, "agent_runtime_mode": mode,
        "agent_execution_seconds": 249,
        "phase_timing": {"agent_execution_start_ns": 1_000_000_000,
                         "agent_execution_end_ns": 250_000_000_000},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    observation = {
        "materialized_child_session_count": int(child),
        "child_session_count": int(child), "max_concurrent_children": int(child),
        "agent_timeline": {"agents": [{"role": "parent", "session_id": "main",
                                       "start_timestamp_ns": 1_000_000_000,
                                       "end_timestamp_ns": 250_000_000_000}]},
        "child_runtime_by_session": {},
    }
    parents = []
    if child:
        observation["child_runtime_by_session"]["agent:test:child"] = {
            "session_id": "child", "label": "Implement feature",
            "start_timestamp_ns": 10_000_000_000,
            "terminal_timestamp_ns": 101_000_000_000,
            "runtime_complete": True, "terminal_timestamp_quality": "native_event",
        }
        text = "Verified implementation with substantial evidence. " * 5
        final = _event("final", None, 100, "assistant", text, stopReason="stop")
        (sessions / "child.jsonl").write_text(json.dumps(final) + "\n")
        notice = _event("notice", None, 200, "user",
                        "[Internal task completion event]\nsource: subagent\n"
                        "session_key: agent:test:child\n\nChild result:\n"
                        f"<prompt-data>\n{text}\n</prompt-data>",
                        provenance={"kind": "inter_session",
                                    "sourceTool": "subagent_announce"})
        parents = [notice, _event("follow", "notice", 202, "assistant",
                                   "Review result and finish", stopReason="stop")]
    (sessions / "main.jsonl").write_text("".join(json.dumps(e) + "\n" for e in parents))
    (run / "runtime_observability.json").write_text(json.dumps(observation))
    (run / "trajectory.jsonl").write_text(json.dumps({"event_type": "done"}) + "\n")
    (run / "spans.jsonl").write_text("")
    return run


def _report(output, name):
    bundle = output / name
    report = json.loads((bundle / f"{name}.json").read_text())
    export = report["result_timing"]
    detail = json.loads((bundle / export["json"]).read_text())
    index = json.loads((bundle / f"{name}-artifact-index.json").read_text())
    # Index must include all timing artifacts, not just copied input evidence.
    encoded = json.dumps(index)
    for key in ("json", "csv", "viewer"):
        assert export[key] in encoded
        assert (bundle / export[key]).is_file()
    assert "result-timing.csv" in (bundle / f"{name}.md").read_text()
    return bundle, report, detail


def test_paired_compare_matches_analyze_timing_and_keeps_both_arms(tmp_path):
    runs = tmp_path / "runs"
    sub = _run(runs, "101.pbs101")
    _run(runs, "102.pbs101", child=False)
    output = tmp_path / "reports"
    assert compare.main([
        "--subagent", "101.pbs101", "--single-agent", "102.pbs101",
        "--runs-root", str(runs), "--output-dir", str(output), "--name", "paired",
        "--strict-trajectory", "--no-plots", "--bootstrap-samples", "100",
    ], omit_candidate_workspace=True) == 0
    bundle, report, detail = _report(output, "paired")
    assert report["paired"]["paired_task_count"] == 1
    assert report["result_timing"]["summary"]["tasks"] == 2
    assert {t["arm"] for t in detail["tasks"]} == {"subagent", "single-agent"}
    assert len(detail["tasks"][0]["children"]) == 1
    assert detail["tasks"][1]["children"] == []
    with (bundle / report["result_timing"]["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert (rows[0]["arm"], rows[0]["seed"]) == ("subagent", "42")
    assert not report["figures"]
    assert analyze.main([str(sub), "--output-dir", str(output), "--name", "one",
                         "--no-plots"], omit_candidate_workspace=True) == 0
    _, _, one = _report(output, "one")
    assert detail["tasks"][0] == one["tasks"][0]


def test_rightmost_failed_retry_excludes_old_timing_from_comparison(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "201.pbs101")
    _run(runs, "202.pbs101", outcome="timeout")
    _run(runs, "203.pbs101", child=False)
    output = tmp_path / "reports"
    assert compare.main([
        "--subagent", "201.pbs101", "202.pbs101", "--single-agent", "203.pbs101",
        "--runs-root", str(runs), "--output-dir", str(output), "--name", "excluded",
        "--no-plots", "--bootstrap-samples", "100",
    ], omit_candidate_workspace=True) == 0
    _, report, detail = _report(output, "excluded")
    assert report["selection"]["excluded_pair_count"] == 1
    assert report["paired"]["paired_task_count"] == 0
    assert detail["tasks"] == []


@pytest.mark.parametrize("arm,child", [("subagent", True), ("single-agent", False)])
def test_single_arm_analyze_exports_timing_without_pairing(tmp_path, arm, child):
    runs = tmp_path / "runs"
    _run(runs, "301.pbs101", child=child)
    output = tmp_path / "reports"
    assert analyze.main([
        "301.pbs101", "--runs-root", str(runs), "--arm", arm,
        "--output-dir", str(output), "--name", "single", "--no-plots",
    ], omit_candidate_workspace=True) == 0
    _, report, detail = _report(output, "single")
    assert report["summary"]["task_count"] == 1
    assert detail["tasks"][0]["arm"] == arm
    assert report["result_timing"]["summary"]["children"] == int(child)


def test_unpaired_compare_also_exports_timing(tmp_path):
    run = _run(tmp_path / "runs", "401.pbs101")
    output = tmp_path / "reports"
    assert compare.main([str(run), "--output-dir", str(output), "--name", "summary",
                         "--no-plots"], omit_candidate_workspace=True) == 0
    _, report, detail = _report(output, "summary")
    assert report["mode"] == "summary"
    assert len(detail["tasks"]) == 1


def test_viewer_associates_each_figure_with_its_arm_and_seed(tmp_path, monkeypatch):
    tasks = []
    for arm, seed in [("subagent", 42), ("single-agent", 42), ("subagent", 7)]:
        tasks.append({"instance_id": "same-roadmap", "arm": arm, "seed": seed,
                      "children": []})
    def plot(tasks, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-image")
        return True
    monkeypatch.setattr(timing, "plot_tasks", plot)
    export = timing.export_analysis(tasks, tmp_path, "viewer")
    text = (tmp_path / export["viewer"]).read_text()
    assert len(export["figure_paths"]) == 3
    for path in export["figure_paths"]:
        assert text.count(path.name) == 1
    assert "single-agent / seed 42" in text
    assert "subagent / seed 7" in text


@pytest.mark.parametrize("mode", ["hermes", "codex"])
def test_other_harness_compare_does_not_infer_openclaw_timing(tmp_path, mode):
    run = _run(tmp_path / "runs", "501.pbs101", mode=mode)
    output = tmp_path / "reports"
    assert compare.main([str(run), "--output-dir", str(output), "--name", "other",
                         "--no-plots"], omit_candidate_workspace=True) == 0
    report = json.loads((output / "other" / "other.json").read_text())
    assert "result_timing" not in report
