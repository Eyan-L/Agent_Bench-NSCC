from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from agent_bench.core import run_artifacts
from agent_bench.core.commands import analyze_main
from scripts import compare_roadmap_runs as compare


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _make_run(
    root: Path,
    *,
    outcome: str = "normal_completion",
    child_count: int = 1,
    with_spans: bool = True,
    with_html: bool = True,
) -> Path:
    run = root / "source-run"
    run.mkdir(parents=True)
    manifest = {
        "instance_id": "task/example",
        "seed": 42,
        "run_id": "run-1",
        "status": "DONE" if outcome == "normal_completion" else "ERROR",
        "agent_execution_outcome": outcome,
        "benchmark_valid": outcome != "runtime_error",
        "verifier_status": "FAIL",
        "reward": 0.5,
        "test_pass_rate": 0.5,
        "agent_execution_seconds": 12.5,
        "phase_timing": {"agent_execution_seconds": 12.5, "verifier_seconds": 1.0},
        "total_prompt_tokens": 100,
        "total_completion_tokens": 20,
        "trajectory_tokens": {"parent": {"input": 80, "output": 15}},
        "serving_tokens": {
            "processed_input": 1000,
            "generated_output": 25,
            "request_count": 3,
        },
        "unknown_future_field": {"unit": "widgets", "value": 7},
        "trajectory_path": str(run / "trajectory.jsonl"),
        "token_accounting_path": "token_accounting.json",
        "runtime_observability_path": str(run / "runtime_observability.json"),
    }
    _write_json(run / "manifest.json", manifest)
    _write_json(
        run / "verdict.json",
        {"status": "FAIL", "reward": 0.5, "unknown_verifier_field": [1, 2]},
    )
    _write_json(
        run / "runtime_observability.json",
        {
            "child_session_count": child_count,
            "max_concurrent_children": child_count,
            "agent_timeline": {
                "agents": [
                    {
                        "role": "parent",
                        "label": "parent",
                        "status": "done",
                        "start_timestamp_ns": 1_000_000_000,
                        "end_timestamp_ns": 2_000_000_000,
                        "duration_seconds": 1.0,
                    }
                ]
            },
        },
    )
    _write_json(
        run / "token_accounting.json",
        {"logical": {"input": 100, "output": 20}, "unknown": "preserved"},
    )
    (run / "trajectory.jsonl").write_text(
        json.dumps(
            {
                "event_type": "model_request",
                "session_key": "parent",
                "timestamp_ns": 1_000_000_000,
            }
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "tool_call",
                "session_key": "parent",
                "timestamp_ns": 2_000_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "steps.jsonl").write_text('{"step_type":"agent"}\n', encoding="utf-8")
    if with_spans:
        (run / "spans.jsonl").write_text(
            json.dumps(
                {
                    "name": "openclaw.model.call",
                    "start_unix_nano": 1_000_000_000,
                    "end_unix_nano": 2_000_000_000,
                    "attrs": {"request_latency_ms": 1000},
                    "resource": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if with_html:
        (run / "spans_timeline.html").write_bytes(b"<html>ORIGINAL TIMELINE</html>\n")
    (run / "util.jsonl").write_text(
        '{"_meta":true}\n'
        '{"ts_ns":1000000000,"cpu":{"pct":5,"mem_used_mb":10},'
        '"gpu":[{"index":0,"util_gpu_pct":70,"util_mem_pct":50,'
        '"mem_used_mb":100,"power_w":200}]}\n',
        encoding="utf-8",
    )
    conversation = run / "conversation"
    (conversation / "sessions").mkdir(parents=True)
    (conversation / "sessions" / "sessions.json").write_text("{}\n")
    (conversation / "sessions" / "parent.jsonl").write_bytes(
        b'{"type":"message","unknown":"kept"}\n'
    )
    (conversation / "cache-trace.jsonl").write_bytes(b'{"messages":["full"]}\n')
    (conversation / "openclaw-agent.sqlite").write_bytes(b"sqlite-bytes\x00\x01")
    for name, content in {
        "openclaw_response.json": b'{"result":"complete"}\n',
        "openclaw_stdout.log": b"stdout\n",
        "openclaw_stderr.log": b"stderr\n",
        "openclaw_gateway.log": b"gateway\n",
        "openclaw_exec_bridge.jsonl": b'{"command":"true"}\n',
        "candidate.patch": b"diff --git a/a b/a\n",
        "verifier-extra.log": b"verifier details\n",
        "serving-metrics.jsonl": b'{"running":1}\n',
        "unknown-new-artifact.bin": b"unknown artifact bytes\x00",
    }.items():
        (run / name).write_bytes(content)
    return run


def _export(tmp_path: Path, run: Path):
    bundle = run_artifacts.prepare_bundle_dir(
        tmp_path / "reports", "bundle", source_dirs=(run,)
    )
    index = run_artifacts.export_run_artifacts(
        run,
        bundle,
        arm="subagent",
        instance_id="task/example",
        seed=42,
        job_id="1.pbs101",
    )
    return bundle, index


def test_complete_export_preserves_bytes_unknown_fields_and_conversation(tmp_path):
    run = _make_run(tmp_path)
    source_files = {path.resolve(): path.read_bytes() for path in run.rglob("*") if path.is_file()}

    bundle, index = _export(tmp_path, run)

    copied = [item for item in index["artifacts"] if item["status"] == "copied"]
    assert len(copied) == len(source_files)
    for item in copied:
        source = Path(item["source_path"])
        destination = bundle / item["bundle_path"]
        assert destination.read_bytes() == source_files[source]
        assert item["size_bytes"] == len(source_files[source])
        assert item["sha256"] == hashlib.sha256(source_files[source]).hexdigest()

    task = "task_example__seed42"
    manifest_copy = (
        bundle
        / "04_final_workspace_score"
        / "subagent"
        / task
        / "manifest.json"
    )
    assert json.loads(manifest_copy.read_text())["unknown_future_field"] == {
        "unit": "widgets",
        "value": 7,
    }
    assert (
        bundle
        / "01_durable_session_records"
        / "subagent"
        / task
        / "conversation"
        / "cache-trace.jsonl"
    ).read_bytes() == b'{"messages":["full"]}\n'
    assert (
        bundle
        / "03_execution_evidence"
        / "subagent"
        / task
        / "unknown-new-artifact.bin"
    ).read_bytes() == b"unknown artifact bytes\x00"


def test_bundle_is_portable_after_sources_are_removed(tmp_path):
    run = _make_run(tmp_path)
    external = tmp_path / "external-telemetry.log"
    external.write_text("external telemetry\n")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["custom_telemetry_path"] = str(external)
    _write_json(run / "manifest.json", manifest)
    bundle, index = _export(tmp_path, run)

    links = [target for target in index["links"].values() if target]
    shutil.rmtree(run)
    external.unlink()
    moved = tmp_path / "moved-bundle"
    shutil.move(str(bundle), moved)

    assert links
    for target in links:
        assert (moved / target).is_file()
    referenced = [
        item
        for item in index["artifacts"]
        if "custom_telemetry_path" in item.get("manifest_fields", [])
    ]
    assert len(referenced) == 1
    assert (moved / referenced[0]["bundle_path"]).read_text() == "external telemetry\n"
    detail = (moved / index["links"]["details"]).read_text()
    assert "file:///" not in detail


def test_optional_missing_artifacts_are_explicit_and_not_zero(tmp_path):
    run = _make_run(tmp_path, with_spans=False, with_html=False)
    (run / "util.jsonl").unlink()
    bundle, index = _export(tmp_path, run)

    missing = {
        Path(item["source_path"]).name: item
        for item in index["artifacts"]
        if item["status"] == "not_collected" and item.get("source_path")
    }
    assert missing["spans.jsonl"]["size_bytes"] is None
    assert missing["spans_timeline.html"]["size_bytes"] is None
    assert missing["util.jsonl"]["size_bytes"] is None
    assert index["viewer"]["status"] == "unavailable"
    assert index["links"]["spans_timeline"] is None
    assert not list(bundle.rglob("spans_timeline.html"))


def test_copy_failure_is_reported_and_no_partial_file_remains(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    original = run_artifacts.shutil.copy2

    def fail_one(source, destination, *args, **kwargs):
        if Path(source).name == "util.jsonl":
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_bytes(b"")
            raise OSError("injected copy failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(run_artifacts.shutil, "copy2", fail_one)
    bundle, index = _export(tmp_path, run)

    failed = [
        item
        for item in index["artifacts"]
        if item["status"] == "copy_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["source_path"].endswith("util.jsonl")
    assert "injected copy failure" in failed[0]["error"]
    assert not (bundle / failed[0]["bundle_path"]).exists()
    assert index["complete"] is False


def test_existing_timeline_is_preserved_and_missing_timeline_is_rendered(tmp_path):
    existing_run = _make_run(tmp_path / "existing")
    existing_bundle, existing_index = _export(tmp_path / "existing", existing_run)
    existing_timeline = existing_bundle / existing_index["links"]["spans_timeline"]
    assert existing_timeline.read_bytes() == b"<html>ORIGINAL TIMELINE</html>\n"
    assert existing_index["viewer"]["status"] == "copied"

    generated_run = _make_run(tmp_path / "generated", with_html=False)
    generated_bundle, generated_index = _export(tmp_path / "generated", generated_run)
    generated_timeline = generated_bundle / generated_index["links"]["spans_timeline"]
    rendered = generated_timeline.read_text()
    assert generated_index["viewer"]["status"] == "generated"
    assert "Span Timeline" in rendered
    assert "Agent Runtime Timeline" in rendered


def test_timeline_generation_failure_marks_bundle_incomplete(tmp_path, monkeypatch):
    run = _make_run(tmp_path, with_html=False)

    def fail_to_load(_path):
        raise ValueError("injected renderer failure")

    monkeypatch.setattr(
        "agent_bench.telemetry.otel.visualize_spans._load_spans",
        fail_to_load,
    )
    _bundle, index = _export(tmp_path, run)

    assert index["viewer"]["status"] == "generation_failed"
    assert "injected renderer failure" in index["viewer"]["error"]
    assert index["links"]["spans_timeline"] is None
    assert index["complete"] is False
    failed = [
        item
        for item in index["artifacts"]
        if item["status"] == "generation_failed"
        and str(item.get("bundle_path", "")).endswith("spans_timeline.html")
    ]
    assert len(failed) == 1


def test_manifest_artifact_paths_drive_analysis_and_timeline(tmp_path):
    run = _make_run(tmp_path / "source", with_html=False)
    external = tmp_path / "external"
    external.mkdir()
    manifest = json.loads((run / "manifest.json").read_text())
    artifact_fields = {
        "telemetry_path": ("spans.jsonl", "model-trace.jsonl"),
        "trajectory_path": ("trajectory.jsonl", "trajectory.jsonl"),
        "step_trace_path": ("steps.jsonl", "steps.jsonl"),
        "runtime_observability_path": (
            "runtime_observability.json",
            "runtime_observability.json",
        ),
        "token_accounting_path": (
            "token_accounting.json",
            "token_accounting.json",
        ),
        "util_path": ("util.jsonl", "resource-samples.jsonl"),
        "verdict_path": ("verdict.json", "score-result.json"),
    }
    for field, (source_name, destination_name) in artifact_fields.items():
        source = run / source_name
        destination = external / destination_name
        shutil.move(str(source), destination)
        manifest[field] = str(destination)
    _write_json(run / "manifest.json", manifest)

    summary = run_artifacts.analyze_run(run)
    assert summary["activity"]["spans"]["status"] == "available"
    assert summary["activity"]["trajectory"]["record_count"] == 2
    assert summary["resources"]["status"] == "available"
    assert summary["source_documents"]["verdict"]["status"] == "available"
    assert summary["artifact_sources"]["spans"] == str(
        external / "model-trace.jsonl"
    )
    assert summary["artifact_sources"]["resources"] == str(
        external / "resource-samples.jsonl"
    )

    _bundle, index = _export(tmp_path / "export", run)
    assert index["viewer"]["status"] == "generated"
    assert not any(
        item["status"] == "not_collected"
        and Path(str(item.get("source_path", ""))).name
        in {"spans.jsonl", "util.jsonl", "verdict.json"}
        for item in index["artifacts"]
    )
    copied_spans = [
        item
        for item in index["artifacts"]
        if "telemetry_path" in item.get("manifest_fields", [])
    ]
    assert len(copied_spans) == 1
    assert copied_spans[0]["status"] == "copied"
    copied_resources = [
        item
        for item in index["artifacts"]
        if "util_path" in item.get("manifest_fields", [])
    ]
    assert len(copied_resources) == 1
    assert copied_resources[0]["status"] == "copied"
    assert index["links"]["resources"] == copied_resources[0]["bundle_path"]
    copied_verdict = [
        item
        for item in index["artifacts"]
        if "verdict_path" in item.get("manifest_fields", [])
    ]
    assert len(copied_verdict) == 1
    assert index["links"]["verdict"] == copied_verdict[0]["bundle_path"]


def test_corrupt_json_is_preserved_and_reported_invalid(tmp_path):
    run = _make_run(tmp_path)
    (run / "verdict.json").write_bytes(b"{not-json\n")

    bundle, index = _export(tmp_path, run)

    copied = next(
        item
        for item in index["artifacts"]
        if str(item.get("source_path", "")).endswith("verdict.json")
        and item["status"] == "copied"
    )
    assert (bundle / copied["bundle_path"]).read_bytes() == b"{not-json\n"
    assert index["summary"]["source_documents"]["verdict"]["status"] == "invalid"


@pytest.mark.parametrize(
    "outcome,child_count",
    [
        ("normal_completion", 2),
        ("blocked_incomplete", 0),
        ("timeout", 0),
        ("runtime_error", 1),
    ],
)
def test_single_run_analysis_supports_all_outcomes(tmp_path, outcome, child_count):
    run = _make_run(tmp_path, outcome=outcome, child_count=child_count)

    summary = run_artifacts.analyze_run(run)

    assert summary["result"]["agent_execution_outcome"] == outcome
    assert summary["lifecycle"]["child_session_count"] == child_count
    assert summary["activity"]["model_request_count"] == 1
    assert summary["activity"]["tool_call_count"] == 1
    assert summary["resources"]["sample_count"] == 1


def test_compare_and_analyze_share_common_run_metrics(tmp_path):
    run = _make_run(tmp_path, child_count=2)
    manifest = json.loads((run / "manifest.json").read_text())
    record = compare.RunRecord(
        arm="subagent",
        job_id="1.pbs101",
        pbs_run_dir=run.parent,
        run_dir=run,
        manifest_path=run / "manifest.json",
        manifest=manifest,
        trajectory_path=run / "trajectory.jsonl",
        trajectory=compare.inspect_trajectory(run / "trajectory.jsonl"),
    )

    compare_row = compare.record_to_dict(record)
    analysis = run_artifacts.analyze_run(run)

    assert compare_row["reward"] == analysis["result"]["reward"]
    assert (
        compare_row["agent_execution_seconds"]
        == analysis["timing"]["agent_execution_seconds"]
    )
    assert (
        compare_row["materialized_subagent_count"]
        == analysis["lifecycle"]["child_session_count"]
    )
    assert compare_row["tool_call_count"] == analysis["activity"]["tool_call_count"]


def test_analyze_cli_remains_compatible_and_exports_bundle(tmp_path, capsys):
    run = _make_run(tmp_path)
    output = tmp_path / "analysis-output"

    assert analyze_main(
        ["--run", str(run), "--json", "--output-dir", str(output)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    bundle = Path(payload["bundle"])
    assert payload["manifest"] == str((run / "manifest.json").resolve())
    assert "manifest.json" in payload["result_files"]
    assert (bundle / "analysis.json").is_file()
    assert (bundle / "artifact-index.json").is_file()
    assert (bundle / "index.html").is_file()
    catalog = json.loads((bundle / "artifact-index.json").read_text())
    indexed = {item["bundle_path"] for item in catalog["bundle_files"]}
    actual = {
        str(path.relative_to(bundle).as_posix())
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "artifact-index.json"
    }
    assert catalog["index_self_excluded_from_hash"] is True
    assert indexed == actual
    assert all(item["size_bytes"] is not None for item in catalog["bundle_files"])
    assert all(item["sha256"] for item in catalog["bundle_files"])


def test_output_overlap_is_rejected_before_source_changes(tmp_path):
    run = _make_run(tmp_path)
    before = {
        str(path.relative_to(run)): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="must not overlap"):
        run_artifacts.prepare_bundle_dir(run, "report", source_dirs=(run,))

    after = {
        str(path.relative_to(run)): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_repeated_report_generation_does_not_modify_source(tmp_path):
    run = _make_run(tmp_path)
    before = {
        str(path.relative_to(run)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in run.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "reports"

    first_bundle, _first = run_artifacts.create_single_run_bundle(
        run, output, name="repeatable"
    )
    first_manifest = (first_bundle / "04_final_workspace_score" / "run" / "task_example__seed42" / "manifest.json").read_bytes()
    second_bundle, _second = run_artifacts.create_single_run_bundle(
        run, output, name="repeatable"
    )

    after = {
        str(path.relative_to(run)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert (
        second_bundle
        / "04_final_workspace_score"
        / "run"
        / "task_example__seed42"
        / "manifest.json"
    ).read_bytes() == first_manifest


def test_existing_compare_selection_policy_is_unchanged_and_explained(tmp_path):
    sub = []
    single = []
    for instance_id, outcome, children in (
        ("included", "normal_completion", 1),
        ("no-child", "normal_completion", 0),
        ("failed", "runtime_error", 1),
    ):
        for arm, target in (("subagent", sub), ("single-agent", single)):
            run = _make_run(
                tmp_path / f"{arm}-{instance_id}",
                outcome=(outcome if arm == "subagent" else "normal_completion"),
                child_count=(children if arm == "subagent" else 0),
            )
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["instance_id"] = instance_id
            _write_json(run / "manifest.json", manifest)
            target.append(
                compare.RunRecord(
                    arm=arm,
                    job_id="1.pbs101",
                    pbs_run_dir=run.parent,
                    run_dir=run,
                    manifest_path=run / "manifest.json",
                    manifest=manifest,
                    trajectory_path=run / "trajectory.jsonl",
                    trajectory=compare.inspect_trajectory(run / "trajectory.jsonl"),
                )
            )

    kept_sub, kept_single = compare._impl._filter_normal_completion_pairs(
        sub, single
    )
    diagnostics = compare._impl._comparison_selection_diagnostics(sub, single)

    assert [record.instance_id for record in kept_sub] == ["included"]
    assert [record.instance_id for record in kept_single] == ["included"]
    assert diagnostics["included_pair_count"] == 1
    assert diagnostics["excluded_pair_count"] == 2
    assert diagnostics["excluded_reason_counts"] == {
        "no_materialized_subagent": 1,
        "subagent_not_normal_completion": 1,
    }


def test_single_run_bundle_exports_shared_serving_and_otel_logs_once(tmp_path):
    pbs_root = tmp_path / "pbs-job"
    run = _make_run(pbs_root / "runs" / "benchmark")
    (pbs_root / "logs" / "otel").mkdir(parents=True)
    (pbs_root / "logs" / "sglang.log").write_bytes(b"serving log\n")
    (pbs_root / "logs" / "otel" / "otel-traces.jsonl").write_bytes(
        b'{"resourceSpans":[]}\n'
    )
    _write_json(pbs_root / "pbs_job_spec.json", {"queue": "gdev"})
    secret = pbs_root / "states" / "run-1" / "agent-bench" / "oc_gateway_env"
    secret.parent.mkdir(parents=True)
    secret.write_text("OPENCLAW_GATEWAY_TOKEN=must-not-leak\n")
    workspace = pbs_root / "workspaces" / "run-1" / "source.py"
    workspace.parent.mkdir(parents=True)
    workspace.write_text("print('must not copy workspace')\n")
    (pbs_root / "unrelated-cache.txt").write_text("must not copy\n")

    bundle, report = run_artifacts.create_single_run_bundle(
        run,
        tmp_path / "reports",
        name="with-job-logs",
    )
    catalog = json.loads((bundle / "artifact-index.json").read_text())

    assert report["artifact_bundle"]["shared_job_artifact_count"] == 1
    assert len(catalog["shared_artifacts"]) == 1
    shared = catalog["shared_artifacts"][0]
    assert shared["status_counts"] == {"copied": 3, "generated": 1}
    job_page = bundle / shared["links"]["job_evidence"]
    assert job_page.is_file()
    assert "sglang.log" in job_page.read_text()
    assert (
        bundle
        / "03_execution_evidence"
        / "run"
        / "_jobs"
        / "pbs-job"
        / "logs"
        / "sglang.log"
    ).read_bytes() == b"serving log\n"
    assert not any(path.name == "oc_gateway_env" for path in bundle.rglob("*"))
    assert not any("workspaces" in path.parts for path in bundle.rglob("*"))
    assert not any(path.name == "unrelated-cache.txt" for path in bundle.rglob("*"))
    assert not any(
        item.get("source_path") in {str(secret), str(workspace)}
        for item in shared["artifacts"]
    )
    assert (
        run_artifacts._excluded_artifact(
            Path("states/run-1/agent-bench/oc_gateway_env")
        )
        == "credential_or_secret_file"
    )


def test_compare_cli_exports_both_arms_with_portable_links(tmp_path, capsys):
    subagent_run = _make_run(tmp_path / "subagent", child_count=2)
    single_run = _make_run(tmp_path / "single-agent", child_count=0)
    output = tmp_path / "reports"

    assert compare.main(
        [
            "--subagent",
            str(subagent_run),
            "--single-agent",
            str(single_run),
            "--output-dir",
            str(output),
            "--name",
            "paired",
            "--no-plots",
            "--bootstrap-samples",
            "100",
        ]
    ) == 0
    capsys.readouterr()

    bundle = output / "paired"
    report = json.loads((bundle / "paired.json").read_text())
    assert report["paired"]["paired_task_count"] == 1
    assert report["artifact_bundle"]["selected_task_count"] == 2
    assert report["artifact_bundle"]["complete_run_count"] == 2

    markdown = (bundle / "paired.md").read_text()
    for arm in ("subagent", "single-agent"):
        detail = (
            bundle
            / "04_final_workspace_score"
            / arm
            / "task_example__seed42"
            / "index.html"
        )
        conversation = (
            bundle
            / "01_durable_session_records"
            / arm
            / "task_example__seed42"
            / "conversation"
            / "sessions"
            / "parent.jsonl"
        )
        assert detail.is_file()
        assert conversation.is_file()
        assert str(detail.relative_to(bundle).as_posix()) in markdown

    catalog = json.loads((bundle / "paired-artifact-index.json").read_text())
    assert len(catalog["runs"]) == 2
    exported_links = {item["arm"]: item["links"] for item in catalog["runs"]}
    assert report["tasks"]["subagent"][0]["artifact_links"] == (
        exported_links["subagent"]
    )
    assert report["tasks"]["single_agent"][0]["artifact_links"] == (
        exported_links["single-agent"]
    )
    assert report["tasks"]["paired"][0]["subagent_artifact_links"] == (
        exported_links["subagent"]
    )
    assert report["tasks"]["paired"][0]["single_agent_artifact_links"] == (
        exported_links["single-agent"]
    )
    assert "subagent: not collected" not in markdown
    assert "single: not collected" not in markdown
    assert not any(
        item["status"] in {"copy_failed", "generation_failed"}
        for run in catalog["runs"]
        for item in run["artifacts"]
    )
