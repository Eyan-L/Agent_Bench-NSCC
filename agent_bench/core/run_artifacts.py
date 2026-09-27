from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_SUBDIRS = (
    "01_durable_session_records",
    "02_timeline_model_trace",
    "03_execution_evidence",
    "04_final_workspace_score",
    "05_figures",
)

MINIMUM_ARTIFACTS = (
    "manifest.json",
    "steps.jsonl",
    "spans.jsonl",
    "spans_timeline.html",
    "trajectory.jsonl",
    "runtime_observability.json",
    "token_accounting.json",
    "util.jsonl",
    "conversation",
    "openclaw_response.json",
    "openclaw_stdout.log",
    "openclaw_stderr.log",
    "openclaw_gateway.log",
    "openclaw_exec_bridge.jsonl",
    "candidate.patch",
    "verdict.json",
)

HERMES_MINIMUM_ARTIFACTS = (
    "manifest.json",
    "steps.jsonl",
    "spans.jsonl",
    "spans_timeline.html",
    "trajectory.jsonl",
    "runtime_observability.json",
    "token_accounting.json",
    "util.jsonl",
    "hermes_sessions.json",
    "hermes_events.jsonl",
    "hermes_state/sessions.jsonl",
    "hermes_state/sessions",
    "hermes_state/state.db",
    "hermes_stdout.log",
    "candidate.patch",
    "verdict.json",
)

_MINIMUM_ARTIFACT_MANIFEST_FIELDS = {
    "steps.jsonl": {"step_trace_path"},
    "spans.jsonl": {"telemetry_path"},
    "spans_timeline.html": {"telemetry_path"},
    "trajectory.jsonl": {"trajectory_path"},
    "runtime_observability.json": {"runtime_observability_path"},
    "token_accounting.json": {"token_accounting_path"},
    "util.jsonl": {"util_path"},
    "verdict.json": {"verdict_path"},
}

_TIMELINE_NAMES = {
    "steps.jsonl",
    "spans.jsonl",
    "spans_raw.jsonl",
    "spans_timeline.html",
    "trajectory.jsonl",
    "token_accounting.json",
}
_FINAL_NAMES = {
    "candidate.patch",
    "manifest.json",
    "post_verifier.patch",
    "verdict.json",
}
_MODEL_OR_IMAGE_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sif",
}
_SENSITIVE_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "netrc",
    "oc_gateway_env",
}
_SHARED_JOB_DIRS = {"logs"}
_SHARED_JOB_FILES = {
    "pbs_job_spec.json",
    "placement.json",
    "placement_manifest.json",
    "swe_atlas_qa_instances.json",
    "terminal_bench_tasks.json",
}
_SHARED_JOB_CONFIG_RE = re.compile(
    r"^(?:config|experiment|placement|resolved[-_].+)\.(?:json|toml|ya?ml)$",
    re.IGNORECASE,
)
_SHARED_JOB_OUTPUT_RE = re.compile(
    r"(?:\.(?:out|err|log)|\.[oe]\d+)$",
    re.IGNORECASE,
)
_MANIFEST_ARTIFACT_KEYS = {
    "runtime_observability_path",
    "step_trace_path",
    "telemetry_path",
    "token_accounting_path",
    "trajectory_path",
    "util_path",
    "verdict_path",
}
_MANIFEST_ARTIFACT_KEY_RE = re.compile(
    r"(?:artifact|log|trace|telemetry|verdict|observability)", re.IGNORECASE
)


@dataclass(frozen=True)
class JsonDocument:
    value: dict[str, Any] | None
    status: str
    error: str | None = None


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "run"


def artifact_task_dir(instance_id: object, seed: object) -> str:
    try:
        normalized_seed = int(seed or 0)
    except (TypeError, ValueError):
        normalized_seed = 0
    return f"{safe_name(instance_id)}__seed{normalized_seed}"


def _json_document(path: Path) -> JsonDocument:
    if not path.is_file():
        return JsonDocument(None, "not_collected")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return JsonDocument(None, "invalid", str(exc))
    if not isinstance(value, dict):
        return JsonDocument(None, "invalid", "top-level JSON value is not an object")
    return JsonDocument(value, "available")


def _finite_number(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_disjoint_output(output: Path, source_dirs: Iterable[Path]) -> None:
    output_resolved = output.expanduser().resolve()
    for source in source_dirs:
        source_resolved = source.expanduser().resolve()
        if (
            output_resolved == source_resolved
            or _is_relative_to(output_resolved, source_resolved)
            or _is_relative_to(source_resolved, output_resolved)
        ):
            raise ValueError(
                "report output must not overlap a source run: "
                f"output={output_resolved}, source={source_resolved}"
            )


def prepare_bundle_dir(
    output_dir: Path,
    basename: str,
    *,
    source_dirs: Iterable[Path] = (),
) -> Path:
    bundle_dir = output_dir.expanduser() / safe_name(basename)
    ensure_disjoint_output(bundle_dir, source_dirs)
    if bundle_dir.exists():
        if bundle_dir.is_symlink():
            raise ValueError(f"refusing to replace symlinked report directory: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    for name in BUNDLE_SUBDIRS:
        (bundle_dir / name).mkdir()
    return bundle_dir


def _artifact_category(relative_path: Path) -> str:
    if relative_path.parts and relative_path.parts[0] == "conversation":
        return BUNDLE_SUBDIRS[0]
    if relative_path.name == "hermes_sessions.json":
        return BUNDLE_SUBDIRS[0]
    if relative_path.parts and relative_path.parts[0] == "hermes_state":
        if len(relative_path.parts) > 1 and (
            relative_path.parts[1] == "sessions"
            or relative_path.parts[1] in {
                "sessions.jsonl", "state.db", "state.db-shm", "state.db-wal"
            }
        ):
            return BUNDLE_SUBDIRS[0]
        if relative_path.name == "events.jsonl":
            return BUNDLE_SUBDIRS[1]
    name = relative_path.name
    if name == "hermes_events.jsonl":
        return BUNDLE_SUBDIRS[1]
    if name in _TIMELINE_NAMES:
        return BUNDLE_SUBDIRS[1]
    if name in _FINAL_NAMES or "verifier" in name.lower():
        return BUNDLE_SUBDIRS[3]
    return BUNDLE_SUBDIRS[2]


def _excluded_artifact(path: Path) -> str | None:
    lowered_parts = {part.lower() for part in path.parts}
    if path.suffix.lower() in _MODEL_OR_IMAGE_SUFFIXES:
        return "model_or_container_image"
    lowered_name = path.name.lower()
    if path.suffix.lower() == ".bin" and any(
        marker in lowered_name for marker in ("model", "pytorch", "weight")
    ):
        return "model_or_container_image"
    if lowered_parts & _SENSITIVE_NAMES:
        return "credential_or_secret_file"
    if lowered_name in _SENSITIVE_NAMES:
        return "credential_or_secret_file"
    return None


def _shared_job_root_file_allowed(path: Path) -> bool:
    name = path.name
    return (
        name in _SHARED_JOB_FILES
        or _SHARED_JOB_CONFIG_RE.fullmatch(name) is not None
        or _SHARED_JOB_OUTPUT_RE.search(name) is not None
    )


def _manifest_reference_candidates(run_dir: Path, raw: str) -> list[Path]:
    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        return [raw_path, run_dir / raw_path.name]
    candidates = [run_dir / raw_path]
    basename_fallback = run_dir / raw_path.name
    if basename_fallback != candidates[0]:
        candidates.append(basename_fallback)
    return candidates


def resolve_run_artifact(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    fallback: str,
) -> Path:
    """Resolve an artifact locally first, then use its manifest reference.

    Copied runs preserve source manifests byte-for-byte, including historical
    absolute paths. The resolved run directory is authoritative when it has
    the requested artifact; external/custom manifest references remain a
    compatibility fallback.
    """
    run_dir = run_dir.expanduser().resolve()
    raw = manifest.get(field)
    referenced = (
        _manifest_reference_candidates(run_dir, raw)
        if isinstance(raw, str) and raw
        else []
    )
    fallback_path = run_dir / fallback
    candidates = [fallback_path]
    for path in referenced:
        local_basename = run_dir / path.name
        if local_basename not in candidates:
            candidates.append(local_basename)
    for path in referenced:
        if path not in candidates:
            candidates.append(path)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _iter_manifest_references(value: object, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            is_artifact_reference = (
                key_text in _MANIFEST_ARTIFACT_KEYS
                or (
                    key_text.endswith(("_path", "_file"))
                    and _MANIFEST_ARTIFACT_KEY_RE.search(key_text)
                )
            )
            if isinstance(child, str) and is_artifact_reference:
                yield child_prefix, child
            yield from _iter_manifest_references(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_manifest_references(child, f"{prefix}[{index}]")


def _entry(
    *,
    source: Path | None,
    bundle_path: Path | None,
    status: str,
    source_kind: str,
    size_bytes: int | None = None,
    sha256: str | None = None,
    error: str | None = None,
    manifest_fields: Sequence[str] = (),
    generated_from: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "source_path": str(source) if source is not None else None,
        "bundle_path": str(bundle_path.as_posix()) if bundle_path is not None else None,
        "status": status,
        "source_kind": source_kind,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "error": error,
        "manifest_fields": list(manifest_fields),
        "generated_from": list(generated_from),
    }


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    bundle_dir: Path,
    source_kind: str,
    manifest_fields: Sequence[str] = (),
) -> dict[str, Any]:
    relative_destination = destination.relative_to(bundle_dir)
    try:
        source_size = source.stat().st_size
        source_hash = _sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_size = destination.stat().st_size
        copied_hash = _sha256(destination)
        if copied_size != source_size or copied_hash != source_hash:
            raise OSError(
                "copy verification failed: "
                f"source={source_size}/{source_hash}, "
                f"destination={copied_size}/{copied_hash}"
            )
        return _entry(
            source=source,
            bundle_path=relative_destination,
            status="copied",
            source_kind=source_kind,
            size_bytes=copied_size,
            sha256=copied_hash,
            manifest_fields=manifest_fields,
        )
    except Exception as exc:  # noqa: BLE001 - every failed artifact must be indexed
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        return _entry(
            source=source,
            bundle_path=relative_destination,
            status="copy_failed",
            source_kind=source_kind,
            error=f"{type(exc).__name__}: {exc}",
            manifest_fields=manifest_fields,
        )


def _event_timestamp_ns(event: Mapping[str, Any]) -> int | None:
    raw = event.get("timestamp_ns")
    try:
        if raw is not None:
            return int(raw)
    except (TypeError, ValueError):
        pass
    raw_timestamp = event.get("timestamp")
    if not raw_timestamp:
        return None
    try:
        return int(
            datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
            .timestamp()
            * 1e9
        )
    except ValueError:
        return None


def summarize_jsonl(path: Path, *, event_field: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_collected", "record_count": None}
    records = 0
    invalid_lines = 0
    event_types: Counter[str] = Counter()
    session_keys: set[str] = set()
    timestamps: list[int] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if not isinstance(item, dict):
                    invalid_lines += 1
                    continue
                records += 1
                if event_field:
                    event_types[str(item.get(event_field) or "unknown")] += 1
                session_key = str(item.get("session_key") or "").strip()
                if session_key:
                    session_keys.add(session_key)
                timestamp = _event_timestamp_ns(item)
                if timestamp is not None and timestamp > 0:
                    timestamps.append(timestamp)
    except OSError as exc:
        return {
            "status": "unreadable",
            "record_count": None,
            "error": str(exc),
        }
    span_seconds = None
    if len(timestamps) >= 2:
        span_seconds = (max(timestamps) - min(timestamps)) / 1e9
    return {
        "status": "available" if invalid_lines == 0 else "partially_invalid",
        "record_count": records,
        "invalid_line_count": invalid_lines,
        "event_types": dict(sorted(event_types.items())),
        "session_keys": sorted(session_keys),
        "timed_record_count": len(timestamps),
        "span_seconds": span_seconds,
    }


def summarize_resources(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "not_collected",
            "source": "direct util.jsonl samples",
            "sample_count": None,
        }
    sample_count = 0
    invalid_lines = 0
    timestamps: list[int] = []
    cpu_values: list[float] = []
    memory_values: list[float] = []
    gpu_values: dict[str, dict[str, list[float]]] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if not isinstance(item, dict) or item.get("_meta"):
                    continue
                sample_count += 1
                try:
                    timestamp = int(item.get("ts_ns") or 0)
                except (TypeError, ValueError):
                    timestamp = 0
                if timestamp:
                    timestamps.append(timestamp)
                cpu = item.get("cpu") or {}
                if isinstance(cpu, dict):
                    if (value := _finite_number(cpu.get("pct"))) is not None:
                        cpu_values.append(value)
                    if (value := _finite_number(cpu.get("mem_used_mb"))) is not None:
                        memory_values.append(value)
                for gpu in item.get("gpu") or []:
                    if not isinstance(gpu, dict):
                        continue
                    index = str(gpu.get("index", "unknown"))
                    values = gpu_values.setdefault(
                        index,
                        {
                            "util_gpu_pct": [],
                            "util_mem_pct": [],
                            "mem_used_mb": [],
                            "power_w": [],
                        },
                    )
                    for field in values:
                        if (number := _finite_number(gpu.get(field))) is not None:
                            values[field].append(number)
    except OSError as exc:
        return {
            "status": "unreadable",
            "source": "direct util.jsonl samples",
            "sample_count": None,
            "error": str(exc),
        }

    def aggregate(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "status": "available" if invalid_lines == 0 else "partially_invalid",
        "source": "direct util.jsonl samples",
        "sample_count": sample_count,
        "invalid_line_count": invalid_lines,
        "start_timestamp_ns": min(timestamps) if timestamps else None,
        "end_timestamp_ns": max(timestamps) if timestamps else None,
        "cpu_pct": aggregate(cpu_values),
        "host_memory_used_mb": aggregate(memory_values),
        "gpus": {
            index: {field: aggregate(values) for field, values in fields.items()}
            for index, fields in sorted(gpu_values.items())
        },
        "notes": [
            "GPU/CPU values are direct sampler observations, not utilization inferred from spans.",
        ],
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest_doc = _json_document(run_dir / "manifest.json")
    manifest = manifest_doc.value or {}
    artifact_paths = {
        "verdict": resolve_run_artifact(
            run_dir, manifest, field="verdict_path", fallback="verdict.json"
        ),
        "runtime_observability": resolve_run_artifact(
            run_dir,
            manifest,
            field="runtime_observability_path",
            fallback="runtime_observability.json",
        ),
        "token_accounting": resolve_run_artifact(
            run_dir,
            manifest,
            field="token_accounting_path",
            fallback="token_accounting.json",
        ),
        "trajectory": resolve_run_artifact(
            run_dir, manifest, field="trajectory_path", fallback="trajectory.jsonl"
        ),
        "steps": resolve_run_artifact(
            run_dir, manifest, field="step_trace_path", fallback="steps.jsonl"
        ),
        "spans": resolve_run_artifact(
            run_dir, manifest, field="telemetry_path", fallback="spans.jsonl"
        ),
        "resources": resolve_run_artifact(
            run_dir, manifest, field="util_path", fallback="util.jsonl"
        ),
    }
    verdict_doc = _json_document(artifact_paths["verdict"])
    observability_doc = _json_document(artifact_paths["runtime_observability"])
    token_doc = _json_document(artifact_paths["token_accounting"])
    verdict = verdict_doc.value or {}
    observability = observability_doc.value or {}

    trajectory = summarize_jsonl(artifact_paths["trajectory"], event_field="event_type")
    steps = summarize_jsonl(artifact_paths["steps"])
    spans = summarize_jsonl(artifact_paths["spans"], event_field="name")
    phase = manifest.get("phase_timing") or {}
    if not isinstance(phase, dict):
        phase = {}
    trajectory_events = trajectory.get("event_types") or {}
    agent_execution_seconds = manifest.get("agent_execution_seconds")
    if agent_execution_seconds is None:
        agent_execution_seconds = phase.get("agent_execution_seconds")

    runtime_observability = manifest.get("runtime_observability") or {}
    if not observability and isinstance(runtime_observability, dict):
        observability = runtime_observability

    logical_tokens = manifest.get("trajectory_tokens")
    serving_tokens = manifest.get("serving_tokens")
    return {
        "schema_version": 1,
        "run": str(run_dir),
        "task": {
            "instance_id": manifest.get("instance_id"),
            "seed": manifest.get("seed"),
            "run_id": manifest.get("run_id") or run_dir.name,
            "condition_id": manifest.get("condition_id"),
            "serving_config_id": manifest.get("serving_config_id"),
        },
        "result": {
            "status": manifest.get("status") or verdict.get("status"),
            "agent_execution_outcome": manifest.get("agent_execution_outcome")
            or verdict.get("agent_completion_status"),
            "agent_runtime_mode": manifest.get("agent_runtime_mode"),
            "agent_exit_code": verdict.get("agent_exit_code")
            if "agent_exit_code" in verdict
            else manifest.get("exit_code"),
            "agent_completion_error": manifest.get("agent_completion_error"),
            "error_message": manifest.get("error_message"),
            "failure_phase": manifest.get("failure_phase"),
            "benchmark_valid": manifest.get("benchmark_valid"),
            "benchmark_invalid_reason": manifest.get("benchmark_invalid_reason"),
            "verifier_eligible": manifest.get("verifier_eligible"),
            "verifier_status": manifest.get("verifier_status")
            or verdict.get("verifier_status"),
            "verifier_skipped": manifest.get("verifier_skipped"),
            "reward": manifest.get("reward", verdict.get("reward")),
            "test_pass_rate": manifest.get(
                "test_pass_rate", verdict.get("test_pass_rate")
            ),
        },
        "timing": {
            "agent_execution_seconds": agent_execution_seconds,
            "total_instance_seconds": manifest.get("total_instance_seconds"),
            "duration_seconds": manifest.get("duration_seconds"),
            "wall_start_ns": manifest.get("wall_start_ns"),
            "wall_end_ns": manifest.get("wall_end_ns"),
            "phases": phase,
        },
        "activity": {
            "steps": steps,
            "spans": spans,
            "trajectory": trajectory,
            "model_request_count": trajectory_events.get("model_request"),
            "tool_call_count": trajectory_events.get("tool_call"),
            "tool_result_count": trajectory_events.get("tool_result"),
            "compaction_count": trajectory_events.get("compaction"),
            "context_overflow_count": trajectory_events.get("context_overflow"),
        },
        "tokens": {
            "logical": {
                "source": "manifest trajectory_tokens/token accounting",
                "total_prompt_tokens": manifest.get("total_prompt_tokens"),
                "total_completion_tokens": manifest.get("total_completion_tokens"),
                "trajectory_tokens": logical_tokens,
                "token_accounting_status": token_doc.status,
            },
            "serving": {
                "source": "serving request processing; repeated context is cumulative",
                "serving_tokens": serving_tokens,
            },
            "notes": [
                "Logical trajectory tokens and serving processed tokens are different metrics.",
                "Cumulative serving input tokens are not unique text tokens or uncached prefill work.",
            ],
        },
        "lifecycle": {
            "source": "runtime_observability.json or manifest runtime_observability",
            "status": observability_doc.status
            if observability_doc.status != "not_collected"
            else ("embedded_in_manifest" if observability else "not_collected"),
            "child_session_count": observability.get("child_session_count"),
            "materialized_child_session_count": observability.get(
                "materialized_child_session_count"
            ),
            "online_index_child_count_at_archive": observability.get(
                "online_index_child_count_at_archive"
            ),
            "restored_archived_child_count": observability.get(
                "restored_archived_child_count"
            ),
            "child_count_source_quality": observability.get(
                "child_count_source_quality"
            ),
            "child_runtime_complete_count": observability.get(
                "child_runtime_complete_count"
            ),
            "child_runtime_unknown_count": observability.get(
                "child_runtime_unknown_count"
            ),
            "max_concurrent_children": observability.get("max_concurrent_children"),
            "max_concurrent_children_lower_bound": observability.get(
                "max_concurrent_children_lower_bound"
            ),
            "child_sum_runtime_seconds": observability.get("child_sum_runtime_seconds"),
            "child_sum_runtime_lower_bound_seconds": observability.get(
                "child_sum_runtime_lower_bound_seconds"
            ),
            "child_union_active_seconds": observability.get("child_union_active_seconds"),
            "child_union_active_lower_bound_seconds": observability.get(
                "child_union_active_lower_bound_seconds"
            ),
            "child_overlap_wall_seconds": observability.get("child_overlap_wall_seconds"),
            "child_concurrency_excess_seconds": observability.get(
                "child_concurrency_excess_seconds"
            ),
            "child_derived_metrics_quality": observability.get(
                "child_derived_metrics_quality"
            ),
            "child_start_timestamp_quality_counts": observability.get(
                "child_start_timestamp_quality_counts"
            ),
            "child_terminal_timestamp_quality_counts": observability.get(
                "child_terminal_timestamp_quality_counts"
            ),
            "child_start_timestamp_conflict_count": observability.get(
                "child_start_timestamp_conflict_count"
            ),
            "child_terminal_identity_qualified_count": observability.get(
                "child_terminal_identity_qualified_count"
            ),
            "reconstructed_child_state_counts": observability.get(
                "reconstructed_child_state_counts"
            ),
            "parent_terminal_child_snapshot": observability.get(
                "parent_terminal_child_snapshot"
            ),
            "child_runtime_by_session": observability.get(
                "child_runtime_by_session"
            ),
            "child_runtime_policy": observability.get("child_runtime_policy"),
            "agent_timeline": observability.get("agent_timeline"),
            "notes": [
                "Child lifetime intervals describe lifecycle, not continuous compute utilization.",
                "Timestamp quality/censored fields are preserved in runtime_observability.json.",
            ],
        },
        "resources": summarize_resources(artifact_paths["resources"]),
        "source_documents": {
            "manifest": {
                "status": manifest_doc.status,
                "error": manifest_doc.error,
                "path": str(run_dir / "manifest.json"),
            },
            "verdict": {
                "status": verdict_doc.status,
                "error": verdict_doc.error,
                "path": str(artifact_paths["verdict"]),
            },
            "runtime_observability": {
                "status": observability_doc.status,
                "error": observability_doc.error,
                "path": str(artifact_paths["runtime_observability"]),
            },
            "token_accounting": {
                "status": token_doc.status,
                "error": token_doc.error,
                "path": str(artifact_paths["token_accounting"]),
            },
        },
        "artifact_sources": {
            name: str(path) for name, path in artifact_paths.items()
        },
        "measurement_notes": [
            "steps.jsonl is a coarse harness step trace; spans.jsonl contains per-call timing when collected.",
            "time_to_first_byte_ms remains TTFB and is not relabeled as TTFT.",
            "Missing fields remain null/not_collected and are not replaced with numeric zero.",
        ],
    }


def _manifest_for_run(run_dir: Path) -> dict[str, Any]:
    document = _json_document(run_dir / "manifest.json")
    return document.value or {}


def _copy_run_tree(
    run_dir: Path,
    bundle_dir: Path,
    *,
    arm: str,
    task_dir: str,
    omit_candidate_workspace: bool = False,
) -> tuple[list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    copied_by_source: dict[Path, dict[str, Any]] = {}

    if omit_candidate_workspace:
        candidate = run_dir / "candidate_workspace"
        if candidate.exists() or candidate.is_symlink():
            entries.append(_entry(
                source=candidate,
                bundle_path=None,
                status="excluded",
                source_kind="run_tree",
                error="candidate_workspace_omitted_from_agent_report",
            ))

    def sources():
        if not omit_candidate_workspace:
            yield from sorted(run_dir.rglob("*"))
            return
        # Prune the entire workspace before traversal. Filtering after rglob
        # would still walk thousands of repository and build-output files.
        for top in sorted(run_dir.iterdir()):
            if top.name == "candidate_workspace":
                continue
            if top.is_dir() and not top.is_symlink():
                yield from sorted(top.rglob("*"))
            else:
                yield top

    for source in sources():
        if source.is_dir():
            continue
        relative = source.relative_to(run_dir)
        category = _artifact_category(relative)
        destination = bundle_dir / category / arm / task_dir / relative
        if source.is_symlink():
            item = _entry(
                source=source,
                bundle_path=destination.relative_to(bundle_dir),
                status="skipped_unsafe",
                source_kind="run_tree",
                error="symbolic links are not followed",
            )
        elif (reason := _excluded_artifact(relative)) is not None:
            item = _entry(
                source=source,
                bundle_path=None,
                status="excluded",
                source_kind="run_tree",
                error=reason,
            )
        else:
            item = _copy_verified(
                source,
                destination,
                bundle_dir=bundle_dir,
                source_kind="run_tree",
            )
        entries.append(item)
        copied_by_source[source.resolve()] = item
    return entries, copied_by_source


def discover_shared_job_root(run_dir: Path) -> Path | None:
    """Find an enclosing experiment/PBS directory without guessing globally."""
    resolved = run_dir.expanduser().resolve()
    for parent in list(resolved.parents)[:5]:
        if (parent / "logs").is_dir() and (parent / "runs").is_dir():
            return parent
    return None


def export_shared_job_artifacts(
    source_root: Path,
    bundle_dir: Path,
    *,
    arm: str,
    job_id: object,
) -> dict[str, Any]:
    """Export an allowlisted set of job-level logs/configuration once."""
    source_root = source_root.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    ensure_disjoint_output(bundle_dir, (source_root,))
    arm = safe_name(arm)
    job_dir = safe_name(job_id)
    destination_root = (
        bundle_dir / BUNDLE_SUBDIRS[2] / arm / "_jobs" / job_dir
    )
    entries: list[dict[str, Any]] = []
    sources: set[Path] = set()
    for name in _SHARED_JOB_DIRS:
        shared_dir = source_root / name
        if shared_dir.is_dir() and not shared_dir.is_symlink():
            for current, directories, filenames in os.walk(
                shared_dir, followlinks=False
            ):
                current_path = Path(current)
                directories[:] = [
                    directory
                    for directory in directories
                    if not (current_path / directory).is_symlink()
                ]
                sources.update(current_path / filename for filename in filenames)
    sources.update(
        path
        for path in source_root.iterdir()
        if path.is_file() and _shared_job_root_file_allowed(path)
    )
    for source in sorted(sources):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_symlink():
            item = _entry(
                source=source,
                bundle_path=destination.relative_to(bundle_dir),
                status="skipped_unsafe",
                source_kind="shared_job_tree",
                error="symbolic links are not followed",
            )
        elif (reason := _excluded_artifact(relative)) is not None:
            item = _entry(
                source=source,
                bundle_path=None,
                status="excluded",
                source_kind="shared_job_tree",
                error=reason,
            )
        else:
            item = _copy_verified(
                source,
                destination,
                bundle_dir=bundle_dir,
                source_kind="shared_job_tree",
            )
        entries.append(item)

    index_path = destination_root / "index.html"
    links = []
    for item in entries:
        if item.get("status") != "copied" or not item.get("bundle_path"):
            continue
        href = _relative_href(
            index_path.relative_to(bundle_dir), str(item["bundle_path"])
        )
        links.append(
            f'<li><a href="{html.escape(href)}">'
            f'{html.escape(str(Path(str(item["bundle_path"])).relative_to(destination_root.relative_to(bundle_dir))))}'
            f'</a> ({item.get("size_bytes")} bytes)</li>'
        )
    generated = _write_generated(
        index_path,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Shared job artifacts</title>"
        "<style>body{font:14px sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}</style>"
        "</head><body><h1>Shared job artifacts</h1>"
        "<p>Serving, OTel, PBS, and experiment-level files shared by selected task runs.</p>"
        f"<ul>{''.join(links) or '<li>No shared job artifacts were collected.</li>'}</ul>"
        "</body></html>",
        bundle_dir=bundle_dir,
        generated_from=(str(source_root),),
    )
    entries.append(generated)
    counts = Counter(str(item.get("status")) for item in entries)
    return {
        "kind": "shared_job_artifacts",
        "arm": arm,
        "job_id": str(job_id),
        "source_root": str(source_root),
        "links": {
            "job_evidence": str(index_path.relative_to(bundle_dir).as_posix())
        },
        "status_counts": dict(sorted(counts.items())),
        "complete": not any(
            item.get("status") in {"copy_failed", "generation_failed"}
            for item in entries
        ),
        "artifacts": entries,
    }


def _copy_manifest_references(
    manifest: Mapping[str, Any],
    run_dir: Path,
    bundle_dir: Path,
    *,
    arm: str,
    task_dir: str,
    entries: list[dict[str, Any]],
    copied_by_source: dict[Path, dict[str, Any]],
) -> None:
    seen_fields: set[tuple[str, str]] = set()
    for field, raw in _iter_manifest_references(manifest):
        if not raw or (field, raw) in seen_fields:
            continue
        seen_fields.add((field, raw))
        candidates = _manifest_reference_candidates(run_dir, raw)
        source = next((path for path in candidates if path.exists()), candidates[0])
        if source.exists() and source.is_dir():
            referenced_files = sorted(path for path in source.rglob("*") if path.is_file())
            if not referenced_files:
                entries.append(
                    _entry(
                        source=source,
                        bundle_path=None,
                        status="not_collected",
                        source_kind="manifest_reference",
                        error="referenced directory is empty",
                        manifest_fields=(field,),
                    )
                )
            for referenced in referenced_files:
                _copy_external_reference(
                    referenced,
                    source,
                    field,
                    bundle_dir,
                    arm,
                    task_dir,
                    entries,
                    copied_by_source,
                )
            continue
        if not source.is_file():
            entries.append(
                _entry(
                    source=source,
                    bundle_path=None,
                    status="source_missing",
                    source_kind="manifest_reference",
                    error=f"manifest field {field} references a missing source",
                    manifest_fields=(field,),
                )
            )
            continue
        _copy_external_reference(
            source,
            None,
            field,
            bundle_dir,
            arm,
            task_dir,
            entries,
            copied_by_source,
        )


def _copy_external_reference(
    source: Path,
    reference_root: Path | None,
    field: str,
    bundle_dir: Path,
    arm: str,
    task_dir: str,
    entries: list[dict[str, Any]],
    copied_by_source: dict[Path, dict[str, Any]],
) -> None:
    resolved = source.resolve()
    existing = copied_by_source.get(resolved)
    if existing is not None:
        fields = existing.setdefault("manifest_fields", [])
        if field not in fields:
            fields.append(field)
        return
    relative = (
        source.relative_to(reference_root)
        if reference_root is not None
        else Path(source.name)
    )
    safe_field = safe_name(field)
    synthetic_relative = Path("manifest_references") / safe_field / relative
    destination = (
        bundle_dir
        / _artifact_category(Path(source.name))
        / arm
        / task_dir
        / synthetic_relative
    )
    if source.is_symlink():
        item = _entry(
            source=source,
            bundle_path=destination.relative_to(bundle_dir),
            status="skipped_unsafe",
            source_kind="manifest_reference",
            error="symbolic links are not followed",
            manifest_fields=(field,),
        )
    elif (reason := _excluded_artifact(source)) is not None:
        item = _entry(
            source=source,
            bundle_path=None,
            status="excluded",
            source_kind="manifest_reference",
            error=reason,
            manifest_fields=(field,),
        )
    else:
        item = _copy_verified(
            source,
            destination,
            bundle_dir=bundle_dir,
            source_kind="manifest_reference",
            manifest_fields=(field,),
        )
    entries.append(item)
    copied_by_source[resolved] = item


def _append_missing_minimum_entries(
    run_dir: Path,
    entries: list[dict[str, Any]],
) -> None:
    present = {
        str(Path(str(item["source_path"])).relative_to(run_dir))
        for item in entries
        if item.get("source_kind") == "run_tree"
        and item.get("source_path")
        and _is_relative_to(Path(str(item["source_path"])), run_dir)
    }
    manifest_reference_names = {
        Path(str(item["source_path"])).name
        for item in entries
        if item.get("source_kind") == "manifest_reference"
        and item.get("status") == "copied"
        and item.get("source_path")
    }
    manifest_reference_fields = {
        str(field).rsplit(".", 1)[-1]
        for item in entries
        for field in item.get("manifest_fields") or []
    }
    is_hermes = any(
        (run_dir / name).exists()
        for name in ("hermes_state", "hermes_sessions.json", "hermes_stdout.log")
    )
    minimum_artifacts = HERMES_MINIMUM_ARTIFACTS if is_hermes else MINIMUM_ARTIFACTS
    for relative in minimum_artifacts:
        if relative in {"conversation", "hermes_state/sessions"}:
            if any(path == relative or path.startswith(f"{relative}/") for path in present):
                continue
        elif (
            relative in present
            or Path(relative).name in manifest_reference_names
            or bool(
                _MINIMUM_ARTIFACT_MANIFEST_FIELDS.get(relative, set())
                & manifest_reference_fields
            )
        ):
            continue
        entries.append(
            _entry(
                source=run_dir / relative,
                bundle_path=None,
                status="not_collected",
                source_kind="expected_artifact",
                error="artifact was not present in the source run",
            )
        )


def _write_generated(
    path: Path,
    content: str,
    *,
    bundle_dir: Path,
    generated_from: Sequence[str] = (),
) -> dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return _entry(
            source=None,
            bundle_path=path.relative_to(bundle_dir),
            status="generated",
            source_kind="report_generated",
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            generated_from=generated_from,
        )
    except Exception as exc:  # noqa: BLE001 - report must expose generation errors
        return _entry(
            source=None,
            bundle_path=path.relative_to(bundle_dir),
            status="generation_failed",
            source_kind="report_generated",
            error=f"{type(exc).__name__}: {exc}",
            generated_from=generated_from,
        )


def _ensure_spans_timeline(
    run_dir: Path,
    manifest: Mapping[str, Any],
    bundle_dir: Path,
    *,
    arm: str,
    task_dir: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    destination = (
        bundle_dir
        / BUNDLE_SUBDIRS[1]
        / arm
        / task_dir
        / "spans_timeline.html"
    )
    copied = next(
        (
            item
            for item in entries
            if item.get("status") == "copied"
            and item.get("bundle_path")
            == str(destination.relative_to(bundle_dir).as_posix())
        ),
        None,
    )
    if copied is not None:
        return {"status": "copied", "path": copied["bundle_path"], "error": None}

    spans_path = resolve_run_artifact(
        run_dir, manifest, field="telemetry_path", fallback="spans.jsonl"
    )
    if not spans_path.is_file() or spans_path.stat().st_size <= 0:
        return {"status": "unavailable", "path": None, "error": "spans.jsonl not collected"}
    try:
        from agent_bench.telemetry.otel.visualize_spans import _load_spans, render_html

        spans = _load_spans(spans_path)
        if not spans:
            return {"status": "unavailable", "path": None, "error": "no valid spans"}
        observability_path = resolve_run_artifact(
            run_dir,
            manifest,
            field="runtime_observability_path",
            fallback="runtime_observability.json",
        )
        observability = _json_document(observability_path).value or {}
        timeline = observability.get("agent_timeline")
        if not isinstance(timeline, dict):
            timeline = None
        item = _write_generated(
            destination,
            render_html(spans, spans_path, agent_timeline=timeline),
            bundle_dir=bundle_dir,
            generated_from=(str(spans_path), str(observability_path)),
        )
        entries.append(item)
        return {
            "status": item["status"],
            "path": item["bundle_path"] if item["status"] == "generated" else None,
            "error": item.get("error"),
        }
    except Exception as exc:  # noqa: BLE001 - source data remains available
        error = f"{type(exc).__name__}: {exc}"
        entries.append(
            _entry(
                source=None,
                bundle_path=destination.relative_to(bundle_dir),
                status="generation_failed",
                source_kind="report_generated",
                error=error,
                generated_from=(str(spans_path),),
            )
        )
        return {
            "status": "generation_failed",
            "path": None,
            "error": error,
        }


def _relative_href(from_path: Path, to_bundle_path: str) -> str:
    return Path(os.path.relpath(to_bundle_path, start=from_path.parent)).as_posix()


def _copied_artifact_link(
    entries: Sequence[Mapping[str, Any]],
    *,
    manifest_field: str | None = None,
    fallback_names: Sequence[str] = (),
) -> str | None:
    """Return the exported path for a logical artifact, not a guessed path."""
    if manifest_field:
        for item in entries:
            fields = {
                str(field).rsplit(".", 1)[-1]
                for field in item.get("manifest_fields") or []
            }
            if (
                item.get("status") == "copied"
                and item.get("bundle_path")
                and manifest_field in fields
            ):
                return str(item["bundle_path"])
    names = set(fallback_names)
    if names:
        for item in entries:
            source = item.get("source_path")
            if (
                item.get("status") == "copied"
                and item.get("bundle_path")
                and source
                and Path(str(source)).name in names
            ):
                return str(item["bundle_path"])
    return None


def _render_conversation_index(
    index_path: Path,
    bundle_dir: Path,
    entries: Sequence[dict[str, Any]],
) -> str:
    links = []
    for item in entries:
        relative = str(item.get("bundle_path") or "")
        if (
            item.get("status") != "copied"
            or not relative.startswith(f"{BUNDLE_SUBDIRS[0]}/")
        ):
            continue
        href = _relative_href(index_path.relative_to(bundle_dir), relative)
        links.append(
            f'<li><a href="{html.escape(href)}">{html.escape(Path(relative).name)}</a> '
            f'({item.get("size_bytes")} bytes)</li>'
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Conversation artifacts</title>"
        "<style>body{font:14px sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}</style>"
        "</head><body><h1>Conversation artifacts</h1><p>Original files are copied byte-for-byte.</p>"
        f"<ul>{''.join(links) or '<li>No conversation files were collected.</li>'}</ul>"
        "</body></html>"
    )


def _render_detail_html(
    summary: Mapping[str, Any],
    detail_path: Path,
    bundle_dir: Path,
    entries: Sequence[dict[str, Any]],
    related_links: Mapping[str, str] | None = None,
) -> str:
    rows = []
    for item in sorted(entries, key=lambda entry: str(entry.get("bundle_path") or entry.get("source_path") or "")):
        relative = item.get("bundle_path")
        if relative and item.get("status") in {"copied", "generated"}:
            href = _relative_href(detail_path.relative_to(bundle_dir), str(relative))
            label = f'<a href="{html.escape(href)}">{html.escape(str(relative))}</a>'
        else:
            label = html.escape(str(relative or item.get("source_path") or "unavailable"))
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{html.escape(str(item.get('status')))}</td>"
            f"<td>{html.escape(str(item.get('source_kind')))}</td>"
            f"<td>{html.escape(str(item.get('size_bytes') if item.get('size_bytes') is not None else 'NA'))}</td>"
            f"<td>{html.escape(str(item.get('error') or ''))}</td>"
            "</tr>"
        )
    task = summary.get("task") or {}
    result = summary.get("result") or {}
    title = f"{task.get('instance_id') or task.get('run_id') or 'Agent Bench run'}"
    related = []
    for label, target in (related_links or {}).items():
        href = _relative_href(detail_path.relative_to(bundle_dir), target)
        related.append(
            f'<li><a href="{html.escape(href)}">{html.escape(label.replace("_", " ").title())}</a></li>'
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:14px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dbe2ea;padding:.45rem;vertical-align:top;text-align:left}}
th{{background:#edf2f7}}pre{{background:#f6f8fa;padding:1rem;overflow:auto}}.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.75rem}}
.card{{border:1px solid #dbe2ea;padding:.8rem;background:#fff}}code{{word-break:break-all}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">
<div class="card"><strong>Agent outcome</strong><br>{html.escape(str(result.get('agent_execution_outcome')))}</div>
<div class="card"><strong>Benchmark valid</strong><br>{html.escape(str(result.get('benchmark_valid')))}</div>
<div class="card"><strong>Verifier</strong><br>{html.escape(str(result.get('verifier_status')))}</div>
<div class="card"><strong>Reward</strong><br>{html.escape(str(result.get('reward')))}</div>
<div class="card"><strong>Failure phase</strong><br>{html.escape(str(result.get('failure_phase')))}</div>
</div>
<h2>Measurement semantics</h2><ul>
<li><code>steps.jsonl</code> is coarse; <code>spans.jsonl</code> is the per-call trace.</li>
<li>Logical trajectory tokens and cumulative serving processed tokens are separate.</li>
<li>Child intervals are lifecycle observations, not continuous compute activity.</li>
<li>Direct, synthesized, inferred, and censored source/quality fields remain in the original files.</li>
<li>TTFB remains TTFB; it is not relabeled as TTFT.</li></ul>
<h2>Related job artifacts</h2><ul>{''.join(related) or '<li>Not collected for this run.</li>'}</ul>
<h2>Artifacts</h2><table><thead><tr><th>Bundle path</th><th>Status</th><th>Source</th><th>Bytes</th><th>Issue</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Run summary</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
</body></html>"""


def export_run_artifacts(
    run_dir: Path,
    bundle_dir: Path,
    *,
    arm: str,
    instance_id: object | None = None,
    seed: object | None = None,
    job_id: object | None = None,
    related_links: Mapping[str, str] | None = None,
    omit_candidate_workspace: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    ensure_disjoint_output(bundle_dir, (run_dir,))
    manifest = _manifest_for_run(run_dir)
    instance_id = instance_id if instance_id is not None else manifest.get("instance_id") or run_dir.name
    seed = seed if seed is not None else manifest.get("seed") or 0
    task_dir = artifact_task_dir(instance_id, seed)
    arm = safe_name(arm)

    entries, copied_by_source = _copy_run_tree(
        run_dir,
        bundle_dir,
        arm=arm,
        task_dir=task_dir,
        omit_candidate_workspace=omit_candidate_workspace,
    )
    _copy_manifest_references(
        manifest,
        run_dir,
        bundle_dir,
        arm=arm,
        task_dir=task_dir,
        entries=entries,
        copied_by_source=copied_by_source,
    )
    _append_missing_minimum_entries(run_dir, entries)
    viewer = _ensure_spans_timeline(
        run_dir,
        manifest,
        bundle_dir,
        arm=arm,
        task_dir=task_dir,
        entries=entries,
    )
    summary = analyze_run(run_dir)

    conversation_index = (
        bundle_dir / BUNDLE_SUBDIRS[0] / arm / task_dir / "index.html"
    )
    conversation_item = _write_generated(
        conversation_index,
        _render_conversation_index(conversation_index, bundle_dir, entries),
        bundle_dir=bundle_dir,
        generated_from=("conversation/",),
    )
    entries.append(conversation_item)

    detail_path = bundle_dir / BUNDLE_SUBDIRS[3] / arm / task_dir / "index.html"
    detail_item = _write_generated(
        detail_path,
        _render_detail_html(
            summary,
            detail_path,
            bundle_dir,
            entries,
            related_links=related_links,
        ),
        bundle_dir=bundle_dir,
        generated_from=("manifest.json", "verdict.json", "runtime_observability.json"),
    )
    entries.append(detail_item)

    counts = Counter(str(item.get("status")) for item in entries)
    copied_files = [
        str(item["bundle_path"])
        for item in entries
        if item.get("status") in {"copied", "generated"} and item.get("bundle_path")
    ]
    links = {
        "details": (
            str(detail_path.relative_to(bundle_dir).as_posix())
            if detail_item.get("status") == "generated"
            else None
        ),
        "spans_timeline": viewer.get("path"),
        "conversation": (
            str(conversation_index.relative_to(bundle_dir).as_posix())
            if conversation_item.get("status") == "generated"
            else None
        ),
        "resources": _copied_artifact_link(
            entries,
            manifest_field="util_path",
            fallback_names=("util.jsonl",),
        ),
        "execution_log": _copied_artifact_link(
            entries,
            fallback_names=("openclaw_gateway.log",),
        ) or _copied_artifact_link(
            entries,
            fallback_names=("hermes_stdout.log",),
        ),
        "candidate": _copied_artifact_link(
            entries,
            fallback_names=("candidate.patch",),
        ),
        "verdict": _copied_artifact_link(
            entries,
            manifest_field="verdict_path",
            fallback_names=("verdict.json",),
        ),
        "job_evidence": (related_links or {}).get("job_evidence"),
    }
    return {
        "arm": arm,
        "instance_id": str(instance_id),
        "seed": int(seed or 0),
        "job_id": str(job_id) if job_id is not None else None,
        "source_run_dir": str(run_dir),
        "task_directory": task_dir,
        "summary": summary,
        "links": links,
        "viewer": viewer,
        "status_counts": dict(sorted(counts.items())),
        "complete": not any(
            item.get("status") in {"copy_failed", "generation_failed"}
            for item in entries
        ),
        "copied_files": copied_files,
        "artifacts": entries,
    }


def render_single_run_markdown(index: Mapping[str, Any]) -> str:
    summary = index.get("summary") or {}
    task = summary.get("task") or {}
    result = summary.get("result") or {}
    timing = summary.get("timing") or {}
    lifecycle = summary.get("lifecycle") or {}
    resources = summary.get("resources") or {}
    links = index.get("links") or {}
    lines = [
        f"# Agent Bench run: {task.get('instance_id') or task.get('run_id')}",
        "",
        f"- agent outcome: `{result.get('agent_execution_outcome')}`",
        f"- benchmark valid: `{result.get('benchmark_valid')}`",
        f"- verifier / reward: `{result.get('verifier_status')}` / `{result.get('reward')}`",
        f"- failure phase / verifier skipped: `{result.get('failure_phase')}` / `{result.get('verifier_skipped')}`",
        f"- agent execution seconds: `{timing.get('agent_execution_seconds')}`",
        f"- materialized children: `{lifecycle.get('child_session_count')}`; "
        f"max concurrent: `{lifecycle.get('max_concurrent_children')}`",
        f"- child timing quality: `{lifecycle.get('child_derived_metrics_quality')}`; "
        f"unknown terminal times: `{lifecycle.get('child_runtime_unknown_count')}`",
        f"- resource samples: `{resources.get('sample_count')}` ({resources.get('status')})",
        "",
        "## View",
        "",
    ]
    for label, key in (
        ("Detailed run page", "details"),
        ("Span timeline", "spans_timeline"),
        ("Conversation", "conversation"),
        ("Resource samples", "resources"),
        ("Execution log", "execution_log"),
        ("Candidate patch", "candidate"),
        ("Verifier result", "verdict"),
        ("Shared serving / OTel / PBS evidence", "job_evidence"),
    ):
        target = links.get(key)
        lines.append(f"- [{label}]({target})" if target else f"- {label}: not collected")
    lines += [
        "",
        "Missing values remain `not_collected`/`null`; they are not converted to zero.",
        "Logical tokens, serving processed tokens, lifecycle intervals, and resource samples retain their distinct source semantics.",
        "",
    ]
    return "\n".join(lines)


def finalize_bundle_index(
    bundle_dir: Path,
    index_path: Path,
    run_indexes: Sequence[Mapping[str, Any]],
    *,
    shared_indexes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write a portable catalog for every final bundle file except itself."""
    bundle_dir = bundle_dir.resolve()
    index_path = index_path.resolve()
    known: dict[str, Mapping[str, Any]] = {}
    all_indexes = [*run_indexes, *shared_indexes]
    for run_index in all_indexes:
        for item in run_index.get("artifacts") or []:
            relative = item.get("bundle_path")
            if relative:
                known[str(relative)] = item

    bundle_files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in bundle_dir.rglob("*") if candidate.is_file()):
        if path.resolve() == index_path:
            continue
        relative = str(path.relative_to(bundle_dir).as_posix())
        original = known.get(relative)
        bundle_files.append(
            {
                "bundle_path": relative,
                "status": (
                    str(original.get("status"))
                    if original is not None
                    else "generated"
                ),
                "source_kind": (
                    str(original.get("source_kind"))
                    if original is not None
                    else "report_generated"
                ),
                "source_path": (
                    original.get("source_path") if original is not None else None
                ),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    source_status_counts: Counter[str] = Counter()
    for run_index in all_indexes:
        source_status_counts.update(
            str(item.get("status")) for item in run_index.get("artifacts") or []
        )
    catalog = {
        "schema_version": 2,
        "bundle_root": ".",
        "index_path": str(index_path.relative_to(bundle_dir).as_posix()),
        "index_self_excluded_from_hash": True,
        "bundle_file_count_excluding_index": len(bundle_files),
        "bundle_files": bundle_files,
        "source_artifact_status_counts": dict(sorted(source_status_counts.items())),
        "runs": list(run_indexes),
        "shared_artifacts": list(shared_indexes),
    }
    index_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return catalog


def create_single_run_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    manifest = _manifest_for_run(run_dir)
    basename = safe_name(name or manifest.get("run_id") or run_dir.name)
    shared_root = discover_shared_job_root(run_dir)
    source_dirs = (run_dir, shared_root) if shared_root is not None else (run_dir,)
    bundle_dir = prepare_bundle_dir(
        output_dir,
        basename,
        source_dirs=source_dirs,
    )
    shared_indexes: list[dict[str, Any]] = []
    related_links: dict[str, str] = {}
    if shared_root is not None:
        shared = export_shared_job_artifacts(
            shared_root,
            bundle_dir,
            arm="run",
            job_id=shared_root.name,
        )
        shared_indexes.append(shared)
        related_links.update(shared["links"])
    index = export_run_artifacts(
        run_dir,
        bundle_dir,
        arm="run",
        instance_id=manifest.get("instance_id") or run_dir.name,
        seed=manifest.get("seed") or 0,
        related_links=related_links,
    )
    artifact_index_path = bundle_dir / "artifact-index.json"
    shared_exports_complete = all(
        bool(item.get("complete")) for item in shared_indexes
    )
    report = {
        "schema_version": 2,
        "mode": "single-run",
        "summary": index["summary"],
        "artifact_bundle": {
            "subdirectories": list(BUNDLE_SUBDIRS),
            "artifact_index": artifact_index_path.name,
            "complete": bool(index["complete"]) and shared_exports_complete,
            "status_counts": index["status_counts"],
            "links": index["links"],
            "shared_job_artifact_count": len(shared_indexes),
            "shared_job_exports_complete": shared_exports_complete,
        },
    }
    (bundle_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "analysis.md").write_text(
        render_single_run_markdown(index),
        encoding="utf-8",
    )
    detail_target = str(index["links"]["details"])
    (bundle_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Agent Bench run</title>"
        f'<p><a href="{html.escape(detail_target)}">Open run details</a></p>',
        encoding="utf-8",
    )
    finalize_bundle_index(
        bundle_dir,
        artifact_index_path,
        (index,),
        shared_indexes=shared_indexes,
    )
    return bundle_dir, report
