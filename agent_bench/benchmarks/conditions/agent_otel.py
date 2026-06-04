"""Shared agent telemetry helpers.

Agent runtimes emit OTLP records to the shared collector. These helpers keep
the per-run extraction contract uniform: slice the collector trace file by
``oc.run_id`` and write both the raw OTLP envelopes and flattened spans.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def ns_to_iso(ns: int) -> str | None:
    if not ns:
        return None
    return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).isoformat()


def optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed or None


def otel_endpoint_for_container(host_address: str) -> str:
    explicit = os.environ.get("AB_TELEMETRY_CONTAINER_ENDPOINT", "").strip()
    if explicit:
        return explicit.rstrip("/")
    inherited = os.environ.get("AB_TELEMETRY_ENDPOINT", "").strip()
    if inherited and not _is_loopback_endpoint(inherited):
        return inherited.rstrip("/")
    port = os.environ.get("AB_TELEMETRY_PORT", "4318")
    return f"http://{host_address}:{port}"


def otel_env(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    extra_attrs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    attrs: dict[str, Any] = {
        "ab.run_id": run_id,
        "oc.run_id": run_id,
        "agent.run_id": run_id,
        "service.name": service_name,
    }
    attrs.update(extra_attrs or {})
    return {
        "AB_TELEMETRY_ENDPOINT": endpoint,
        "OTEL_RESOURCE_ATTRIBUTES": _resource_attrs_to_env(attrs),
        "OTEL_SERVICE_NAME": service_name,
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
    }


def otel_config_values(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    extra_attrs: Mapping[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, str]:
    env = otel_env(
        service_name=service_name,
        run_id=run_id,
        endpoint=endpoint,
        extra_attrs=extra_attrs,
    )
    return {
        "AB_RUN_ID": _config_string(run_id),
        "OTEL_ENABLED": "true" if enabled else "false",
        "OTEL_ENDPOINT": _config_string(env["AB_TELEMETRY_ENDPOINT"]),
        "OTEL_SERVICE_NAME": _config_string(service_name),
        "OTEL_RESOURCE_ATTRIBUTES": _config_string(env["OTEL_RESOURCE_ATTRIBUTES"]),
        "OTEL_PROTOCOL": "http/json",
    }


def _config_string(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_spans_and_otlp(
    spans_path: Path,
    spans: Iterable[Mapping[str, Any]],
    *,
    scope_name: str,
    resource_defaults: Mapping[str, Any] | None = None,
) -> Path:
    """Write flat spans and a synthetic OTLP JSONL envelope.

    This is kept for static/unit-test fallbacks. Runtime agent telemetry should
    prefer ``snapshot_otel_spans`` so ``spans_raw.jsonl`` comes from the real
    collector. Existing flat spans are preserved instead of being overwritten.
    """
    span_list = [dict(span) for span in spans]
    all_spans = _read_flat_spans(spans_path) + span_list
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    with spans_path.open("w") as fh:
        for span in all_spans:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")

    raw_path = spans_path.with_name("spans_raw.jsonl")
    try:
        record = _flat_spans_to_otlp(
            all_spans,
            scope_name=scope_name,
            resource_defaults=resource_defaults or {},
        )
        raw_path.write_text(
            json.dumps(record, ensure_ascii=False) + ("\n" if all_spans else ""),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("OTLP raw span write failed: %s", exc)
    return spans_path


def otel_trace_file_from_env() -> str:
    return os.environ.get(
        "AB_TELEMETRY_TRACE_FILE",
        os.path.join(os.environ.get("AB_TELEMETRY_DATA_DIR", "/tmp/otel"), "otel-traces.jsonl"),
    )


def otel_trace_offset(trace_file: str | None = None) -> tuple[str, int]:
    path = trace_file or otel_trace_file_from_env()
    try:
        return path, os.path.getsize(path)
    except FileNotFoundError:
        return path, 0


def wait_for_otel_flush() -> None:
    try:
        delay = float(os.environ.get("AB_TELEMETRY_FLUSH_WAIT_SECONDS", "1.5"))
    except ValueError:
        delay = 1.5
    if delay > 0:
        time.sleep(delay)


def snapshot_otel_spans(
    trace_file: str,
    pre_offset: int,
    run_dir: Path,
    *,
    expected_run_id: str | None = None,
) -> Path | None:
    """Slice collector output and append flattened spans for one run."""
    try:
        cur_size = os.path.getsize(trace_file)
    except FileNotFoundError:
        return None
    if cur_size <= pre_offset:
        return None

    try:
        with open(trace_file, "rb") as fh:
            fh.seek(pre_offset)
            chunk = fh.read(cur_size - pre_offset)
    except OSError as exc:
        logger.warning("OTel trace snapshot failed: %s", exc)
        return None

    raw_lines: list[str] = []
    spans: list[dict[str, Any]] = []
    for line in chunk.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            continue
        kept_resource_spans = []
        for resource_span in envelope.get("resourceSpans") or []:
            resource = _otlp_attrs(resource_span.get("resource", {}).get("attributes"))
            if expected_run_id is not None and _resource_run_id(resource) != expected_run_id:
                continue
            kept_resource_spans.append(resource_span)
            for scope_span in resource_span.get("scopeSpans", []) or []:
                for span in scope_span.get("spans", []) or []:
                    spans.append(_flatten_otlp_span(span, resource))
        if kept_resource_spans:
            raw_lines.append(json.dumps({"resourceSpans": kept_resource_spans}, ensure_ascii=False))

    raw_path = run_dir / "spans_raw.jsonl"
    try:
        raw_path.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")
    except OSError as exc:
        logger.warning("OTel raw write failed: %s", exc)
        return None

    spans_path = run_dir / "spans.jsonl"
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    with spans_path.open("a") as fh:
        for span in spans:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")
    logger.info("captured %d OTel span(s) -> %s", len(spans), spans_path.name)
    return spans_path


def _resource_run_id(resource: Mapping[str, Any]) -> Any:
    return (
        resource.get("ab.run_id")
        or resource.get("oc.run_id")
        or resource.get("agent.run_id")
    )


def _normalize_resource_run_ids(resource: dict[str, Any]) -> None:
    run_id = (
        resource.get("ab.run_id")
        or resource.get("oc.run_id")
        or resource.get("agent.run_id")
    )
    if run_id is None:
        return
    resource.setdefault("ab.run_id", run_id)
    resource.setdefault("oc.run_id", run_id)
    resource.setdefault("agent.run_id", run_id)


def _flat_spans_to_otlp(
    spans: Sequence[Mapping[str, Any]],
    *,
    scope_name: str,
    resource_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for span in spans:
        resource = dict(resource_defaults)
        resource.update(_string_key_dict(span.get("resource")))
        _normalize_resource_run_ids(resource)
        key = json.dumps(resource, sort_keys=True, default=str)
        if key not in grouped:
            grouped[key] = (resource, [])
        grouped[key][1].append(_flat_span_to_otlp_span(span))

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(resource)},
                "scopeSpans": [
                    {
                        "scope": {"name": scope_name},
                        "spans": otlp_spans,
                    }
                ],
            }
            for resource, otlp_spans in grouped.values()
        ]
    }


def _flat_span_to_otlp_span(span: Mapping[str, Any]) -> dict[str, Any]:
    start = _as_int(span.get("start_unix_nano"))
    end = _as_int(span.get("end_unix_nano"))
    attrs = dict(_string_key_dict(span.get("attrs")))
    if span.get("canonical_name"):
        attrs.setdefault("agent.canonical_name", span.get("canonical_name"))
    if span.get("synthesized") is not None:
        attrs.setdefault("agent.span.synthesized", bool(span.get("synthesized")))
    out = {
        "name": str(span.get("name") or span.get("canonical_name") or "agent.span"),
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": _attrs(attrs),
    }
    if span.get("duration_ms") is not None:
        out["attributes"].append({
            "key": "agent.duration_ms",
            "value": _otel_value(span.get("duration_ms")),
        })
    return out


def _flatten_otlp_span(span: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    start = _as_int(span.get("startTimeUnixNano"))
    end = _as_int(span.get("endTimeUnixNano"))
    attrs = _otlp_attrs(span.get("attributes"))
    name = str(span.get("name") or "")
    canonical = _canonical_name(name)
    if canonical:
        attrs.setdefault("agent.canonical_name", canonical)
    _add_standard_agent_attrs(name, attrs)
    return {
        "name": name,
        "canonical_name": canonical or name,
        "start_unix_nano": start,
        "end_unix_nano": end,
        "start_iso": ns_to_iso(start),
        "end_iso": ns_to_iso(end),
        "duration_ms": (end - start) / 1e6 if end and start else 0,
        "attrs": attrs,
        "resource": dict(resource),
    }


def _attrs(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in values.items():
        if value is None:
            continue
        out.append({"key": str(key), "value": _otel_value(value)})
    return out


def _otlp_attrs(items: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        key = item.get("key")
        value = item.get("value") or {}
        if not key or not isinstance(value, Mapping):
            continue
        out[str(key)] = next(iter(value.values()), None)
    return out


def _canonical_name(name: str) -> str | None:
    if name in {"agent.model.usage", "openclaw.model.usage"}:
        return "agent.model.usage"
    if name in {"agent.tool.call", "openclaw.tool.call"}:
        return "agent.tool.call"
    return None


def _add_standard_agent_attrs(name: str, attrs: dict[str, Any]) -> None:
    canonical = _canonical_name(name)
    if canonical == "agent.model.usage":
        attrs.setdefault("agent.channel", attrs.get("openclaw.channel"))
        attrs.setdefault("agent.provider", attrs.get("openclaw.provider"))
        attrs.setdefault("agent.model", attrs.get("openclaw.model"))
        attrs.setdefault("agent.api", attrs.get("openclaw.api"))
        attrs.setdefault("agent.run_id", attrs.get("openclaw.runId"))
        attrs.setdefault("agent.session_id", attrs.get("openclaw.sessionId"))
        attrs.setdefault("agent.llm.request_index", attrs.get("openclaw.llm.request_index"))
        attrs.setdefault("agent.http.status_code", attrs.get("openclaw.http.status_code"))
        attrs.setdefault("agent.tokens.input", attrs.get("openclaw.tokens.input"))
        attrs.setdefault("agent.tokens.output", attrs.get("openclaw.tokens.output"))
        attrs.setdefault("agent.tokens.cache_read", attrs.get("openclaw.tokens.cache_read"))
        attrs.setdefault("agent.tokens.cache_write", attrs.get("openclaw.tokens.cache_write"))
        attrs.setdefault("agent.tokens.total", attrs.get("openclaw.tokens.total"))
        attrs.setdefault("agent.ttft_ms", attrs.get("openclaw.ttft_ms"))
        attrs.setdefault("agent.tpot_ms", attrs.get("openclaw.tpot_ms"))
        attrs.setdefault("agent.headers_ms", attrs.get("openclaw.headers_ms"))
        attrs.setdefault("agent.chunks.count", attrs.get("openclaw.chunks.count"))
    elif canonical == "agent.tool.call":
        attrs.setdefault("agent.channel", attrs.get("openclaw.channel"))
        attrs.setdefault("agent.tool.name", attrs.get("openclaw.tool.name"))
        attrs.setdefault("agent.tool.call_id", attrs.get("openclaw.tool.callId"))
        attrs.setdefault("agent.tool.is_error", attrs.get("openclaw.tool.isError"))
        attrs.setdefault("agent.run_id", attrs.get("openclaw.runId"))
        attrs.setdefault("agent.session_id", attrs.get("openclaw.sessionId"))
        attrs.setdefault("agent.tool.arguments_json", attrs.get("openclaw.tool.arguments_json"))
        attrs.setdefault("agent.tool.result_preview", attrs.get("openclaw.tool.result_preview"))
        attrs.setdefault("agent.tool.result_chars", attrs.get("openclaw.tool.result_chars"))


def _otel_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otel_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"stringValue": json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {"stringValue": str(value)}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0


def _read_flat_spans(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    spans: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(span, dict):
                spans.append(span)
    except OSError:
        return []
    return spans


def _string_key_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _resource_attrs_to_env(attrs: Mapping[str, Any]) -> str:
    return ",".join(
        f"{key}={_escape_resource_value(value)}"
        for key, value in attrs.items()
        if value is not None
    )


def _escape_resource_value(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=")


def _is_loopback_endpoint(endpoint: str) -> bool:
    try:
        host = urlparse(endpoint).hostname
    except ValueError:
        return False
    return (host or "").lower() in {"127.0.0.1", "localhost", "::1"}
