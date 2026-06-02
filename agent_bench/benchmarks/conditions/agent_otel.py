"""Shared agent telemetry helpers.

OpenClaw can emit real OTLP records through its diagnostics plugin. Other
agent runtimes often expose structured events rather than native OTLP. These
helpers let those runtimes write the same per-run artifacts OpenClaw writes:
flat ``spans.jsonl`` plus an OTLP-shaped ``spans_raw.jsonl`` envelope.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    return os.environ.get("AB_TELEMETRY_ENDPOINT", f"http://{host_address}:4318")


def otel_env(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    extra_attrs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    attrs: dict[str, Any] = {
        "oc.run_id": run_id,
        "agent.run_id": run_id,
        "service.name": service_name,
    }
    attrs.update(extra_attrs or {})
    return {
        "OTEL_RESOURCE_ATTRIBUTES": _resource_attrs_to_env(attrs),
        "OTEL_SERVICE_NAME": service_name,
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
    }


def write_spans_and_otlp(
    spans_path: Path,
    spans: Iterable[Mapping[str, Any]],
    *,
    scope_name: str,
    resource_defaults: Mapping[str, Any] | None = None,
) -> Path:
    """Write flat spans and a synthetic OTLP JSONL envelope.

    ``spans.jsonl`` keeps the flattened format consumed by the existing result
    analysis code. ``spans_raw.jsonl`` mirrors OpenClaw's raw snapshot file so
    smoke/debug tools can inspect all harnesses with one reader.
    """
    span_list = [dict(span) for span in spans]
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    with spans_path.open("w") as fh:
        for span in span_list:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")

    raw_path = spans_path.with_name("spans_raw.jsonl")
    try:
        record = _flat_spans_to_otlp(
            span_list,
            scope_name=scope_name,
            resource_defaults=resource_defaults or {},
        )
        raw_path.write_text(
            json.dumps(record, ensure_ascii=False) + ("\n" if span_list else ""),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("OTLP raw span write failed: %s", exc)
    return spans_path


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


def _attrs(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in values.items():
        if value is None:
            continue
        out.append({"key": str(key), "value": _otel_value(value)})
    return out


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
