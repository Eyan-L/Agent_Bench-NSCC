#!/usr/bin/env python3
"""Ad-hoc OTel JSONL inspector for oc-eval runs.

Reads the file exporter's output (default /tmp/otel/otel-traces.jsonl) and
prints one line per span / metric / log record. Useful for "did the patched
gateway dispatch actually emit model.usage?" sanity checks without standing
up a full agent_bench.serving.

Usage:
    scripts/otel_query.py                 # default path, all signals
    scripts/otel_query.py --kind spans
    scripts/otel_query.py --name openclaw.model.usage
    AB_TELEMETRY_DATA_DIR=data/otel scripts/otel_query.py --kind spans --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def _attrs(items) -> Dict[str, Any]:
    """OTLP attribute list -> flat dict (first scalar value of each entry)."""
    out: Dict[str, Any] = {}
    for a in items or []:
        v = a.get("value", {})
        # value is one of {stringValue, intValue, doubleValue, boolValue, arrayValue, ...}
        if not v:
            continue
        out[a["key"]] = next(iter(v.values()))
    return out


def _iter_records(path: Path):
    if not path.exists():
        sys.exit(f"trace file not found: {path}")
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _spans(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for rs in record.get("resourceSpans", []) or []:
        res_attrs = _attrs(rs.get("resource", {}).get("attributes"))
        for ss in rs.get("scopeSpans", []) or []:
            for sp in ss.get("spans", []) or []:
                start = int(sp.get("startTimeUnixNano", 0))
                end = int(sp.get("endTimeUnixNano", 0))
                yield {
                    "kind": "span",
                    "name": sp.get("name"),
                    "duration_ms": (end - start) / 1e6 if end and start else 0,
                    "attrs": _attrs(sp.get("attributes")),
                    "resource": res_attrs,
                }


def _metrics(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for rm in record.get("resourceMetrics", []) or []:
        res_attrs = _attrs(rm.get("resource", {}).get("attributes"))
        for sm in rm.get("scopeMetrics", []) or []:
            for m in sm.get("metrics", []) or []:
                yield {
                    "kind": "metric",
                    "name": m.get("name"),
                    "unit": m.get("unit"),
                    "resource": res_attrs,
                }


def _logs(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for rl in record.get("resourceLogs", []) or []:
        res_attrs = _attrs(rl.get("resource", {}).get("attributes"))
        for sl in rl.get("scopeLogs", []) or []:
            for lr in sl.get("logRecords", []) or []:
                body = lr.get("body", {})
                yield {
                    "kind": "log",
                    "severity": lr.get("severityText"),
                    "body": body.get("stringValue") if isinstance(body, dict) else body,
                    "attrs": _attrs(lr.get("attributes")),
                    "resource": res_attrs,
                }


def main() -> None:
    default_path = Path(
        os.environ.get("AB_TELEMETRY_DATA_DIR", "/tmp/otel")
    ) / "otel-traces.jsonl"
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=str(default_path), help="JSONL trace file")
    p.add_argument("--kind", choices=("spans", "metrics", "logs", "all"), default="all")
    p.add_argument("--name", help="filter by name (substring match)")
    p.add_argument("--json", action="store_true", help="emit JSON lines instead of text")
    args = p.parse_args()

    extractors = {
        "spans": _spans,
        "metrics": _metrics,
        "logs": _logs,
    }
    selected = list(extractors.values()) if args.kind == "all" else [extractors[args.kind]]

    seen = 0
    for record in _iter_records(Path(args.path)):
        for fn in selected:
            for item in fn(record):
                if args.name and args.name not in (item.get("name") or ""):
                    continue
                seen += 1
                if args.json:
                    print(json.dumps(item, ensure_ascii=False))
                elif item["kind"] == "span":
                    print(
                        f"span    {item['name']:40s} {item['duration_ms']:8.1f}ms  "
                        f"{json.dumps(item['attrs'], ensure_ascii=False)[:200]}"
                    )
                elif item["kind"] == "metric":
                    print(f"metric  {item['name']:40s}  unit={item.get('unit') or '-'}")
                elif item["kind"] == "log":
                    body = (item.get("body") or "")[:120]
                    print(f"log     [{item.get('severity') or '?'}] {body}")
    print(f"--- {seen} records", file=sys.stderr)


if __name__ == "__main__":
    main()
