#!/usr/bin/env python3
"""Render a flattened spans.jsonl file as a self-contained HTML timeline."""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


PALETTE = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
    "#0f766e",
    "#ca8a04",
]


def _load_spans(path: Path) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            start = int(item.get("start_unix_nano") or 0)
            end = int(item.get("end_unix_nano") or 0)
            if not start or not end or end < start:
                continue
            attrs = item.get("attrs") or {}
            resource = item.get("resource") or {}
            name = item.get("name") or "unknown"
            channel = (
                attrs.get("agent_bench.channel")
                or attrs.get("openclaw.channel")
                or attrs.get("agent.channel")
                or "unknown"
            )
            spans.append(
                {
                    "line": line_no,
                    "name": name,
                    "channel": channel,
                    "start": start,
                    "end": end,
                    "duration_ms": (end - start) / 1e6,
                    "tool": (
                        attrs.get("openclaw.tool.name")
                        or attrs.get("agent.tool.name")
                        or ""
                    ),
                    "returncode": attrs.get("agent_bench.returncode"),
                    "run_id": (
                        attrs.get("openclaw.runId")
                        or attrs.get("agent.run_id")
                        or attrs.get("agent.runId")
                        or resource.get("oc.run_id")
                        or resource.get("hm.run_id")
                        or resource.get("agent.run_id")
                        or ""
                    ),
                    "attrs": attrs,
                    "resource": resource,
                }
            )
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans


def _assign_lanes(spans: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    names = sorted({span["name"] for span in spans})
    return {name: i for i, name in enumerate(names)}


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def _fmt_attrs(span: Dict[str, Any]) -> str:
    payload = {
        "line": span["line"],
        "name": span["name"],
        "channel": span["channel"],
        "duration_ms": round(span["duration_ms"], 3),
        "tool": span["tool"],
        "returncode": span["returncode"],
        "run_id": span["run_id"],
        "attrs": span["attrs"],
        "resource": span["resource"],
    }
    return html.escape(json.dumps(payload, ensure_ascii=False, indent=2))


def render_html(spans: List[Dict[str, Any]], source: Path) -> str:
    if not spans:
        raise SystemExit(f"no valid spans found in {source}")

    min_start = min(span["start"] for span in spans)
    max_end = max(span["end"] for span in spans)
    total_ms = max((max_end - min_start) / 1e6, 1.0)
    lanes = _assign_lanes(spans)
    channels = sorted({span["channel"] for span in spans})
    colors = {ch: PALETTE[i % len(PALETTE)] for i, ch in enumerate(channels)}

    left_pad = 290
    right_pad = 40
    top_pad = 92
    row_h = 30
    plot_w = 1200
    width = left_pad + plot_w + right_pad
    height = top_pad + max(1, len(lanes)) * row_h + 70

    def x_for(ns: int) -> float:
        return left_pad + ((ns - min_start) / 1e6 / total_ms) * plot_w

    rows = []
    for name, lane in lanes.items():
        y = top_pad + lane * row_h
        rows.append(
            f'<text class="label" x="{left_pad - 12}" y="{y + 18}" text-anchor="end">'
            f"{html.escape(name)}</text>"
        )
        rows.append(
            f'<line class="rowline" x1="{left_pad}" x2="{left_pad + plot_w}" '
            f'y1="{y + 11}" y2="{y + 11}" />'
        )

    rects = []
    details = {}
    for idx, span in enumerate(spans):
        lane = lanes[span["name"]]
        y = top_pad + lane * row_h
        x = x_for(span["start"])
        w = max(x_for(span["end"]) - x, 2.0)
        detail_id = f"span-{idx}"
        details[detail_id] = {
            "line": span["line"],
            "name": span["name"],
            "channel": span["channel"],
            "duration_ms": round(span["duration_ms"], 3),
            "start_iso": _iso(span["start"]),
            "end_iso": _iso(span["end"]),
            "tool": span["tool"],
            "returncode": span["returncode"],
            "run_id": span["run_id"],
            "attrs": span["attrs"],
            "resource": span["resource"],
        }
        title = (
            f"{span['name']}\\n"
            f"channel: {span['channel']}\\n"
            f"duration_ms: {span['duration_ms']:.3f}\\n"
            f"start: {_iso(span['start'])}\\n"
            f"end: {_iso(span['end'])}\\n"
            f"tool: {span['tool']}\\n"
            f"run_id: {span['run_id']}"
        )
        rects.append(
            f'<g class="span-hit" tabindex="0" role="button" data-span-id="{detail_id}" '
            f'aria-label="{html.escape(span["name"])} details">'
            f'<rect class="span" '
            f'x="{x:.2f}" y="{y + 4}" width="{w:.2f}" height="16" '
            f'rx="3" fill="{colors[span["channel"]]}">'
            f"<title>{html.escape(title)}</title></rect></g>"
        )

    tick_count = 8
    ticks = []
    for i in range(tick_count + 1):
        frac = i / tick_count
        x = left_pad + frac * plot_w
        ms = frac * total_ms
        ticks.append(f'<line class="tick" x1="{x:.2f}" x2="{x:.2f}" y1="64" y2="{height - 52}" />')
        ticks.append(f'<text class="ticklabel" x="{x:.2f}" y="{height - 28}" text-anchor="middle">{ms:.1f} ms</text>')

    legend = []
    for i, ch in enumerate(channels):
        x = left_pad + i * 170
        legend.append(f'<rect x="{x}" y="44" width="12" height="12" fill="{colors[ch]}" rx="2" />')
        legend.append(f'<text class="legend" x="{x + 18}" y="55">{html.escape(ch)}</text>')

    counts = Counter(span["name"] for span in spans)
    summary = ", ".join(f"{html.escape(k)}={v}" for k, v in sorted(counts.items()))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Span Timeline</title>
<style>
body {{ margin: 0; font: 13px system-ui, -apple-system, Segoe UI, sans-serif; color: #111827; background: #f8fafc; }}
header {{ padding: 18px 24px 10px; background: white; border-bottom: 1px solid #e5e7eb; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
.meta {{ color: #4b5563; line-height: 1.45; }}
.wrap {{ padding: 16px 24px 28px; overflow-x: auto; }}
svg {{ background: white; border: 1px solid #e5e7eb; }}
.label {{ fill: #374151; font-size: 12px; }}
.legend, .ticklabel {{ fill: #4b5563; font-size: 12px; }}
.tick {{ stroke: #e5e7eb; stroke-width: 1; }}
.rowline {{ stroke: #eef2f7; stroke-width: 1; }}
.span-hit {{ cursor: pointer; outline: none; }}
.span {{ opacity: 0.88; pointer-events: all; }}
.span-hit:hover .span, .span-hit:focus .span {{ opacity: 1; stroke: #111827; stroke-width: 1; }}
.span-hit.selected .span {{ opacity: 1; stroke: #111827; stroke-width: 2; }}
.drawer {{ position: fixed; top: 0; right: 0; width: min(720px, 86vw); height: 100vh; background: #111827; color: #e5e7eb; box-shadow: -8px 0 24px #0003; transform: translateX(100%); transition: transform 160ms ease; z-index: 20; display: flex; flex-direction: column; }}
.drawer.open {{ transform: translateX(0); }}
.drawer-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #374151; }}
.drawer-title {{ font-weight: 700; color: white; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.drawer-close {{ border: 1px solid #4b5563; background: #1f2937; color: white; border-radius: 6px; padding: 5px 9px; cursor: pointer; }}
#detail {{ white-space: pre-wrap; margin: 0; padding: 14px; overflow: auto; flex: 1; }}
.scrim {{ position: fixed; inset: 0; background: #0005; opacity: 0; pointer-events: none; transition: opacity 160ms ease; z-index: 19; }}
.scrim.open {{ opacity: 1; pointer-events: all; }}
</style>
</head>
<body>
<header>
<h1>Span Timeline</h1>
<div class="meta">Source: {html.escape(str(source))}<br>
Spans: {len(spans)} | Window: {_iso(min_start)} to {_iso(max_end)} | Duration: {total_ms:.3f} ms<br>
Counts: {summary}</div>
</header>
<div class="wrap">
<svg width="{width}" height="{height}" role="img" aria-label="Span timeline">
<text x="24" y="28" font-size="14" font-weight="600">Channels</text>
{''.join(legend)}
{''.join(ticks)}
{''.join(rows)}
{''.join(rects)}
</svg>
</div>
<div id="scrim" class="scrim"></div>
<aside id="drawer" class="drawer" aria-label="Span details" aria-hidden="true">
  <div class="drawer-head">
    <div id="drawer-title" class="drawer-title">Span Details</div>
    <button id="drawer-close" class="drawer-close" type="button">Close</button>
  </div>
  <pre id="detail">Click a span to inspect its attributes.</pre>
</aside>
<script>
const SPAN_DETAILS = {json.dumps(details, ensure_ascii=False)};
const detailEl = document.getElementById('detail');
const drawerEl = document.getElementById('drawer');
const drawerTitleEl = document.getElementById('drawer-title');
const scrimEl = document.getElementById('scrim');
const closeEl = document.getElementById('drawer-close');
function openDrawer() {{
  drawerEl.classList.add('open');
  scrimEl.classList.add('open');
  drawerEl.setAttribute('aria-hidden', 'false');
}}
function closeDrawer() {{
  drawerEl.classList.remove('open');
  scrimEl.classList.remove('open');
  drawerEl.setAttribute('aria-hidden', 'true');
}}
function showSpan(node) {{
  const id = node.getAttribute('data-span-id');
  const detail = SPAN_DETAILS[id];
  if (!detail) return;
  document.querySelectorAll('.span.selected').forEach(function (item) {{
    item.classList.remove('selected');
  }});
  node.classList.add('selected');
  drawerTitleEl.textContent = detail.name + ' · ' + detail.channel;
  detailEl.textContent = JSON.stringify(detail, null, 2);
  openDrawer();
}}
closeEl.addEventListener('click', closeDrawer);
scrimEl.addEventListener('click', closeDrawer);
document.addEventListener('keydown', function (event) {{
  if (event.key === 'Escape') closeDrawer();
}});
document.querySelectorAll('.span-hit').forEach(function (node) {{
  node.addEventListener('click', function () {{ showSpan(node); }});
  node.addEventListener('keydown', function (event) {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      showSpan(node);
    }}
  }});
}});
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spans_jsonl", type=Path, help="flattened spans.jsonl file")
    parser.add_argument("-o", "--output", type=Path, default=Path("spans_timeline.html"))
    args = parser.parse_args()

    spans = _load_spans(args.spans_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(spans, args.spans_jsonl))
    print(f"wrote {args.output} ({len(spans)} spans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
