#!/usr/bin/env python3
"""Temporary RoadmapBench paired comparison excluding eight known tasks.

This wrapper intentionally leaves ``compare_roadmap_runs.py`` unchanged. It
reuses the current comparison implementation, including rightmost-input retry
precedence, but removes the same fixed eight task IDs from BOTH arms before arm
summaries, pairing, strict-trajectory validation, CSV/JSON output, and plots.

Excluded tasks
--------------
* mko-5.0.0-roadmap   -- invalid workspace prime; must be rerun
* mko-7.0.0-roadmap   -- invalid workspace prime; must be rerun
* fbr-2.27.0-roadmap  -- current subagent sample has zero spawn events
* fbr-3.0.0-roadmap   -- current subagent sample has zero spawn events
* mko-4.0.0-roadmap   -- current subagent sample has zero spawn events
* pyg-2.1.0-roadmap   -- current subagent sample has zero spawn events
* glz-7.0.0-roadmap   -- current single-agent sample has no verifier reward
* pyg-2.0.0-roadmap   -- current single-agent sample has no verifier reward

Example::

    python scripts/compare_roadmap_runs_skip6.py \
      --subagent \
        16056107.pbs101 \
        16056108.pbs101 \
        16097006.pbs101 \
      --single-agent \
        16104466.pbs101 \
        16104470.pbs101 \
      --strict-trajectory

The resulting paired comparison should contain 12 tasks when both original arms
contain the full RoadmapBench subset20. The script intentionally keeps the legacy
``-skip6`` output suffix so rerunning the same command overwrites the previous
temporary report and PNGs in place.
"""

from __future__ import annotations

try:  # Package import.
    from . import compare_roadmap_runs as base
except ImportError:  # Direct CLI execution.
    import compare_roadmap_runs as base  # type: ignore


EXCLUDED_TASKS = frozenset(
    {
        "mko-5.0.0-roadmap",
        "mko-7.0.0-roadmap",
        "fbr-2.27.0-roadmap",
        "fbr-3.0.0-roadmap",
        "mko-4.0.0-roadmap",
        "pyg-2.1.0-roadmap",
        "glz-7.0.0-roadmap",
        "pyg-2.0.0-roadmap",
    }
)

_EXCLUSION_REASONS = {
    "mko-5.0.0-roadmap": "invalid workspace prime; rerun required",
    "mko-7.0.0-roadmap": "invalid workspace prime; rerun required",
    "fbr-2.27.0-roadmap": "current subagent sample has zero spawn events",
    "fbr-3.0.0-roadmap": "current subagent sample has zero spawn events",
    "mko-4.0.0-roadmap": "current subagent sample has zero spawn events",
    "pyg-2.1.0-roadmap": "current subagent sample has zero spawn events",
    "glz-7.0.0-roadmap": "current single-agent sample has no verifier reward",
    "pyg-2.0.0-roadmap": "current single-agent sample has no verifier reward",
}

_original_select_best_records = base.select_best_records
_original_render_markdown = base.render_markdown
_original_summarize_arm = base.summarize_arm
_original_default_name = base._default_name


def select_best_records(records):
    """Apply normal retry precedence, then remove the fixed exclusion set."""
    selected, duplicates = _original_select_best_records(records)
    selected = [
        record for record in selected if record.instance_id not in EXCLUDED_TASKS
    ]
    return selected, duplicates


def summarize_arm(records):
    """Keep normal statistics and make the temporary exclusion machine-readable."""
    summary = _original_summarize_arm(records)
    summary["analysis_exclusions"] = {
        "policy": "temporary_fixed_skip8",
        "applied_to": "both arms before pairing",
        "task_ids": sorted(EXCLUDED_TASKS),
        "reasons": {
            task: _EXCLUSION_REASONS[task] for task in sorted(EXCLUDED_TASKS)
        },
    }
    return summary


def render_markdown(*args, **kwargs):
    """Render the normal paired report plus an explicit exclusion audit section."""
    text = _original_render_markdown(*args, **kwargs)
    lines = [
        "",
        "## Temporary analysis exclusions",
        "",
        "This temporary comparison removes the same eight tasks from **both** "
        "subagent and single-agent arms before summaries, pairing, strict "
        "trajectory validation, and plotting:",
        "",
    ]
    for task in sorted(EXCLUDED_TASKS):
        lines.append(f"- `{task}`: {_EXCLUSION_REASONS[task]}")
    lines += [
        "",
        "This is an analysis-only filter; the underlying run artifacts are not "
        "modified. With the full subset20 present in both arms, the expected "
        "paired task count is **12**.",
        "",
        "The output filename keeps the legacy `-skip6` suffix only so rerunning "
        "the same command overwrites the previous temporary report and figures.",
        "",
    ]
    return text.rstrip() + "\n" + "\n".join(lines)


def _default_name(subagent, single_agent=None):
    """Reuse the previous temporary basename so regenerated files overwrite it."""
    return _original_default_name(subagent, single_agent) + "-skip6"


# compare_roadmap_runs.main is the implementation module's main(), which resolves
# helper functions from that implementation module at runtime. Patch that
# namespace so all downstream outputs consistently see the filtered records.
base._impl.select_best_records = select_best_records
base._impl.summarize_arm = summarize_arm
base._impl.render_markdown = render_markdown
base._impl._default_name = _default_name

main = base._impl.main


if __name__ == "__main__":
    raise SystemExit(main())
