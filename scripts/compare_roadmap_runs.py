#!/usr/bin/env python3
"""RoadmapBench comparison CLI with explicit retry precedence.

The full analyzer implementation lives in ``compare_roadmap_runs_impl.py``.
This entry point keeps its existing analysis/reporting behavior but changes one
important retry rule: for duplicate ``(instance_id, seed)`` records, the later
PBS job/run directory supplied on the command line is authoritative.

Example::

    python scripts/compare_roadmap_runs.py \
      --subagent original_a original_b smoke2 smoke4 \
      --single-agent single_a single_b \
      --strict-trajectory

If ``smoke4`` contains the same task+seed as any earlier subagent input, the
``smoke4`` record is used, regardless of reward/status/trajectory quality.
"""

from __future__ import annotations

try:  # Package import, e.g. ``import scripts.compare_roadmap_runs``.
    from . import compare_roadmap_runs_impl as _impl
except ImportError:  # Direct CLI execution: ``python scripts/compare_roadmap_runs.py``.
    import compare_roadmap_runs_impl as _impl  # type: ignore


# Re-export the implementation API so existing imports keep working.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


_original_render_markdown = _impl.render_markdown


def select_best_records(records):
    """Select duplicate task+seed records by CLI input order: rightmost wins.

    ``collect_records`` appends records while iterating the user-provided input
    specs from left to right. Therefore the last record in each duplicate group
    belongs to the latest/rightmost matching input and is authoritative. No
    reward/status/trajectory quality heuristic is applied.
    """
    grouped = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record)

    selected = []
    duplicates = {}
    for key, group in grouped.items():
        # Store duplicate records with the selected/rightmost record first so
        # the JSON retry audit is easy to read.
        ordered = list(reversed(group))
        selected.append(ordered[0])
        if len(ordered) > 1:
            duplicates[f"{key[0]}::seed={key[1]}"] = ordered
    return sorted(selected, key=lambda record: record.key), duplicates


def render_markdown(*args, **kwargs):
    """Render the existing report while documenting rightmost-input precedence."""
    text = _original_render_markdown(*args, **kwargs)
    old = (
        "For each `instance_id + seed`, the analyzer prefers a run with verifier "
        "reward, then non-ERROR status, trajectory presence, and the latest "
        "terminal timestamp. A focused smoke/retry can therefore replace "
        "an earlier infrastructure ERROR without double-counting the task."
    )
    new = (
        "For each `instance_id + seed`, command-line input order is authoritative: "
        "the later/rightmost PBS job or run directory replaces earlier records "
        "for that task. No reward, status, or trajectory-quality heuristic is "
        "used, so focused retries can explicitly supersede prior runs."
    )
    return text.replace(old, new)


# ``main`` resolves these functions from the implementation module's globals,
# so patch that namespace before handing control to it.
_impl.select_best_records = select_best_records
_impl.render_markdown = render_markdown
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
