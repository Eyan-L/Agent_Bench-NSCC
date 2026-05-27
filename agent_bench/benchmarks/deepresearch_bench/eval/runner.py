"""Standalone DR-Bench evaluation CLI.

Re-grade persisted agent reports without re-running the agent. Pairs with
``DR_SKIP_EVAL=1`` (used by the ``make run_dr_*_agent`` targets), which
makes the harness collect the report into ``run_dir/report.md`` and skip
RACE + FACT. This script then walks the sweep dir and grades each report.

Usage:
    python -m agent_bench.benchmarks.deepresearch_bench.eval.runner \\
        --sweep data/experiments --app-leaf dr_smoke1 \\
        --instance-data data/deepresearch_bench/instances.json

By default, ``--sweep data/experiments --app-leaf <leaf>`` picks the most
recent experiment containing ``runs/<leaf>``. You can also point ``--sweep``
directly at a run directory parent. With ``--force`` it re-grades even if a
non-DEFERRED ``verdict.json`` already exists.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from agent_bench.benchmarks.deepresearch_bench.eval import judge_runner

logger = logging.getLogger(__name__)


def _load_instance_map(path: Path) -> Dict[str, Dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        return rows
    return {r["instance_id"]: r for r in rows}


def _has_run_dirs(path: Path) -> bool:
    try:
        return any(
            p.is_dir() and (p / "manifest.json").is_file()
            for p in path.iterdir()
        )
    except OSError:
        return False


def _resolve_run_parent(path: Path, app_leaf: Optional[str]) -> Optional[Path]:
    if _has_run_dirs(path):
        return path
    runs_dir = path / "runs"
    if not runs_dir.is_dir():
        return None
    if app_leaf:
        candidate = runs_dir / app_leaf
        return candidate if _has_run_dirs(candidate) else None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and _has_run_dirs(p)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _pick_stamp_dir(sweep_dir: Path, stamp: Optional[str], app_leaf: Optional[str]) -> Path:
    direct = _resolve_run_parent(sweep_dir, app_leaf)
    if direct is not None:
        return direct
    if stamp:
        p = sweep_dir / stamp
        if not p.is_dir():
            raise SystemExit(f"--stamp {stamp} not under {sweep_dir}")
        resolved = _resolve_run_parent(p, app_leaf)
        if resolved is None:
            suffix = f"/runs/{app_leaf}" if app_leaf else "/runs/<app-leaf>"
            raise SystemExit(f"no run dirs under {p}{suffix}")
        return resolved
    candidates = sorted(
        [p for p in sweep_dir.iterdir() if p.is_dir() and p.name[0].isdigit()],
        reverse=True,
    )
    for candidate in candidates:
        resolved = _resolve_run_parent(candidate, app_leaf)
        if resolved is not None:
            return resolved
    if not candidates:
        raise SystemExit(f"no stamped subdirs or run dirs under {sweep_dir}")
    suffix = f" containing runs/{app_leaf}" if app_leaf else " containing run dirs"
    raise SystemExit(f"no experiment under {sweep_dir}{suffix}")


def _needs_grading(verdict_path: Path, force: bool) -> bool:
    if force:
        return True
    if not verdict_path.exists():
        return True
    try:
        v = json.loads(verdict_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    # DEFERRED = agent-only phase wrote it; needs grading.
    return v.get("status") == "DEFERRED" or v.get("race") is None


def _merge_verdict(existing: Dict, judge_result, host_report: Path, judge_dir: Path) -> Dict:
    """Compose the final verdict by overlaying judge scores onto the agent-phase verdict."""
    race = asdict(judge_result.race) if judge_result.race else None
    fact = asdict(judge_result.fact) if judge_result.fact else None
    overall = float((race or {}).get("overall", 0.0))
    threshold = float(os.environ.get("DR_PASS_THRESHOLD", "0.5"))
    passed = race is not None and overall >= threshold and judge_result.race_error is None
    status = (
        "PASS" if passed
        else ("ERROR" if judge_result.race_error and not race else "FAIL")
    )
    out = dict(existing or {})
    out.update({
        "status": status,
        "test_pass_rate": overall,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "race": race,
        "fact": fact,
        "race_error": judge_result.race_error,
        "fact_error": judge_result.fact_error,
        "report_path": str(host_report),
        "evaluation_dir": str(judge_dir),
        "error_message": None,
    })
    return out


def grade_run(run_dir: Path, inst: Dict, *, force: bool, workers: Optional[int]) -> Dict:
    # Resolve to absolute — judge_runner spawns vendor subprocesses with
    # cwd=VENDOR_DIR, so any relative path here would resolve under vendor/
    # (not the user's CWD) and silently fail.
    run_dir = run_dir.resolve()
    report = run_dir / "report.md"
    verdict_path = run_dir / "verdict.json"

    if not report.is_file() or report.stat().st_size == 0:
        return {"run_dir": str(run_dir), "skipped": "no report.md"}

    if not _needs_grading(verdict_path, force):
        return {"run_dir": str(run_dir), "skipped": "verdict already complete"}

    existing = {}
    if verdict_path.exists():
        try:
            existing = json.loads(verdict_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}

    article = report.read_text(encoding="utf-8")
    judge_dir = run_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Grading %s (instance=%s, %d chars)",
                run_dir.name, inst["instance_id"], len(article))

    result = judge_runner.run(
        instance_id=inst["instance_id"],
        prompt=inst["prompt"],
        language=inst.get("language", "en"),
        article=article,
        output_dir=judge_dir,
        run_id=run_dir.name,
        workers=workers,
    )

    verdict = _merge_verdict(existing, result, report, judge_dir)
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    logger.info("  %s: race=%.4f fact=%s/%s",
                run_dir.name,
                verdict.get("test_pass_rate", 0.0),
                (verdict.get("fact") or {}).get("effective_citations", "-"),
                (verdict.get("fact") or {}).get("total_citations", "-"))
    return {"run_dir": str(run_dir), "verdict": verdict}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path,
                        help="experiment root, legacy sweep root, or run-dir parent")
    parser.add_argument("--app-leaf", default=None,
                        help="app output leaf under data/experiments/<stamp>/runs/")
    parser.add_argument("--instance-data", required=True, type=Path,
                        help="instances.json from the loader")
    parser.add_argument("--stamp", default=None,
                        help="stamped subdir; defaults to most recent")
    parser.add_argument("--force", action="store_true",
                        help="re-grade runs with an existing non-DEFERRED verdict")
    parser.add_argument("--workers", type=int, default=None,
                        help="override DRB_JUDGE_WORKERS")
    parser.add_argument("--instance", default=None,
                        help="grade only this rid (matches dir name); default all")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    inst_map = _load_instance_map(args.instance_data)
    stamp_dir = _pick_stamp_dir(args.sweep, args.stamp, args.app_leaf)
    logger.info("Grading sweep %s", stamp_dir)

    run_dirs = [p for p in stamp_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    if args.instance:
        run_dirs = [p for p in run_dirs if p.name == args.instance]
    if not run_dirs:
        logger.warning("No run dirs with manifest.json under %s", stamp_dir)
        return 0

    n_graded = n_skipped = n_failed = 0
    for run_dir in sorted(run_dirs):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        instance_id = manifest.get("instance_id")
        inst = inst_map.get(instance_id)
        if not inst:
            logger.warning("%s: instance_id %r not in instance_data — skipping",
                           run_dir.name, instance_id)
            n_skipped += 1
            continue
        try:
            r = grade_run(run_dir, inst, force=args.force, workers=args.workers)
            if "skipped" in r:
                n_skipped += 1
                logger.info("  %s: skipped (%s)", run_dir.name, r["skipped"])
            else:
                n_graded += 1
        except Exception as e:
            logger.exception("%s: grading failed", run_dir.name)
            n_failed += 1

    logger.info("Done: graded=%d skipped=%d failed=%d", n_graded, n_skipped, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
