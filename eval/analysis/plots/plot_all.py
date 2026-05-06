"""Run all RQ1/RQ2/RQ3 plot scripts in one shot.

Run:
    python -m eval.analysis.plots.plot_all \
        --runs-dir data/runs --out-dir data/figures
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval.analysis.plots import rq1_composition, rq2_resources, rq3_metrics

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render every RQ1/RQ2/RQ3 figure to <out-dir>/{rq1,rq2,rq3}/"
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--also-pdf", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    common = ["--runs-dir", str(args.runs_dir),
              "--log-level", args.log_level]
    if args.also_pdf:
        common.append("--also-pdf")

    rcs = []
    for mod, sub in [
        (rq1_composition, "rq1"),
        (rq2_resources,   "rq2"),
        (rq3_metrics,     "rq3"),
    ]:
        argv_sub = common + ["--out-dir", str(args.out_dir / sub)]
        rc = mod.main(argv_sub)
        rcs.append(rc)
    return max(rcs) if rcs else 0


if __name__ == "__main__":
    sys.exit(main())
