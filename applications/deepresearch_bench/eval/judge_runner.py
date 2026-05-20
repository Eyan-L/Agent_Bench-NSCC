"""Host-side RACE + FACT runner for one DR-Bench instance.

Shells out to the vendored ``deepresearch_bench`` scripts, which we patched
to accept ``LLM_BACKEND=deepseek``. See ``vendor/PATCHES.md``.

Per-instance contract — one ``run()`` call evaluates one ``(id, prompt,
article)`` tuple and writes ``score.json`` to ``output_dir``. Crashes on
one instance leave others untouched (each gets its own ``target_model``
name + private result subdir).

The agent's report text comes in as ``article`` (str). We stage it as

    vendor/data/test_data/raw_data/<target_model>.jsonl

with one JSON line, then invoke the vendored CLIs. ``target_model`` is
``run_<run_id>_<instance_id>`` so simultaneous runs do not collide on the
shared ``raw_data/`` / ``cleaned_data/`` directories.

Required env (resolved by ``run()``):

    DEEPSEEK_API_KEY    LLM judge backend (RACE + FACT)
    JINA_READER_API     Jina Reader for FACT citation scraping
                        (aliased internally to JINA_API_KEY)

Optional:
    RACE_MODEL          override the deepseek-chat default
    FACT_MODEL          override the deepseek-chat default
    DRB_JUDGE_WORKERS   max RACE thread workers (default 4)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dr_bench.judge_runner")

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
QUERY_FILE = VENDOR_DIR / "data" / "prompt_data" / "query.jsonl"
RAW_DATA_DIR = VENDOR_DIR / "data" / "test_data" / "raw_data"
CLEANED_DATA_DIR = VENDOR_DIR / "data" / "test_data" / "cleaned_data"


@dataclass
class RaceScores:
    comprehensiveness: float
    insight: float
    instruction_following: float
    readability: float
    overall: float


@dataclass
class FactScores:
    citation_accuracy: float
    effective_citations: float
    total_citations: int


@dataclass
class JudgeResult:
    instance_id: str
    race: Optional[RaceScores]
    fact: Optional[FactScores]
    race_error: Optional[str] = None
    fact_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def _prepare_env() -> Dict[str, str]:
    """Build the subprocess env: forces deepseek backend + aliases JINA key."""
    env = os.environ.copy()
    if "DEEPSEEK_API_KEY" not in env:
        raise RuntimeError("DEEPSEEK_API_KEY must be set in host env")
    env["LLM_BACKEND"] = "deepseek"
    # vendor's WebScrapingJinaTool reads JINA_API_KEY; user's env uses
    # JINA_READER_API. Map without mutating the user's shell.
    if "JINA_READER_API" in env and "JINA_API_KEY" not in env:
        env["JINA_API_KEY"] = env["JINA_READER_API"]
    return env


# ---------------------------------------------------------------------------
# RACE
# ---------------------------------------------------------------------------

def _stage_raw_data(target_model: str, instance_row: Dict) -> Path:
    """Write the one-row JSONL the RACE script will read."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / f"{target_model}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(instance_row, ensure_ascii=False) + "\n")
    return raw_path


def _run_race(
    target_model: str,
    language: str,
    output_subdir: Path,
    env: Dict[str, str],
    workers: int,
) -> RaceScores:
    """Invoke ``deepresearch_bench_race.py`` for a single instance."""
    output_subdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "deepresearch_bench_race.py",
        target_model,
        "--raw_data_dir", str(RAW_DATA_DIR),
        "--cleaned_data_dir", str(CLEANED_DATA_DIR),
        "--max_workers", str(workers),
        "--query_file", str(QUERY_FILE),
        "--output_dir", str(output_subdir),
        "--force",  # don't reuse stale results across runs
    ]
    cmd.append("--only_zh" if language == "zh" else "--only_en")

    logger.info("RACE: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(VENDOR_DIR), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"RACE script exited {proc.returncode}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )

    # Per-task scores are in raw_results.jsonl (preferred — it has the per-row
    # numbers). race_result.txt is the aggregate "averages over N rows" file,
    # which for our N=1 case equals the per-row numbers anyway.
    raw_results = output_subdir / "raw_results.jsonl"
    if not raw_results.exists():
        raise RuntimeError(f"RACE produced no raw_results.jsonl in {output_subdir}")
    with raw_results.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    successful = [r for r in rows if "error" not in r]
    if not successful:
        msg = rows[0].get("error", "unknown") if rows else "no rows"
        raise RuntimeError(f"RACE row failed: {msg}")
    row = successful[0]
    return RaceScores(
        comprehensiveness=float(row.get("comprehensiveness", 0.0)),
        insight=float(row.get("insight", 0.0)),
        instruction_following=float(row.get("instruction_following", 0.0)),
        readability=float(row.get("readability", 0.0)),
        overall=float(row.get("overall_score", 0.0)),
    )


# ---------------------------------------------------------------------------
# FACT
# ---------------------------------------------------------------------------

def _run_fact(
    target_model: str,
    raw_data_path: Path,
    output_subdir: Path,
    env: Dict[str, str],
    workers: int,
) -> FactScores:
    """Walk the 5-stage FACT pipeline for a single instance.

    `workers` controls cross-instance parallelism; per-instance we hard-cap
    `validate` at 1 to avoid blowing DeepSeek-V4-Pro rate limits with
    concurrent reasoning-model calls. With workers>=2, validate.py retries
    3× and then sets validate_error, which stat.py silently skips
    (`continue`) → total_citations=0 even though scrape returned good
    content. Verified manually on dr_001's report: workers=1 yields 33
    citations with 79% supported; workers=4 yielded 0.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    extracted = output_subdir / "extracted.jsonl"
    dedup = output_subdir / "deduplicated.jsonl"
    scraped = output_subdir / "scraped.jsonl"
    validated = output_subdir / "validated.jsonl"
    result_txt = output_subdir / "fact_result.txt"

    # Per-instance, validate makes ~5-10 long-context DeepSeek calls; >1
    # worker reliably trips rate limits. Other stages tolerate workers>1.
    validate_workers = int(os.environ.get("DRB_VALIDATE_WORKERS", "1"))

    stages = [
        ("extract",
         [sys.executable, "-u", "-m", "utils.extract",
          "--raw_data_path", str(raw_data_path),
          "--output_path", str(extracted),
          "--query_data_path", str(QUERY_FILE),
          "--n_total_process", str(workers)]),
        ("deduplicate",
         [sys.executable, "-u", "-m", "utils.deduplicate",
          "--raw_data_path", str(extracted),
          "--output_path", str(dedup),
          "--query_data_path", str(QUERY_FILE),
          "--n_total_process", str(workers)]),
        ("scrape",
         [sys.executable, "-u", "-m", "utils.scrape",
          "--raw_data_path", str(dedup),
          "--output_path", str(scraped),
          "--n_total_process", str(workers)]),
        ("validate",
         [sys.executable, "-u", "-m", "utils.validate",
          "--raw_data_path", str(scraped),
          "--output_path", str(validated),
          "--query_data_path", str(QUERY_FILE),
          "--n_total_process", str(validate_workers)]),
        ("stat",
         [sys.executable, "-u", "-m", "utils.stat",
          "--input_path", str(validated),
          "--output_path", str(result_txt)]),
    ]
    for stage_name, cmd in stages:
        logger.info("FACT[%s]: %s", stage_name, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(VENDOR_DIR), env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"FACT stage {stage_name!r} exited {proc.returncode}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )

    if not result_txt.exists():
        raise RuntimeError(f"FACT produced no fact_result.txt in {output_subdir}")
    return _parse_fact_result(result_txt)


def _parse_fact_result(path: Path) -> FactScores:
    """Parse the human-readable ``fact_result.txt`` (key: value lines).

    Upstream's ``utils/stat.py`` writes three fields with underscores:
      total_citations:        avg (supp+unsup) per task
      total_valid_citations:  avg supp per task
      valid_rate:             supp / (supp+unsup), the citation accuracy
    Mapping:
      citation_accuracy   ← valid_rate
      effective_citations ← total_valid_citations
      total_citations     ← total_citations
    """
    fields: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().lower()] = v.strip()

    def _pick(*keys: str) -> Optional[str]:
        for k in keys:
            if k in fields:
                return fields[k]
        return None

    def _float(s: Optional[str], default: float = 0.0) -> float:
        if s is None:
            return default
        try:
            return float(s.rstrip("%"))
        except ValueError:
            return default

    def _int(s: Optional[str], default: int = 0) -> int:
        if s is None:
            return default
        try:
            return int(float(s))
        except ValueError:
            return default

    return FactScores(
        citation_accuracy=_float(_pick("valid_rate", "citation_accuracy", "citation accuracy", "accuracy")),
        effective_citations=_float(_pick("total_valid_citations", "effective_citations", "effective citations", "effective citation")),
        total_citations=_int(_pick("total_citations", "total citations", "total citation")),
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run(
    *,
    instance_id: str,
    prompt: str,
    language: str,
    article: str,
    output_dir: Path,
    run_id: str,
    skip_fact: bool = False,
    workers: Optional[int] = None,
) -> JudgeResult:
    """Score one ``(instance_id, article)`` pair. Writes ``score.json``.

    Args:
        instance_id: e.g. ``"dr_001"`` — used in the staged target_model name.
        prompt: original task prompt (must match an entry in query.jsonl,
            since RACE looks up the criteria and reference by prompt string).
        language: ``"en"`` or ``"zh"`` (drives the only_zh/only_en flag).
        article: agent-generated report markdown.
        output_dir: per-instance directory; receives ``score.json`` plus the
            vendor's intermediate ``race/`` and ``fact/`` subdirs.
        run_id: run identifier; appended to target_model so parallel runs
            don't collide on the shared raw_data/cleaned_data dirs.
        skip_fact: only run RACE (for cheap smoke testing). Defaults False.
        workers: thread/process pool size; defaults to env DRB_JUDGE_WORKERS
            or 4.

    Returns:
        ``JudgeResult`` with race + fact scores or per-pipeline error messages.
    """
    if language not in ("en", "zh"):
        raise ValueError(f"language must be 'en' or 'zh', got {language!r}")
    workers = workers or int(os.environ.get("DRB_JUDGE_WORKERS", "4"))
    env = _prepare_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_model = f"run_{run_id}_{instance_id}"
    try:
        instance_id_int = int(instance_id.lstrip("dr_"))
    except ValueError:
        instance_id_int = abs(hash(instance_id)) % (10 ** 9)
    instance_row = {"id": instance_id_int, "prompt": prompt, "article": article}

    raw_path = _stage_raw_data(target_model, instance_row)

    race_scores: Optional[RaceScores] = None
    race_error: Optional[str] = None
    try:
        race_scores = _run_race(
            target_model=target_model,
            language=language,
            output_subdir=output_dir / "race",
            env=env,
            workers=workers,
        )
    except Exception as e:
        race_error = str(e)
        logger.exception("RACE failed for %s", instance_id)

    fact_scores: Optional[FactScores] = None
    fact_error: Optional[str] = None
    if skip_fact:
        fact_error = "skipped"
    else:
        try:
            fact_scores = _run_fact(
                target_model=target_model,
                raw_data_path=raw_path,
                output_subdir=output_dir / "fact",
                env=env,
                workers=workers,
            )
        except Exception as e:
            fact_error = str(e)
            logger.exception("FACT failed for %s", instance_id)

    result = JudgeResult(
        instance_id=instance_id,
        race=race_scores,
        fact=fact_scores,
        race_error=race_error,
        fact_error=fact_error,
    )
    _write_score_json(output_dir / "score.json", result)
    _cleanup_staging(target_model)
    return result


def _write_score_json(path: Path, result: JudgeResult) -> None:
    payload = {
        "instance_id": result.instance_id,
        "race": asdict(result.race) if result.race else None,
        "fact": asdict(result.fact) if result.fact else None,
        "race_error": result.race_error,
        "fact_error": result.fact_error,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_staging(target_model: str) -> None:
    """Remove the per-instance staged JSONL files from the vendor data dir."""
    for d in (RAW_DATA_DIR, CLEANED_DATA_DIR):
        p = d / f"{target_model}.jsonl"
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser(description="Run RACE+FACT on one DR-Bench report.")
    p.add_argument("--instance-id", required=True)
    p.add_argument("--prompt", help="Inline prompt; or --prompt-file.")
    p.add_argument("--prompt-file", help="Path to a file with the prompt.")
    p.add_argument("--language", choices=["en", "zh"], required=True)
    p.add_argument("--report", required=True, help="Path to agent's report markdown.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", default="cli")
    p.add_argument("--skip-fact", action="store_true")
    args = p.parse_args()

    if args.prompt is None and args.prompt_file is None:
        p.error("--prompt or --prompt-file is required")
    prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
    article = Path(args.report).read_text(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    result = run(
        instance_id=args.instance_id,
        prompt=prompt,
        language=args.language,
        article=article,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        skip_fact=args.skip_fact,
    )
    print(json.dumps({
        "instance_id": result.instance_id,
        "race": asdict(result.race) if result.race else None,
        "fact": asdict(result.fact) if result.fact else None,
        "race_error": result.race_error,
        "fact_error": result.fact_error,
    }, ensure_ascii=False, indent=2))
    return 0 if (result.race is not None) else 2


if __name__ == "__main__":
    sys.exit(_cli())
