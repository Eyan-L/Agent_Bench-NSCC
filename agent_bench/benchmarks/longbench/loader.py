"""LongBench v1 loader — materializes per-subtask ``instance_data`` for the driver.

Reads LongBench's raw per-subtask JSONL (from ``data.zip``, unpacked to
``AB_BENCHMARK_DATA_ROOT``). We deliberately bypass ``datasets.load_dataset`` because
LongBench v1 is a *script-based* dataset that ``datasets>=4`` can no longer load.
Each JSONL row::

    {input, context, answers, length, dataset, language, all_classes, _id}

One ``instance`` == one LongBench subtask (``instance_id`` == subtask name). The
harness loops over ``instance_data["examples"]`` (capped via AB_BENCHMARK_EXAMPLE_CAP),
and per example: ingests ``context`` into ClawRAG as markdown, has the agent
retrieve + answer, then scores with the subtask's LongBench metric. RAG-only — the
``context`` is NOT placed in the prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_CONFIG_DIR = _HERE / "longbench_config"

# 21 standard LongBench v1 subtasks (the *_e LongBench-E variants are excluded).
SUBTASKS: List[str] = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

# subtask -> LongBench metric function name (from LongBench/eval.py `dataset2metric`).
DATASET2METRIC: Dict[str, str] = {
    "narrativeqa": "qa_f1_score", "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score", "multifieldqa_zh": "qa_f1_zh_score",
    "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
    "dureader": "rouge_zh_score", "gov_report": "rouge_score", "qmsum": "rouge_score",
    "multi_news": "rouge_score", "vcsum": "rouge_zh_score",
    "trec": "classification_score", "triviaqa": "qa_f1_score",
    "samsum": "rouge_score", "lsht": "classification_score",
    "passage_count": "count_score", "passage_retrieval_en": "retrieval_score",
    "passage_retrieval_zh": "retrieval_zh_score",
    "lcc": "code_sim_score", "repobench-p": "code_sim_score",
}


def _data_dir() -> Path:
    return Path(os.environ.get("AB_BENCHMARK_DATA_ROOT", "/mnt/vdb/hongrui/longbench/data"))


def _example_cap() -> int:
    return int(os.environ.get("AB_BENCHMARK_EXAMPLE_CAP", "20"))


def _load_maxlen() -> Dict[str, int]:
    p = _CONFIG_DIR / "dataset2maxlen.json"
    try:
        return {k: int(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return {}


_MAXLEN = _load_maxlen()


def list_task_ids() -> List[str]:
    return list(SUBTASKS)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_task(task_id: str, cap: Optional[int] = None) -> Dict:
    """Build the ``instance_data`` dict for one LongBench subtask."""
    if task_id not in DATASET2METRIC:
        raise ValueError(f"unknown LongBench subtask {task_id!r}; expected one of {SUBTASKS}")
    cap = _example_cap() if cap is None else cap
    src = _data_dir() / f"{task_id}.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"LongBench data not found: {src} — unzip data.zip into AB_BENCHMARK_DATA_ROOT "
            f"(see agent_bench/benchmarks/longbench/loader.py docstring)."
        )
    rows = _read_jsonl(src)
    # Deterministic subset: first `cap` rows in upstream order (cap<=0 -> all rows).
    selected = rows if cap <= 0 else rows[:cap]
    examples = [
        {
            "_id": r.get("_id"),
            "input": r["input"],
            "context": r["context"],
            "answers": r.get("answers", []),
            "all_classes": r.get("all_classes"),
            "length": r.get("length"),
        }
        for r in selected
    ]
    language = selected[0].get("language", "en") if selected else "en"
    metric = DATASET2METRIC[task_id]
    return {
        "instance_id": task_id,
        "benchmark": "longbench",
        "task_name": task_id,
        "language": language,
        "metric": metric,
        "max_gen": _MAXLEN.get(task_id, 512),
        "example_cap": cap,
        "num_examples": len(examples),
        "examples": examples,
        # OC sandbox image for the agent; overridden by harness/config if set.
        "docker_image": os.environ.get(
            "AB_BENCHMARK_SANDBOX_IMAGE", "longbench/clawrag-sandbox.UNUSED:latest"
        ),
        "workspace": "/workspace",
        "agent_timeout_sec": int(os.environ.get("AB_BENCHMARK_AGENT_TIMEOUT_SEC", "600")),
        # SWE-bench-shaped echoes so generic readers (manifest emitter, etc.) don't NPE.
        "repo": "longbench",
        "base_commit": "longbench-v1",
        "problem_statement": f"LongBench subtask {task_id} ({metric}); "
        f"{len(examples)} examples, RAG via ClawRAG",
    }


def materialize_dataset(task_ids: List[str], out_path: Path, cap: Optional[int] = None) -> None:
    """Write a JSON file with one entry per subtask (no indent — contexts are large)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [load_task(t, cap=cap) for t in task_ids]
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)


# CLI: python -m agent_bench.benchmarks.longbench.loader <out.json> [subtask ...]
if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("usage: loader.py <out.json> [subtask ...]", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    ids = sys.argv[2:] or list_task_ids()
    materialize_dataset(ids, out)
    print(f"wrote {len(ids)} subtasks -> {out}")
