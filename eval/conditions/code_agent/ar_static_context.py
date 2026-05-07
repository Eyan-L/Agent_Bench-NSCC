"""AR-static-context baseline: issue + offline static context.

Builds context offline from the repo container using:
  1. Top-2-level file tree (via `docker exec find`)
  2. BM25 retrieval over .py file CONTENT to pick top-K relevant files

The "offline" framing is important: this context is constructed BEFORE the LLM
call, non-iteratively, so the LLM never interacts with the repo. This is what
distinguishes it from OpenClaw agent in the RQ1 composition analysis.

BM25 implementation is pure stdlib — no external dependency. For Phase 0/1
throughput this is fine; if you scale to many parallel runs consider caching
the index per (repo, base_commit) tuple.
"""
from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .._llm_client import call_llm, extract_patch

logger = logging.getLogger(__name__)


# BM25 hyperparameters — defaults from the standard Okapi BM25 paper.
_BM25_K1 = 1.5
_BM25_B = 0.75

# Token regex: split on non-alphanumeric/underscore. Good enough for code +
# natural language. Case-folded so "DataFrame" matches "dataframe".
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

# File size caps — don't index or include enormous files; they pollute BM25
# and blow up the prompt.
_MAX_FILE_BYTES = 200_000
_MAX_SNIPPET_BYTES = 2000


class ARStaticContextCondition:
    """AR generation with offline static context (file tree + BM25 top-K)."""

    def __init__(self, model_config, max_turns: int = 1, top_k_files: int = 5):
        self.model_config = model_config
        self.max_turns = max_turns
        self.top_k_files = top_k_files

    def run(
        self,
        instance_data: Dict,
        run_dir: Path,
        seed: int,
        workspace_root: str,
        container_id: str,
        env_info: Optional[Dict] = None,
    ) -> Dict:
        logger.info("Running AR-static-context for %s", instance_data["instance_id"])

        issue_text = instance_data["problem_statement"]
        steps = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        patch_content = ""

        # Step 0: build static context (TOOL_READ with offline=true flag).
        step_start_ns = time.time_ns()
        static_context = self._build_static_context(container_id, workspace_root, issue_text)
        step_end_ns = time.time_ns()

        steps.append({
            "trajectory_id": run_dir.name,
            "attempt_id": 1,
            "step_id": 0,
            "parent_step_id": None,
            "step_type": "TOOL_READ",
            "step_subtype": "TOOL_READ_OFFLINE_CONTEXT",
            "tool_name": "static_context_builder",
            "command": f"bm25+tree on {workspace_root}",
            "file_path": None,
            "start_ts": step_start_ns,
            "end_ts": step_end_ns,
            "duration_ms": (step_end_ns - step_start_ns) / 1e6,
            "prompt_tokens": None,
            "completion_tokens": None,
            "exit_code": 0,
            "error_type": None,
            "repo_id": instance_data["repo"],
            "instance_id": instance_data["instance_id"],
            "config_id": "ar_static_context",
            "model_name": self.model_config.name,
            "offline": True,
        })

        # LLM_INFER steps.
        for turn in range(self.max_turns):
            step_start_ns = time.time_ns()
            prompt = self._build_prompt(issue_text, static_context, turn)

            text, pt, ct, err = call_llm(self.model_config, prompt, seed)

            step_end_ns = time.time_ns()
            total_prompt_tokens += pt
            total_completion_tokens += ct

            steps.append({
                "trajectory_id": run_dir.name,
                "attempt_id": 1,
                "step_id": turn + 1,
                "parent_step_id": 0,
                "step_type": "LLM_INFER",
                "tool_name": None,
                "command": None,
                "file_path": None,
                "start_ts": step_start_ns,
                "end_ts": step_end_ns,
                "duration_ms": (step_end_ns - step_start_ns) / 1e6,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "exit_code": None,
                "error_type": err or None,
                "repo_id": instance_data["repo"],
                "instance_id": instance_data["instance_id"],
                "config_id": "ar_static_context",
                "model_name": self.model_config.name,
            })

            patch_content = extract_patch(text)
            if patch_content:
                break

        step_trace_path = run_dir / "steps.jsonl"
        with open(step_trace_path, "w") as f:
            for step in steps:
                f.write(json.dumps(step) + "\n")

        patch_path = run_dir / "candidate.patch"
        patch_path.write_text(patch_content)

        return {
            "step_trace_path": str(step_trace_path),
            "patch_path": str(patch_path),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }

    # ------------------------------------------------------------------
    # Static context construction (inside container, via docker exec)
    # ------------------------------------------------------------------

    def _build_static_context(
        self, container_id: str, workspace_root: str, issue_text: str
    ) -> str:
        parts = []

        tree = self._get_file_tree(container_id, workspace_root)
        if tree:
            parts.append("Repository structure (top 2 levels):\n" + tree)

        relevant = self._bm25_retrieve(
            container_id, workspace_root, issue_text, self.top_k_files
        )
        if relevant:
            parts.append("\nTop relevant files by BM25:")
            for path, snippet in relevant:
                parts.append(f"\n--- {path} ---\n{snippet}")

        return "\n".join(parts) if parts else "(no context available)"

    def _get_file_tree(self, container_id: str, workspace_root: str) -> str:
        """Use find inside the container to get a 2-level directory listing."""
        r = subprocess.run(
            [
                "docker", "exec", container_id,
                "find", workspace_root,
                "-maxdepth", "2",
                "-not", "-path", "*/.*",       # skip hidden
                "-not", "-path", "*/__pycache__*",
                "-type", "f", "-o", "-type", "d",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning("File tree listing failed: %s", r.stderr[:200])
            return ""
        # Truncate — don't blow up the prompt.
        lines = r.stdout.splitlines()
        if len(lines) > 200:
            lines = lines[:200] + [f"... (+{len(r.stdout.splitlines()) - 200} more entries)"]
        return "\n".join(lines)

    def _list_py_files(self, container_id: str, workspace_root: str) -> List[str]:
        """Return relative paths of .py files in the repo."""
        r = subprocess.run(
            [
                "docker", "exec", container_id,
                "find", workspace_root,
                "-name", "*.py",
                "-not", "-path", "*/.*",
                "-not", "-path", "*/__pycache__*",
                "-type", "f",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return []
        return [p for p in r.stdout.splitlines() if p.strip()]

    def _read_file(self, container_id: str, path: str) -> str:
        """Read a single file from the container; return empty string on failure."""
        r = subprocess.run(
            ["docker", "exec", container_id, "cat", path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return ""
        if len(r.stdout) > _MAX_FILE_BYTES:
            return ""  # Skip oversized files — they skew BM25.
        return r.stdout

    # ------------------------------------------------------------------
    # BM25 implementation (pure stdlib)
    # ------------------------------------------------------------------

    def _bm25_retrieve(
        self,
        container_id: str,
        workspace_root: str,
        query: str,
        top_k: int,
    ) -> List[Tuple[str, str]]:
        """Index all .py files in the repo and return top-K scored by BM25 on `query`.

        Returns [(file_path, snippet)], sorted by descending score.
        """
        files = self._list_py_files(container_id, workspace_root)
        if not files:
            logger.warning("No .py files found under %s — BM25 returned empty", workspace_root)
            return []

        # Read all files. This is O(N) docker exec calls — slow for big repos.
        # Acceptable for Phase 0 (single instance); for scale, batch via tar.
        docs: List[Tuple[str, List[str]]] = []
        for path in files:
            content = self._read_file(container_id, path)
            if not content:
                continue
            tokens = self._tokenize(content)
            if tokens:
                docs.append((path, tokens))

        if not docs:
            return []

        # BM25 setup
        N = len(docs)
        doc_lens = [len(toks) for _, toks in docs]
        avgdl = sum(doc_lens) / N

        # Document frequency per term
        df: Counter = Counter()
        for _, tokens in docs:
            for term in set(tokens):
                df[term] += 1

        # Score each doc against the query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[Tuple[float, str]] = []
        for (path, tokens), dl in zip(docs, doc_lens):
            tf = Counter(tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
                tf_qt = tf[qt]
                numerator = tf_qt * (_BM25_K1 + 1)
                denominator = tf_qt + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                score += idf * (numerator / denominator)
            if score > 0:
                scored.append((score, path))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        # Build result with snippets (first _MAX_SNIPPET_BYTES of each file)
        results: List[Tuple[str, str]] = []
        for _, path in top:
            content = self._read_file(container_id, path)
            snippet = content[:_MAX_SNIPPET_BYTES]
            if len(content) > _MAX_SNIPPET_BYTES:
                snippet += f"\n... (truncated, full file is {len(content)} bytes)"
            results.append((path, snippet))
        return results

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, issue_text: str, static_context: str, turn: int) -> str:
        return (
            "You are a software engineer fixing a bug in a Python repository.\n\n"
            f"Issue:\n{issue_text}\n\n"
            f"Repository context:\n{static_context}\n\n"
            "Generate a git patch to fix this issue. Output ONLY the patch in "
            "unified diff format. No explanation, no markdown fences — just the "
            "raw diff starting with 'diff --git' or '--- a/...'."
        )