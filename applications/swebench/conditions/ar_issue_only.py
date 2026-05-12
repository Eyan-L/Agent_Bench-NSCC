"""AR-issue-only baseline: single-shot LLM generation from issue text.

No repo exploration, no retries, no tool use. One LLM_INFER step per turn.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict

from applications.conditions._llm_client import call_llm, extract_patch

logger = logging.getLogger(__name__)


class ARIssueOnlyCondition:
    """Single-shot AR generation from issue text only."""

    def __init__(self, model_config, max_turns: int = 1):
        self.model_config = model_config
        self.max_turns = max_turns

    def run(self, instance_data: Dict, run_dir: Path, seed: int) -> Dict:
        logger.info("Running AR-issue-only for %s", instance_data["instance_id"])

        issue_text = instance_data["problem_statement"]
        steps = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        patch_content = ""

        for turn in range(self.max_turns):
            step_start_ns = time.time_ns()
            prompt = self._build_prompt(issue_text, turn)

            text, pt, ct, err = call_llm(self.model_config, prompt, seed)

            step_end_ns = time.time_ns()
            total_prompt_tokens += pt
            total_completion_tokens += ct

            steps.append({
                "trajectory_id": run_dir.name,
                "attempt_id": 1,
                "step_id": turn + 1,
                "parent_step_id": None,
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
                "config_id": "ar_issue_only",
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

    def _build_prompt(self, issue_text: str, turn: int) -> str:
        if turn == 0:
            return (
                "You are a software engineer fixing a bug in a Python repository.\n\n"
                "Issue:\n"
                f"{issue_text}\n\n"
                "Generate a git patch to fix this issue. Output ONLY the patch in "
                "unified diff format. No explanation, no markdown fences — just the "
                "raw diff starting with 'diff --git' or '--- a/...'."
            )
        return "Refine the patch based on feedback."