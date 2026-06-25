"""DeepResearch-Bench specialization of the Hermes agent condition."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition


class DeepResearchHermesCondition(HermesAgentCondition):
    """Hermes agent writing a DeepResearch report in the task sandbox."""

    def _build_message(self, instance_data: Dict) -> str:
        return (
            "Read /workspace/PROMPT.md and write the final research report to /workspace/report.md. "
            "Use terminal/file tools to gather evidence, but keep tool outputs compact: save raw "
            "search/fetch results under /workspace/notes and print only short, relevant evidence "
            "summaries back to the conversation. If BRAVE_API_KEY is available, search with "
            "https://api.search.brave.com/res/v1/web/search and header "
            "'X-Subscription-Token: $BRAVE_API_KEY'; do not scrape search.brave.com HTML. "
            "When fetching pages, prefer Jina Reader if JINA_READER_API or JINA_API_KEY is set, "
            "then extract only evidence needed for the prompt. When writing the final report, "
            "write directly to /workspace/report.md in manageable chunks instead of emitting the "
            "whole report in one large tool call; append sections if needed and make each command "
            "print only a short confirmation. Include inline markdown citations for factual "
            "claims. Reply with a single line DONE only after /workspace/report.md has been "
            "written and verified to exist.\n\n"
            f"Task id: {instance_data.get('instance_id')}"
        )

    def _collect_artifacts(
        self,
        env_info: Dict,
        container_id: str,
        workspace_root: str,
    ) -> str:
        return "# deepresearch_bench: no patch - report.md is scored by judge\n"

    def _collect_sessions(
        self,
        env_info: Dict,
        container_id: str,
        run_dir: Path,
        workspace_root: str,
    ) -> None:
        harness = env_info.get("_harness")
        if harness is None or not hasattr(harness, "exec_in_container"):
            return
        harness.exec_in_container(
            container_id,
            ["hermes", "sessions", "export", "sessions.jsonl"],
            timeout=120,
            env=env_info.get("agent_env"),
            workdir=workspace_root,
            check=False,
        )
        subprocess.run(
            ["docker", "cp", f"{container_id}:{workspace_root}/sessions.jsonl", str(run_dir / "sessions.jsonl")],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
