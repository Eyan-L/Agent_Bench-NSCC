from __future__ import annotations

from typing import Dict

from agent_bench.benchmarks.swebench.conditions.hermes_agent import HermesAgentCondition


class SWEAtlasQAHermesCondition(HermesAgentCondition):
    """Hermes Agent driving a SWE-Atlas-QA task container."""

    def _build_message(self, instance_data: Dict) -> str:
        question = instance_data.get('question') or instance_data.get('problem_statement') or instance_data.get('prompt') or ''
        instructions = (
            "You can execute bash commands or use your file tools to explore the codebase and gather evidence for your answer. "
            "Do NOT modify any files in the repository.\n\n"
            "## CRITICAL MANDATE: TASK COMPLETION AND ANSWER SUBMISSION\n\n"
            "When you are confident in your answer, you MUST write your complete final answer to disk before terminating. "
            "The evaluation harness validates your run solely based on these exact file paths. "
            "You MUST write your comprehensive technical report to BOTH of the following exact locations:\n\n"
            "1. Workspace Root Path: Write a markdown-formatted response to `/app/answer.md`\n"
            "2. Harness Log Path: Write the same markdown-formatted response to `/logs/agent/answer.txt`\n\n"
            "You can write these files using your file tools or by executing bash commands (e.g. using redirect or cat).\n\n"
            "### Strict Execution Constraints:\n"
            "- BANNED PATHS: Do NOT write to `/tmp/` (e.g., `/tmp/kitty-rc-explained.md`). Writing to `/tmp/` will fail due to environment restrictions. Always use `/app/answer.md`.\n"
            "- NO TAGS: Do NOT wrap your answer in literal text tags like `<<FINAL_ANSWER>>`. Write clean, raw markdown.\n"
            "- Do NOT invoke your session-closing or execution-halt commands until you have successfully verified that both files exist and are non-empty.\n\n"
            "Please begin by analyzing the codebase and answering the question below.\n\n"
            f"Question:\n{question}"
        )
        return instructions
