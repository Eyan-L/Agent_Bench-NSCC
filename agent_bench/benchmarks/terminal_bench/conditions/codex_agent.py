"""Terminal-Bench prompt for the generic Codex CLI runner."""
from agent_bench.harnesses.codex.condition import CodexAgentCondition


class TerminalBenchCodexCondition(CodexAgentCondition):
    def _build_message(self, instance_data):
        return (
            "Complete the following task in the current workspace. Inspect the "
            "environment, implement the requested changes, and verify your work. "
            "Read BOOTSTRAP.md in the workspace for task-runtime guidance. "
            "Reply DONE only when the task is complete.\n\n"
            + instance_data["instruction"]
        )
