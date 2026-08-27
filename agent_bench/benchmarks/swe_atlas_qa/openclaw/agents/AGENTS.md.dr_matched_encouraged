# AGENTS.md

## Task

You receive a repository question.
Your job:

1. Read the repository question carefully — note the exact scope and any specific deliverables the question asks for.
2. Plan the work: decompose the question into sub-questions and decide what evidence each sub-question needs.
3. When a sub-question can be investigated independently and would benefit from separate focused work, consider using an available subagent to investigate it.
4. Explore the repository with the available tools to find relevant evidence.
5. Read each promising source of evidence carefully.
6. Synthesize the evidence into a coherent technical answer.
7. When the answer is complete, follow the answer-submission requirements below.

Do NOT modify any files in the repository.

## CRITICAL MANDATE: TASK COMPLETION AND ANSWER SUBMISSION

When you are confident in your answer, you **MUST** write your complete final answer to disk before terminating. The evaluation harness validates your run solely based on these exact file paths. 

You **MUST** invoke your native `write` tool to save your comprehensive technical report to **BOTH** of the following exact locations:

1. **Workspace Root Path:** Write a markdown-formatted response to `/app/answer.md`
2. **Harness Log Path:** Write the same markdown-formatted response to `/logs/agent/answer.txt`

### Strict Execution Constraints:
- **BANNED PATHS:** Do NOT write to `/tmp/` (e.g., `/tmp/kitty-rc-explained.md`). Writing to `/tmp/` will fail due to environment restrictions. Always use `/app/answer.md`.
- **NO TAGS:** Do NOT wrap your answer in literal text tags like `<<FINAL_ANSWER>>`. Write clean, raw markdown.
- Do NOT invoke your session-closing or execution-halt commands until you have successfully verified that both files exist and are non-empty.

### Required Final Step Sequence:
1. **Step 1:** Call your native `write` tool with argument `path: "/app/answer.md"` containing your full report.
2. **Step 2:** Call your native `write` tool with argument `path: "/logs/agent/answer.txt"` containing your full report.
3. **Step 3:** In a **SEPARATE, final follow-up command**, call your `exec` tool with the submission trigger:
   ```bash
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
   ```
