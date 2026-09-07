# AGENTS.md

You can execute bash commands or use your file tools to explore the codebase and gather evidence for your answer. Do NOT modify any files in the repository.

## Command Execution Rules

You are operating in an environment where:
1. You issue at least one command or tool call.
2. The system executes the action.
3. You see the result(s).
4. You write your next action(s).

Each response should include:
1. **Reasoning text** where you explain your analysis and plan.
2. At least one tool call with your command or action.

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include reasoning text explaining what you're doing.
- Your response MUST include AT LEAST ONE tool call (`exec`, `read`, `write`, etc.).
- Directory or environment variable changes inside `exec` are not persistent. Every shell action is executed in a new subshell.
- You can prefix any bash action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files.

---

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
   
Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

[Makes bash tool call with {"command": "ls -la"} as arguments]
</example_response>

## Useful command examples

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Search for patterns:

```bash
grep -rn "pattern" /app/src/
```

### Explore directory structure:

```bash
find /app -type f -name "*.py" | head -20
```
