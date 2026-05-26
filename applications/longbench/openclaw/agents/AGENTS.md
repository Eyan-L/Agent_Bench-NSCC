# AGENTS.md

You are answering ONE question using a private knowledge base (a RAG system),
inside a Linux sandbox. The document(s) relevant to the question have already
been ingested into the knowledge base — you must **retrieve** from it, not guess.

## What to do (every session = one question)

1. Read the question in the user message.
2. Retrieve relevant passages by running the search tool:
   ```
   python3 /workspace/clawrag_search.py "<a focused search query>"
   ```
   It prints the top passages from the knowledge base (separated by `---`).
   You may run it more than once with different queries if the first result
   is not enough. Optionally pass a second arg for top-k, e.g. `... "query" 8`.
3. Answer the question **using ONLY the retrieved passages**. If the passages
   do not contain the answer, answer with your best extraction from them — do
   not invent facts not present in the passages.
4. Reply with **ONLY the final answer text** — as concise as possible
   (a phrase, name, number, or one short sentence). No explanation, no
   preamble, no markdown, no quotes around it, no "The answer is".

The `clawrag_retrieve` skill (details on the retrieval step) is available in the
workspace at `/workspace/skills/clawrag_retrieve/SKILL.md` — read THAT path if you
want it. Do NOT try to read `~/.openclaw/...` or `/root/.openclaw/...`: those are
outside the sandbox and will fail with "Path escapes sandbox root". Everything you
need is under `/workspace`.

## Hard rules

- Always retrieve first with `clawrag_search.py` before answering. The context
  is NOT in your prompt; it lives in the knowledge base.
- Keep the answer short and direct — the grader compares your answer text to a
  short reference answer; extra words lower the score.
- Do not write files. Do not ask the user anything. Do not reply with `DONE`
  or any wrapper — your final message IS the answer.
- Only `/workspace` paths are valid for any tool input (incl. reading the skill).
