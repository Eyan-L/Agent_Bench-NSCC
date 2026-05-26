# BOOTSTRAP.md

This is a single-question Retrieval-Augmented QA task. Bootstrap is trivial —
there is nothing to install or set up.

- The document for this question is already ingested into a knowledge base.
- Your tools run inside `/workspace`. Only `/workspace` paths are valid.
- Answer via the **`clawrag_retrieve` skill** (it is listed in your available
  skills). Load it first: `read /workspace/skills/clawrag_retrieve/SKILL.md`,
  then follow it. In short: retrieve, then reply. Run (with the collection name
  given in the user message):
      python3 /workspace/clawrag_search.py "<focused query>" 5 <collection>
  then answer the question as concisely as possible using ONLY the returned
  passages.
- Do NOT read paths outside `/workspace` (e.g. `~/.openclaw`, host paths) — they
  error with "Path escapes sandbox root".

Bootstrap is COMPLETE as soon as you have read this file. Then proceed to answer
the question; your final reply must be ONLY the answer text, no preamble.
