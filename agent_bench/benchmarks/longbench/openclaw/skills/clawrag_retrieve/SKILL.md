---
name: clawrag_retrieve
description: Retrieve relevant passages from the ClawRAG knowledge base for the current question. Use this BEFORE answering — the source document is in the knowledge base, not in your prompt. Run the seeded search script and read the returned passages.
---

# clawrag_retrieve

The question's source document has been ingested into a local ClawRAG knowledge
base. You retrieve from it with a seeded helper script — there is no document in
your prompt.

## How to invoke

Run, via the exec tool:

```
python3 /workspace/clawrag_search.py "<focused search query>"
```

- Arg 1: your search query (phrase the information need, not the whole question).
- Arg 2 (optional): top-k, default 5. Use a larger k (e.g. 8) if the first
  passages are not enough: `python3 /workspace/clawrag_search.py "<query>" 8`.

It prints the top passages, separated by lines of `---`. The script already
knows which knowledge-base collection to search and how to reach the service —
you only supply the query.

## Using the output

- Read the printed passages and extract the answer from them.
- If the passages don't answer the question, run the script again with a
  different query (synonyms, named entities, narrower/broader phrasing).
- Cite nothing and add nothing not present in the passages. Then give a short,
  direct final answer (see AGENTS.md).

## What it does NOT do

- It does not ingest or write anything — retrieval only.
- It does not answer the question for you — it returns raw passages; you do the
  reading and the concise answer.
