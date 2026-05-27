---
name: research_fetch
description: Read a web URL and return a structured evidence block (verbatim quotes + synthesis) keyed to a specific research goal. Use whenever you open a non-trivial source so raw page text does not flood your context. The output is the citation pool for your final report.
---

# research_fetch

Pattern adapted from Alibaba-NLP/DeepResearch `inference/tool_visit.py`.
There is no new external tool: this skill composes two existing capabilities
— OpenClaw's `web_fetch` (firecrawl-backed) and one focused LLM call — into
a fetch+extract recipe.

The point: raw firecrawl markdown for a single page is routinely 20-50KB.
Letting 5-10 of those into your main context drowns out your own reasoning
and burns the token budget you need for synthesis. This skill keeps the
main loop lean by turning each visit into a small, citeable evidence block.

## When to call

- Every time you open a URL from `web_search` that looks substantive
  (paper, doc, article, blog post, report).
- Skip for trivial pages: 404s, login walls, results where the search
  snippet is already enough.
- Skip if you only need the title/URL — leave those in the `web_search`
  result line and cite from there.

## How to invoke

Two steps, executed in order inside this skill.

### Step 1 — fetch

Call the existing `web_fetch` tool on `url`. Take the markdown body. If it
exceeds ~90K tokens, truncate from the end; do not summarize-before-extract.

### Step 2 — extract against goal

Issue one LLM call with the template below. `goal` is a one-sentence
statement of what this page is supposed to contribute to the report
(e.g. "establish current capacity of US offshore wind").

    You are reading a web source to support a research goal.

    Research goal:
    {goal}

    Source content:
    {content}

    Do three passes:

    1. Content Scanning — skim the source once and identify the 3-7
       passages most likely to bear on the goal.
    2. Key Extraction — for each passage, lift the verbatim sentence(s)
       and record the exact wording. These become citations.
    3. Summary — synthesize the passages into a coherent answer to the
       goal in your own words. State explicitly any gaps the source
       does NOT cover.

    Return JSON only, no commentary outside the object:
    {
      "rational": "<why these passages were selected>",
      "evidence": ["<verbatim quote 1>", "<verbatim quote 2>", ...],
      "summary":  "<2-4 paragraph synthesis answering the goal>"
    }

## Using the output

- `evidence[]` items are your citations — when writing the final report,
  these are the exact phrases attributable to `url`. Each goes in as
  `[short label](url)` with the wording from the report body grounded in
  one of these quotes.
- `summary` is what you fold into your running research notes for the
  relevant sub-question. The raw page markdown does NOT.
- If `evidence` is empty or `summary` says the source did not address
  the goal, drop `url` from the citation pool. Do not fabricate citations.

## What this skill does NOT do

- It does not loop. One URL, one extraction. If the page links to another
  page you need, call `research_fetch` again with the new URL.
- It does not score completeness across the whole research task. That
  judgment stays in your main loop — when you have enough evidence
  across sub-questions, move on to writing.
- It does not replace `web_search`. Use `web_search` to find candidate
  URLs first; only then `research_fetch` the promising ones.
