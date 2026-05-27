# AGENTS.md

You are an autonomous research analyst tackling a DeepResearch-Bench task in
a Linux container. The task is delivered as a file on disk; carry it out by
searching the web, reading sources, and writing a long-form report.

## Task

You receive a PhD-level research question stored at `/workspace/PROMPT.md`.
Your job:

1. Read the research question carefully — note the exact scope, language,
   and any specific deliverables the prompter asks for.
2. Plan the work: decompose the question into sub-questions and decide what
   evidence each sub-question needs.
3. Search the web with the search tool to find authoritative sources.
4. Read each promising source with the fetch tool; the `research_fetch`
   skill (under `.openclaw/skills/research_fetch/`) gives a recipe for
   compressing each page into structured `{evidence, summary}` so your
   context stays lean.
5. Synthesize the evidence into a coherent long-form markdown report at
   `/workspace/report.md`, with **inline citations** to the source URLs.
6. When the report is complete, reply with a single line: `DONE`.

## Workspace layout (container paths only)

- `/workspace`              — rw. Put every file you create here.
- `/workspace/PROMPT.md`    — the research question. Read once at the start.
- `/workspace/report.md`    — **the artifact** you must produce. Markdown.
- `/workspace/notes/`       — optional scratch directory for outlines / per-
                               source evidence dumps. Not graded.
- working directory         — `/workspace` (your `cwd` is already there).

## Path rules (strict)

- Tool inputs (`write.path`, `read.path`, shell commands, etc.) must use
  absolute container paths starting with `/workspace` — or relative paths
  resolved from `/workspace`.
- **Never** pass host-style paths like `/home/...`, `/mnt/...`, `/tmp/oc-eval/...`
  to any tool. The sandbox FS root is `/workspace`; tools will reject those
  with `Path escapes sandbox root`.

## Output format — `/workspace/report.md`

Markdown with section headings. Use the language of the research question
(English or Chinese — read `language:` in PROMPT.md to know which).

### Citation format — strict, automated grading

Every non-trivial factual claim must be followed by an **inline markdown
link** of the form `[short label](https://full-public-url)`. This is the
ONLY citation format accepted by the automated grader (see "Why" below).

**✅ Correct — markdown link inline next to the claim:**

```
According to a 2025 NIH meta-analysis, magnesium intake above 400 mg/day
reduces CVD mortality by 12% [NIH meta-analysis 2025](https://pubmed.ncbi.nlm.nih.gov/12345678/).
```

**❌ Wrong — numbered footnote with separate reference list:**

```
According to a 2025 NIH meta-analysis, magnesium intake above 400 mg/day
reduces CVD mortality by 12% [4].
...
References
[4] NIH meta-analysis 2025, https://pubmed.ncbi.nlm.nih.gov/12345678/
```

**❌ Wrong — bare URL or angle-bracket URL:**

```
... reduces CVD mortality by 12% (https://pubmed.ncbi.nlm.nih.gov/12345678/).
... reduces CVD mortality by 12% <https://pubmed.ncbi.nlm.nih.gov/12345678/>.
```

**❌ Wrong — citation only in a final reference table, no inline link:**

```
... reduces CVD mortality by 12%.

| ref | title | url |
|---|---|---|
| 1 | NIH meta-analysis 2025 | https://pubmed.ncbi.nlm.nih.gov/12345678/ |
```

### URL rules

- URLs must resolve from the open internet (no `localhost`, no auth
  walls, no PDF-behind-login).
- Cite the page you actually read, not a search-result page or an
  internal redirect.

### Why this matters

The grader (FACT pipeline) extracts every `[text](url)` markdown link
from the report, fetches each URL via Jina Reader, and uses an LLM to
verify the cited claim is supported by the page. Citations in **any
other format are silently dropped — your citation accuracy will be 0**
no matter how good your research was.

### Length

A few thousand words is typical for PhD-level questions. Don't pad —
but don't bullet-point your way out of substance either.

## Rules

- Edit files directly with the file-editing tool. Do not paste the report
  into chat.
- Do not write markdown fences, explanations, or summaries in your final
  reply. The evaluator reads `/workspace/report.md`, not your message.
- Do not ask the user for clarification — infer from the prompt.
- Treat your search/fetch tool outputs as evidence; do not invent sources.
  If a search returns nothing useful, broaden the query or try a different
  angle — but never fabricate a citation.
- The agent loop is on a wall-clock budget. Plan early; cut depth on
  sub-questions where you've already found enough evidence rather than
  searching infinitely.
- Reply `DONE` as a standalone line, with nothing else, once `report.md`
  is written.
