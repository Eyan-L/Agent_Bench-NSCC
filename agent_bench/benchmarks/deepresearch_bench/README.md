# DeepResearch-Bench application

100 PhD-level open-ended research prompts (50 EN + 50 ZH, 22 domains) from
[muset-ai/DeepResearch-Bench-Dataset](https://huggingface.co/datasets/muset-ai/DeepResearch-Bench-Dataset)
([paper](https://arxiv.org/abs/2506.11763),
[repo](https://github.com/Ayanami0730/deep_research_bench)).

The agent reads the research question from `/workspace/PROMPT.md`, searches
the web, and writes a long-form markdown report with inline citations to
`/workspace/report.md`. After the agent says `DONE`, the harness collects
the report and runs the vendored RACE + FACT evaluation pipelines against
it on the host using an OpenAI-compatible judge backend.

## Layout

```
agent_bench/benchmarks/deepresearch_bench/
  app.py                              DeepResearchHarness
  loader.py                           reads vendor query.jsonl → instance_data
  conditions/openclaw_agent.py        DeepResearchOpenClawCondition
  eval/judge_runner.py                RACE + FACT host-side runner
  openclaw/
    agents/                           per-app prompts (AGENTS.md, BOOTSTRAP.md, ...)
    skills/research_fetch/            "fetch + LLM extraction" recipe
  eval/
    judge_runner.py                   RACE + FACT host-side runner
    runner.py                         standalone CLI: re-judge persisted reports
    vendor/                           git-vendored upstream eval scripts (flat)
      COMMIT.txt                      pinned upstream sha + Apache-2.0 attribution
      PATCHES.md                      list of in-tree edits to the vendor
      data/, utils/, prompt/, deepresearch_bench_race.py
```

## Required env

| Var               | Used for                                             |
|-------------------|------------------------------------------------------|
| `BRAVE_API_KEY`   | OC native `web_search` tool (passed to OC container) |
| `JINA_READER_API` | FACT pipeline citation scraping (aliased to `JINA_API_KEY`) |
| `OPENAI_API_KEY`  | RACE + FACT judge model; optional for local OpenAI-compatible endpoints |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible judge endpoint override  |

If `LLM_BACKEND=openai` and `OPENAI_API_KEY` is unset, the judge uses `EMPTY`
as a dummy key and lets the endpoint validate the request.

vLLM endpoint at `http://localhost:8000/v1` for the agent model (mirrors
TB / SWE-bench convention).

## Images

The sandbox image is published on Docker Hub — `docker run` will pull it
on first use, no local build step:

```
docker pull hongrui0/dr-sandbox:1.0   # python:3.11-slim + bash/curl/jq/git, ~330 MB
```

The OpenClaw harness image is the same one TB / SWE-bench use:

```
bash agent_bench/harnesses/openclaw/docker/build.sh   # agent-bench/openclaw:<v>
```

## Run

```
make run_dr_smoke           # 3 tasks (dr_001 ZH + dr_051 EN + dr_075 EN)
make run_dr_full            # all 100 tasks
```

Override model/serving separately from the harness:

```
make run_dr_smoke AB_HARNESS_CONFIG=openclaw_deepresearch_bench_brave AB_MODEL_CONFIG=qwen36_35b_a3b_local_research AB_SERVING_CONFIG=vllm/nscc
```

Override `AB_DEPLOYMENT_CONTAINER_BACKEND` is **not** supported — DR-Bench is docker-only
(no singularity backend; not implemented).

## Outputs

Per task, the harness writes to
`data/experiments/<stamp>/runs/dr_<sweep>/<run_id>/`:

```
<run_id>/
  candidate.patch          (empty — DR has no patch artifact)
  verdict.json             { status, race, fact, test_pass_rate, ... }
  judge/
    report.md              the agent-written report
    score.json             { race: {...}, fact: {...}, errors }
    race/                  vendored RACE intermediate outputs
    fact/                  vendored FACT intermediate outputs
```

`status = PASS` when RACE overall score >= `AB_BENCHMARK_PASS_THRESHOLD` (default
0.5; tune via env). FACT scores are reported but don't gate the verdict.

## Evaluation details

- **RACE**: four 0–10 dimensions (Comprehensiveness, Insight,
  Instruction-Following, Readability) scored by the judge LLM against a
  reference report, with dynamic per-task criteria. Aggregated to an
  overall ratio in `[0,1]`.
- **FACT**: extract URLs from the report → Jina-scrape each → judge
  decides "does this URL support the cited claim" → citation accuracy +
  effective-citations count.

Both pipelines live under `eval/vendor/`; the
in-tree edits add OpenAI-compatible backend selection, including optional
DeepSeek support via `LLM_BACKEND=deepseek`
(see [vendor/PATCHES.md](vendor/PATCHES.md)).

## Architecture

Two-container model identical to Terminal-Bench:

```
host  ── vLLM (agent) / OpenAI-compatible judge / Brave / Jina
                                  ▲
                                  │
    ┌────────────────────────────────────┐
    │ oc-<rid>  openclaw container       │
    │ - env: BRAVE_API_KEY                │
    └───────────┬────────────────────────┘
                │ exec-bridge socket (bash + fs)
                ↓
    ┌────────────────────────────────────┐
    │ openclaw-sandbox-<sess>            │
    │ image: hongrui0/dr-sandbox:1.0     │
    │ - /workspace (rw)                  │
    └────────────────────────────────────┘
```

Web tools live in the OC container — OC reaches Brave / firecrawl
directly with the API keys we passed through. The dispatcher only
intercepts bash + filesystem operations (the same tools it intercepts
for TB and SWE-bench). No new OC patch.
