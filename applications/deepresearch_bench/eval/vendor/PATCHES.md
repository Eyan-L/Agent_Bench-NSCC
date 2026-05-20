# In-tree edits applied to the vendored upstream

Upstream pin: see [COMMIT.txt](COMMIT.txt).

We keep these edits minimal and surgical — the goal is to ride upstream
forward without merge pain.

## `utils/api.py`

Two related changes, both adding a DeepSeek backend so the harness can use
`DEEPSEEK_API_KEY` directly without monkey-patching at runtime.

### 1. Add `deepseek` to `_BACKEND_DEFAULTS`

```python
"deepseek": {
    "base_url": "https://api.deepseek.com",
    "key_env":  "DEEPSEEK_API_KEY",
    "race":     "deepseek-v4-pro",
    "fact":     "deepseek-v4-pro",
},
```

Also extended `_BASE_URL` resolution to read `DEEPSEEK_BASE_URL` when
`LLM_BACKEND=deepseek`.

### 2. Gate `reasoning_effort` by backend

DeepSeek's OpenAI-compatible API rejects the `reasoning_effort` field
(it's a GPT-5.x reasoning-model parameter). Wrapped its insertion into
the chat-completions payload behind:

```python
if LLM_BACKEND in ("openrouter", "openai"):
    payload["reasoning_effort"] = stage_cfg["reasoning_effort"]
```

## Why edit the vendor instead of monkey-patching?

The upstream `_BACKEND_DEFAULTS` table and the doc-comment at the top of
`api.py` explicitly invite adding new backends — that's the supported
extension surface. A monkey-patch would either need to reach into module
internals (fragile) or shadow the whole module (heavy-handed). A 10-line
in-place edit, recorded here, is honest and easy to re-apply on upstream
refresh.

## How to verify after editing

```
cd applications/deepresearch_bench/eval/vendor
LLM_BACKEND=deepseek DEEPSEEK_API_KEY=... python -c \
  "from utils.api import AIClient, Model, FACT_Model; \
   print('models:', Model, FACT_Model); \
   print('ok:', AIClient().generate('say hi', stage='fact')[:60])"
```

## Pruned files (not needed for our eval driver)

After a fresh `git clone`, delete these to slim 173 MB → 16 MB. None are
imported by RACE or the FACT pipeline that `judge_runner.py` invokes;
they're upstream-author tooling, sample submissions, or README assets.

```
cd applications/deepresearch_bench/eval/vendor
rm -rf results pics
rm -f run_benchmark.sh requirements.txt setup.py
rm -f README.md LICENSE .gitignore
rm -f data/test_data/raw_data/claude-3-7-sonnet-latest.jsonl
rm -f data/test_data/cleaned_data/claude-3-7-sonnet-latest.jsonl
rm -f utils/generate_criteria.py
rm -f prompt/criteria_prompt_en.py prompt/criteria_prompt_zh.py
find . -name __pycache__ -type d -exec rm -rf {} +
```

If you plan to redistribute this repo publicly, *do not delete LICENSE* —
Apache-2.0 §4 requires retaining the license text. See [COMMIT.txt](COMMIT.txt).

Why each one:
- `results/` (149 MB) — upstream's own baseline runs
- `pics/` (3.9 MB) — README screenshots
- `run_benchmark.sh` / `requirements.txt` / `setup.py` — upstream's
  install + wrapper; our driver invokes the entry points directly
- `claude-3-7-sonnet-latest.jsonl` (×2) — sample target submission
- `generate_criteria.py` + `criteria_prompt_{en,zh}.py` — re-generates
  `data/criteria_data/criteria.jsonl`; upstream already ships it and
  our eval just reads it
