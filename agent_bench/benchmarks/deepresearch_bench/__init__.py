"""DeepResearch-Bench application — open-ended long-form research benchmark.

100 PhD-level research prompts (50 EN + 50 ZH, 22 domains). Agent writes a
markdown report; host-side judge (RACE + FACT, DeepSeek-V4) scores it.

See ``app.py`` for the harness, ``loader.py`` for the query loader, and
``eval/judge_runner.py`` for the per-instance evaluator.
"""
