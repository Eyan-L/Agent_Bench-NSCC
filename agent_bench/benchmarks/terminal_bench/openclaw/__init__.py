"""agent_bench.benchmarks.terminal_bench.openclaw — OpenClaw customization for Terminal-Bench.

Holds per-(TB x OpenClaw) prompts (agents/) and skills (skills/) that get
mounted into the OC container at run start. TB-flavoured glue logic itself
(harness, condition) lives in `agent_bench/benchmarks/terminal_bench/app.py` and
`agent_bench/benchmarks/terminal_bench/conditions/openclaw_agent.py`; shared helpers
(gateway port pool, singularity argv wrapper) live in
`agent_bench/benchmarks/conditions/gateway_helpers.py`.
"""
