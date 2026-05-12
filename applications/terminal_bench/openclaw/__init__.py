"""applications.terminal_bench.openclaw — OpenClaw customization for Terminal-Bench.

Holds per-(TB × OpenClaw) prompts (agents/) and skills (skills/) that get
mounted into the OC container at run start. TB-flavoured glue logic itself
(harness, condition) lives in `applications/terminal_bench/app.py` and
`applications/terminal_bench/conditions/openclaw_agent.py`; shared helpers
(gateway port pool, singularity argv wrapper) live in
`applications/conditions/gateway_helpers.py`.
"""
