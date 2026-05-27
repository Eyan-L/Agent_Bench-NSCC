"""harnesses — Layer-2 of the three-layer abstraction.

Each subpackage is one agent runtime (OpenClaw, future OpenHands, ...).
Only harness-wide shared code lives here. Per-(app x harness) customization
lives under agent_bench/benchmarks/<app>/<harness>/.
"""
