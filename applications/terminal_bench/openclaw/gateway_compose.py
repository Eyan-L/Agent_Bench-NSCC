"""Placeholder for Terminal-Bench's OpenClaw gateway compose helpers.

When TB starts running on OpenClaw, the per-app glue analogous to
applications.swebench.openclaw.gateway_compose belongs here: TB-specific
mount construction (the .sif binds, /logs/verifier write paths) and the
gateway-start invocation tuned for singularity instances rather than docker
containers.

Today's TB condition (applications/terminal_bench/conditions/react.py)
does not use OpenClaw, so this module exposes nothing.
"""
