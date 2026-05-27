"""OpenHands harness — Layer-2 shared code for the OpenHands SDK runtime.

Per-(app x harness) glue lives under agent_bench/benchmarks/<app>/openhands/.
The container image is built from agent_bench/harnesses/openhands/docker/Dockerfile and
bakes in:

  - openhands-sdk == VERSION (pinned in patch/VERSION)
  - openhands-tools == VERSION
  - The bridge-aware driver (agent_bench/harnesses/openhands/driver.py) at
    /opt/openhands_driver.py — the harness invokes this entry point inside
    the container; it reads OPENHANDS_EXEC_BRIDGE_* env to route tool
    execution into the SWE-bench sandbox over the shared exec-bridge.
"""
