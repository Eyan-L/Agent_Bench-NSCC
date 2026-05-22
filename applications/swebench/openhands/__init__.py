"""applications.swebench.openhands — OpenHands customization for SWE-bench.

Per-(application x harness) glue for running the OpenHands SDK against
SWE-bench tasks. Mirrors `applications.swebench.openclaw`:

  - `skills/verify/SKILL.md` — bridge-aware pytest validation playbook.
    The skill text is loaded into the agent's context via the SDK skill
    loader (see harnesses/openhands/driver.py); the actual `pytest` exec
    rides the shared exec-bridge into the SWE-bench sandbox container.

  - `compose.py` — host-side helpers that render a per-run config.toml
    from config/harness/openhands_swebench.json plus config/model/*.json and compute the docker
    mounts the harness needs.

The dispatcher that brokers tool calls between the OH harness container and
the sandbox lives at `orchestration/dispatcher/openhands.py` — it is shared
across applications, not specific to SWE-bench.
"""
