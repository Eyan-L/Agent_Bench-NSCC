"""SWE-bench application — shared helpers across harness variants.

Harness classes live next to their per-(app x harness) prompts/skills:
    * `agent_bench/benchmarks/swebench/openclaw/app.py`  → SWEBenchHarness + SingularityHarness
    * `agent_bench/benchmarks/swebench/openhands/app.py` → SWEBenchOpenHandsHarness

This file keeps only what both harnesses (plus the orchestration driver
and the Terminal-Bench OC variant) need to share: the workspace cleanup
helper, the OC-compatible sandbox-name predictor, the default OpenClaw
image tag, and the fairness checker used by the driver.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _force_rmtree(path: Path, *, ignore_errors: bool = False) -> None:
    """`shutil.rmtree` that survives root-owned files left by the OC container.

    The OC container runs as --user 0 (needed for docker.sock access) and
    writes per-session subdirs (`identity/`, `logs/`, `tasks/`, ...) into the
    bind-mounted state_dir as root. The driver runs as the host user, so a
    plain `shutil.rmtree` fails with EACCES on the next run's setup.

    On EACCES we spin up a throwaway alpine container to `chown -R` the path
    back to the host UID:GID, then retry. The container does the chown as
    root via the docker socket — no sudo required on the host.
    """
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        return
    except (PermissionError, OSError):
        pass
    # Chown back via a throwaway container running as root.
    uid = os.getuid()
    gid = os.getgid()
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{path}:/x",
             "alpine", "chown", "-R", f"{uid}:{gid}", "/x"],
            capture_output=True, check=True, timeout=60,
        )
    except Exception as e:
        if ignore_errors:
            logger.warning("chown via docker failed for %s: %s", path, e)
            return
        raise
    try:
        shutil.rmtree(path)
    except Exception:
        if not ignore_errors:
            raise


# ----------------------------------------------------------------------
# OpenClaw sandbox-name prediction.
#
# OC computes its sandbox container name as `containerPrefix + slugify(scopeKey)`
# (see openclaw/src/agents/sandbox/shared.ts::slugifySessionKey and
# docker.ts::ensureSandboxContainer). We mirror that algorithm so we can
# pre-start the task env container ourselves with the exact name OC will
# look for; OC's ensureSandboxContainer then sees the container already
# exists, hits a config-hash mismatch (no `openclaw.configHash` label), but
# its "hot container" path (running + no registry entry) keeps it and just
# logs a warning — and OC `docker exec`s into our container for tool calls.
# Net effect: we own the task env lifecycle; OC only owns its own container.
#
# The OpenHands harness reuses the same name format so trace consumers can
# pivot on `oc-sbx-*` regardless of which harness was active.
# ----------------------------------------------------------------------


def _slugify_session_key(value: str) -> str:
    trimmed = (value or "").strip() or "session"
    h = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"[^a-z0-9._-]+", "-", trimmed.lower())
    safe = re.sub(r"^-+|-+$", "", safe)
    base = safe[:32] or "session"
    return f"{base}-{h}"


def _oc_sandbox_name(prefix: str, scope_key: str) -> str:
    """Predict the sandbox container name OC will compute for a given prefix+scopeKey."""
    return (prefix + _slugify_session_key(scope_key))[:63]


# ----------------------------------------------------------------------
# Image tag for the patched OpenClaw runtime built by
# agent_bench/harnesses/openclaw/docker/build.sh. Override via env.
# ----------------------------------------------------------------------
DEFAULT_OC_IMAGE = os.environ.get("OC_IMAGE_TAG", "agent-bench/openclaw:2026.4.22")


# ----------------------------------------------------------------------
# Fairness checker (run_configs come from agent_bench.core.config)
# ----------------------------------------------------------------------


def validate_fairness_constraints(run_configs: List):
    instance_sets: Dict[str, set] = {}
    model_families: Dict[str, set] = {}
    serving_endpoints: Dict[str, set] = {}
    repo_commits_per_instance: Dict[str, set] = {}
    container_images_per_instance: Dict[str, set] = {}
    wall_budgets_per_instance: Dict[str, set] = {}

    for cfg in run_configs:
        sweep_key = cfg.condition.condition_id
        instance_sets.setdefault(sweep_key, set()).add(cfg.instance_id)
        model_families.setdefault(sweep_key, set()).add(cfg.condition.model.name)
        serving_endpoints.setdefault(sweep_key, set()).add(cfg.condition.model.endpoint)
        repo_commits_per_instance.setdefault(cfg.instance_id, set()).add(
            (cfg.repo_id, cfg.base_commit)
        )
        container_images_per_instance.setdefault(cfg.instance_id, set()).add(
            cfg.container_image
        )
        wall_budgets_per_instance.setdefault(cfg.instance_id, set()).add(
            cfg.wall_clock_budget_seconds
        )

    all_instance_sets = list(instance_sets.values())
    if len(set(frozenset(s) for s in all_instance_sets)) > 1:
        raise ValueError(f"FAIRNESS VIOLATION: instance sets: {instance_sets}")
    for k, v in model_families.items():
        if len(v) > 1:
            logger.warning("Multiple model families in %s: %s", k, v)
    for k, v in serving_endpoints.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: endpoints in {k}: {v}")
    for inst, v in repo_commits_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: commits for {inst}: {v}")
    for inst, v in container_images_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: images for {inst}: {v}")
    for inst, v in wall_budgets_per_instance.items():
        if len(v) > 1:
            raise ValueError(f"FAIRNESS VIOLATION: budgets for {inst}: {v}")
