from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_harness_workspace_base(workspace_base: Optional[Path] = None) -> Path:
    if workspace_base is not None:
        return Path(workspace_base).expanduser()

    explicit = os.environ.get("AB_HARNESS_WORKSPACE_BASE")
    if explicit:
        return Path(explicit).expanduser()

    run_dir = (
        os.environ.get("AGENT_BENCH_RUN_DIR")
        or os.environ.get("AB_RUN_DIR")
        or os.environ.get("AB_EXPERIMENT_DIR")
    )
    if run_dir:
        return Path(run_dir).expanduser() / "workspaces"

    return Path("~/.cache/oc-eval/workspaces").expanduser()
