"""Dispatcher ABC — harness-agnostic contract for routing agent tool calls.

A *dispatcher* owns the per-run state that lets an agent runtime (OpenClaw,
OpenHands, …) reach an arbitrary execution backend (docker exec, singularity
exec, …) without each harness reinventing the spawn/wait/cleanup glue.

Lifecycle, driven by the harness:

    d = SomeDispatcher(...)
    d.start(state_dir=..., backend="docker", target=container_name)
    # harness reads d.agent_mounts_and_env() (or legacy agent_container_args)
    # to wire the agent runtime container.
    # ... agent runs ...
    d.stop()

For cross-process cleanup (when only the serialized env_info survives — e.g.
a driver crash recovery or split prepare_environment / cleanup callers), each
impl ships a `stop_external(**state)` classmethod that mirrors stop() but
operates on the dict returned by `env_info()`.

ExecRequest/ExecResult shapes deliberately live OUTSIDE this ABC — they're
wire-protocol details of the specific runtime (OpenClaw's HMAC+JSONL frames,
OpenHands' action/observation, …) and don't generalize.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class Dispatcher(ABC):
    """Per-run owner of the bridge between an agent runtime and a backend."""

    @abstractmethod
    def start(self, *, state_dir: Path, backend: str, target: str) -> None:
        """Bring up whatever per-run state the dispatcher needs (subprocess,
        socket, token, …). Must leave the dispatcher ready to be reached
        from inside the agent container started with `agent_container_args()`.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear down the per-run state created by `start()`. Idempotent."""

    @classmethod
    @abstractmethod
    def stop_external(cls, **state: Any) -> None:
        """Cross-process variant of `stop()` operating on serialized state
        (as returned by `env_info()`). Used by harness cleanup paths that
        only have access to the env_info dict, not the live instance."""

    def agent_container_args(self) -> List[str]:
        """Legacy `docker run` extras (volume mounts + env vars) the harness
        used to splice into the agent runtime container. Returned as a flat
        list, e.g. ["-v", "src:dst:ro", "-e", "FOO=bar", ...].

        Superseded by `agent_mounts_and_env()` which is runtime-agnostic
        (returns Mount + dict, harness translates). Kept on the ABC as a
        non-abstract default to avoid forcing every dispatcher to implement
        the docker-specific shape. New dispatchers should implement
        `agent_mounts_and_env` instead."""
        return []

    @abstractmethod
    def env_info(self) -> Dict[str, Any]:
        """Serializable per-run state for the harness to merge into the
        env_info dict it returns from prepare_environment. Consumed later
        by cleanup paths via `stop_external(**env_info_subset)`."""
