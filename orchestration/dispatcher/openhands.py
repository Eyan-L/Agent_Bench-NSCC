"""OpenHands dispatcher — placeholder.

OpenHands has its own runtime container with an action-server endpoint (no
shared OC bridge protocol). When this lands, `start()` will spawn / health-
check that endpoint; `agent_container_args()` will inject the OpenHands
runtime image's bind-mounts + endpoint URL env var.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from orchestration.dispatcher.base import Dispatcher


class OpenHandsDispatcher(Dispatcher):
    backend_id = "openhands"

    def start(self, *, state_dir: Path, backend: str, target: str) -> None:
        raise NotImplementedError("OpenHandsDispatcher not implemented yet")

    def stop(self) -> None:
        raise NotImplementedError("OpenHandsDispatcher not implemented yet")

    @classmethod
    def stop_external(cls, **state: Any) -> None:
        raise NotImplementedError("OpenHandsDispatcher not implemented yet")

    def agent_container_args(self) -> List[str]:
        raise NotImplementedError("OpenHandsDispatcher not implemented yet")

    def env_info(self) -> Dict[str, Any]:
        raise NotImplementedError("OpenHandsDispatcher not implemented yet")
