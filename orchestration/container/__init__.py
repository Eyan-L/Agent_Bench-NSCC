"""Container runtime abstraction shared by application and harness layers.

Selected at driver time by HARNESS_BACKEND; injected into harness instances
so neither application logic nor harness logic carries `docker`/`singularity`
literals.
"""
from orchestration.container.base import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)
from orchestration.container.docker import DockerContainer
from orchestration.container.singularity import SingularityContainer

__all__ = [
    "Container",
    "ContainerHandle",
    "ContainerSpec",
    "Mount",
    "DockerContainer",
    "SingularityContainer",
]
