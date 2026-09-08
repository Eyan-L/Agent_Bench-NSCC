"""Container runtime abstraction shared by application and harness layers.

Selected at driver time by AB_DEPLOYMENT_CONTAINER_BACKEND; injected into harness instances
so neither application logic nor harness logic carries `docker`/`singularity`
literals.
"""
from agent_bench.deployments.containers.base import (
    Container,
    ContainerHandle,
    ContainerSpec,
    Mount,
)
from agent_bench.deployments.containers.docker import DockerContainer
from agent_bench.deployments.containers.singularity import SingularityContainer
from agent_bench.deployments.containers.singularity_liveness import (
    install_singularity_main_process_liveness,
)

install_singularity_main_process_liveness()

__all__ = [
    "Container",
    "ContainerHandle",
    "ContainerSpec",
    "Mount",
    "DockerContainer",
    "SingularityContainer",
]
