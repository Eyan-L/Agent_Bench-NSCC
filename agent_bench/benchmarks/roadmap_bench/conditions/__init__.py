"""RoadmapBench condition implementations."""

# Roadmap's pinned OpenClaw SIF emits some recovery diagnostics only on the
# agent CLI stderr. Install the narrowly-scoped measurement adapter before the
# condition module is imported; the shared builder remains unchanged for every
# other benchmark/agent id.
from .measurement_compat import install_roadmap_measurement_compat

install_roadmap_measurement_compat()
