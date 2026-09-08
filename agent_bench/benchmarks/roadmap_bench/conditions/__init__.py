"""RoadmapBench condition implementations."""

# Roadmap spans multiple pinned OpenClaw runtimes while the baseline is being
# migrated. Install narrow compatibility adapters before the condition module
# is imported; the shared builder remains unchanged for other benchmarks.
from .measurement_compat import install_roadmap_measurement_compat
from .otel_compat import install_roadmap_otel_activity_compat
from .runtime_policy import install_roadmap_gateway_only_policy
from .session_compat import install_roadmap_session_compat
from .sqlite_session_compat import install_roadmap_sqlite_session_compat
from .sqlite_measurement_compat import install_roadmap_sqlite_measurement_compat

install_roadmap_measurement_compat()
install_roadmap_otel_activity_compat()
install_roadmap_gateway_only_policy()
install_roadmap_session_compat()
install_roadmap_sqlite_session_compat()
install_roadmap_sqlite_measurement_compat()
