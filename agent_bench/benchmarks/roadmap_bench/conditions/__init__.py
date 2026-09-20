"""RoadmapBench condition implementations and runtime compatibility."""

# These adapters are idempotent and narrowly scoped to Roadmap/OpenClaw data.
# They accept both legacy JSON sessions/spans and v2026.9.2 SQLite/native OTel
# without changing the shared schemas used by other benchmarks.
from .otel_compat import install_roadmap_otel_activity_compat
from .sqlite_measurement_compat import install_roadmap_sqlite_measurement_compat
from .sqlite_session_compat import install_roadmap_sqlite_session_compat

install_roadmap_otel_activity_compat()
install_roadmap_sqlite_session_compat()
install_roadmap_sqlite_measurement_compat()
