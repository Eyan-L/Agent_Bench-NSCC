"""
Raw trace schema for OpenClaw + SWE-bench instrumentation.

All events carry the full required schema. Step taxonomy is enforced at runtime.
Schema version is bumped on breaking changes.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time

SCHEMA_VERSION = "1.0.0"


class StepType(Enum):
    """Top-level step taxonomy. Must be one of these five categories."""
    
    LLM_INFER = "LLM_INFER"
    TOOL_READ = "TOOL_READ"
    TOOL_WRITE = "TOOL_WRITE"
    TOOL_EXEC = "TOOL_EXEC"
    WAIT_STALL = "WAIT_STALL"
    
    @classmethod
    def validate(cls, step_type: str) -> "StepType":
        """Validate and return StepType enum. Raises ValueError on unknown type."""
        try:
            return cls(step_type)
        except ValueError:
            raise ValueError(
                f"Unknown step_type: {step_type}. Must be one of: "
                f"{', '.join(t.value for t in cls)}"
            )


@dataclass
class StepEvent:
    """
    One atomic step in a trajectory.
    
    Required for all steps:
    - trajectory_id, attempt_id, step_id, parent_step_id
    - step_type, timestamps, duration
    - repo_id, instance_id, config_id, model_name, schema_version
    
    Conditionally required:
    - LLM_INFER: prompt_tokens, completion_tokens, ttft_ms, tpot_ms
    - TOOL_*: tool_name
    - TOOL_EXEC: command, exit_code
    - TOOL_READ/WRITE: file_path (if applicable)
    """
    
    # Identity
    trajectory_id: str
    attempt_id: int
    step_id: int
    parent_step_id: Optional[int]
    
    # Type classification
    step_type: str  # validated to StepType enum
    step_subtype: Optional[str] = None
    
    # Tool context
    tool_name: Optional[str] = None
    command: Optional[str] = None
    file_path: Optional[str] = None
    
    # Timestamps (nanoseconds)
    start_ts_mono: int = 0  # CLOCK_MONOTONIC_RAW
    end_ts_mono: int = 0
    start_ts_wall: int = 0  # wall-clock time.time_ns()
    end_ts_wall: int = 0
    duration_ms: float = 0.0
    
    # LLM-specific metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    token_source: str = "api"  # "api" or "tokenizer"
    
    # Execution status
    exit_code: Optional[int] = None
    error_type: Optional[str] = None
    
    # Context
    repo_id: str = ""
    instance_id: str = ""
    config_id: str = ""
    model_name: str = ""
    
    # Versioning
    schema_version: str = SCHEMA_VERSION
    
    def __post_init__(self):
        """Validate step_type on construction."""
        StepType.validate(self.step_type)
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    def finalize(self, end_ts_mono: Optional[int] = None, end_ts_wall: Optional[int] = None):
        """
        Finalize step by setting end timestamps and computing duration.
        
        If end timestamps not provided, uses current time.
        Duration is always computed from monotonic clock.
        """
        if end_ts_mono is None:
            end_ts_mono = time.monotonic_ns()
        if end_ts_wall is None:
            end_ts_wall = time.time_ns()
        
        self.end_ts_mono = end_ts_mono
        self.end_ts_wall = end_ts_wall
        self.duration_ms = (self.end_ts_mono - self.start_ts_mono) / 1_000_000.0


@dataclass
class TelemetrySample:
    """
    Resource telemetry sample collected at regular intervals.
    
    All samples are timestamped with CLOCK_MONOTONIC_RAW for alignment
    with step events.
    """
    
    # Timestamp (nanoseconds, monotonic)
    timestamp_mono: int
    timestamp_wall: int
    
    # GPU metrics (null if no GPU or NVML unavailable)
    gpu_utilization_pct: Optional[float] = None
    sm_occupancy_pct: Optional[float] = None
    hbm_bandwidth_gb_s: Optional[float] = None
    gpu_power_w: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    
    # CPU metrics
    cpu_utilization_pct: float = 0.0
    subprocess_cpu_time_s: float = 0.0
    
    # Memory
    rss_mb: float = 0.0
    
    # I/O (optional, may be omitted if not available)
    pcio_read_mb_s: Optional[float] = None
    pcio_write_mb_s: Optional[float] = None
    
    # Context
    trajectory_id: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dict for CSV/Parquet serialization."""
        return asdict(self)


@dataclass
class TrajectoryManifest:
    """
    Metadata for one trajectory execution.
    
    Written at trajectory start, updated at completion.
    Used for recovery and partial-trajectory marking.
    """
    
    trajectory_id: str
    instance_id: str
    repo_id: str
    config_id: str
    model_name: str
    
    start_ts_mono: int
    start_ts_wall: int
    end_ts_mono: Optional[int] = None
    end_ts_wall: Optional[int] = None
    
    total_attempts: int = 0
    total_steps: int = 0
    
    partial: bool = False  # True if trajectory did not complete normally
    error_message: Optional[str] = None
    
    schema_version: str = SCHEMA_VERSION
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
