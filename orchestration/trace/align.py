"""
Timestamp alignment utility.

Merges step events with telemetry samples on monotonic clock.
Computes per-step and per-trajectory resource roll-ups.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import bisect

from .schema import StepEvent, TelemetrySample


@dataclass
class StepResourceProfile:
    """Resource roll-up for one step."""
    
    step_id: int
    step_type: str
    duration_ms: float
    
    # GPU metrics
    gpu_utilization_mean: Optional[float] = None
    gpu_utilization_p95: Optional[float] = None
    gpu_idle_fraction: Optional[float] = None
    gpu_memory_mean_mb: Optional[float] = None
    
    # CPU metrics
    cpu_utilization_mean: Optional[float] = None
    subprocess_cpu_time_s: Optional[float] = None
    rss_mean_mb: Optional[float] = None
    
    # Sample count
    sample_count: int = 0


def align_telemetry_to_steps(
    step_events: List[StepEvent],
    telemetry_samples: List[TelemetrySample],
) -> List[StepResourceProfile]:
    """
    Align telemetry samples to step events on monotonic clock.
    
    For each step, finds all telemetry samples whose timestamps fall within
    the step's [start_ts_mono, end_ts_mono] window and computes resource
    roll-ups.
    
    Args:
        step_events: List of step events (must be sorted by start_ts_mono)
        telemetry_samples: List of telemetry samples (must be sorted by timestamp_mono)
    
    Returns:
        List of StepResourceProfile, one per step
    """
    # Sort inputs (defensive)
    step_events = sorted(step_events, key=lambda e: e.start_ts_mono)
    telemetry_samples = sorted(telemetry_samples, key=lambda s: s.timestamp_mono)
    
    # Build timestamp index for binary search
    telemetry_timestamps = [s.timestamp_mono for s in telemetry_samples]
    
    profiles = []
    
    for event in step_events:
        # Find samples in [start_ts_mono, end_ts_mono]
        start_idx = bisect.bisect_left(telemetry_timestamps, event.start_ts_mono)
        end_idx = bisect.bisect_right(telemetry_timestamps, event.end_ts_mono)
        
        samples = telemetry_samples[start_idx:end_idx]
        
        if not samples:
            # No telemetry coverage
            profiles.append(StepResourceProfile(
                step_id=event.step_id,
                step_type=event.step_type,
                duration_ms=event.duration_ms,
                sample_count=0,
            ))
            continue
        
        # Compute roll-ups
        gpu_utils = [s.gpu_utilization_pct for s in samples if s.gpu_utilization_pct is not None]
        gpu_mems = [s.gpu_memory_used_mb for s in samples if s.gpu_memory_used_mb is not None]
        cpu_utils = [s.cpu_utilization_pct for s in samples if s.cpu_utilization_pct is not None]
        rss_vals = [s.rss_mb for s in samples if s.rss_mb is not None]
        
        # GPU
        gpu_util_mean = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
        gpu_util_p95 = _percentile(gpu_utils, 0.95) if gpu_utils else None
        gpu_idle_frac = sum(1 for u in gpu_utils if u < 10.0) / len(gpu_utils) if gpu_utils else None
        gpu_mem_mean = sum(gpu_mems) / len(gpu_mems) if gpu_mems else None
        
        # CPU
        cpu_util_mean = sum(cpu_utils) / len(cpu_utils) if cpu_utils else None
        rss_mean = sum(rss_vals) / len(rss_vals) if rss_vals else None
        
        # Subprocess CPU time (take max, as it's cumulative)
        subprocess_cpu = max(
            (s.subprocess_cpu_time_s for s in samples if s.subprocess_cpu_time_s is not None),
            default=None,
        )
        
        profiles.append(StepResourceProfile(
            step_id=event.step_id,
            step_type=event.step_type,
            duration_ms=event.duration_ms,
            gpu_utilization_mean=gpu_util_mean,
            gpu_utilization_p95=gpu_util_p95,
            gpu_idle_fraction=gpu_idle_frac,
            gpu_memory_mean_mb=gpu_mem_mean,
            cpu_utilization_mean=cpu_util_mean,
            subprocess_cpu_time_s=subprocess_cpu,
            rss_mean_mb=rss_mean,
            sample_count=len(samples),
        ))
    
    return profiles


def _percentile(values: List[float], p: float) -> float:
    """Compute percentile (simple implementation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_trajectory_resource_summary(
    profiles: List[StepResourceProfile],
) -> Dict[str, float]:
    """
    Compute trajectory-level resource summary from step profiles.
    
    Returns dict with aggregated metrics:
    - total_duration_ms
    - gpu_utilization_mean
    - gpu_idle_fraction
    - cpu_utilization_mean
    - etc.
    """
    if not profiles:
        return {}
    
    total_duration = sum(p.duration_ms for p in profiles)
    
    # Weighted means by duration
    def weighted_mean(values_and_weights: List[Tuple[float, float]]) -> Optional[float]:
        """Compute weighted mean from (value, weight) pairs."""
        valid = [(v, w) for v, w in values_and_weights if v is not None]
        if not valid:
            return None
        total_weight = sum(w for _, w in valid)
        if total_weight == 0:
            return None
        return sum(v * w for v, w in valid) / total_weight
    
    gpu_util_mean = weighted_mean([
        (p.gpu_utilization_mean, p.duration_ms)
        for p in profiles
    ])
    
    gpu_idle_frac = weighted_mean([
        (p.gpu_idle_fraction, p.duration_ms)
        for p in profiles
    ])
    
    cpu_util_mean = weighted_mean([
        (p.cpu_utilization_mean, p.duration_ms)
        for p in profiles
    ])
    
    # Max RSS
    rss_max = max(
        (p.rss_mean_mb for p in profiles if p.rss_mean_mb is not None),
        default=None,
    )
    
    return {
        "total_duration_ms": total_duration,
        "gpu_utilization_mean": gpu_util_mean,
        "gpu_idle_fraction": gpu_idle_frac,
        "cpu_utilization_mean": cpu_util_mean,
        "rss_max_mb": rss_max,
    }
