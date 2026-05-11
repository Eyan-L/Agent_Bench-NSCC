"""
Resource telemetry collector sidecar.

Collects GPU (NVML), CPU (psutil), memory metrics at regular intervals.
Runs as background thread or separate process.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .schema import TelemetrySample


class ResourceCollector:
    """
    Background resource telemetry collector.
    
    Usage:
        collector = ResourceCollector(
            trajectory_id="...",
            sample_callback=lambda sample: writer.write(sample),
            poll_interval_ms=200,
        )
        collector.start()
        # ... do work ...
        collector.stop()
    """
    
    def __init__(
        self,
        trajectory_id: str,
        sample_callback: Callable[[TelemetrySample], None],
        poll_interval_ms: int = 200,
        gpu_index: int = 0,
    ):
        """
        Args:
            trajectory_id: ID to tag all samples
            sample_callback: Function called with each TelemetrySample
            poll_interval_ms: Sampling interval (100-500 ms recommended)
            gpu_index: GPU device index (0 for single-GPU)
        """
        self.trajectory_id = trajectory_id
        self.sample_callback = sample_callback
        self.poll_interval_s = poll_interval_ms / 1000.0
        self.gpu_index = gpu_index
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Initialize NVML if available
        self._nvml_handle = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                print(f"Warning: NVML initialization failed: {e}")
        
        # Process handle for CPU tracking
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def start(self):
        """Start background collection thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
    
    def stop(self, timeout: float = 5.0):
        """Stop collection thread."""
        if not self._thread:
            return
        
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        
        # Cleanup NVML
        if NVML_AVAILABLE and self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _collection_loop(self):
        """Main collection loop (runs in background thread)."""
        while not self._stop_event.wait(timeout=self.poll_interval_s):
            sample = self._collect_sample()
            if sample:
                self.sample_callback(sample)
    
    def _collect_sample(self) -> Optional[TelemetrySample]:
        """Collect one telemetry sample."""
        timestamp_mono = time.monotonic_ns()
        timestamp_wall = time.time_ns()
        
        # GPU metrics
        gpu_util = None
        sm_occ = None
        hbm_bw = None
        gpu_power = None
        gpu_mem = None
        
        if self._nvml_handle:
            try:
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_mem = mem_info.used / (1024 ** 2)  # MB
                
                # Power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                gpu_power = power_mw / 1000.0  # W
                
                # SM occupancy (may not be available on all GPUs)
                try:
                    sm_occ = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).memory
                except:
                    pass
                
            except Exception as e:
                # Silently skip on transient errors
                pass
        
        # CPU metrics
        cpu_util = 0.0
        subprocess_cpu = 0.0
        rss_mb = 0.0
        
        if self._process:
            try:
                cpu_util = self._process.cpu_percent(interval=None)
                rss_mb = self._process.memory_info().rss / (1024 ** 2)
                
                # Subprocess CPU time (sum of all children)
                try:
                    children = self._process.children(recursive=True)
                    subprocess_cpu = sum(
                        c.cpu_times().user + c.cpu_times().system
                        for c in children
                    )
                except:
                    pass
                
            except:
                pass
        
        return TelemetrySample(
            timestamp_mono=timestamp_mono,
            timestamp_wall=timestamp_wall,
            gpu_utilization_pct=gpu_util,
            sm_occupancy_pct=sm_occ,
            hbm_bandwidth_gb_s=hbm_bw,
            gpu_power_w=gpu_power,
            gpu_memory_used_mb=gpu_mem,
            cpu_utilization_pct=cpu_util,
            subprocess_cpu_time_s=subprocess_cpu,
            rss_mb=rss_mb,
            trajectory_id=self.trajectory_id,
        )
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
