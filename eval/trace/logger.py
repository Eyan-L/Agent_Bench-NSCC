"""
Step-trace logger with buffering, thread safety, and crash recovery.

Design:
- In-memory ring buffer, flush every 64 events or 1 second
- Thread-safe via lock
- fsync on trajectory termination
- Partial trajectory marker on crash/interrupt
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from .schema import StepEvent, TrajectoryManifest, StepType


class StepLogger:
    """
    Thread-safe step-trace logger.
    
    Usage:
        logger = StepLogger(output_path="trace.jsonl", trajectory_id="...")
        
        # Manual start/end
        step_id = logger.start_step(step_type="LLM_INFER", ...)
        # ... do work ...
        logger.end_step(step_id, prompt_tokens=100, completion_tokens=50, ...)
        
        # Context manager (preferred)
        with logger.step(step_type="TOOL_READ", tool_name="Read") as step:
            # ... do work ...
            step.exit_code = 0
    
    Flush behavior:
    - Automatic flush every `flush_interval_s` seconds or `flush_threshold` events
    - Explicit flush via flush()
    - Final flush + fsync on close()
    """
    
    def __init__(
        self,
        output_path: str,
        trajectory_id: str,
        instance_id: str,
        repo_id: str,
        config_id: str,
        model_name: str,
        flush_threshold: int = 64,
        flush_interval_s: float = 1.0,
    ):
        self.output_path = Path(output_path)
        self.trajectory_id = trajectory_id
        self.instance_id = instance_id
        self.repo_id = repo_id
        self.config_id = config_id
        self.model_name = model_name
        
        self.flush_threshold = flush_threshold
        self.flush_interval_s = flush_interval_s
        
        # State
        self._buffer: list[StepEvent] = []
        self._lock = threading.Lock()
        self._file = None
        self._next_step_id = 1
        self._current_attempt = 1
        self._last_flush_time = time.monotonic()
        
        # Active steps (for context manager)
        self._active_steps: Dict[int, StepEvent] = {}
        
        # Manifest
        self._manifest = TrajectoryManifest(
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            repo_id=repo_id,
            config_id=config_id,
            model_name=model_name,
            start_ts_mono=time.monotonic_ns(),
            start_ts_wall=time.time_ns(),
        )
        
        # Open file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w")
        
        # Write manifest as first line (will be updated on close)
        self._write_manifest()
        
        # Start flush timer thread
        self._stop_flush_thread = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _write_manifest(self):
        """Write or update manifest at top of file."""
        # For simplicity, we append manifest updates as special lines
        # The analyzer can read the last manifest line for final state
        manifest_line = json.dumps({
            "_type": "manifest",
            **self._manifest.to_dict()
        })
        self._file.write(manifest_line + "\n")
    
    def start_step(
        self,
        step_type: str,
        parent_step_id: Optional[int] = None,
        step_subtype: Optional[str] = None,
        tool_name: Optional[str] = None,
        command: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Start a new step. Returns step_id.
        
        Caller must call end_step(step_id, ...) when done.
        """
        with self._lock:
            step_id = self._next_step_id
            self._next_step_id += 1
            
            event = StepEvent(
                trajectory_id=self.trajectory_id,
                attempt_id=self._current_attempt,
                step_id=step_id,
                parent_step_id=parent_step_id,
                step_type=step_type,
                step_subtype=step_subtype,
                tool_name=tool_name,
                command=command,
                file_path=file_path,
                start_ts_mono=time.monotonic_ns(),
                start_ts_wall=time.time_ns(),
                repo_id=self.repo_id,
                instance_id=self.instance_id,
                config_id=self.config_id,
                model_name=self.model_name,
                **kwargs,
            )
            
            self._active_steps[step_id] = event
            return step_id
    
    def end_step(
        self,
        step_id: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        ttft_ms: Optional[float] = None,
        tpot_ms: Optional[float] = None,
        token_source: str = "api",
        exit_code: Optional[int] = None,
        error_type: Optional[str] = None,
        **kwargs,
    ):
        """
        End a step started with start_step().
        
        Sets end timestamps, computes duration, updates LLM/execution fields.
        """
        with self._lock:
            if step_id not in self._active_steps:
                raise ValueError(f"Step {step_id} not found in active steps")
            
            event = self._active_steps.pop(step_id)
            event.finalize()
            
            # Update optional fields
            if prompt_tokens is not None:
                event.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                event.completion_tokens = completion_tokens
            if ttft_ms is not None:
                event.ttft_ms = ttft_ms
            if tpot_ms is not None:
                event.tpot_ms = tpot_ms
            if token_source:
                event.token_source = token_source
            if exit_code is not None:
                event.exit_code = exit_code
            if error_type is not None:
                event.error_type = error_type
            
            # Update from kwargs
            for k, v in kwargs.items():
                if hasattr(event, k):
                    setattr(event, k, v)
            
            # Buffer
            self._buffer.append(event)
            self._manifest.total_steps += 1
            
            # Flush if threshold reached
            if len(self._buffer) >= self.flush_threshold:
                self._flush_unlocked()
    
    @contextmanager
    def step(
        self,
        step_type: str,
        parent_step_id: Optional[int] = None,
        step_subtype: Optional[str] = None,
        tool_name: Optional[str] = None,
        command: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """
        Context manager for step execution.
        
        Usage:
            with logger.step(step_type="TOOL_READ", tool_name="Read") as step:
                # ... do work ...
                step.exit_code = 0
        
        Step is automatically finalized on context exit.
        Caller can set fields on the returned StepEvent object.
        """
        step_id = self.start_step(
            step_type=step_type,
            parent_step_id=parent_step_id,
            step_subtype=step_subtype,
            tool_name=tool_name,
            command=command,
            file_path=file_path,
        )
        
        # Yield the event so caller can set fields
        event = self._active_steps[step_id]
        
        try:
            yield event
        finally:
            # End step (pulls from active_steps, finalizes, buffers)
            self.end_step(step_id)
    
    def new_attempt(self):
        """Mark start of new attempt within the trajectory."""
        with self._lock:
            self._current_attempt += 1
            self._manifest.total_attempts = self._current_attempt
    
    def _flush_unlocked(self):
        """Flush buffer to disk (caller must hold lock)."""
        if not self._buffer:
            return
        
        for event in self._buffer:
            line = json.dumps(event.to_dict())
            self._file.write(line + "\n")
        
        self._buffer.clear()
        self._last_flush_time = time.monotonic()
    
    def flush(self):
        """Explicit flush of buffered events."""
        with self._lock:
            self._flush_unlocked()
    
    def _flush_loop(self):
        """Background thread for time-based flushing."""
        while not self._stop_flush_thread.wait(timeout=self.flush_interval_s):
            with self._lock:
                if time.monotonic() - self._last_flush_time >= self.flush_interval_s:
                    self._flush_unlocked()
    
    def close(self, partial: bool = False, error_message: Optional[str] = None):
        """
        Close logger, flush remaining events, fsync, write final manifest.
        
        Args:
            partial: If True, marks trajectory as incomplete
            error_message: Optional error message if trajectory failed
        """
        # Stop flush thread
        self._stop_flush_thread.set()
        self._flush_thread.join(timeout=2.0)
        
        with self._lock:
            # Flush remaining events
            self._flush_unlocked()
            
            # Update manifest
            self._manifest.end_ts_mono = time.monotonic_ns()
            self._manifest.end_ts_wall = time.time_ns()
            self._manifest.partial = partial
            self._manifest.error_message = error_message
            
            # Write final manifest
            self._write_manifest()
            
            # fsync
            self._file.flush()
            import os
            os.fsync(self._file.fileno())
            
            # Close
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Mark as partial if exception occurred
        partial = exc_type is not None
        error_msg = str(exc_val) if exc_val else None
        self.close(partial=partial, error_message=error_msg)
