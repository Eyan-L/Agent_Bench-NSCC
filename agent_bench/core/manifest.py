"""Per-run manifest handling."""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class RunManifest:
    """Manifest for a single run."""
    run_id: str
    condition_id: str
    instance_id: str
    seed: int
    serving_config_id: str
    
    status: str  # PENDING, RUNNING, PASS, FAIL, TIMEOUT, ERROR
    
    step_trace_path: Optional[str] = None
    telemetry_path: Optional[str] = None
    verdict_path: Optional[str] = None
    
    wall_start_ns: Optional[int] = None
    wall_end_ns: Optional[int] = None
    duration_seconds: Optional[float] = None
    # Cleanup-inclusive wall time; duration_seconds keeps its historical pre-cleanup boundary.
    total_instance_seconds: Optional[float] = None
    # Logical agent wall time. For RoadmapBench this starts at the
    # initial parent dispatch and ends only after the lifecycle hook
    # has quiesced all agent writers. Setup/freeze/verifier are excluded.
    agent_execution_seconds: Optional[float] = None
    phase_timing: Optional[dict[str, Any]] = None
    
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    agent_runtime_mode: Optional[str] = None
    agent_execution_outcome: Optional[str] = None
    agent_completion_error: Optional[str] = None
    agent_lifecycle: Optional[dict[str, Any]] = None
    benchmark_valid: Optional[bool] = None
    benchmark_invalid_reason: Optional[str] = None
    verifier_eligible: Optional[bool] = None
    verifier_status: Optional[str] = None
    reward: Optional[float] = None
    test_pass_rate: Optional[float] = None
    trajectory_path: Optional[str] = None
    token_accounting_path: Optional[str] = None
    trajectory_tokens: Optional[dict[str, Any]] = None
    serving_tokens: Optional[dict[str, Any]] = None
    runtime_observability_path: Optional[str] = None
    runtime_observability: Optional[dict[str, Any]] = None
    notes: str = ""
    
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "serving_config_id": self.serving_config_id,
            "status": self.status,
            "step_trace_path": self.step_trace_path,
            "telemetry_path": self.telemetry_path,
            "verdict_path": self.verdict_path,
            "wall_start_ns": self.wall_start_ns,
            "wall_end_ns": self.wall_end_ns,
            "duration_seconds": self.duration_seconds,
            "total_instance_seconds": self.total_instance_seconds,
            "agent_execution_seconds": self.agent_execution_seconds,
            "phase_timing": self.phase_timing,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "agent_runtime_mode": self.agent_runtime_mode,
            "agent_execution_outcome": self.agent_execution_outcome,
            "agent_completion_error": self.agent_completion_error,
            "agent_lifecycle": self.agent_lifecycle,
            "benchmark_valid": self.benchmark_valid,
            "benchmark_invalid_reason": self.benchmark_invalid_reason,
            "verifier_eligible": self.verifier_eligible,
            "verifier_status": self.verifier_status,
            "reward": self.reward,
            "test_pass_rate": self.test_pass_rate,
            "trajectory_path": self.trajectory_path,
            "token_accounting_path": self.token_accounting_path,
            "trajectory_tokens": self.trajectory_tokens,
            "serving_tokens": self.serving_tokens,
            "runtime_observability_path": self.runtime_observability_path,
            "runtime_observability": self.runtime_observability,
            "notes": self.notes,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        return cls(
            run_id=data["run_id"],
            condition_id=data["condition_id"],
            instance_id=data["instance_id"],
            seed=data["seed"],
            serving_config_id=data["serving_config_id"],
            status=data["status"],
            step_trace_path=data.get("step_trace_path"),
            telemetry_path=data.get("telemetry_path"),
            verdict_path=data.get("verdict_path"),
            wall_start_ns=data.get("wall_start_ns"),
            wall_end_ns=data.get("wall_end_ns"),
            duration_seconds=data.get("duration_seconds"),
            total_instance_seconds=data.get("total_instance_seconds"),
            agent_execution_seconds=data.get("agent_execution_seconds"),
            phase_timing=data.get("phase_timing"),
            exit_code=data.get("exit_code"),
            error_message=data.get("error_message"),
            agent_runtime_mode=data.get("agent_runtime_mode"),
            agent_execution_outcome=data.get("agent_execution_outcome"),
            agent_completion_error=data.get("agent_completion_error"),
            agent_lifecycle=data.get("agent_lifecycle"),
            benchmark_valid=data.get("benchmark_valid"),
            benchmark_invalid_reason=data.get("benchmark_invalid_reason"),
            verifier_eligible=data.get("verifier_eligible"),
            verifier_status=data.get("verifier_status"),
            reward=data.get("reward"),
            test_pass_rate=data.get("test_pass_rate"),
            trajectory_path=data.get("trajectory_path"),
            token_accounting_path=data.get("token_accounting_path"),
            trajectory_tokens=data.get("trajectory_tokens"),
            serving_tokens=data.get("serving_tokens"),
            runtime_observability_path=data.get("runtime_observability_path"),
            runtime_observability=data.get("runtime_observability"),
            notes=data.get("notes", ""),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
        )
    
    def save(self, path: Path):
        """Save manifest to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        """Load manifest from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def exists(cls, path: Path) -> bool:
        """Check if manifest exists."""
        return path.exists()
    
    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self.status in ("PASS", "FAIL", "TIMEOUT", "ERROR")
    
    def should_retry(self) -> bool:
        """Check if this run should be retried (ERROR only)."""
        return self.status == "ERROR"


def get_manifest_path(run_id: str, output_dir: Path) -> Path:
    """Get manifest path for a run."""
    return output_dir / run_id / "manifest.json"


def load_or_create_manifest(run_id: str, condition_id: str, instance_id: str,
                            seed: int, serving_config_id: str, output_dir: Path) -> RunManifest:
    """Load existing manifest or create new one."""
    manifest_path = get_manifest_path(run_id, output_dir)
    
    if manifest_path.exists():
        return RunManifest.load(manifest_path)
    
    return RunManifest(
        run_id=run_id,
        condition_id=condition_id,
        instance_id=instance_id,
        seed=seed,
        serving_config_id=serving_config_id,
        status="PENDING",
    )
