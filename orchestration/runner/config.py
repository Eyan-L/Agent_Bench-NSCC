"""Configuration schema for experiment runs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import hashlib
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """LLM model configuration."""
    name: str
    endpoint: str  # e.g., "anthropic://claude-sonnet-4.6" or "vllm://localhost:8000/v1"
    temperature: float = 0.0
    max_tokens: int = 4096
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass
class ServingConfig:
    """Serving engine configuration."""
    config_id: str  # e.g. sglang-default, vllm-latency
    preset: str  # default, latency, throughput, context
    max_num_seqs: int
    enable_prefix_caching: bool
    backend: str = "vllm"  # vllm | sglang
    additional_flags: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "preset": self.preset,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "backend": self.backend,
            "additional_flags": self.additional_flags,
        }


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    max_steps: int = 500
    max_time_seconds: int = 1800
    retry_limit: int = 10
    tool_profile: str = "code-agent-minimal"
    allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep"
    ])
    
    def to_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "max_time_seconds": self.max_time_seconds,
            "retry_limit": self.retry_limit,
            "tool_profile": self.tool_profile,
            "allowed_tools": self.allowed_tools,
        }


@dataclass
class ConditionConfig:
    """Configuration for one experimental condition."""
    condition_id: str
    condition_type: str  # ar_issue_only, ar_static_context, openclaw_agent, ablation
    description: str
    model: ModelConfig
    agent: Optional[AgentConfig] = None
    ablation_flags: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        d = {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type,
            "description": self.description,
            "model": self.model.to_dict(),
            "ablation_flags": self.ablation_flags,
        }
        if self.agent:
            d["agent"] = self.agent.to_dict()
        return d


@dataclass
class RunConfig:
    """Configuration for a single run."""
    condition: ConditionConfig
    instance_id: str
    seed: int
    serving: ServingConfig
    plan_version: str
    
    repo_id: str
    base_commit: str
    container_image: str
    wall_clock_budget_seconds: int
    
    workspace_root: str = "/testbed"
    output_dir: Path = Path("data/runs")
    
    def compute_run_id(self) -> str:
        """Deterministic run ID from hash of key parameters."""
        hash_input = {
            "condition_id": self.condition.condition_id,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "config_id": self.serving.config_id,
            "model_name": self.condition.model.name,
            "plan_version": self.plan_version,
        }
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.compute_run_id(),
            "condition": self.condition.to_dict(),
            "instance_id": self.instance_id,
            "seed": self.seed,
            "serving": self.serving.to_dict(),
            "plan_version": self.plan_version,
            "repo_id": self.repo_id,
            "base_commit": self.base_commit,
            "container_image": self.container_image,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "workspace_root": self.workspace_root,
        }


@dataclass
class SweepConfig:
    """Configuration for a full sweep (Phase 0/1/2)."""
    sweep_name: str
    plan_version: str
    instances: List[str]
    conditions: List[ConditionConfig]
    serving_configs: List[ServingConfig]
    seeds: List[int]
    
    repo_mapping: Dict[str, dict]  # instance_id -> {repo_id, base_commit}
    container_image: str
    wall_clock_budget_seconds: int
    output_dir: Path
    
    parallel_limit_gpu: int = 1
    parallel_limit_cpu_cores: int = 8
    
    def generate_runs(self) -> List[RunConfig]:
        """Generate all RunConfig objects for this sweep."""
        runs = []
        for condition in self.conditions:
            for instance_id in self.instances:
                for seed in self.seeds:
                    for serving in self.serving_configs:
                        repo_info = self.repo_mapping[instance_id]
                        run = RunConfig(
                            condition=condition,
                            instance_id=instance_id,
                            seed=seed,
                            serving=serving,
                            plan_version=self.plan_version,
                            repo_id=repo_info["repo_id"],
                            base_commit=repo_info["base_commit"],
                            container_image=self.container_image,
                            wall_clock_budget_seconds=self.wall_clock_budget_seconds,
                            output_dir=self.output_dir,
                        )
                        runs.append(run)
        return runs


def load_composed_config(
    app_path: Path,
    harness_path: Path,
    backend_path: Path,
) -> SweepConfig:
    """Compose a SweepConfig from three independent JSON configs.

    Folder convention:
        config/application/<app>/<exp>.json   instances/repo_mapping/container/budget/output
        config/harness/<name>.json             model + (optional) openclaw block +
                                                conditions[] (condition_id/type + agent)
        config/backend/<engine>/<name>.json   engine runtime knobs (TP/GPU/port/...)
                                                + serving_configs presets; no model

    Model selection lives ONLY in the harness JSON — single source of truth
    for which model the run targets. The driver passes model.name down to
    every Condition; serve.sh and applications/swebench/app.py also read
    it from there to keep vLLM, OC provider config, and the agent's model
    selection in lockstep.

    Each call timestamps the output directory under the app JSON's
    `output_dir`. Override via `OC_RUN_STAMP` for stable / human-named runs.

    `_doc` keys at the top level of any file are ignored — they're a
    JSON-friendly stand-in for the comments YAML would otherwise allow.
    """
    from datetime import datetime
    import os as _os

    with open(app_path) as f:
        app = json.load(f) or {}
    with open(harness_path) as f:
        harness = json.load(f) or {}
    with open(backend_path) as f:
        backend = json.load(f) or {}

    # --- model (from harness) ----------------------------------------
    if "model" not in harness:
        raise ValueError(
            f"harness config {harness_path} is missing a top-level `model` block "
            "(model selection lives in the harness config, not the backend)"
        )
    model_data = harness["model"]
    model = ModelConfig(
        name=model_data["name"],
        endpoint=model_data["endpoint"],
        temperature=model_data.get("temperature", 0.0),
        max_tokens=model_data.get("max_tokens", 4096),
    )

    # --- serving configs (from backend) ------------------------------
    serving_configs = []
    for sc in backend.get("serving_configs", []):
        serving_configs.append(ServingConfig(
            config_id=sc["config_id"],
            preset=sc["preset"],
            max_num_seqs=sc["max_num_seqs"],
            enable_prefix_caching=sc["enable_prefix_caching"],
            backend=backend.get("backend", "vllm"),
            additional_flags=sc.get("additional_flags", {}),
        ))

    # --- conditions (from harness) -----------------------------------
    conditions = []
    for cond in harness.get("conditions", []):
        agent_cfg = None
        if "agent" in cond:
            ad = cond["agent"]
            agent_cfg = AgentConfig(
                max_steps=ad.get("max_steps", 500),
                max_time_seconds=ad.get("max_time_seconds", 1800),
                retry_limit=ad.get("retry_limit", 10),
                tool_profile=ad.get("tool_profile", "code-agent-minimal"),
                allowed_tools=ad.get("allowed_tools", [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ]),
            )
        conditions.append(ConditionConfig(
            condition_id=cond["condition_id"],
            condition_type=cond["condition_type"],
            description=cond.get("description", ""),
            model=model,
            agent=agent_cfg,
            ablation_flags=cond.get("ablation_flags", {}),
        ))

    # --- application-level (instances/budget/output) -----------------
    stamp = _os.environ.get("OC_RUN_STAMP") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(app["output_dir"]) / stamp

    return SweepConfig(
        sweep_name=app["sweep_name"],
        plan_version=app["plan_version"],
        instances=app["instances"] or [],
        conditions=conditions,
        serving_configs=serving_configs,
        seeds=app.get("seeds", [42]),
        repo_mapping=app.get("repo_mapping") or {},
        container_image=app["container_image"],
        wall_clock_budget_seconds=app["wall_clock_budget_seconds"],
        output_dir=output_dir,
        parallel_limit_gpu=app.get("parallel_limit_gpu", 1),
        parallel_limit_cpu_cores=app.get("parallel_limit_cpu_cores", 8),
    )
