"""Configuration schema for experiment runs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import hashlib
import json
import yaml
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


def load_sweep_config(yaml_path: Path) -> SweepConfig:
    """Load sweep configuration from YAML."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    # Parse serving configs
    serving_configs = []
    for sc in data["serving_configs"]:
        serving_configs.append(ServingConfig(
            config_id=sc["config_id"],
            preset=sc["preset"],
            max_num_seqs=sc["max_num_seqs"],
            enable_prefix_caching=sc["enable_prefix_caching"],
            additional_flags=sc.get("additional_flags", {}),
        ))
    
    # Parse model config (shared across conditions in this example)
    model_data = data["model"]
    model = ModelConfig(
        name=model_data["name"],
        endpoint=model_data["endpoint"],
        temperature=model_data.get("temperature", 0.0),
        max_tokens=model_data.get("max_tokens", 4096),
    )
    
    # Parse conditions
    conditions = []
    for cond in data["conditions"]:
        agent_cfg = None
        if "agent" in cond:
            agent_data = cond["agent"]
            agent_cfg = AgentConfig(
                max_steps=agent_data.get("max_steps", 500),
                max_time_seconds=agent_data.get("max_time_seconds", 1800),
                retry_limit=agent_data.get("retry_limit", 10),
                tool_profile=agent_data.get("tool_profile", "code-agent-minimal"),
                allowed_tools=agent_data.get("allowed_tools", [
                    "Read", "Write", "Edit", "Bash", "Glob", "Grep"
                ]),
            )
        
        conditions.append(ConditionConfig(
            condition_id=cond["condition_id"],
            condition_type=cond["condition_type"],
            description=cond["description"],
            model=model,
            agent=agent_cfg,
            ablation_flags=cond.get("ablation_flags", {}),
        ))
    
    return SweepConfig(
        sweep_name=data["sweep_name"],
        plan_version=data["plan_version"],
        instances=data["instances"],
        conditions=conditions,
        serving_configs=serving_configs,
        seeds=data.get("seeds", [42]),
        repo_mapping=data["repo_mapping"],
        container_image=data["container_image"],
        wall_clock_budget_seconds=data["wall_clock_budget_seconds"],
        output_dir=Path(data["output_dir"]),
        parallel_limit_gpu=data.get("parallel_limit_gpu", 1),
        parallel_limit_cpu_cores=data.get("parallel_limit_cpu_cores", 8),
    )
