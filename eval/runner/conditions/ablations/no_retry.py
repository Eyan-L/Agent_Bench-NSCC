"""No-retry ablation: terminate after first attempt."""

from ..code_agent.openclaw_agent import OpenClawAgentCondition


class NoRetryAblation(OpenClawAgentCondition):
    """OpenClaw with retry limit = 1 (no retries)."""
    
    def __init__(self, model_config, agent_config):
        super().__init__(model_config, agent_config)
        # Override retry limit
        self.agent_config.retry_limit = 1
