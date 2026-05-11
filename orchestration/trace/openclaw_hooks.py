"""
OpenClaw integration layer.

Provides two integration paths:
1. Hook-based: assumes OpenClaw exposes lifecycle hooks
2. Wrapper-based: wraps tool dispatcher and LLM client at adapter boundary

Default: wrapper-based (more portable).
"""

import time
from typing import Optional, Any, Callable
from contextlib import contextmanager

from .logger import StepLogger
from .llm_wrapper import LLMWrapper


# ============================================================================
# Path 1: Hook-based integration
# ============================================================================
# Assumptions:
# - OpenClaw runtime exposes hooks: on_tool_dispatch, on_llm_call, on_retry
# - Hooks provide structured event data (tool name, command, etc.)
# ============================================================================

class OpenClawHooks:
    """
    Hook-based integration for OpenClaw.
    
    Register this with OpenClaw runtime's lifecycle hooks.
    
    Usage:
        logger = StepLogger(...)
        hooks = OpenClawHooks(logger)
        
        # Register with OpenClaw (pseudocode, depends on actual API)
        openclaw.register_hook("on_tool_dispatch", hooks.on_tool_dispatch)
        openclaw.register_hook("on_llm_call", hooks.on_llm_call)
        openclaw.register_hook("on_retry", hooks.on_retry)
    """
    
    def __init__(self, logger: StepLogger):
        self.logger = logger
        self._tool_step_stack = []  # Stack of (step_id, tool_name)
        self._llm_step_stack = []
    
    def on_tool_dispatch(self, event: dict):
        """
        Hook called when OpenClaw dispatches a tool.
        
        Expected event fields:
        - phase: "start" or "end"
        - tool_name: str (e.g., "Read", "Edit", "Bash")
        - command: Optional[str] (for Bash)
        - file_path: Optional[str] (for Read/Write/Edit)
        - exit_code: Optional[int] (for end phase)
        - error_type: Optional[str]
        """
        if event["phase"] == "start":
            # Classify step type
            tool_name = event["tool_name"]
            step_type = self._classify_tool_step_type(tool_name)
            
            step_id = self.logger.start_step(
                step_type=step_type,
                tool_name=tool_name,
                command=event.get("command"),
                file_path=event.get("file_path"),
            )
            self._tool_step_stack.append((step_id, tool_name))
        
        elif event["phase"] == "end":
            if not self._tool_step_stack:
                return
            
            step_id, tool_name = self._tool_step_stack.pop()
            self.logger.end_step(
                step_id=step_id,
                exit_code=event.get("exit_code"),
                error_type=event.get("error_type"),
            )
    
    def on_llm_call(self, event: dict):
        """
        Hook called when OpenClaw makes an LLM call.
        
        Expected event fields:
        - phase: "start" or "end"
        - prompt_tokens: int
        - completion_tokens: int
        - ttft_ms: float
        - tpot_ms: float
        """
        if event["phase"] == "start":
            step_id = self.logger.start_step(step_type="LLM_INFER")
            self._llm_step_stack.append(step_id)
        
        elif event["phase"] == "end":
            if not self._llm_step_stack:
                return
            
            step_id = self._llm_step_stack.pop()
            self.logger.end_step(
                step_id=step_id,
                prompt_tokens=event.get("prompt_tokens"),
                completion_tokens=event.get("completion_tokens"),
                ttft_ms=event.get("ttft_ms"),
                tpot_ms=event.get("tpot_ms"),
            )
    
    def on_retry(self, event: dict):
        """
        Hook called when OpenClaw starts a new attempt.
        
        Expected event fields:
        - attempt_id: int
        """
        self.logger.new_attempt()
    
    def _classify_tool_step_type(self, tool_name: str) -> str:
        """Map tool name to step type."""
        read_tools = {"Read", "Glob", "Grep"}
        write_tools = {"Write", "Edit"}
        exec_tools = {"Bash"}
        
        if tool_name in read_tools:
            return "TOOL_READ"
        elif tool_name in write_tools:
            return "TOOL_WRITE"
        elif tool_name in exec_tools:
            return "TOOL_EXEC"
        else:
            # Default fallback
            return "TOOL_READ"


# ============================================================================
# Path 2: Wrapper-based integration
# ============================================================================
# Assumptions:
# - OpenClaw tool calls and LLM calls are invoked via adapters
# - We can wrap the adapter functions to inject instrumentation
# ============================================================================

class InstrumentedToolDispatcher:
    """
    Wrapper for OpenClaw tool dispatcher.
    
    Usage:
        logger = StepLogger(...)
        original_dispatcher = openclaw.tool_dispatcher
        instrumented = InstrumentedToolDispatcher(original_dispatcher, logger)
        openclaw.tool_dispatcher = instrumented
    """
    
    def __init__(self, dispatcher: Any, logger: StepLogger):
        self.dispatcher = dispatcher
        self.logger = logger
    
    def dispatch(self, tool_name: str, **kwargs) -> Any:
        """
        Dispatch tool call and log as step.
        
        Infers step_type from tool_name.
        """
        # Classify
        step_type = self._classify_tool_step_type(tool_name)
        
        # Extract context
        command = kwargs.get("command")
        file_path = kwargs.get("file_path")
        
        # Log step
        with self.logger.step(
            step_type=step_type,
            tool_name=tool_name,
            command=command,
            file_path=file_path,
        ) as step:
            try:
                result = self.dispatcher.dispatch(tool_name, **kwargs)
                step.exit_code = 0
                return result
            except Exception as e:
                step.exit_code = 1
                step.error_type = type(e).__name__
                raise
    
    def _classify_tool_step_type(self, tool_name: str) -> str:
        """Map tool name to step type."""
        read_tools = {"Read", "Glob", "Grep"}
        write_tools = {"Write", "Edit"}
        exec_tools = {"Bash"}
        
        if tool_name in read_tools:
            return "TOOL_READ"
        elif tool_name in write_tools:
            return "TOOL_WRITE"
        elif tool_name in exec_tools:
            return "TOOL_EXEC"
        else:
            return "TOOL_READ"


class InstrumentedLLMClient:
    """
    Wrapper for OpenClaw LLM client.
    
    Usage:
        logger = StepLogger(...)
        original_client = openclaw.llm_client
        instrumented = InstrumentedLLMClient(original_client, logger)
        openclaw.llm_client = instrumented
    """
    
    def __init__(self, client: Any, logger: StepLogger):
        self.wrapper = LLMWrapper(client)
        self.logger = logger
    
    def chat_completions_create(self, model: str, messages: list, **kwargs):
        """
        Wrap chat.completions.create() with step logging.
        
        Returns full response (non-streaming) with metrics captured.
        """
        with self.logger.step(step_type="LLM_INFER") as step:
            # Collect streaming response
            chunks = []
            for chunk in self.wrapper.stream(model=model, messages=messages, **kwargs):
                chunks.append(chunk)
            
            # Update step with metrics
            metrics = self.wrapper.last_metrics
            step.prompt_tokens = metrics.prompt_tokens
            step.completion_tokens = metrics.completion_tokens
            step.ttft_ms = metrics.ttft_ms
            step.tpot_ms = metrics.tpot_ms
            step.token_source = metrics.token_source
            
            return "".join(chunks)


# ============================================================================
# Recommended integration
# ============================================================================

def instrument_openclaw_wrapper(
    openclaw_runtime: Any,
    logger: StepLogger,
) -> None:
    """
    Wrapper-based instrumentation (recommended).
    
    Wraps OpenClaw tool dispatcher and LLM client.
    
    Args:
        openclaw_runtime: OpenClaw runtime instance with .tool_dispatcher and .llm_client
        logger: StepLogger instance
    
    Side effects:
        Replaces openclaw_runtime.tool_dispatcher and .llm_client with instrumented wrappers
    """
    openclaw_runtime.tool_dispatcher = InstrumentedToolDispatcher(
        openclaw_runtime.tool_dispatcher,
        logger,
    )
    openclaw_runtime.llm_client = InstrumentedLLMClient(
        openclaw_runtime.llm_client,
        logger,
    )


def instrument_openclaw_hooks(
    openclaw_runtime: Any,
    logger: StepLogger,
) -> OpenClawHooks:
    """
    Hook-based instrumentation (if OpenClaw exposes lifecycle hooks).
    
    Returns:
        OpenClawHooks instance for manual registration with runtime
    """
    hooks = OpenClawHooks(logger)
    
    # If OpenClaw has a register_hook API:
    if hasattr(openclaw_runtime, 'register_hook'):
        openclaw_runtime.register_hook("on_tool_dispatch", hooks.on_tool_dispatch)
        openclaw_runtime.register_hook("on_llm_call", hooks.on_llm_call)
        openclaw_runtime.register_hook("on_retry", hooks.on_retry)
    
    return hooks
