"""
LLM wrapper for OpenAI-compatible streaming clients.

Captures:
- TTFT: time from request start to first non-empty token
- TPOT: mean inter-token time across completion
- Token counts from API usage block or tokenizer fallback
"""

import time
from typing import Iterator, Optional, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class LLMMetrics:
    """Metrics captured from one LLM request."""
    
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    token_source: str  # "api" or "tokenizer"
    total_latency_ms: float


class LLMWrapper:
    """
    Wrapper for OpenAI-compatible streaming client.
    
    Usage:
        wrapper = LLMWrapper(client=openai_client)
        
        for chunk in wrapper.stream(model="...", messages=[...]):
            # process chunk
            print(chunk)
        
        metrics = wrapper.last_metrics
        print(f"TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"TPOT: {metrics.tpot_ms:.1f} ms")
    """
    
    def __init__(
        self,
        client: Any,
        tokenizer: Optional[Callable[[str], int]] = None,
    ):
        """
        Args:
            client: OpenAI-compatible client with chat.completions.create() method
            tokenizer: Optional tokenizer function (str -> token count) for fallback
        """
        self.client = client
        self.tokenizer = tokenizer
        self.last_metrics: Optional[LLMMetrics] = None
    
    def stream(
        self,
        model: str,
        messages: list,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream LLM completion and capture metrics.
        
        Yields content chunks. After iteration completes, metrics are available
        in self.last_metrics.
        """
        request_start = time.monotonic_ns()
        first_token_ts = None
        token_timestamps = []
        
        content_chunks = []
        usage_data = None
        
        # Make streaming request
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        
        for chunk in stream:
            chunk_ts = time.monotonic_ns()
            
            # Extract content
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None)
                
                if content:
                    # First non-empty token
                    if first_token_ts is None and content.strip():
                        first_token_ts = chunk_ts
                    
                    token_timestamps.append(chunk_ts)
                    content_chunks.append(content)
                    yield content
            
            # Check for usage data (may come in final chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
        
        request_end = time.monotonic_ns()
        
        # Compute metrics
        self.last_metrics = self._compute_metrics(
            request_start=request_start,
            request_end=request_end,
            first_token_ts=first_token_ts,
            token_timestamps=token_timestamps,
            content_chunks=content_chunks,
            messages=messages,
            usage_data=usage_data,
        )
    
    def _compute_metrics(
        self,
        request_start: int,
        request_end: int,
        first_token_ts: Optional[int],
        token_timestamps: list[int],
        content_chunks: list[str],
        messages: list,
        usage_data: Optional[Any],
    ) -> LLMMetrics:
        """Compute LLM metrics from captured timing data."""
        
        # TTFT
        if first_token_ts:
            ttft_ms = (first_token_ts - request_start) / 1_000_000.0
        else:
            ttft_ms = 0.0
        
        # TPOT (mean inter-token time)
        if len(token_timestamps) > 1:
            inter_token_times = [
                (token_timestamps[i] - token_timestamps[i-1]) / 1_000_000.0
                for i in range(1, len(token_timestamps))
            ]
            tpot_ms = sum(inter_token_times) / len(inter_token_times)
        else:
            tpot_ms = 0.0
        
        # Token counts
        token_source = "api"
        
        if usage_data and hasattr(usage_data, 'prompt_tokens'):
            prompt_tokens = usage_data.prompt_tokens
            completion_tokens = usage_data.completion_tokens
        else:
            # Fallback to tokenizer
            token_source = "tokenizer"
            if self.tokenizer:
                prompt_text = self._messages_to_text(messages)
                completion_text = "".join(content_chunks)
                prompt_tokens = self.tokenizer(prompt_text)
                completion_tokens = self.tokenizer(completion_text)
            else:
                # Crude fallback: approximate tokens
                prompt_text = self._messages_to_text(messages)
                completion_text = "".join(content_chunks)
                prompt_tokens = len(prompt_text.split()) * 4 // 3
                completion_tokens = len(completion_text.split()) * 4 // 3
        
        total_latency_ms = (request_end - request_start) / 1_000_000.0
        
        return LLMMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            token_source=token_source,
            total_latency_ms=total_latency_ms,
        )
    
    def _messages_to_text(self, messages: list) -> str:
        """Convert messages to text for tokenizer fallback."""
        return "\n".join(
            f"{msg.get('role', '')}: {msg.get('content', '')}"
            for msg in messages
        )
