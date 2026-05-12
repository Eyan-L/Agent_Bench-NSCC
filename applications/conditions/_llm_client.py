"""Shared OpenAI-compatible LLM client for AR conditions.

One place to update the API call. Both ar_issue_only and ar_static_context
call `call_llm` here so that thinking-mode handling, token accounting, and
error reporting stay consistent.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Tuple

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^\s*```(?:diff|patch)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def call_llm(
    model_config,
    prompt: str,
    seed: int,
    timeout_seconds: int = 600,
    disable_thinking: bool = True,
) -> Tuple[str, int, int, str]:
    """POST to an OpenAI-compatible /v1/chat/completions endpoint.

    Returns (text, prompt_tokens, completion_tokens, error_message).
    """
    endpoint = model_config.endpoint.rstrip("/")
    url = f"{endpoint}/chat/completions"

    body = {
        "model": model_config.name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
        "seed": seed,
    }
    if disable_thinking:
        # Qwen3.6 quirk: without this, reasoning tokens fill `message.reasoning`
        # and `content` comes back null, producing empty patches.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return "", 0, 0, f"HTTP {e.code} from {url}: {detail}"
    except urllib.error.URLError as e:
        return "", 0, 0, f"URL error from {url}: {e.reason}"
    except Exception as e:
        return "", 0, 0, f"LLM call failed: {e}"

    choice = data["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or msg.get("reasoning") or ""

    usage = data.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    if choice.get("finish_reason") == "length":
        logger.warning(
            "LLM hit max_tokens=%s (finish_reason=length); output may be truncated",
            model_config.max_tokens,
        )

    return text, prompt_tokens, completion_tokens, ""


def extract_patch(response: str) -> str:
    """Pull a unified diff out of an LLM response.

    Handles three formats:
    1. Pure diff (no fences)
    2. ```diff ... ``` fence wrapping the whole patch
    3. Diff embedded in free text (take from first `diff --git` or `--- a/`)
    """
    if not response:
        return ""

    m = _FENCE_RE.match(response.strip())
    if m:
        body = m.group(1)
        if body.strip().startswith(("diff --git", "--- a/", "--- ")):
            return body if body.endswith("\n") else body + "\n"

    for marker in ("diff --git", "--- a/"):
        idx = response.find(marker)
        if idx >= 0:
            patch = response[idx:]
            fence_idx = patch.rfind("\n```")
            if fence_idx > 0:
                patch = patch[:fence_idx]
            if not patch.endswith("\n"):
                patch += "\n"
            return patch

    return ""