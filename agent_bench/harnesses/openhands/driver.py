"""Headless OpenHands SDK driver — bridge-aware.

Lives inside the OpenHands harness container at /opt/openhands_driver.py.
Invoked by `SWEBenchOpenHandsHarness` (and its condition) once per run.

Reads the OPENHANDS_EXEC_BRIDGE_* env vars set by OpenHandsDispatcher; when
present, every BashTool / TerminalTool / FileEditorTool call from the SDK
is re-routed through the bridge CLI (oh-exec-bridge → unix socket →
host server → docker exec sandbox). When unset, tool calls execute in
the harness container's own process — useful for unit tests but not what
the SWE-bench harness uses.

Wire to the world:

    --config   <path>        TOML rendered from profiles/harnesses/openhands_*.yaml
    --task     <message>     initial user message
    --events   <jsonl>       NDJSON event log written here
    --workspace <dir>        agent's workspace (in-container path)

Tolerant to SDK shape drift: wraps the resolved executable tools after the
SDK has initialized the Conversation. If the bridge is configured but the
required execution tools cannot be wrapped, the driver fails fast instead of
letting tools run in the harness container.
"""
from __future__ import annotations

import argparse
import base64
import base64
import inspect
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover — 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
# Default runtime to "process" — when the bridge wraps tool execs, the SDK's
# own Runtime layer becomes a passthrough. We explicitly avoid "docker" so
# the SDK never tries to spawn its own sandbox.
os.environ.setdefault("RUNTIME", "process")

logger = logging.getLogger("openhands_driver")
logging.basicConfig(
    level=os.environ.get("OPENHANDS_DRIVER_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)


def _accepts_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return (
        name in sig.parameters
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )


# ----------------------------------------------------------------------
# Event sink (NDJSON)
# ----------------------------------------------------------------------


class EventSink:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", buffering=1)

    def write(self, event: Dict[str, Any]) -> None:
        event.setdefault("driver_ts", datetime.now(timezone.utc).isoformat())
        self._fh.write(json.dumps(event, default=str) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _emit(sink: EventSink, kind: str, **payload) -> None:
    sink.write({"driver_event": kind, **payload})


def _emit_error(sink: EventSink, kind: str, msg: str, **extra) -> None:
    sink.write({"driver_event": "error", "kind": kind, "message": msg, **extra})


class _OtelEmitter:
    """Best-effort OTLP/HTTP span emitter used inside the harness container."""

    def __init__(
        self,
        *,
        service_name: str,
        run_id: str,
        endpoint: str | None = None,
        resource_attributes: str | None = None,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.run_id = run_id
        endpoint = (
            os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or os.environ.get("AB_TELEMETRY_ENDPOINT")
            or endpoint
            or ""
        ).strip().rstrip("/")
        self.url = ""
        if enabled and endpoint:
            self.url = endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"
        self.trace_id = os.urandom(16).hex()
        self.resource = _resource_attrs_from_env(resource_attributes)
        self.resource.update({
            "ab.run_id": run_id,
            "oc.run_id": run_id,
            "agent.run_id": run_id,
            "oh.run_id": run_id,
            "service.name": service_name,
        })

    def emit_span(self, name: str, start_ns: int, end_ns: int, attrs: Dict[str, Any]) -> None:
        if not self.url or not self.run_id:
            return
        try:
            span_attrs = {
                "agent.canonical_name": name,
                "agent.run_id": self.run_id,
                "agent.session_id": self.run_id,
                "agent.span.source": "openhands.driver",
            }
            span_attrs.update(attrs)
            now_ns = time.time_ns()
            body = {
                "resourceSpans": [{
                    "resource": {"attributes": _otel_attrs(self.resource)},
                    "scopeSpans": [{
                        "scope": {"name": self.service_name},
                        "spans": [{
                            "traceId": self.trace_id,
                            "spanId": os.urandom(8).hex(),
                            "name": name,
                            "kind": 1,
                            "startTimeUnixNano": str(int(start_ns or end_ns or now_ns)),
                            "endTimeUnixNano": str(int(end_ns or start_ns or now_ns)),
                            "attributes": _otel_attrs(span_attrs),
                        }],
                    }],
                }],
            }
            req = urllib.request.Request(
                self.url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0):
                pass
        except Exception:
            logger.debug("OTel span export failed for %s", name, exc_info=True)


def _resource_attrs_from_env(raw: str | None = None) -> Dict[str, Any]:
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or raw or ""
    attrs: Dict[str, Any] = {}
    current = ""
    escaped = False
    parts: List[str] = []
    for ch in raw:
        if escaped:
            current += ch
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ",":
            parts.append(current)
            current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key:
            attrs[key] = value
    return attrs


def _otel_attrs(values: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"key": str(key), "value": _otel_value(value)}
        for key, value in values.items()
        if value is not None
    ]


def _otel_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otel_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"stringValue": json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {"stringValue": str(value)}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0


def _get_value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                continue
    return None


def _usage_dict(value: Any, *, depth: int = 0) -> Dict[str, Any]:
    if value is None or depth > 4:
        return {}
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            return usage
        for key in ("response", "data", "result", "output"):
            nested = _usage_dict(value.get(key), depth=depth + 1)
            if nested:
                return nested
        return value if any(k in value for k in ("prompt_tokens", "completion_tokens", "total_tokens")) else {}
    usage = _get_value(value, "usage")
    if usage is not None:
        if isinstance(usage, dict):
            return usage
        serialized = _serialize(usage)
        if isinstance(serialized, dict):
            return serialized
    serialized = _serialize(value)
    if isinstance(serialized, dict) and serialized is not value:
        return _usage_dict(serialized, depth=depth + 1)
    return {}


def _extract_model_usage(value: Any) -> Dict[str, int]:
    usage = _usage_dict(value)
    details = usage.get("prompt_tokens_details") or {}
    prompt = _as_int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt")
    )
    completion = _as_int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion")
    )
    cache_read = _as_int(
        usage.get("cache_read_tokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cached_tokens")
        or details.get("cached_tokens")
    )
    cache_write = _as_int(
        usage.get("cache_write_tokens")
        or usage.get("cache_creation_tokens")
        or usage.get("cache_creation_input_tokens")
    )
    total = _as_int(usage.get("total_tokens") or usage.get("total"))
    if total == 0:
        total = prompt + completion
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total,
    }


def _is_stream_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict, list, tuple, set)):
        return False
    return hasattr(value, "__iter__") and not hasattr(value, "model_dump")


class _LLMTelemetryProxy:
    """Proxy OpenHands SDK LLM calls and emit one model_usage event per call."""

    def __init__(
        self,
        llm: Any,
        sink: EventSink,
        llm_cfg: Dict[str, Any],
        otel: _OtelEmitter | None = None,
    ):
        self._ab_llm = llm
        self._ab_sink = sink
        self._ab_llm_cfg = llm_cfg
        self._ab_otel = otel
        self._ab_request_index = 0

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._ab_llm, name)
        if name.startswith("_") or not callable(value):
            return value
        return self._wrap_callable(name, value)

    def __call__(self, *args, **kwargs):
        return self._wrap_callable("__call__", self._ab_llm)(*args, **kwargs)

    def _wrap_callable(self, name: str, fn: Callable):
        if inspect.iscoroutinefunction(fn):
            async def awrapped(*args, **kwargs):
                start_ns = time.time_ns()
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    self._emit_usage(name, start_ns, time.time_ns(), None, error=str(exc))
                    raise
                self._emit_usage(name, start_ns, time.time_ns(), result)
                return result
            return awrapped

        def wrapped(*args, **kwargs):
            start_ns = time.time_ns()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self._emit_usage(name, start_ns, time.time_ns(), None, error=str(exc))
                raise
            if _is_stream_like(result):
                return self._wrap_stream(name, result, start_ns)
            self._emit_usage(name, start_ns, time.time_ns(), result)
            return result
        return wrapped

    def _wrap_stream(self, name: str, stream: Any, start_ns: int):
        chunks = []

        def gen():
            last = None
            try:
                for item in stream:
                    chunks.append(item)
                    last = item
                    yield item
            finally:
                usage_source = last if last is not None else (chunks[-1] if chunks else None)
                self._emit_usage(
                    name,
                    start_ns,
                    time.time_ns(),
                    usage_source,
                    chunks_count=len(chunks),
                )
        return gen()

    def _emit_usage(
        self,
        method: str,
        start_ns: int,
        end_ns: int,
        result: Any,
        *,
        error: str | None = None,
        chunks_count: int | None = None,
    ) -> None:
        self._ab_request_index += 1
        usage = _extract_model_usage(result)
        event = {
            "type": "model.usage",
            "channel": "openhands.llm",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": max((end_ns - start_ns) / 1e6, 0.0),
            "request_index": self._ab_request_index,
            "provider": self._ab_llm_cfg.get("provider") or "openhands",
            "model": self._ab_llm_cfg.get("model"),
            "api": method,
            "http_status": 200 if error is None else None,
            "error": error,
            "chunks_count": chunks_count,
            **usage,
        }
        _emit(self._ab_sink, "model_usage", **event)
        if self._ab_otel is not None:
            self._ab_otel.emit_span(
                "agent.model.usage",
                start_ns,
                end_ns,
                {
                    "agent.channel": "openhands.llm",
                    "agent.provider": event["provider"],
                    "agent.model": event.get("model"),
                    "agent.api": method,
                    "agent.llm.request_index": self._ab_request_index,
                    "agent.llm.requestIndex": self._ab_request_index,
                    "agent.http.status_code": event["http_status"],
                    "agent.http.statusCode": event["http_status"],
                    "agent.tokens.input": usage["input_tokens"],
                    "agent.tokens.output": usage["output_tokens"],
                    "agent.tokens.cache_read": usage["cache_read_tokens"],
                    "agent.tokens.cache_write": usage["cache_write_tokens"],
                    "agent.tokens.total": usage["total_tokens"],
                    "agent.chunks.count": chunks_count,
                    "agent.error": error,
                },
            )


_LLM_TELEMETRY_METHODS = (
    "completion",
    "acompletion",
    "async_completion",
    "chat",
    "achat",
    "chat_completion",
    "achat_completion",
    "generate",
    "agenerate",
    "invoke",
    "ainvoke",
    "call",
)


def _instrument_llm_for_telemetry(
    llm: Any,
    sink: EventSink,
    llm_cfg: Dict[str, Any],
    otel: _OtelEmitter | None = None,
) -> Any:
    """Patch inference methods on the SDK LLM while preserving its class identity.

    OpenHands SDK 1.21 validates ``Agent.llm`` as an actual ``LLM`` instance.
    Replacing it with a proxy object breaks Pydantic validation, so telemetry
    has to be installed in place on the original object.
    """
    proxy = _LLMTelemetryProxy(llm, sink, llm_cfg, otel)
    wrapped: list[str] = []
    skipped: list[str] = []
    for name in _LLM_TELEMETRY_METHODS:
        try:
            fn = getattr(llm, name)
        except Exception:
            continue
        if not callable(fn):
            continue
        try:
            object.__setattr__(llm, name, proxy._wrap_callable(name, fn))
            wrapped.append(name)
        except Exception as exc:
            skipped.append(f"{name}:{type(exc).__name__}")
            logger.debug("could not wrap LLM method %s", name, exc_info=True)
    _emit(
        sink,
        "llm_telemetry_wrap",
        methods_wrapped=wrapped,
        methods_skipped=skipped,
    )
    return llm


# ----------------------------------------------------------------------
# Bridge invocation
# ----------------------------------------------------------------------


_BRIDGE_BIN = os.environ.get("OPENHANDS_EXEC_BRIDGE", "")
_BRIDGE_SOCKET = os.environ.get("OPENHANDS_EXEC_BRIDGE_SOCKET", "")
_BRIDGE_TOKEN = os.environ.get("OPENHANDS_EXEC_BRIDGE_TOKEN", "")
_BRIDGE_AGENT_WORKSPACE = os.environ.get("OPENHANDS_EXEC_BRIDGE_AGENT_WORKSPACE", "")
_BRIDGE_TASK_WORKSPACE = os.environ.get("OPENHANDS_EXEC_BRIDGE_TASK_WORKSPACE", "")


def bridge_active() -> bool:
    return bool(_BRIDGE_BIN and _BRIDGE_SOCKET and _BRIDGE_TOKEN)


def _translate_workspace_path_to_task(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    agent_root = _BRIDGE_AGENT_WORKSPACE.rstrip("/")
    task_root = _BRIDGE_TASK_WORKSPACE.rstrip("/")
    if not agent_root or not task_root or agent_root == task_root:
        return value
    if value == agent_root:
        return task_root
    if value.startswith(agent_root + "/"):
        return task_root + value[len(agent_root):]
    return value


def _translate_workspace_path_to_agent(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    agent_root = _BRIDGE_AGENT_WORKSPACE.rstrip("/")
    task_root = _BRIDGE_TASK_WORKSPACE.rstrip("/")
    if not agent_root or not task_root or agent_root == task_root:
        return value
    if value == task_root:
        return agent_root
    if value.startswith(task_root + "/"):
        return agent_root + value[len(task_root):]
    return value


def _translate_workspace_refs_to_task(command: str) -> str:
    agent_root = _BRIDGE_AGENT_WORKSPACE.rstrip("/")
    task_root = _BRIDGE_TASK_WORKSPACE.rstrip("/")
    if not agent_root or not task_root or agent_root == task_root:
        return command
    return command.replace(agent_root + "/", task_root + "/")


def bridge_exec_shell(
    command: str,
    *,
    workdir: str = "/testbed",
    env: Optional[Dict[str, str]] = None,
    timeout_ms: Optional[int] = None,
    stdin_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Send a shell command via the bridge CLI; return {stdout, stderr, exit}."""
    return _bridge_exec(
        command=command,
        workdir=workdir,
        env=env,
        timeout_ms=timeout_ms,
        stdin_bytes=stdin_bytes,
    )


def bridge_exec_argv(
    argv_json: List[str],
    *,
    workdir: str = "/testbed",
    env: Optional[Dict[str, str]] = None,
    timeout_ms: Optional[int] = None,
    stdin_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Send raw argv via the bridge CLI; used for fs-like file_editor ops."""
    return _bridge_exec(
        argv_json=argv_json,
        workdir=workdir,
        env=env,
        timeout_ms=timeout_ms,
        stdin_bytes=stdin_bytes,
    )


def _bridge_exec(
    *,
    command: str | None = None,
    argv_json: List[str] | None = None,
    workdir: str,
    env: Optional[Dict[str, str]] = None,
    timeout_ms: Optional[int] = None,
    stdin_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if not bridge_active():
        raise RuntimeError("bridge exec called but bridge env not set")
    if (command is None) == (argv_json is None):
        raise RuntimeError("bridge exec requires exactly one of command or argv_json")
    argv = [
        _BRIDGE_BIN,
        "--workdir", workdir,
        "--env-json", json.dumps(env or {}),
    ]
    if argv_json is None:
        argv += ["--command", command or ""]
    else:
        argv += ["--argv-json", json.dumps(argv_json)]
    if timeout_ms:
        argv += ["--timeout-ms", str(int(timeout_ms))]
    proc_env = {
        **os.environ,
        "OPENCLAW_EXEC_BRIDGE_SOCKET": _BRIDGE_SOCKET,
        "OPENCLAW_EXEC_BRIDGE_TOKEN": _BRIDGE_TOKEN,
    }
    proc = subprocess.run(
        argv,
        input=stdin_bytes,
        capture_output=True,
        env=proc_env,
        timeout=(timeout_ms / 1000.0 + 60) if timeout_ms else 3600.0,
    )
    return {
        "exit_code": proc.returncode,
        "stdout_bytes": proc.stdout or b"",
        "stderr_bytes": proc.stderr or b"",
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
    }


# ----------------------------------------------------------------------
# SDK tool executor wrapping
# ----------------------------------------------------------------------


def _json_preview(value: Any, *, limit: int = 8192) -> tuple[str, bool, int]:
    try:
        serialized = _serialize(value)
        if isinstance(serialized, str):
            text = serialized
        else:
            text = json.dumps(serialized, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    original_len = len(text)
    if original_len > limit:
        return text[:limit], True, original_len
    return text, False, original_len


def _tool_result_is_error(result: Any, error: str | None) -> bool:
    if error:
        return True
    if isinstance(result, dict):
        if result.get("is_error") is not None:
            return bool(result.get("is_error"))
        exit_code = result.get("exit_code")
        return exit_code not in (None, 0, "0")
    for attr in ("is_error", "error"):
        if hasattr(result, attr):
            try:
                if getattr(result, attr):
                    return True
            except Exception:
                pass
    return False


def _emit_tool_span(
    otel: _OtelEmitter | None,
    tool_name: str,
    action: Any,
    result: Any,
    start_ns: int,
    end_ns: int,
    *,
    bridged: bool,
    error: str | None,
) -> None:
    if otel is None:
        return
    args_json, args_truncated, _args_len = _json_preview(action)
    result_preview, result_truncated, result_chars = _json_preview(result, limit=4096)
    call_id = (
        getattr(action, "tool_call_id", None)
        or getattr(action, "id", None)
        or getattr(action, "call_id", None)
        or f"{tool_name}:{start_ns}"
    )
    exit_code = result.get("exit_code") if isinstance(result, dict) else getattr(result, "exit_code", None)
    otel.emit_span(
        "agent.tool.call",
        start_ns,
        end_ns,
        {
            "agent.channel": "openhands.tool",
            "agent.tool.name": str(tool_name or "unknown"),
            "agent.tool.call_id": str(call_id),
            "agent.tool.callId": str(call_id),
            "agent.tool.is_error": _tool_result_is_error(result, error),
            "agent.tool.isError": _tool_result_is_error(result, error),
            "agent.tool.bridged": bool(bridged),
            "agent.tool.arguments_json": args_json,
            "agent.tool.arguments_truncated": args_truncated,
            "agent.tool.result_preview": result_preview,
            "agent.tool.result_chars": result_chars,
            "agent.tool.result_truncated": result_truncated,
            "agent.tool.exit_code": exit_code,
            "agent.error": error,
        },
    )


_BRIDGE_EXEC_TOOL_NAMES = {"terminal", "bash", "file_editor"}
_BRIDGE_REQUIRED_TOOL_NAMES = {"terminal", "file_editor"}


@dataclass
class _BridgeCall:
    command: str
    workdir: str
    timeout_ms: int | None = None
    stdin_bytes: bytes | None = None


class _UnsupportedBridgeAction(Exception):
    pass


_run_id = os.environ.get("AB_RUN_ID") or os.environ.get("OPENHANDS_RUN_ID") or "default"
_FILE_EDITOR_HISTORY_PATH = Path(
    os.environ.get(
        "OPENHANDS_FILE_EDITOR_HISTORY",
        f"/tmp/ab_openhands_file_editor_history_{_run_id}.json",
    )
)
_FILE_EDITOR_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_FILE_EDITOR_MAX_RESPONSE_CHARS = 20000


class _FileEditorBridgeError(Exception):
    pass


def _file_editor_action_payload(action: Any) -> dict[str, Any]:
    op = getattr(action, "command", None) or getattr(action, "op", None)
    path = getattr(action, "path", None) or getattr(action, "file_path", None)
    if not op or not path:
        raise _UnsupportedBridgeAction("file_editor action requires command and path")
    return {
        "command": op,
        "path": _translate_workspace_path_to_task(path),
        "file_text": getattr(action, "file_text", None),
        "view_range": getattr(action, "view_range", None),
        "old_str": getattr(action, "old_str", None),
        "new_str": getattr(action, "new_str", None),
        "insert_line": getattr(action, "insert_line", None),
    }


def _file_editor_make_obs(
    command: str,
    path: str | None,
    text: str,
    *,
    is_error: bool = False,
    prev_exist: bool = True,
    old_content: str | None = None,
    new_content: str | None = None,
    content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "content": content if content is not None else _text_content(text),
        "is_error": bool(is_error),
        "command": command,
        "path": path,
        "prev_exist": bool(prev_exist),
        "old_content": old_content,
        "new_content": new_content,
    }


def _file_editor_truncate(text: str) -> str:
    if len(text) <= _FILE_EDITOR_MAX_RESPONSE_CHARS:
        return text
    return text[:_FILE_EDITOR_MAX_RESPONSE_CHARS] + "\n<response clipped by Agent_Bench bridge>\n"


def _file_editor_line_output(content: str, description: str, start_line: int = 1) -> str:
    content = _file_editor_truncate(content)
    numbered = "\n".join(
        f"{idx + start_line:6}\t{line}"
        for idx, line in enumerate(content.split("\n"))
    )
    return f"Here's the result of running `cat -n` on {description}:\n{numbered}\n"


def _file_editor_load_history() -> dict[str, list[str]]:
    try:
        data = json.loads(_FILE_EDITOR_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _file_editor_save_history(history: dict[str, list[str]]) -> None:
    try:
        _FILE_EDITOR_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FILE_EDITOR_HISTORY_PATH.write_text(json.dumps(history), encoding="utf-8")
    except Exception:
        pass


def _file_editor_push_history(path: str, content: str) -> None:
    history = _file_editor_load_history()
    entries = history.setdefault(path, [])
    entries.append(content)
    del entries[:-10]
    _file_editor_save_history(history)


def _file_editor_pop_history(path: str) -> str | None:
    history = _file_editor_load_history()
    entries = history.get(path) or []
    if not entries:
        return None
    content = entries.pop()
    if entries:
        history[path] = entries
    else:
        history.pop(path, None)
    _file_editor_save_history(history)
    return content


def _file_editor_bridge_result_or_error(res: dict[str, Any], action: str) -> dict[str, Any]:
    if int(res.get("exit_code", 0) or 0) == 0:
        return res
    message = _combined_output(res).strip() or f"file_editor bridge {action} failed"
    raise _FileEditorBridgeError(message)


def _remote_path_kind(path: str, workdir: str) -> str:
    script = (
        'if [ -d "$1" ]; then printf dir; '
        'elif [ -e "$1" ]; then printf file; '
        'else printf missing; fi'
    )
    res = bridge_exec_argv(
        ["sh", "-c", script, "openhands-file-stat", path],
        workdir=workdir,
        timeout_ms=30000,
    )
    _file_editor_bridge_result_or_error(res, "stat")
    kind = (res.get("stdout") or "").strip()
    if kind not in {"dir", "file", "missing"}:
        raise _FileEditorBridgeError(f"unexpected file stat response for {path}: {kind!r}")
    return kind


def _remote_read_bytes(path: str, workdir: str) -> bytes:
    res = bridge_exec_argv(
        ["sh", "-c", 'cat "$1"', "openhands-file-read", path],
        workdir=workdir,
        timeout_ms=30000,
    )
    _file_editor_bridge_result_or_error(res, "read")
    return bytes(res.get("stdout_bytes") or b"")


def _remote_read_text(path: str, workdir: str) -> str:
    return _remote_read_bytes(path, workdir).decode("utf-8")


def _remote_write_bytes(path: str, data: bytes, workdir: str) -> None:
    res = bridge_exec_argv(
        ["sh", "-c", 'cat > "$1"', "openhands-file-write", path],
        workdir=workdir,
        stdin_bytes=data,
        timeout_ms=30000,
    )
    _file_editor_bridge_result_or_error(res, "write")


def _remote_write_text(path: str, content: str, workdir: str) -> None:
    _remote_write_bytes(path, content.encode("utf-8"), workdir)


def _remote_list_dir(path: str, workdir: str) -> str:
    script = r'''
root=$1
printf '%s/\n' "$root"
find "$root" -mindepth 1 -maxdepth 2 \( -name '.*' -o -path '*/.*' \) -prune -o -print 2>/dev/null \
  | sort \
  | while IFS= read -r p; do
      [ "$p" = "$root" ] && continue
      if [ -d "$p" ]; then printf '%s/\n' "$p"; else printf '%s\n' "$p"; fi
    done
printf '\n__AB_HIDDEN_COUNT__\n'
find "$root" -mindepth 1 -maxdepth 1 -name '.*' -print 2>/dev/null | wc -l
'''.strip()
    res = bridge_exec_argv(
        ["sh", "-c", script, "openhands-file-list", path],
        workdir=workdir,
        timeout_ms=30000,
    )
    _file_editor_bridge_result_or_error(res, "list")
    output = res.get("stdout") or ""
    entries_text, marker, hidden_text = output.partition("\n__AB_HIDDEN_COUNT__\n")
    lines = [line for line in entries_text.rstrip("\n").splitlines() if line]
    message = (
        f"Here's the files and directories up to 2 levels deep in {path}, "
        "excluding hidden items:\n" + "\n".join(lines)
    )
    if marker:
        try:
            hidden = int(hidden_text.strip() or "0")
        except ValueError:
            hidden = 0
        if hidden:
            message += (
                f"\n{hidden} hidden files/directories in this directory are excluded. "
                f"You can use 'ls -la {path}' to see them."
            )
    return message


def _file_editor_prev_exists(path: str | None, workdir: str) -> bool:
    if not path:
        return True
    try:
        return _remote_path_kind(path, workdir) != "missing"
    except Exception:
        return True


def _file_editor_validate(command: str, path: str, workdir: str) -> str:
    if not path.startswith("/"):
        raise _FileEditorBridgeError(f"The path should be an absolute path: {path}")
    kind = _remote_path_kind(path, workdir)
    if command == "create":
        if kind != "missing":
            raise _FileEditorBridgeError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        parent = str(Path(path).parent)
        if _remote_path_kind(parent, workdir) != "dir":
            raise _FileEditorBridgeError(f"Parent directory does not exist: {parent}")
        return kind
    if kind == "missing":
        raise _FileEditorBridgeError(f"The path {path} does not exist. Please provide a valid path.")
    if command != "view" and kind == "dir":
        raise _FileEditorBridgeError(
            f"The path {path} is a directory and only the `view` command can be used on directories."
        )
    return kind


def _file_editor_do_view(command: str, path: str, payload: dict[str, Any], workdir: str, kind: str) -> dict[str, Any]:
    view_range = payload.get("view_range")
    if kind == "dir":
        if view_range:
            raise _FileEditorBridgeError("The `view_range` parameter is not allowed for directories.")
        return _file_editor_make_obs(command, path, _remote_list_dir(path, workdir), prev_exist=True)

    if Path(path).suffix.lower() in _FILE_EDITOR_IMAGE_EXTS:
        image_bytes = _remote_read_bytes(path, workdir)
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"
        content = [
            {
                "type": "text",
                "text": f"Image file {path} read successfully. Displaying image content.",
                "cache_prompt": False,
            },
            {
                "type": "image",
                "image_urls": [
                    f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                ],
                "cache_prompt": False,
            },
        ]
        return _file_editor_make_obs(command, path, "", prev_exist=True, content=content)

    file_content = _remote_read_text(path, workdir)
    lines = file_content.splitlines()
    if view_range:
        if len(view_range) != 2:
            raise _FileEditorBridgeError("view_range should be a list of two integers.")
        start, end = int(view_range[0]), int(view_range[1])
        if start < 1 or start > max(len(lines), 1):
            raise _FileEditorBridgeError(f"view_range start {start} is outside the file line range.")
        if end == -1 or end > len(lines):
            end = len(lines)
        if end < start:
            raise _FileEditorBridgeError("view_range end should be greater than or equal to start.")
        shown = "\n".join(lines[start - 1:end])
        return _file_editor_make_obs(
            command,
            path,
            _file_editor_line_output(shown, path, start),
            prev_exist=True,
        )
    return _file_editor_make_obs(
        command,
        path,
        _file_editor_line_output(file_content, path, 1),
        prev_exist=True,
    )


def _file_editor_do_create(command: str, path: str, payload: dict[str, Any], workdir: str) -> dict[str, Any]:
    file_text = payload.get("file_text")
    if file_text is None:
        raise _FileEditorBridgeError("Missing required parameter `file_text` for create.")
    file_text = str(file_text)
    _remote_write_text(path, file_text, workdir)
    return _file_editor_make_obs(
        command,
        path,
        f"File created successfully at: {path}",
        prev_exist=False,
        new_content=file_text,
    )


def _file_editor_do_str_replace(command: str, path: str, payload: dict[str, Any], workdir: str) -> dict[str, Any]:
    old_str = payload.get("old_str")
    new_str = payload.get("new_str")
    if old_str is None:
        raise _FileEditorBridgeError("Missing required parameter `old_str` for str_replace.")
    if new_str is None:
        raise _FileEditorBridgeError("Missing required parameter `new_str` for str_replace.")
    if old_str == new_str:
        raise _FileEditorBridgeError("No replacement was performed. `new_str` and `old_str` must be different.")
    file_content = _remote_read_text(path, workdir)
    search_old = str(old_str)
    search_new = str(new_str)
    matches = list(re.finditer(re.escape(search_old), file_content))
    if not matches:
        search_old = str(old_str).strip()
        search_new = str(new_str).strip()
        matches = list(re.finditer(re.escape(search_old), file_content))
    if not matches:
        raise _FileEditorBridgeError(
            f"No replacement was performed, old_str `{search_old}` did not appear verbatim in {path}."
        )
    if len(matches) > 1:
        lines = sorted({file_content.count("\n", 0, m.start()) + 1 for m in matches})
        raise _FileEditorBridgeError(
            f"No replacement was performed. Multiple occurrences of old_str `{search_old}` "
            f"in lines {lines}. Please ensure it is unique."
        )
    match = matches[0]
    new_content = file_content[:match.start()] + search_new + file_content[match.end():]
    _file_editor_push_history(path, file_content)
    _remote_write_text(path, new_content, workdir)
    return _file_editor_make_obs(
        command,
        path,
        f"The file {path} has been edited. Review the changes and make sure they are as expected.",
        prev_exist=True,
        old_content=file_content,
        new_content=new_content,
    )


def _file_editor_do_insert(command: str, path: str, payload: dict[str, Any], workdir: str) -> dict[str, Any]:
    insert_line = payload.get("insert_line")
    new_str = payload.get("new_str")
    if insert_line is None:
        raise _FileEditorBridgeError("Missing required parameter `insert_line` for insert.")
    if new_str is None:
        raise _FileEditorBridgeError("Missing required parameter `new_str` for insert.")
    insert_line = int(insert_line)
    file_content = _remote_read_text(path, workdir)
    lines = file_content.splitlines(True)
    if insert_line < 0 or insert_line > len(lines):
        raise _FileEditorBridgeError(f"insert_line should be within the range of allowed values: {[0, len(lines)]}")
    inserted = str(new_str)
    if not inserted.endswith("\n"):
        inserted += "\n"
    new_content = "".join(lines[:insert_line]) + inserted + "".join(lines[insert_line:])
    _file_editor_push_history(path, file_content)
    _remote_write_text(path, new_content, workdir)
    return _file_editor_make_obs(
        command,
        path,
        f"The file {path} has been edited. Review the changes and make sure they are as expected.",
        prev_exist=True,
        old_content=file_content,
        new_content=new_content,
    )


def _file_editor_do_undo(command: str, path: str, workdir: str) -> dict[str, Any]:
    current = _remote_read_text(path, workdir)
    previous = _file_editor_pop_history(path)
    if previous is None:
        raise _FileEditorBridgeError(f"No edit history found for {path}.")
    _remote_write_text(path, previous, workdir)
    return _file_editor_make_obs(
        command,
        path,
        f"Last edit to {path} undone successfully. {_file_editor_line_output(previous, path)}",
        prev_exist=True,
        old_content=current,
        new_content=previous,
    )


def _run_file_editor_payload(payload: dict[str, Any], workdir: str) -> dict[str, Any]:
    command = payload.get("command") or "view"
    path = str(payload.get("path") or "")
    try:
        kind = _file_editor_validate(command, path, workdir)
        if command == "view":
            return _file_editor_do_view(command, path, payload, workdir, kind)
        if command == "create":
            return _file_editor_do_create(command, path, payload, workdir)
        if command == "str_replace":
            return _file_editor_do_str_replace(command, path, payload, workdir)
        if command == "insert":
            return _file_editor_do_insert(command, path, payload, workdir)
        if command == "undo_edit":
            return _file_editor_do_undo(command, path, workdir)
        raise _FileEditorBridgeError(f"Unrecognized file_editor command: {command}")
    except Exception as exc:
        return _file_editor_make_obs(
            command,
            path or None,
            str(exc),
            is_error=True,
            prev_exist=_file_editor_prev_exists(path or None, workdir),
        )


def _run_file_editor_action(action: Any, default_workdir: str) -> dict[str, Any]:
    workdir = str(_translate_workspace_path_to_task(default_workdir))
    payload = _file_editor_action_payload(action)
    return _restore_file_editor_observation_paths(_run_file_editor_payload(payload, workdir))


class _BridgeToolExecutor:
    """Executor wrapper that routes OpenHands tool calls through Agent_Bench bridge."""

    def __init__(
        self,
        tool_name: str,
        original_executor: Any,
        default_workdir: str,
        otel: _OtelEmitter | None = None,
    ):
        self.tool_name = tool_name
        self.original_executor = original_executor
        self.default_workdir = default_workdir
        self.otel = otel

    def __call__(self, action: Any, conversation: Any = None) -> Any:  # noqa: ARG002
        start_ns = time.time_ns()
        bridged = False
        result = None
        error_text = None
        try:
            if self.tool_name.lower() == "file_editor":
                bridged = True
                logger.info("bridge[%s] agent-side file adapter", self.tool_name)
                result = _run_file_editor_action(action, self.default_workdir)
                return result

            try:
                call = _action_to_bridge_call(
                    self.tool_name,
                    action,
                    self.default_workdir,
                )
            except _UnsupportedBridgeAction as exc:
                error_text = str(exc)
                result = _bridge_error_observation(
                    self.tool_name,
                    action,
                    error_text,
                    self.default_workdir,
                )
                return result

            bridged = True
            logger.info("bridge[%s] %s", self.tool_name, call.command[:120])
            res = bridge_exec_shell(
                call.command,
                workdir=call.workdir,
                timeout_ms=call.timeout_ms,
                stdin_bytes=call.stdin_bytes,
            )
            result = _shell_result_to_observation(
                self.tool_name,
                action,
                res,
                workdir=call.workdir,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            result = _bridge_error_observation(
                self.tool_name,
                action,
                error_text,
                self.default_workdir,
            )
            return result
        finally:
            _emit_tool_span(
                self.otel,
                self.tool_name,
                action,
                result,
                start_ns,
                time.time_ns(),
                bridged=bridged,
                error=error_text,
            )

    def close(self) -> None:
        close = getattr(self.original_executor, "close", None)
        if callable(close):
            close()


def _wrap_tool_executors(
    tools: Any,
    default_workdir: str,
    otel: _OtelEmitter | None = None,
) -> int:
    """Replace resolved OpenHands tool executors with bridge-routing wrappers."""
    if not bridge_active():
        return 0

    wrapped = 0
    for key, tool in _iter_tools(tools):
        name = str(getattr(tool, "name", key if key is not None else repr(tool)))
        if name.lower() not in _BRIDGE_EXEC_TOOL_NAMES:
            continue
        executor = getattr(tool, "executor", None)
        if executor is None:
            logger.warning("bridge: tool %s has no .executor; cannot wrap", name)
            continue
        if isinstance(executor, _BridgeToolExecutor):
            wrapped += 1
            continue

        wrapper = _BridgeToolExecutor(name, executor, default_workdir, otel)
        try:
            replacement = (
                tool.set_executor(wrapper)
                if callable(getattr(tool, "set_executor", None))
                else _replace_executor_in_place(tool, wrapper)
            )
            _store_tool_replacement(tools, key, replacement)
            wrapped += 1
            logger.info("bridge-wrapped tool: %s", name)
        except Exception as e:  # noqa: BLE001
            logger.warning("bridge: cannot wrap executor for tool=%s: %s", name, e)
    return wrapped


def _iter_tools(tools: Any) -> list[tuple[Any, Any]]:
    if isinstance(tools, dict):
        return list(tools.items())
    return list(enumerate(list(tools or [])))


def _store_tool_replacement(tools: Any, key: Any, replacement: Any) -> None:
    if isinstance(tools, dict):
        tools[key] = replacement
    elif isinstance(tools, list) and isinstance(key, int):
        tools[key] = replacement


def _replace_executor_in_place(tool: Any, executor: Any) -> Any:
    setattr(tool, "executor", executor)
    return tool


def _tool_by_name(tools: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, tool in _iter_tools(tools):
        name = str(getattr(tool, "name", key if key is not None else repr(tool))).lower()
        out[name] = tool
    return out


def _tool_is_bridge_wrapped(tool: Any) -> bool:
    return isinstance(getattr(tool, "executor", None), _BridgeToolExecutor)


def _missing_required_bridge_tools(tools: Any) -> list[str]:
    by_name = _tool_by_name(tools)
    missing: list[str] = []
    for name in sorted(_BRIDGE_REQUIRED_TOOL_NAMES):
        tool = by_name.get(name)
        if tool is None or not _tool_is_bridge_wrapped(tool):
            missing.append(name)
    return missing


def _bridge_wrapped_tool_names(tools: Any) -> list[str]:
    return sorted(
        name
        for name, tool in _tool_by_name(tools).items()
        if _tool_is_bridge_wrapped(tool)
    )


def _action_to_bridge_call(
    tool_name: str,
    action: Any,
    default_workdir: str,
) -> _BridgeCall:
    """Map an OpenHands SDK Action to a bridge invocation."""
    name = tool_name.lower()
    if name in ("terminal", "bash"):
        if getattr(action, "is_input", False):
            raise _UnsupportedBridgeAction(
                "terminal input to a running process is not supported by "
                "the Agent_Bench stateless exec bridge"
            )
        cmd = getattr(action, "command", None) or getattr(action, "shell_command", None)
        if not isinstance(cmd, str):
            raise _UnsupportedBridgeAction("terminal action did not contain a shell command")
        cmd = _translate_workspace_refs_to_task(cmd)
        timeout = getattr(action, "timeout", None)
        timeout_ms = None
        if timeout is not None:
            try:
                timeout_ms = max(int(float(timeout) * 1000), 1)
            except (TypeError, ValueError):
                timeout_ms = None
        return _BridgeCall(
            command=cmd,
            workdir=str(_translate_workspace_path_to_task(
                getattr(action, "workdir", None) or default_workdir
            )),
            timeout_ms=timeout_ms,
        )

    if name == "file_editor":
        raise _UnsupportedBridgeAction(
            "file_editor actions are handled by the agent-side bridge adapter"
        )

    raise _UnsupportedBridgeAction(f"tool {tool_name!r} is not bridge-routable")


def _text_content(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "cache_prompt": False}]


def _combined_output(res: Dict[str, Any]) -> str:
    out = (res.get("stdout") or "") + (res.get("stderr") or "")
    if len(out) > 65_536:
        out = out[:65_536] + "\n... [truncated by bridge driver]\n"
    return out


def _terminal_observation(
    action: Any,
    text: str,
    *,
    exit_code: int,
    workdir: str,
    is_error: bool,
) -> dict[str, Any]:
    command = getattr(action, "command", None) or getattr(action, "shell_command", None) or ""
    return {
        "content": _text_content(text),
        "is_error": bool(is_error),
        "command": command,
        "exit_code": int(exit_code),
        "timeout": False,
        "metadata": {
            "exit_code": int(exit_code),
            "pid": -1,
            "username": None,
            "hostname": None,
            "working_dir": _translate_workspace_path_to_agent(workdir),
            "py_interpreter_path": None,
            "prefix": "",
            "suffix": f"\n[The command completed with exit code {int(exit_code)}.]",
        },
        "full_output_save_dir": None,
    }


def _file_editor_observation(
    action: Any,
    text: str,
    *,
    is_error: bool,
) -> dict[str, Any]:
    command = getattr(action, "command", None) or getattr(action, "op", None) or "view"
    path = getattr(action, "path", None) or getattr(action, "file_path", None)
    return {
        "content": _text_content(text),
        "is_error": bool(is_error),
        "command": command,
        "path": _translate_workspace_path_to_agent(path),
        "prev_exist": True,
        "old_content": None,
        "new_content": None,
    }


def _restore_file_editor_observation_paths(observation: dict[str, Any]) -> dict[str, Any]:
    restored = dict(observation)
    restored["path"] = _translate_workspace_path_to_agent(restored.get("path"))
    return restored


def _bridge_error_observation(
    tool_name: str,
    action: Any,
    message: str,
    default_workdir: str,
) -> dict[str, Any]:
    if tool_name.lower() == "file_editor":
        return _file_editor_observation(action, message, is_error=True)
    return _terminal_observation(
        action,
        message,
        exit_code=125,
        workdir=default_workdir,
        is_error=True,
    )


def _shell_result_to_observation(
    tool_name: str,
    action: Any,
    res: Dict[str, Any],
    *,
    workdir: str,
) -> Any:
    """Build an observation-shaped return value the SDK can consume.

    The SDK consumes whatever object the executor returns; for tools whose
    executors return a pydantic Observation, we mirror that shape via a
    plain object. The harness reads events.jsonl, not the observation
    objects directly, so structural fidelity at the SDK boundary is more
    important than the exact pydantic class identity.
    """
    name = tool_name.lower()
    exit_code = int(res.get("exit_code", 0) or 0)
    if name == "file_editor":
        stdout = res.get("stdout") or ""
        try:
            payload = json.loads(stdout)
            if isinstance(payload, dict) and payload.get("ab_bridge_file_editor"):
                observation = payload.get("observation")
                if isinstance(observation, dict):
                    return _restore_file_editor_observation_paths(observation)
        except Exception:
            pass
        return _file_editor_observation(
            action,
            _combined_output(res),
            is_error=True,
        )

    return _terminal_observation(
        action,
        _combined_output(res),
        exit_code=exit_code,
        workdir=workdir,
        is_error=exit_code != 0,
    )


# ----------------------------------------------------------------------
# Conversation metrics extraction (token totals)
# ----------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            pass
    return repr(obj)


def _conversation_events(conv: Any) -> list:
    state = getattr(conv, "state", None)
    if state is None:
        return []
    for attr in ("events", "history"):
        v = getattr(state, attr, None)
        if v is None:
            continue
        try:
            return list(v)
        except Exception:
            continue
    return []


def _conversation_metrics(conv: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt_tokens": 0, "completion_tokens": 0,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0,
    }
    stats = getattr(conv, "conversation_stats", None)
    if stats is None:
        return out
    d = _serialize(stats) if not isinstance(stats, dict) else stats
    usage_map = d.get("usage_to_metrics") or d.get("service_to_metrics") or {}
    if isinstance(usage_map, dict):
        for metrics in usage_map.values():
            m = _serialize(metrics) if not isinstance(metrics, dict) else metrics
            usage = m.get("accumulated_token_usage") or {}
            if isinstance(usage, dict):
                out["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                out["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                out["cache_read_tokens"] += int(
                    usage.get("cache_read_tokens")
                    or usage.get("cache_read_input_tokens") or 0
                )
                out["cache_creation_tokens"] += int(
                    usage.get("cache_write_tokens")
                    or usage.get("cache_creation_tokens")
                    or usage.get("cache_creation_input_tokens") or 0
                )
            out["cost_usd"] += float(m.get("accumulated_cost") or 0.0)
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _import_sdk():
    from openhands.sdk import LLM, Conversation, LocalWorkspace  # type: ignore
    from openhands.tools import get_default_agent, get_default_tools  # type: ignore
    return {
        "LLM": LLM, "Conversation": Conversation, "LocalWorkspace": LocalWorkspace,
        "get_default_agent": get_default_agent,
        "get_default_tools": get_default_tools,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="OpenHands SDK headless driver")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--workspace", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=0)
    args = ap.parse_args(argv)

    sink = EventSink(args.events)
    try:
        config = _load_config(args.config)
    except Exception as e:
        _emit_error(sink, "config_load", f"{e}", config_path=str(args.config))
        sink.close()
        return 2

    llm_cfg = config.get("llm") or {}
    agent_cfg = config.get("agent") or {}
    run_cfg = config.get("run") or {}
    workspace = str(args.workspace or run_cfg.get("workspace") or os.getcwd())
    max_iter = int(run_cfg.get("max_iterations") or 100)
    telemetry_cfg = (config.get("telemetry") or {}).get("otel") or {}
    run_id = (
        os.environ.get("AB_RUN_ID")
        or os.environ.get("OPENHANDS_RUN_ID")
        or os.environ.get("OPENCLAW_RUN_ID")
        or str(telemetry_cfg.get("run_id") or "")
    )
    otel = _OtelEmitter(
        service_name=str(telemetry_cfg.get("service_name") or "agent-bench-openhands"),
        run_id=run_id,
        endpoint=telemetry_cfg.get("endpoint"),
        resource_attributes=telemetry_cfg.get("resource_attributes"),
        enabled=telemetry_cfg.get("enabled", True) is not False,
    )

    _emit(sink, "boot",
          workspace=workspace,
          max_iterations=max_iter,
          model=llm_cfg.get("model"),
          base_url=llm_cfg.get("base_url"),
          bridge_active=bridge_active(),
          bridge_socket=_BRIDGE_SOCKET,
          otel_endpoint=otel.url,
          ab_run_id=run_id,
          python_version=sys.version.split()[0])

    try:
        sdk = _import_sdk()
    except ImportError as e:
        _emit_error(sink, "sdk_import", str(e))
        sink.close()
        return 3

    LLM = sdk["LLM"]
    Conversation = sdk["Conversation"]
    LocalWorkspace = sdk["LocalWorkspace"]
    get_default_agent = sdk["get_default_agent"]

    try:
        llm_kwargs = {
            "model": llm_cfg.get("model"),
            "base_url": llm_cfg.get("base_url"),
            "api_key": llm_cfg.get("api_key") or "EMPTY",
            "temperature": float(llm_cfg.get("temperature") or 0.0),
        }
        if llm_cfg.get("max_output_tokens"):
            llm_kwargs["max_output_tokens"] = int(llm_cfg["max_output_tokens"])
        context_param_name = None
        if llm_cfg.get("max_input_tokens"):
            max_input_tokens = int(llm_cfg["max_input_tokens"])
            for param_name in (
                "max_input_tokens",
                "context_window",
                "max_context_tokens",
            ):
                if _accepts_kwarg(LLM, param_name):
                    llm_kwargs[param_name] = max_input_tokens
                    context_param_name = param_name
                    break
            else:
                _emit(
                    sink,
                    "llm_context_window_ignored",
                    context_window=max_input_tokens,
                    reason="unsupported_llm_parameter",
                )
        try:
            llm = LLM(**llm_kwargs)
        except TypeError as e:
            if not context_param_name:
                raise
            max_input_tokens = llm_kwargs.pop(context_param_name)
            _emit(
                sink,
                "llm_context_window_ignored",
                context_window=max_input_tokens,
                parameter=context_param_name,
                reason=f"constructor_rejected: {e}",
            )
            llm = LLM(**llm_kwargs)
    except Exception as e:
        _emit_error(sink, "llm_init", f"{e}", traceback=traceback.format_exc())
        sink.close()
        return 4

    llm = _instrument_llm_for_telemetry(llm, sink, llm_cfg, otel)

    try:
        agent = get_default_agent(llm=llm, cli_mode=True)
    except Exception as e:
        _emit_error(sink, "agent_init", f"{e}", traceback=traceback.format_exc())
        sink.close()
        return 5

    conv = None
    try:
        ws = LocalWorkspace(working_dir=workspace)
        persistence_dir = args.events.parent / "sdk-persistence"
        persistence_dir.mkdir(parents=True, exist_ok=True)
        conv = Conversation(
            agent=agent,
            workspace=ws,
            persistence_dir=str(persistence_dir),
            max_iteration_per_run=max_iter,
        )
    except Exception as e:
        _emit_error(sink, "conversation_init", f"{e}",
                    traceback=traceback.format_exc())
        sink.close()
        return 6

    try:
        if bridge_active():
            ensure_ready = getattr(conv, "_ensure_agent_ready", None)
            if not callable(ensure_ready):
                raise RuntimeError(
                    "OpenHands Conversation does not expose _ensure_agent_ready(); "
                    "cannot wrap resolved tool executors"
                )
            ensure_ready()
            tools_map = conv.agent.tools_map
            n_wrapped = _wrap_tool_executors(tools_map, workspace, otel=otel)
            missing_required = _missing_required_bridge_tools(tools_map)
            _emit(
                sink,
                "bridge_wrap",
                tools_wrapped=n_wrapped,
                total_tools=len(tools_map),
                wrapped_tools=_bridge_wrapped_tool_names(tools_map),
                required_tools=sorted(_BRIDGE_REQUIRED_TOOL_NAMES),
                missing_required_tools=missing_required,
                resolved=True,
            )
            if missing_required:
                raise RuntimeError(
                    "OpenHands exec bridge failed to wrap required tool(s): "
                    + ", ".join(missing_required)
                )
        else:
            _emit(
                sink,
                "bridge_wrap",
                tools_wrapped=0,
                total_tools=len(getattr(agent, "tools", []) or []),
                wrapped_tools=[],
                required_tools=sorted(_BRIDGE_REQUIRED_TOOL_NAMES),
                missing_required_tools=[],
                resolved=False,
            )
    except Exception as e:
        _emit_error(sink, "bridge_wrap", f"{e}", traceback=traceback.format_exc())
        try:
            conv.close()
        except Exception:
            pass
        sink.close()
        return 5

    start = time.time()
    try:
        conv.send_message(args.task)
        conv.run()
    except Exception as e:
        _emit_error(sink, "run", f"{e}", traceback=traceback.format_exc())
    elapsed = time.time() - start

    for evt in _conversation_events(conv):
        try:
            sink.write({"sdk_event": _serialize(evt)})
        except Exception as e:
            _emit_error(sink, "event_serialize", f"{e}")

    metrics = _conversation_metrics(conv)
    _emit(sink, "summary",
          elapsed_seconds=elapsed,
          metrics=metrics,
          stopped_reason=str(getattr(getattr(conv, "state", None), "agent_state", None)))
    try:
        conv.close()
    except Exception:
        pass
    sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
