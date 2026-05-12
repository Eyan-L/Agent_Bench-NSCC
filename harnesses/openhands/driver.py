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

    --config   <path>        TOML rendered from config/harness/openhands_*.json
    --task     <message>     initial user message
    --events   <jsonl>       NDJSON event log written here
    --workspace <dir>        agent's workspace (in-container path)

Tolerant to SDK shape drift: tries the most natural interception point
(replacing the tool's `executor`) and falls back to a wrapper Tool subclass
if the executor attribute isn't writable. Errors during tool wrapping are
logged but never raise — a partial bridge is better than no agent.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
import traceback
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


# ----------------------------------------------------------------------
# Bridge invocation
# ----------------------------------------------------------------------


_BRIDGE_BIN = os.environ.get("OPENHANDS_EXEC_BRIDGE", "")
_BRIDGE_SOCKET = os.environ.get("OPENHANDS_EXEC_BRIDGE_SOCKET", "")
_BRIDGE_TOKEN = os.environ.get("OPENHANDS_EXEC_BRIDGE_TOKEN", "")


def bridge_active() -> bool:
    return bool(_BRIDGE_BIN and _BRIDGE_SOCKET and _BRIDGE_TOKEN)


def bridge_exec_shell(
    command: str,
    *,
    workdir: str = "/testbed",
    env: Optional[Dict[str, str]] = None,
    timeout_ms: Optional[int] = None,
    stdin_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Send a shell command via the bridge CLI; return {stdout, stderr, exit}."""
    if not bridge_active():
        raise RuntimeError("bridge_exec_shell called but bridge env not set")
    argv = [
        _BRIDGE_BIN,
        "--workdir", workdir,
        "--env-json", json.dumps(env or {}),
        "--command", command,
    ]
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
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
    }


# ----------------------------------------------------------------------
# SDK tool executor wrapping
# ----------------------------------------------------------------------


def _wrap_tool_executors(tools: List[Any], default_workdir: str) -> int:
    """Replace each tool's executor with a bridge-routing wrapper.

    Returns the number of tools we successfully wrapped. Failures are logged
    and skipped; an agent with a partial wrap still runs (it just executes
    some tools locally — the harness will see those in the event log).
    """
    if not bridge_active():
        return 0

    wrapped = 0
    for tool in tools:
        name = getattr(tool, "name", repr(tool))
        executor = getattr(tool, "executor", None)
        if executor is None:
            logger.debug("tool %s has no .executor; skipping wrap", name)
            continue
        original_call = getattr(executor, "__call__", None)
        if original_call is None:
            logger.debug("tool %s executor not callable; skipping wrap", name)
            continue

        # Bind name into closure so the wrapper can identify itself.
        def _make_wrapper(tool_name: str, orig):
            def _bridged(action):
                """Route a tool action through the exec-bridge.

                We can't anticipate every Action schema across openhands-tools
                versions, so we map a few well-known shapes (terminal /
                file_editor) explicitly and fall back to invoking the
                original executor (in-process) when we don't recognise the
                action — that fallback runs in the OH harness container,
                visible in trace as un-bridged.
                """
                try:
                    cmd = _action_to_shell(tool_name, action)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "bridge: could not convert action for tool=%s (%s); "
                        "falling back to original executor", tool_name, e,
                    )
                    return orig(action)

                if cmd is None:
                    return orig(action)

                workdir = getattr(action, "workdir", None) or default_workdir
                logger.info("bridge[%s] %s", tool_name, cmd[:120])
                res = bridge_exec_shell(cmd, workdir=workdir)
                return _shell_result_to_observation(tool_name, action, res)
            return _bridged

        try:
            executor.__call__ = _make_wrapper(name, original_call)
            wrapped += 1
            logger.info("bridge-wrapped tool: %s", name)
        except (AttributeError, TypeError):
            # Some pydantic-based executors are frozen — fall back to a
            # subclass-style override.
            try:
                executor.__class__.__call__ = _make_wrapper(name, original_call)
                wrapped += 1
                logger.info("bridge-wrapped tool (class-level): %s", name)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "bridge: cannot wrap executor for tool=%s: %s", name, e,
                )
    return wrapped


def _action_to_shell(tool_name: str, action: Any) -> Optional[str]:
    """Best-effort mapping of an SDK Action to a shell command string.

    Returns None when we don't know how to bridge this tool — the caller
    will fall back to the in-process executor.
    """
    name = tool_name.lower()
    # TerminalTool / BashTool — the action carries a `command` field.
    if name in ("terminal", "bash"):
        cmd = getattr(action, "command", None) or getattr(action, "shell_command", None)
        if isinstance(cmd, str):
            return cmd

    # FileEditorTool — multiple ops (view/create/str_replace/insert/undo_edit).
    if name == "file_editor":
        op = getattr(action, "command", None) or getattr(action, "op", None)
        path = getattr(action, "path", None) or getattr(action, "file_path", None)
        if not path:
            return None
        q = lambda s: "'" + str(s).replace("'", "'\\''") + "'"
        if op == "view":
            view_range = getattr(action, "view_range", None)
            if view_range and len(view_range) == 2:
                a, b = view_range
                return f"sed -n '{a},{b}p' {q(path)}"
            return f"cat -n {q(path)}"
        if op == "create":
            text = getattr(action, "file_text", "") or ""
            return f"cat > {q(path)} <<'__OH_EOF__'\n{text}\n__OH_EOF__"
        # str_replace / insert / undo_edit are best handled inline by SDK
        # to avoid byte-perfect newline issues; fall through to in-process.

    return None


def _shell_result_to_observation(
    tool_name: str, action: Any, res: Dict[str, Any]
) -> Any:
    """Build an observation-shaped return value the SDK can consume.

    The SDK consumes whatever object the executor returns; for tools whose
    executors return a pydantic Observation, we mirror that shape via a
    plain object. The harness reads events.jsonl, not the observation
    objects directly, so structural fidelity at the SDK boundary is more
    important than the exact pydantic class identity.
    """
    out = (res.get("stdout") or "") + (res.get("stderr") or "")
    # Truncate huge outputs to keep trajectory size bounded.
    if len(out) > 65_536:
        out = out[:65_536] + "\n... [truncated by bridge driver]\n"
    return {
        "tool_name": tool_name,
        "exit_code": res.get("exit_code", 0),
        "content": out,
        "is_error": res.get("exit_code", 0) != 0,
    }


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

    _emit(sink, "boot",
          workspace=workspace,
          max_iterations=max_iter,
          model=llm_cfg.get("model"),
          base_url=llm_cfg.get("base_url"),
          bridge_active=bridge_active(),
          bridge_socket=_BRIDGE_SOCKET,
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
        llm = LLM(**llm_kwargs)
    except Exception as e:
        _emit_error(sink, "llm_init", f"{e}", traceback=traceback.format_exc())
        sink.close()
        return 4

    try:
        agent = get_default_agent(llm=llm, cli_mode=True)
        n_wrapped = _wrap_tool_executors(list(agent.tools), workspace)
        _emit(sink, "bridge_wrap",
              tools_wrapped=n_wrapped,
              total_tools=len(agent.tools))
    except Exception as e:
        _emit_error(sink, "agent_init", f"{e}", traceback=traceback.format_exc())
        sink.close()
        return 5

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
