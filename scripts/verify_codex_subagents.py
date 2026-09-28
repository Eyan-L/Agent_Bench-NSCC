#!/usr/bin/env python3
"""Exercise the official CLI with a scripted model, without changing Codex.

The child runs a four-second shell command. The parent writes a marker before
waiting. Native rollouts prove whether the parent worked during the child turn.
This is a protocol/scheduling test, not a model-quality benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_bench.harnesses.codex.responses_adapter import AdapterServer
from agent_bench.harnesses.codex.runtime import write_config
from agent_bench.harnesses.codex.telemetry import summarize


def shell_call(tools, command):
    for tool in tools:
        function = tool["function"]
        name = function["name"].split("__")[-1]
        if name not in {"exec_command", "shell_command", "shell"}:
            continue
        properties = function["parameters"]["properties"]
        if "cmd" in properties:
            arguments = {"cmd": command, "yield_time_ms": 10000, "max_output_tokens": 100}
        elif properties.get("command", {}).get("type") == "array":
            arguments = {"command": ["sh", "-c", command], "timeout_ms": 10000}
        else:
            arguments = {"command": command, "timeout_ms": 10000}
        return function["name"], arguments
    raise RuntimeError("Codex did not expose a native shell tool")


class ScriptedModel(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tools, messages = request.get("tools") or [], request["messages"]
        root = any(m["role"] == "user" and "CODEX_ASYNC_PARENT" in (m.get("content") or "") for m in messages)
        outputs = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        selected = arguments = call_id = None
        answer = "DONE"
        if not root:
            self.server.child_started.set()
            if "probe_child_work" not in outputs:
                selected, arguments = shell_call(tools, "sleep 4 && date +%s%N > child-work.marker")
                call_id = "probe_child_work"
        elif "probe_spawn" not in outputs:
            selected = next(t["function"]["name"] for t in tools if t["function"]["name"].endswith("__spawn_agent"))
            arguments = {"message": "CODEX_ASYNC_CHILD: Run your independent check.", "fork_context": False}
            call_id = "probe_spawn"
        elif "probe_parent_work" not in outputs:
            if not self.server.child_started.wait(10):
                raise RuntimeError("Child did not start after native spawn")
            selected, arguments = shell_call(tools, "date +%s%N > parent-work.marker")
            call_id = "probe_parent_work"
        elif "probe_wait" not in outputs:
            child_id = re.search(r'"agent_id"\s*:\s*"([^"]+)"', outputs["probe_spawn"])
            if not child_id:
                raise RuntimeError("Native spawn did not return an agent ID: " + outputs["probe_spawn"])
            selected = next(t["function"]["name"] for t in tools if t["function"]["name"].endswith("__wait_agent"))
            arguments = {"targets": [child_id.group(1)], "timeout_ms": 10000}
            call_id = "probe_wait"
        message = {"role": "assistant", "content": answer if not selected else None}
        if selected:
            message["tool_calls"] = [{"id": call_id, "type": "function", "function": {"name": selected, "arguments": json.dumps(arguments)}}]
        reply = {"id": "probe_" + str(time.time_ns()), "choices": [{"index": 0, "message": message,
                  "finish_reason": "tool_calls" if selected else "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sif", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    home, workspace = root / "codex_state", root / "work"
    workspace.mkdir()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModel)
    upstream.child_started = threading.Event()
    config = {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1", "token": "probe-only",
              "model": "Qwen/Qwen3.6-35B-A3B", "upstream_api": "chat_completions", "max_tokens": 4096}
    adapter = AdapterServer(("127.0.0.1", 0), config, root)
    for server in (upstream, adapter):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    write_config(home, model=config["model"], base_url=f"http://127.0.0.1:{adapter.server_port}/v1",
                 context_window=262144, settings={"max_subagents": 16})
    env = dict(os.environ, SINGULARITYENV_CODEX_HOME="/opt/probe/codex_state", SINGULARITYENV_AB_CODEX_API_KEY="probe-only")
    command = ["singularity", "exec", "--bind", f"{root}:/opt/probe", args.runtime_sif,
               "/opt/codex/bin/codex", "exec", "--strict-config", "--skip-git-repo-check", "--ignore-rules",
               "--json", "--color", "never", "--cd", "/opt/probe/work", "-"]
    try:
        result = subprocess.run(command, input="CODEX_ASYNC_PARENT: Spawn a child, continue independent work, then wait and finish.",
                                env=env, text=True, capture_output=True, timeout=90)
    finally:
        for server in (adapter, upstream):
            server.shutdown()
            server.server_close()
    (root / "codex_stdout.jsonl").write_text(result.stdout)
    (root / "codex_stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Codex exited {result.returncode}: {result.stderr[-3000:]}\n{result.stdout[-1000:]}")
    sessions, obs = summarize(root)
    parent_marker = int((workspace / "parent-work.marker").read_text())
    child_marker = int((workspace / "child-work.marker").read_text())
    assert len(sessions) == 2 and obs["child_runtime_unknown_count"] == 0, obs
    assert parent_marker < child_marker, "Parent waited for the child before executing its own command"
    assert obs["parent_overlapping_tool_calls"] > 0, obs
    summary = {"cli_version": sessions[0]["cli_version"], "model": "scripted scheduling test",
               "nonblocking_spawn_verified": True, "parent_work_before_child_finish_seconds": (child_marker-parent_marker)/1e9,
               "observability": obs}
    (root / "probe_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
