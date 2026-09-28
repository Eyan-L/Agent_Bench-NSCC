"""Concurrent Responses/Chat protocol adapter; Codex owns all orchestration."""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def text(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    parts = []
    for part in content:
        if part.get("type") not in {"input_text", "output_text", "text", "summary_text"}:
            raise ValueError(f"Unsupported content: {part.get('type')}")
        parts.append(str(part.get("text") or ""))
    return "\n".join(parts)


def flatten_tools(tools):
    flat, identities = [], {}
    def add(tool, namespace=None):
        if tool["type"] == "namespace":
            for nested in tool["tools"]:
                add(nested, tool["name"])
            return
        if tool["type"] not in {"function", "custom"}:
            raise ValueError(f"Unsupported tool: {tool['type']}")
        name = (namespace + "__" if namespace else "") + tool["name"]
        if name in identities:
            raise ValueError(f"Ambiguous tool: {name}")
        identities[name] = {"name": tool["name"], "namespace": namespace, "custom": tool["type"] == "custom"}
        parameters = tool.get("parameters") or {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"], "additionalProperties": False}
        description = tool.get("description", "")
        if tool["type"] == "custom":
            description += "\nSupply the complete tool input as the input string."
        flat.append({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
    for tool in tools:
        add(tool)
    return flat, identities


def to_chat(request, config):
    if request.get("previous_response_id"):
        raise ValueError("Stateless adapter requires complete input history")
    tools, identities = flatten_tools(request.get("tools") or [])
    messages = []
    if request.get("instructions"):
        messages.append({"role": "system", "content": request["instructions"]})
    inputs = request.get("input") or []
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    for item in inputs:
        kind = item.get("type", "message")
        if kind == "message":
            messages.append({"role": "system" if item["role"] == "developer" else item["role"], "content": text(item.get("content"))})
        elif kind in {"function_call", "custom_tool_call"}:
            name = (item.get("namespace") + "__" if item.get("namespace") else "") + item["name"]
            args = item.get("arguments", "{}") if kind == "function_call" else json.dumps({"input": item["input"]})
            call = {"id": item["call_id"], "type": "function", "function": {"name": name, "arguments": args}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": text(item.get("output"))})
        elif kind == "reasoning":
            if item.get("summary"):
                messages.append({"role": "assistant", "content": text(item["summary"])})
        else:
            raise ValueError(f"Unsupported input item: {kind}")
    # Codex can insert developer messages between turns. Qwen's chat template
    # requires its system block first. Keep all high-priority text in original
    # order in that block and retain the order of user/assistant/tool history.
    system = [m["content"] for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]
    if system:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
    chat = {"model": config.get("model") or request["model"], "messages": messages, "stream": False,
            "temperature": config.get("temperature", 0.0), "max_tokens": config.get("max_tokens", 16384)}
    if tools:
        chat.update(tools=tools, tool_choice="auto")
    return chat, identities


def from_chat(chat, identities, model):
    choice = chat["choices"][0]
    if choice.get("finish_reason") not in {"stop", "tool_calls", "length"}:
        raise ValueError(f"Unsupported finish reason: {choice.get('finish_reason')}")
    message, output = choice["message"], []
    if message.get("content"):
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex, "role": "assistant", "status": "completed",
                       "phase": "commentary" if message.get("tool_calls") else "final_answer",
                       "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    for call in message.get("tool_calls") or []:
        function = call["function"]
        if function["name"] not in identities:
            raise ValueError(f"Unknown returned tool: {function['name']}")
        identity = identities[function["name"]]
        item = {"type": "function_call", "id": "fc_" + uuid.uuid4().hex, "call_id": call["id"], "name": identity["name"], "arguments": function["arguments"], "status": "completed"}
        if identity["namespace"]:
            item["namespace"] = identity["namespace"]
        if identity["custom"]:
            item["type"] = "custom_tool_call"
            item["input"] = json.loads(item.pop("arguments"))["input"]
        output.append(item)
    usage = chat.get("usage") or {}
    prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    truncated = choice.get("finish_reason") == "length"
    return {"id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()), "model": model,
            "status": "incomplete" if truncated else "completed", "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "error": None, "output": output, "usage": {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion,
            "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}, "output_tokens_details": {"reasoning_tokens": 0}}}


def response_events(response):
    yield {"type": "response.created", "response": dict(response, status="in_progress", output=[], usage=None)}
    for index, item in enumerate(response["output"]):
        yield {"type": "response.output_item.added", "output_index": index, "item": item}
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    yield {"type": "response.completed", "response": response}


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, config, state):
        super().__init__(address, Handler)
        self.config, self.state, self.log_lock = config, Path(state), threading.Lock()
        self.state.mkdir(parents=True, exist_ok=True)

    def record(self, row):
        with self.log_lock, (self.state / "codex_api.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self):
        config = self.server.config
        if self.headers.get("Authorization") != "Bearer " + config["token"]:
            self._json(401, {"error": "invalid adapter token"})
            return
        if self.path not in {"/v1/responses", "/responses"}:
            self._json(404, {"error": {"message": "unsupported endpoint"}})
            return
        row = {"start_ns": time.time_ns(), "thread_id": self.headers.get("thread-id"), "session_id": self.headers.get("session-id")}
        streaming = False
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            request = json.loads(raw)
            row["request"] = request
            self.server.record(dict(row, event="request_start"))
            mode = config.get("upstream_api", "chat_completions")
            if mode == "responses":
                upstream_request, suffix = request, "/responses"
            elif mode == "chat_completions":
                upstream_request, identities = to_chat(request, config)
                suffix = "/chat/completions"
            else:
                raise ValueError(f"Unknown upstream_api: {mode}")
            row["upstream_request"] = upstream_request
            req = urllib.request.Request(config["base_url"].rstrip("/") + suffix, data=json.dumps(upstream_request).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + config.get("api_key", "EMPTY")})
            with urllib.request.urlopen(req, timeout=config.get("timeout", 600)) as reply:
                if mode == "responses" and request.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    streaming = True
                    chunks = []
                    for line in reply:
                        chunks.append(line)
                        self.wfile.write(line)
                        self.wfile.flush()
                    row["response_sse"] = b"".join(chunks).decode(errors="replace")
                    return
                upstream_reply = json.load(reply)
            row["upstream_response"] = upstream_reply
            response = from_chat(upstream_reply, identities, request["model"]) if mode == "chat_completions" else upstream_reply
            row["response"] = response
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                streaming = True
                for sequence, event in enumerate(response_events(response)):
                    event["sequence_number"] = sequence
                    self.wfile.write(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            else:
                self._json(200, response)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            detail = exc.read().decode(errors="replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
            row["error"] = detail
            if not streaming:
                self._json(502, {"error": {"message": detail, "type": "adapter_error"}})
        finally:
            row["end_ns"] = time.time_ns()
            self.server.record(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    AdapterServer(("0.0.0.0", args.port), json.loads(Path(args.config).read_text()), args.state).serve_forever()


if __name__ == "__main__":
    main()
