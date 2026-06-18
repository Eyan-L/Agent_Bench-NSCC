#!/usr/bin/env python3
"""OpenClaw exec bridge — runs INSIDE the OC container.

Two callers patch in via env var OPENCLAW_EXEC_BRIDGE:

1. **bash-tools.exec-runtime** (the agent's `exec` tool). Invoked as:
       bridge --workdir <wd> --env-json <j> --command <shell-cmd>
   The shell command is wrapped in /bin/sh -lc and sent to the server.

2. **fs-bridge runCommand / readPinnedFile** (the agent's read/write/edit
   tools). Invoked as:
       bridge --workdir <wd> --env-json <j> --argv-json <argv> [--no-tty]
   With raw argv (typically `["sh","-c","<plan-script>","openclaw-sandbox-fs",...]`).
   Binary stdin from our own stdin is forwarded; binary stdout/stderr from
   server is replayed to our own stdio.

Protocol mirrors openclaw/infra/exec-host.ts (HMAC-SHA256 + JSONL). Wire
fields use base64 for stdin/stdout/stderr so binary payloads survive JSON.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import sys
import time
import uuid


def _sign(token: str, nonce: str, ts: int, request_json: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        f"{nonce}:{ts}:{request_json}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _send_and_recv(socket_path: str, frame: dict, timeout_s: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(socket_path)
    payload = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
    s.sendall(payload)
    buf = bytearray()
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in chunk:
            break
    s.close()
    line, _, _ = bytes(buf).partition(b"\n")
    return json.loads(line.decode("utf-8"))


def _read_stdin_bytes() -> bytes:
    """Drain our stdin into bytes for forwarding. Returns b'' if nothing piped."""
    if sys.stdin.isatty():
        return b""
    try:
        return sys.stdin.buffer.read()
    except Exception:
        return b""


def _resolve_log_path() -> str:
    explicit = os.environ.get("OPENCLAW_EXEC_BRIDGE_LOG_PATH")
    if explicit:
        return explicit
    log_dir = os.environ.get("OPENCLAW_EXEC_BRIDGE_LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, "openclaw_exec_bridge.jsonl")
    return "/root/.openclaw/logs/openclaw_exec_bridge.jsonl"


def _append_log_entry(log_path: str, entry: dict) -> None:
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--env-json", default="{}")
    ap.add_argument("--command", default=None,
                    help="Shell command string (wrapped in /bin/sh -lc). "
                         "Mutually exclusive with --argv-json.")
    ap.add_argument("--argv-json", default=None,
                    help="JSON-encoded raw argv array (e.g. for fs-bridge "
                         "scripts run as `sh -c ...`).")
    ap.add_argument("--timeout-ms", type=int, default=0)
    args = ap.parse_args()

    if (args.command is None) == (args.argv_json is None):
        sys.stderr.write(
            "oc-exec-bridge: exactly one of --command / --argv-json required\n"
        )
        return 2

    socket_path = os.environ.get("OPENCLAW_EXEC_BRIDGE_SOCKET", "")
    token = os.environ.get("OPENCLAW_EXEC_BRIDGE_TOKEN", "")
    if not socket_path or not token:
        sys.stderr.write(
            "oc-exec-bridge: OPENCLAW_EXEC_BRIDGE_SOCKET / _TOKEN must be set\n"
        )
        return 127

    try:
        env_map = json.loads(args.env_json)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"oc-exec-bridge: invalid --env-json: {e}\n")
        return 2

    if args.argv_json is not None:
        try:
            command = json.loads(args.argv_json)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"oc-exec-bridge: invalid --argv-json: {e}\n")
            return 2
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            sys.stderr.write("oc-exec-bridge: --argv-json must be a list of strings\n")
            return 2
    else:
        command = ["/bin/sh", "-lc", args.command]

    stdin_bytes = _read_stdin_bytes()
    log_path = _resolve_log_path()

    request = {
        "command": command,
        "cwd": args.workdir,
        "env": env_map,
        "timeoutMs": args.timeout_ms or None,
        "agentId": os.environ.get("OPENCLAW_AGENT_ID"),
        "sessionKey": os.environ.get("OPENCLAW_SESSION_KEY"),
        "stdinB64": base64.b64encode(stdin_bytes).decode("ascii") if stdin_bytes else None,
    }
    request_json = json.dumps(request, separators=(",", ":"))
    nonce = secrets.token_hex(16)
    ts = int(time.time() * 1000)
    frame = {
        "type": "exec",
        "id": str(uuid.uuid4()),
        "nonce": nonce,
        "ts": ts,
        "hmac": _sign(token, nonce, ts, request_json),
        "requestJson": request_json,
    }

    sock_timeout = (args.timeout_ms / 1000.0 + 30) if args.timeout_ms else 3600.0
    send_start = time.monotonic()
    try:
        response = _send_and_recv(socket_path, frame, sock_timeout)
        elapsed_ms = int((time.monotonic() - send_start) * 1000)
    except Exception as e:
        elapsed_ms = int((time.monotonic() - send_start) * 1000)
        _append_log_entry(log_path, {
            "ts": int(time.time() * 1000),
            "elapsedMs": elapsed_ms,
            "commandName": command[0] if command else None,
            "command": command,
            "cwd": args.workdir,
            "stdout": "",
            "stderr": "",
            "stdin": stdin_bytes.decode("utf-8", errors="replace"),
            "stdoutSize": 0,
            "stderrSize": 0,
            "stdinSize": len(stdin_bytes),
            "ok": False,
            "error": {"message": str(e)},
            "agentId": request.get("agentId"),
            "sessionKey": request.get("sessionKey"),
        })
        sys.stderr.write(f"oc-exec-bridge: socket call failed: {e}\n")
        return 125

    if response.get("type") != "exec-res":
        _append_log_entry(log_path, {
            "ts": int(time.time() * 1000),
            "elapsedMs": elapsed_ms,
            "commandName": command[0] if command else None,
            "command": command,
            "cwd": args.workdir,
            "stdout": "",
            "stderr": "",
            "stdin": stdin_bytes.decode("utf-8", errors="replace"),
            "stdoutSize": 0,
            "stderrSize": 0,
            "stdinSize": len(stdin_bytes),
            "ok": False,
            "error": {"message": f"bad response type: {response.get('type')!r}"},
            "agentId": request.get("agentId"),
            "sessionKey": request.get("sessionKey"),
        })
        sys.stderr.write(f"oc-exec-bridge: bad response type: {response.get('type')!r}\n")
        return 125
    if not response.get("ok"):
        err = response.get("error", {})
        _append_log_entry(log_path, {
            "ts": int(time.time() * 1000),
            "elapsedMs": elapsed_ms,
            "commandName": command[0] if command else None,
            "command": command,
            "cwd": args.workdir,
            "stdout": "",
            "stderr": "",
            "stdin": stdin_bytes.decode("utf-8", errors="replace"),
            "stdoutSize": 0,
            "stderrSize": 0,
            "stdinSize": len(stdin_bytes),
            "ok": False,
            "error": {"code": err.get("code"), "message": err.get("message")},
            "agentId": request.get("agentId"),
            "sessionKey": request.get("sessionKey"),
        })
        sys.stderr.write(
            f"oc-exec-bridge: server error "
            f"[{err.get('code', '?')}] {err.get('message', '?')}\n"
        )
        return 126
    payload = response.get("payload", {})
    # New base64 fields take precedence over legacy text fields. Binary safety
    # matters for fs-bridge `cat <file>` of arbitrary content.
    stdout_b64 = payload.get("stdoutB64")
    stderr_b64 = payload.get("stderrB64")
    if stdout_b64 is not None:
        stdout_bytes = base64.b64decode(stdout_b64)
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        sys.stdout.buffer.write(stdout_bytes)
    else:
        stdout_text = payload.get("stdout", "")
        stdout_bytes = stdout_text.encode("utf-8")
        sys.stdout.write(stdout_text)
    sys.stdout.flush()
    if stderr_b64 is not None:
        stderr_bytes = base64.b64decode(stderr_b64)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        sys.stderr.buffer.write(stderr_bytes)
    else:
        stderr_text = payload.get("stderr", "")
        stderr_bytes = stderr_text.encode("utf-8")
        sys.stderr.write(stderr_text)
    sys.stderr.flush()
    _append_log_entry(log_path, {
        "ts": int(time.time() * 1000),
        "elapsedMs": elapsed_ms,
        "commandName": command[0] if command else None,
        "command": command,
        "cwd": args.workdir,
        "exitCode": payload.get("exitCode"),
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdin": stdin_bytes.decode("utf-8", errors="replace"),
        "stdoutSize": len(stdout_bytes),
        "stderrSize": len(stderr_bytes),
        "stdinSize": len(stdin_bytes),
        "ok": True,
        "agentId": request.get("agentId"),
        "sessionKey": request.get("sessionKey"),
    })
    return int(payload.get("exitCode", 0))


if __name__ == "__main__":
    sys.exit(main())