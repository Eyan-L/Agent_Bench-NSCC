from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Sequence
from urllib import error, request


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if importlib.util.find_spec("sglang") is None:
        print("sglang is not importable in this Python environment", file=sys.stderr)
        return 1
    if importlib.util.find_spec("sglang_router") is None:
        print(
            "sglang_router is not importable; install the SGLang router package before using PD disaggregation",
            file=sys.stderr,
        )
        return 1

    children: list[subprocess.Popen] = []

    def stop_children(*_: object) -> None:
        for proc in reversed(children):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 30
        for proc in reversed(children):
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    try:
        _ensure_ports_available(
            (
                ("router", args.router_host, args.router_port),
                ("prefill", args.prefill_host, args.prefill_port),
                ("decode", args.decode_host, args.decode_port),
                ("bootstrap", "0.0.0.0", args.bootstrap_port),
            )
        )

        prefill_cmd = _server_cmd(args, "prefill", args.prefill_host, args.prefill_port)
        decode_cmd = _server_cmd(args, "decode", args.decode_host, args.decode_port)
        decode_cmd.extend(["--max-running-requests", str(args.decode_max_running_requests)])

        prefill_proc = _spawn(
            "prefill",
            prefill_cmd,
            cuda_visible_devices=args.prefill_cuda_visible_devices,
            args=args,
        )
        children.append(prefill_proc)
        _wait_for_port(
            args.prefill_host,
            args.prefill_port,
            "prefill",
            args.startup_timeout,
            proc=prefill_proc,
        )

        decode_proc = _spawn(
            "decode",
            decode_cmd,
            cuda_visible_devices=args.decode_cuda_visible_devices,
            args=args,
        )
        children.append(decode_proc)
        _wait_for_port(
            args.decode_host,
            args.decode_port,
            "decode",
            args.startup_timeout,
            proc=decode_proc,
        )

        router_cmd = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            f"http://127.0.0.1:{args.prefill_port}",
            str(args.bootstrap_port),
            "--decode",
            f"http://127.0.0.1:{args.decode_port}",
            "--host",
            args.router_host,
            "--port",
            str(args.router_port),
        ]
        router_proc = _spawn("router", router_cmd, cuda_visible_devices="", args=args)
        children.append(router_proc)
        _wait_for_port(
            args.router_host,
            args.router_port,
            "router",
            args.startup_timeout,
            proc=router_proc,
        )
        if args.warmup_count > 0:
            _run_router_warmup(args)

        while True:
            for proc in children:
                code = proc.poll()
                if code is not None:
                    stop_children()
                    print(f"sglang PD child exited: pid={proc.pid} code={code}", file=sys.stderr)
                    return code or 1
            time.sleep(1)
    finally:
        stop_children()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SGLang PD disaggregation processes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--router-host", default="0.0.0.0")
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--prefill-host", default="0.0.0.0")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--prefill-cuda-visible-devices", default="0")
    parser.add_argument("--decode-host", default="0.0.0.0")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--decode-cuda-visible-devices", default="1")
    parser.add_argument("--transfer-backend", default="mooncake")
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--decode-max-running-requests", type=int, default=128)
    parser.add_argument("--mooncake-custom-mem-pool")
    parser.add_argument("--server-flag", action="append", default=[])
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--warmup-timeout", type=int, default=180)
    parser.add_argument("--warmup-max-tokens", type=int, default=16)
    parser.add_argument(
        "--warmup-prompt",
        default="Warm up the serving engine. Reply with OK.",
    )
    return parser.parse_args(argv)


def _server_cmd(args: argparse.Namespace, mode: str, host: str, port: int) -> list[str]:
    flags = _without_flags(args.server_flag, {"--model-path", "--host", "--port"})
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--disaggregation-mode",
        mode,
        "--disaggregation-transfer-backend",
        args.transfer_backend,
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
        *flags,
    ]


def _without_flags(flags: Sequence[str], option_names: set[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for item in flags:
        if skip_next:
            skip_next = False
            continue
        if item in option_names:
            skip_next = True
            continue
        out.append(item)
    return out


def _spawn(
    label: str,
    cmd: Sequence[str],
    *,
    cuda_visible_devices: str,
    args: argparse.Namespace,
) -> subprocess.Popen:
    env = os.environ.copy()
    python_bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if args.mooncake_custom_mem_pool:
        env["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = args.mooncake_custom_mem_pool
    print(f"[sglang-pd] starting {label}: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env)


def _ensure_ports_available(ports: Sequence[tuple[str, str, int]]) -> None:
    seen: dict[int, str] = {}
    for label, host, port in ports:
        previous = seen.get(port)
        if previous is not None:
            raise RuntimeError(
                f"sglang PD port {port} is assigned to both {previous} and {label}; "
                "choose distinct SGLANG_*_PORT values"
            )
        seen[port] = label
        _assert_port_available(host, port, label)


def _assert_port_available(host: str, port: int, label: str) -> None:
    bind_host = "0.0.0.0" if host in {"", "0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in bind_host and bind_host != "0.0.0.0" else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"sglang PD {label} port {host}:{port} is already in use; "
            "stop stale SGLang serving processes or set distinct SGLANG_*_PORT values"
        ) from exc


def _wait_for_port(
    host: str,
    port: int,
    label: str,
    timeout: int = 600,
    *,
    proc: subprocess.Popen | None = None,
) -> None:
    target_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"{label} exited before listening on {target_host}:{port}: "
                    f"pid={proc.pid} code={code}"
                )
        try:
            with socket.create_connection((target_host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"{label} did not listen on {target_host}:{port} within {timeout}s")


def _run_router_warmup(args: argparse.Namespace) -> None:
    url = f"http://127.0.0.1:{args.router_port}/v1/chat/completions"
    payload = {
        "model": args.model_path,
        "messages": [{"role": "user", "content": args.warmup_prompt}],
        "temperature": 0.0,
        "max_tokens": args.warmup_max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    print(
        f"[sglang-pd] warming router with {args.warmup_count} chat request(s): {url}",
        flush=True,
    )
    for idx in range(1, args.warmup_count + 1):
        started = time.time()
        deadline = started + args.warmup_timeout
        last_error = ""
        while True:
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                remaining = max(1.0, deadline - time.time())
                with request.urlopen(req, timeout=min(30, remaining)) as res:
                    status = res.status
                    response_body = res.read(4096).decode("utf-8", errors="replace")
                if 200 <= status < 300:
                    break
                last_error = f"status={status} body={response_body[:500]}"
            except error.HTTPError as exc:
                response_body = exc.read(4096).decode("utf-8", errors="replace")
                last_error = f"status={exc.code} body={response_body[:500]}"
                if exc.code < 500:
                    raise RuntimeError(
                        f"router warmup {idx}/{args.warmup_count} failed: {last_error}"
                    ) from exc
            except error.URLError as exc:
                last_error = str(exc)
            if time.time() >= deadline:
                raise RuntimeError(
                    f"router warmup {idx}/{args.warmup_count} failed after "
                    f"{args.warmup_timeout}s: {last_error}"
                )
            print(
                f"[sglang-pd] warmup {idx}/{args.warmup_count} waiting for "
                f"healthy PD workers: {last_error[:200]}",
                flush=True,
            )
            time.sleep(2)
        elapsed_ms = (time.time() - started) * 1000
        print(
            f"[sglang-pd] warmup {idx}/{args.warmup_count} ok "
            f"status={status} elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
