# OpenHands harness image

Pinned to `openhands-sdk==1.21.0` + `openhands-tools==1.21.0`. Built from
`Dockerfile` in this directory using `build.sh`.

## Build

```bash
bash agent_bench/harnesses/openhands/docker/build.sh
```

## Smoke

```bash
bash agent_bench/harnesses/openhands/docker/smoke.sh
```

Verifies the image imports the SDK and the driver script is executable.

## Runtime model

The OpenHands harness container is spawned by `SWEBenchOpenHandsHarness`
(`agent_bench/benchmarks/swebench/app.py`). It bind-mounts:

| host path                                       | in-container path                        |
|-------------------------------------------------|------------------------------------------|
| `<state_dir>` (per-run, rendered from harness JSON) | `/etc/agent-bench/state` (ro)        |
| `<workspace_host>` (primed from sandbox image)  | `/workspace` (rw)                        |
| `<dispatcher.bridge_script>`                    | `/usr/local/bin/oh-exec-bridge` (ro)     |

And sets:

| env                              | value                                              |
|----------------------------------|----------------------------------------------------|
| `OPENHANDS_EXEC_BRIDGE`          | `/usr/local/bin/oh-exec-bridge`                    |
| `OPENHANDS_EXEC_BRIDGE_SOCKET`   | `/etc/agent-bench/state/exec-bridge.sock`          |
| `OPENHANDS_EXEC_BRIDGE_TOKEN`    | `<random>`                                         |
| `RUNTIME`                        | `process` (disables SDK's built-in DockerRuntime)  |

The driver (`/opt/openhands_driver.py`) reads those env vars on startup and
re-wires the SDK's BashTool / TerminalTool / FileEditorTool executors to
shell out to `oh-exec-bridge` instead of running locally. Tool execution
lands in the SWE-bench sandbox container over the unix socket.
