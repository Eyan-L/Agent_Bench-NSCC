# Claude Code runtime for Agent Bench

The image contains unmodified Claude Code **2.1.278** and a LiteLLM **1.102.0**
proxy. `profiles/harnesses/claude_code_terminal_bench.yaml` selects the reusable
Terminal-Bench harness. It extracts the native CLI into the task runtime and
runs the API adapter as a sidecar:

```text
Claude Code in the task container
  -> Anthropic /v1/messages
  -> LiteLLM sidecar
  -> OpenAI-compatible /v1/chat/completions
  -> experiment model endpoint
```

Build the Docker image or a portable SIF from the repository root:

```bash
bash agent_bench/harnesses/claude_code/docker/build.sh
bash agent_bench/harnesses/claude_code/build_sif.sh
```

The SIF builder audits the CLI payload and proxy dependencies before replacing
the output. Set `AB_AGENT_RUNTIME_SIF` to the installed runtime for Singularity
deployments. Image building happens on the build host; benchmark preparation
reports a missing runtime before starting the agent.

The Terminal-Bench persona requires native Agent-tool delegation. CLI events
and materialized child sessions are retained as execution evidence. Native
completion and isolated verifier reward are recorded separately.
