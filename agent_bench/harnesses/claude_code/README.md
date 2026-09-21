# Claude Code runtime for Agent Bench

This image contains an unmodified Claude Code CLI and a LiteLLM proxy. The
Terminal-Bench harness extracts the portable CLI into the task runtime and
runs the proxy as a sidecar:

```text
Claude Code in task SIF
  -> Anthropic /v1/messages
  -> LiteLLM adapter SIF
  -> OpenAI /v1/chat/completions
  -> experiment SGLang endpoint
```

Build locally from the repository root:

```bash
bash agent_bench/harnesses/claude_code/build_sif.sh
```

The script audits the completed image and prints the upload command. The NSCC
runtime path for the pinned versions is:

```text
$SCRATCH/sif_images/claude_code_2.1.278_litellm_1.102.0.sif
```

Agent Bench never builds this image as part of a GPU benchmark. Stage 0 fails
closed with an actionable message if the uploaded SIF is absent.
