# stress: simulated trace replay

Go-based load generator. Replays recorded / synthetic LLM request traces
against an OpenAI-compatible endpoint (vLLM or sglang).

## Module name vs directory

Go module is `loadgen` (lifted from Agent_Bench). Directory `stress/` is the
platform-level slot (paired with `eval/`). Internal Go imports keep the
`loadgen/...` path prefix.

## Workloads

| Workload      | Go file                  | Sample config                      |
|---------------|--------------------------|------------------------------------|
| code_agent    | `loadgen/agent.go`       | `configs/stress/agent.json`        |
| doc_analysis  | `loadgen/alibaba.go`     | `configs/stress/alibaba.json`      |
| websearch     | `loadgen/azure.go`       | `configs/stress/azure.json`        |

## Run

```bash
SERVING_BACKEND=sglang bash backend/common/serve.sh &
./stress/loadgen-bin agent --config configs/stress/agent.json
```

The endpoint URL is derived via `backend/common/endpoint.py` (or the
`API_BASE` field in the JSON config — keep them in sync).
