# code_agent stress workload (simulated)

Multi-turn agent trace replay against an OpenAI-compatible endpoint.

- Go implementation: `stress/loadgen/agent.go` (`AgentLoadGen`)
- Trace synthesizer: `utils/synthetic_code_trace_generator.py`
- Sample config: `configs/stress/agent.json` (copied from Agent_Bench)
- Sample agent_dirs input: `configs/stress/agent_dirs.example.json`

Invoke via:

```bash
loadgen-bin agent --config configs/stress/agent.json \
  --agent-dirs-json configs/stress/agent_dirs.example.json
```
