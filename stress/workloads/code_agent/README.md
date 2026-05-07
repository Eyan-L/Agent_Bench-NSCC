# code_agent stress workload (simulated)

Multi-turn agent trace replay against an OpenAI-compatible endpoint.

- Go implementation: `stress/loadgen/agent.go` (`AgentLoadGen`)
- Trace synthesizer: `utils/synthetic_code_trace_generator.py`
- Sample config: `config/stress/agent.json` (copied from Agent_Bench)
- Sample agent_dirs input: `config/stress/agent_dirs.example.json`

Invoke via:

```bash
loadgen-bin agent --config config/stress/agent.json \
  --agent-dirs-json config/stress/agent_dirs.example.json
```
