# websearch stress workload (simulated)

Azure trace replay with downsampling (random / Wasserstein).

- Go implementation: `stress/loadgen/azure.go` (`AzureLoadGen`)
- Trace synthesizer: `utils/synthetic_search_trace_generator.py`
- Sample config: `config/stress/azure.json` (copied from Agent_Bench)

Invoke via: `loadgen-bin azure --config config/stress/azure.json`
