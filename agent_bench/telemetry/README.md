# agent_bench.telemetry — Instrumentation for OpenClaw + SWE-bench Study

Runnable Python instrumentation stack that produces raw traces for the OpenClaw + SWE-bench code-agent validation study.

---

## Overview

This package implements:

1. **Step-trace logger** (`logger.py`) — captures trajectory/attempt/step structure with full schema
2. **LLM wrapper** (`llm_wrapper.py`) — captures TTFT, TPOT from OpenAI-compatible streaming clients
3. **OpenClaw integration** (`openclaw_hooks.py`) — hook-based and wrapper-based paths for instrumenting OpenClaw
4. **Resource telemetry** (`telemetry.py`) — sidecar collector for GPU (NVML), CPU (psutil), memory
5. **Timestamp alignment** (`align.py`) — merges step events with telemetry on monotonic clock
6. **Writers** (`writers.py`) — JSONL for steps, Parquet (fallback CSV) for telemetry

---

## Design Decisions

### Clock Source
- **Duration math:** `time.monotonic_ns()` (CLOCK_MONOTONIC_RAW equivalent in Python)
- **Wall-clock:** `time.time_ns()` for external log correlation only
- **Never use wall-clock for latency computation** — subject to NTP corrections, container clock skew

### Buffering
- In-memory ring buffer in `StepLogger`
- Flush every **64 events** or **1 second**, whichever first
- Background flush thread for time-based flushing

### Crash Safety
- `fsync()` on trajectory termination
- Partial trajectories marked with `partial=true` in final manifest line
- JSONL is line-oriented; incomplete last line can be discarded

### Telemetry Placement
- One `ResourceCollector` sidecar per trajectory
- Runs as background thread in same process (can be adapted to separate process)
- Poll interval: **200 ms** (configurable 100–500 ms)
- Pinning via cgroup not implemented (add if needed)

### Schema Evolution
- `schema_version` field in all events
- Bump version on breaking changes
- Old files never mutated

---

## Integration Paths

### Path 1: Wrapper-Based (Recommended)

Assumes OpenClaw exposes a tool dispatcher and LLM client that can be wrapped.

