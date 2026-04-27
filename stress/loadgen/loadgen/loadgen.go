// Package loadgen defines the unified interface for replaying LLM request
// traces against OpenAI-compatible endpoints. Three concrete implementations
// live alongside this file:
//
//   - AlibabaLoadGen — session-aware replay of qwen-bailian JSONL traces.
//   - AzureLoadGen   — replay of Azure inference traces with down-sampling
//     (random or Wasserstein) and alibaba-style timestamp compression.
//   - AgentLoadGen   — multi-agent replay of cache_trace_to_plan plans with
//     incremental message-history bookkeeping.
package loadgen

import (
	"context"
	"flag"
	"time"
)

// LoadGenerator is the contract shared by all trace replayers.
//
// Prepare performs all trace loading, filtering and sampling.
//
// Run blocks until the trace has been fully replayed or ctx is cancelled.
// Per-request results are streamed to disk via the underlying result.Writer;
// the returned error is non-nil only for fatal setup failures (not for
// individual request errors, which are recorded and counted separately).
type LoadGenerator interface {
	Prepare(ctx context.Context) error
	Run(ctx context.Context) error
}

// CommonConfig holds fields that every replayer needs. Each subclass embeds
// it and adds a handful of trace-specific knobs.
type CommonConfig struct {
	APIBase         string
	APIKey          string
	Model           string
	TimeoutSec      float64 // parsed flag value; Timeout is derived in Finalize
	Timeout         time.Duration
	Speedup         float64
	StartOffset     float64
	Duration        float64
	MaxRequests     int
	MaxInflight     int
	OutputPath      string
	LogEvery        int
	RecordBody      bool
	ContinueOnError bool
}

// RegisterFlags binds every CommonConfig field onto the given flag set using
// the canonical names shared across all three subcommands.
func (c *CommonConfig) RegisterFlags(fs *flag.FlagSet) {
	fs.StringVar(&c.APIBase, "api-base", "http://127.0.0.1:30000/v1", "OpenAI-compatible API base URL")
	fs.StringVar(&c.APIKey, "api-key", "EMPTY", "Bearer token (empty string = no Authorization header)")
	fs.StringVar(&c.Model, "model", "default", "Model name sent in the request payload")
	fs.Float64Var(&c.TimeoutSec, "timeout", 3000, "Per-request HTTP timeout in seconds")
	fs.Float64Var(&c.Speedup, "speedup", 1.0, "Divide trace timestamps by this factor (>1 = faster)")
	fs.Float64Var(&c.StartOffset, "start-offset", 0, "Skip records before this trace-second")
	fs.Float64Var(&c.Duration, "duration", 0, "Replay only this many trace-seconds after start-offset (0 = all)")
	fs.IntVar(&c.MaxRequests, "max-requests", 0, "Cap total requests after sampling (0 = no cap)")
	fs.IntVar(&c.MaxInflight, "max-inflight", 1024, "Max concurrent in-flight requests")
	fs.StringVar(&c.OutputPath, "output", "", "Optional path to write per-request JSONL results")
	fs.IntVar(&c.LogEvery, "log-every", 500, "Log one line per N fired requests (also on failure)")
	fs.BoolVar(&c.RecordBody, "record-body", true, "Capture HTTP response bodies in the results file")
	fs.BoolVar(&c.ContinueOnError, "continue-on-error", true, "Continue after request failures")
}

// Finalize must be called after flag.Parse() to materialize fields whose
// final value depends on parsed flags.
func (c *CommonConfig) Finalize() {
	c.Timeout = time.Duration(c.TimeoutSec * float64(time.Second))
}
