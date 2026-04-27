package loadgen

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"loadgen/client"
	"loadgen/result"
	"loadgen/sampling"
)

// AzureLoadGen replays an Azure LLM inference trace against an
// OpenAI-compatible /v1/completions endpoint.
//
// Trace format: JSON array whose first element is a common-header object
// (e.g. {"Content-Type": "application/json"}), followed by records of the
// shape {timestamp, data, attributes}. Timestamps are strings in
// "HH:MM:SS.ffffff" form and are converted into trace-seconds relative to
// the first record.
//
// Sampling: only *down*-sampling is supported here (SamplingRate in [0, 1]).
// For up-sampling, use -speedup to compress inter-request delays — this
// mirrors the Alibaba replayer and sidesteps the request-duplication
// approach LLMLoadgen took in its UpscalerTrace / UpscaleAvgRate paths.
type AzureLoadGen struct {
	Common CommonConfig

	TracePath     string
	SamplingRate  float64 // 0.0..1.0; >=1 is no-op
	PromptName    string  // "None" => random downsample, else Wasserstein on data[PromptName]
	RecurStep     int     // Wasserstein trial count per roll-down step
	SamplingSeed  int64
	SaveSampled   string // optional: write sampled trace to this path and exit
	MaxDriftMilli int    // abort the request if scheduled drift exceeds this (0 = disabled)

	records      []azureRecord
	commonHeader map[string]any
	url          string
	headers      http.Header
	writer       *result.Writer
	http         *http.Client
}

type azureTraceEntry struct {
	Timestamp  string         `json:"timestamp"`
	Data       map[string]any `json:"data"`
	Attributes map[string]any `json:"attributes"`
}

// azureRecord is the preprocessed form with TimeDelta in seconds.
type azureRecord struct {
	TimeDeltaSec float64
	Data         map[string]any
	Attributes   map[string]any
}

type azureResultRecord struct {
	RequestIndex   int             `json:"requestIndex"`
	TraceTimestamp float64         `json:"trace_timestamp"`
	ScheduledAt    float64         `json:"scheduled_sec"`
	FiredAt        float64         `json:"fired_sec"`
	DriftMilli     float64         `json:"drift_ms"`
	OK             bool            `json:"ok"`
	Status         int             `json:"status"`
	LatencySec     float64         `json:"latency_sec"`
	Error          string          `json:"error,omitempty"`
	Response       json.RawMessage `json:"response,omitempty"`
	// Surface a couple of trace attributes for downstream analysis without
	// dumping the full data/attributes blob.
	InputSize  int `json:"input_size,omitempty"`
	OutputSize int `json:"output_size,omitempty"`
}

func (g *AzureLoadGen) RegisterFlags(fs *flag.FlagSet) {
	g.Common.RegisterFlags(fs)
	fs.StringVar(&g.TracePath, "trace", "", "Path to the preprocessed Azure trace JSON (required)")
	fs.Float64Var(&g.SamplingRate, "sampling-rate", 1.0, "Down-sampling fraction in [0, 1] (1.0 = keep all)")
	fs.StringVar(&g.PromptName, "prompt-name", "None", "Field name of the prompt in data; 'None' = random downsample")
	fs.IntVar(&g.RecurStep, "recur-step", 10, "Wasserstein roll-down trials per step")
	fs.Int64Var(&g.SamplingSeed, "seed", 42, "Seed for sampling RNG")
	fs.StringVar(&g.SaveSampled, "save-sampled", "", "If set, dump the sampled trace (as azureRecord JSON) to this path and exit without sending")
	fs.IntVar(&g.MaxDriftMilli, "max-drift-ms", 0, "Abort a request if drift (fired - scheduled) exceeds this many ms (0 = disabled)")
}

func (g *AzureLoadGen) Validate() error {
	g.Common.Finalize()
	if g.TracePath == "" {
		return fmt.Errorf("-trace is required")
	}
	if g.Common.Speedup <= 0 {
		return fmt.Errorf("-speedup must be > 0")
	}
	if g.SamplingRate < 0 || g.SamplingRate > 1 {
		return fmt.Errorf("-sampling-rate must be in [0, 1]")
	}
	return nil
}

func (g *AzureLoadGen) Prepare(ctx context.Context) error {
	_ = ctx
	if err := g.Validate(); err != nil {
		return err
	}

	raw, err := os.ReadFile(g.TracePath)
	if err != nil {
		return fmt.Errorf("read trace: %w", err)
	}

	// First unmarshal to extract the common-header map at position 0.
	var rawList []map[string]any
	if err := json.Unmarshal(raw, &rawList); err != nil {
		return fmt.Errorf("parse trace header: %w", err)
	}
	if len(rawList) == 0 {
		return fmt.Errorf("trace is empty")
	}
	g.commonHeader = rawList[0]

	// Second unmarshal onto the structured entry shape. We accept that the
	// first entry will fail to populate (it has no timestamp) — filter it
	// out by checking for an empty Timestamp.
	var entries []azureTraceEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("parse trace entries: %w", err)
	}
	if len(entries) < 2 {
		return fmt.Errorf("trace has no records (len=%d)", len(entries))
	}
	entries = entries[1:]

	initialTs, err := parseAzureTimestamp(entries[0].Timestamp)
	if err != nil {
		return fmt.Errorf("parse first timestamp %q: %w", entries[0].Timestamp, err)
	}

	records := make([]azureRecord, 0, len(entries))
	for _, e := range entries {
		if e.Timestamp == "" {
			continue
		}
		ts, err := parseAzureTimestamp(e.Timestamp)
		if err != nil {
			log.Printf("azure: skipping invalid timestamp %q: %v", e.Timestamp, err)
			continue
		}
		delta := ts.Sub(initialTs).Seconds()
		if delta < g.Common.StartOffset {
			continue
		}
		if g.Common.Duration > 0 && delta > g.Common.StartOffset+g.Common.Duration {
			continue
		}
		records = append(records, azureRecord{
			TimeDeltaSec: delta,
			Data:         e.Data,
			Attributes:   e.Attributes,
		})
	}
	log.Printf("azure: loaded %d records from %d entries (start_offset=%.1fs duration=%.1fs)", len(records), len(entries), g.Common.StartOffset, g.Common.Duration)

	// Down-sampling: random (PromptName == "None") or Wasserstein.
	if g.SamplingRate < 1.0 && g.SamplingRate > 0.0 {
		if strings.EqualFold(g.PromptName, "None") || g.PromptName == "" {
			target := int(float64(len(records)) * g.SamplingRate)
			records = sampling.RandomSubset(records, target, g.SamplingSeed)
			log.Printf("azure: random downsample -> %d records (rate=%.3f, seed=%d)", len(records), g.SamplingRate, g.SamplingSeed)
		} else {
			prompt := g.PromptName
			records = sampling.WassersteinDownsample(records, g.SamplingRate, g.RecurStep, g.SamplingSeed, func(r azureRecord) float64 {
				v, ok := r.Data[prompt]
				if !ok {
					return 0
				}
				switch x := v.(type) {
				case string:
					return float64(len(x))
				case []any:
					return float64(len(x))
				}
				return 0
			})
		}
	}

	sort.SliceStable(records, func(i, j int) bool { return records[i].TimeDeltaSec < records[j].TimeDeltaSec })
	if g.Common.MaxRequests > 0 && len(records) > g.Common.MaxRequests {
		records = records[:g.Common.MaxRequests]
	}
	g.records = records

	if g.SaveSampled != "" {
		if err := g.dumpSampled(); err != nil {
			return fmt.Errorf("save sampled: %w", err)
		}
		log.Printf("azure: saved sampled trace to %s (%d records); not sending", g.SaveSampled, len(g.records))
		return nil
	}

	g.url = strings.TrimRight(g.Common.APIBase, "/") + "/completions"
	g.headers = client.Headers(g.Common.APIKey)
	// Also forward any string-valued header from the trace's commonHeader
	// (e.g. "Content-Type"). Overrides our defaults deliberately — traces
	// often set an explicit Content-Type.
	for k, v := range g.commonHeader {
		if s, ok := v.(string); ok {
			g.headers.Set(k, s)
		}
	}
	g.http = client.New(g.Common.Timeout, g.Common.MaxInflight)

	if g.Common.OutputPath != "" {
		w, err := result.New(g.Common.OutputPath)
		if err != nil {
			return fmt.Errorf("open output: %w", err)
		}
		g.writer = w
	}
	return nil
}

func (g *AzureLoadGen) Run(ctx context.Context) error {
	if g.SaveSampled != "" {
		// Prepare already wrote the sampled dump and asked us to stop.
		return nil
	}
	cfg := &g.Common
	defer func() {
		if g.writer != nil {
			_ = g.writer.Close()
		}
	}()

	var (
		sentCnt atomic.Int64
		failCnt atomic.Int64
		fired   atomic.Int64
	)

	sem := make(chan struct{}, cfg.MaxInflight)
	var wg sync.WaitGroup

	var (
		started   bool
		t0        time.Time
		scheduled int
	)

	for idx, rec := range g.records {
		sendDelay := (rec.TimeDeltaSec - cfg.StartOffset) / cfg.Speedup
		scheduled++

		if !started {
			t0 = time.Now()
			started = true
		}
		target := t0.Add(time.Duration(sendDelay * float64(time.Second)))
		if d := time.Until(target); d > 0 {
			t := time.NewTimer(d)
			select {
			case <-t.C:
			case <-ctx.Done():
				t.Stop()
				break
			}
		}
		if ctx.Err() != nil {
			break
		}

		sem <- struct{}{}
		wg.Add(1)
		go func(idx int, rec azureRecord, scheduledSec float64) {
			defer wg.Done()
			defer func() { <-sem }()

			firedAt := time.Since(t0).Seconds()
			drift := (firedAt - scheduledSec) * 1000.0 // ms
			inputSize, outputSize := extractAzureSizes(rec.Data)

			if g.MaxDriftMilli > 0 && drift > float64(g.MaxDriftMilli) {
				failCnt.Add(1)
				g.writer.Write(azureResultRecord{
					RequestIndex: idx, TraceTimestamp: rec.TimeDeltaSec,
					ScheduledAt: scheduledSec, FiredAt: firedAt, DriftMilli: drift,
					Error: fmt.Sprintf("drift %.1fms exceeds max %dms", drift, g.MaxDriftMilli),
					InputSize: inputSize, OutputSize: outputSize,
				})
				return
			}

			payload, err := json.Marshal(rec.Data)
			if err != nil {
				failCnt.Add(1)
				g.writer.Write(azureResultRecord{
					RequestIndex: idx, TraceTimestamp: rec.TimeDeltaSec,
					ScheduledAt: scheduledSec, FiredAt: firedAt, DriftMilli: drift,
					Error: "marshal: " + err.Error(),
					InputSize: inputSize, OutputSize: outputSize,
				})
				return
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url, bytes.NewReader(payload))
			if err != nil {
				failCnt.Add(1)
				g.writer.Write(azureResultRecord{
					RequestIndex: idx, TraceTimestamp: rec.TimeDeltaSec,
					ScheduledAt: scheduledSec, FiredAt: firedAt, DriftMilli: drift,
					Error: "newreq: " + err.Error(),
					InputSize: inputSize, OutputSize: outputSize,
				})
				return
			}
			for k, v := range g.headers {
				req.Header[k] = v
			}
			for k, v := range rec.Attributes {
				if s, ok := v.(string); ok {
					req.Header.Set(k, s)
				}
			}

			start := time.Now()
			resp, err := g.http.Do(req)
			latency := time.Since(start).Seconds()

			if err != nil {
				failCnt.Add(1)
				g.writer.Write(azureResultRecord{
					RequestIndex: idx, TraceTimestamp: rec.TimeDeltaSec,
					ScheduledAt: scheduledSec, FiredAt: firedAt, DriftMilli: drift,
					LatencySec: latency, Error: err.Error(),
					InputSize: inputSize, OutputSize: outputSize,
				})
				return
			}
			var bodyRaw json.RawMessage
			if cfg.RecordBody {
				bodyBytes, readErr := io.ReadAll(resp.Body)
				resp.Body.Close()
				if readErr == nil && json.Valid(bodyBytes) {
					bodyRaw = json.RawMessage(bodyBytes)
				} else if readErr == nil {
					wrapped, _ := json.Marshal(map[string]string{"raw_text": string(bodyBytes)})
					bodyRaw = json.RawMessage(wrapped)
				}
			} else {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}

			ok := resp.StatusCode < 400
			if ok {
				sentCnt.Add(1)
			} else {
				failCnt.Add(1)
			}

			g.writer.Write(azureResultRecord{
				RequestIndex: idx, TraceTimestamp: rec.TimeDeltaSec,
				ScheduledAt: scheduledSec, FiredAt: firedAt, DriftMilli: drift,
				OK: ok, Status: resp.StatusCode, LatencySec: latency,
				Response: bodyRaw,
				InputSize: inputSize, OutputSize: outputSize,
			})

			n := fired.Add(1)
			if cfg.LogEvery > 0 && (n%int64(cfg.LogEvery) == 0 || !ok) {
				log.Printf("azure fired=%d ok=%v status=%d latency=%.2fs scheduled=%.3fs fired_at=%.3fs drift=%.1fms input=%d output=%d",
					n, ok, resp.StatusCode, latency, scheduledSec, firedAt, drift, inputSize, outputSize)
			}
		}(idx, rec, sendDelay)
	}

	log.Printf("azure: all records dispatched, waiting for %d in-flight...", cfg.MaxInflight)
	wg.Wait()

	elapsed := time.Duration(0)
	if started {
		elapsed = time.Since(t0)
	}
	log.Printf("azure: done elapsed=%s scheduled=%d sent=%d failed=%d",
		elapsed.Round(time.Millisecond), scheduled, sentCnt.Load(), failCnt.Load())

	if g.writer != nil {
		log.Printf("azure: wrote %s (%d records)", cfg.OutputPath, g.writer.Count())
	}
	if failCnt.Load() > 0 {
		return fmt.Errorf("azure: %d failed requests", failCnt.Load())
	}
	return nil
}

func (g *AzureLoadGen) dumpSampled() error {
	out := make([]map[string]any, 0, len(g.records))
	for _, r := range g.records {
		out = append(out, map[string]any{
			"time_delta_sec": r.TimeDeltaSec,
			"data":           r.Data,
			"attributes":     r.Attributes,
		})
	}
	buf, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dirOf(g.SaveSampled), 0o755); err != nil {
		return err
	}
	return os.WriteFile(g.SaveSampled, buf, 0o644)
}

func dirOf(p string) string {
	i := strings.LastIndex(p, "/")
	if i <= 0 {
		return "."
	}
	return p[:i]
}

// parseAzureTimestamp handles the "HH:MM:SS.ffffff" format used by LLMLoadgen's
// preprocessed Azure traces. Returns a time.Time anchored at Unix epoch (we
// only care about the difference between two timestamps).
func parseAzureTimestamp(s string) (time.Time, error) {
	// time.Parse accepts "15:04:05.000000"; the day/year default to zero
	// which is fine since we only take differences.
	return time.Parse("15:04:05.000000", s)
}

// extractAzureSizes pulls integer-ish "input_size"/"output_size" fields from
// the trace record's data payload for inclusion in the result record.
func extractAzureSizes(data map[string]any) (int, int) {
	return intFromAny(data["input_size"]), intFromAny(data["output_size"])
}

func intFromAny(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	}
	return 0
}
