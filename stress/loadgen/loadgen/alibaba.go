package loadgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"loadgen/client"
	"loadgen/result"
)

// AlibabaLoadGen replays a token-expanded qwen-bailian JSONL trace against
// an OpenAI-compatible /v1/completions endpoint (e.g. SGLang), honoring the
// original wall-clock timestamps.
//
// Ported from scripts/go/qwen_trace_replayer/main.go. The send loop and
// session helpers are preserved byte-for-byte to keep regression output
// identical.
type AlibabaLoadGen struct {
	Common CommonConfig

	TracePath         string
	TypeFilter        string
	SessionSampleRate float64
	SessionSampleSeed int64

	// populated by Prepare
	records []alibabaRecord
	url     string
	headers http.Header
	writer  *result.Writer
	http    *http.Client

	// populated by Run
	typeSet map[string]struct{}
}

type alibabaRecord struct {
	ChatID       int64   `json:"chat_id"`
	ParentChatID int64   `json:"parent_chat_id"`
	SessionID    int64   `json:"session_id,omitempty"`
	Timestamp    float64 `json:"timestamp"`
	InputLength  int     `json:"input_length"`
	OutputLength int     `json:"output_length"`
	Type         string  `json:"type"`
	Turn         int     `json:"turn"`
	TokenIDs     []int   `json:"token_ids"`
}

type alibabaCompletionRequest struct {
	Model        string   `json:"model"`
	Prompt       []int    `json:"prompt"`
	MaxTokens    int      `json:"max_tokens"`
	MinTokens    int      `json:"min_tokens"`
	Temperature  float64  `json:"temperature"`
	IgnoreEOS    bool     `json:"ignore_eos"`
	Stop         []string `json:"stop"`
	StopTokenIDs []int    `json:"stop_token_ids"`
}

type alibabaResultRecord struct {
	RequestIndex   int             `json:"requestIndex"`
	ChatID         int64           `json:"chat_id"`
	SessionID      int64           `json:"session_id"`
	Type           string          `json:"type"`
	TraceTimestamp float64         `json:"trace_timestamp"`
	ScheduledAt    float64         `json:"scheduled_sec"`
	FiredAt        float64         `json:"fired_sec"`
	PromptLen      int             `json:"prompt_len"`
	MaxTokens      int             `json:"max_tokens"`
	OK             bool            `json:"ok"`
	Status         int             `json:"status"`
	LatencySec     float64         `json:"latency_sec"`
	Error          string          `json:"error,omitempty"`
	Response       json.RawMessage `json:"response,omitempty"`
}

// RegisterFlags wires alibaba-specific flags onto fs after CommonConfig's
// flags have been registered.
func (g *AlibabaLoadGen) RegisterFlags(fs *flag.FlagSet) {
	g.Common.RegisterFlags(fs)
	fs.StringVar(&g.TracePath, "trace", "", "Path to *_tokens.jsonl trace file (required)")
	fs.StringVar(&g.TypeFilter, "type-filter", "", "Comma-separated record types to keep (empty = all)")
	fs.Float64Var(&g.SessionSampleRate, "session-sample-rate", 1.0, "Fraction of sessions to keep (1.0 = all)")
	fs.Int64Var(&g.SessionSampleSeed, "session-sample-seed", 42, "Seed for deterministic session sampling")
}

// Validate runs after flag parsing.
func (g *AlibabaLoadGen) Validate() error {
	g.Common.Finalize()
	if g.TracePath == "" {
		return fmt.Errorf("-trace is required")
	}
	if g.Common.Speedup <= 0 {
		return fmt.Errorf("-speedup must be > 0")
	}
	if g.SessionSampleRate < 0 || g.SessionSampleRate > 1 {
		return fmt.Errorf("-session-sample-rate must be in [0, 1]")
	}
	return nil
}

func (g *AlibabaLoadGen) Prepare(ctx context.Context) error {
	_ = ctx
	if err := g.Validate(); err != nil {
		return err
	}

	g.typeSet = map[string]struct{}{}
	if g.TypeFilter != "" {
		for _, t := range strings.Split(g.TypeFilter, ",") {
			t = strings.TrimSpace(t)
			if t != "" {
				g.typeSet[t] = struct{}{}
			}
		}
	}

	f, err := os.Open(g.TracePath)
	if err != nil {
		return fmt.Errorf("open trace: %w", err)
	}
	defer f.Close()

	records, skipped := loadFilteredAlibabaRecords(f, &g.Common, g.typeSet)
	log.Printf("alibaba: loaded %d records after pre-filter (skipped=%d)", len(records), skipped)

	records = resolveAlibabaSessions(records)
	records = sampleAlibabaSessions(records, g.SessionSampleRate, g.SessionSampleSeed)
	sort.SliceStable(records, func(i, j int) bool { return records[i].Timestamp < records[j].Timestamp })
	if g.Common.MaxRequests > 0 && len(records) > g.Common.MaxRequests {
		records = records[:g.Common.MaxRequests]
	}
	g.records = records
	log.Printf("alibaba: replaying %d records across %d sessions", len(records), countAlibabaSessions(records))

	g.url = strings.TrimRight(g.Common.APIBase, "/") + "/completions"
	g.headers = client.Headers(g.Common.APIKey)
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

func (g *AlibabaLoadGen) Run(ctx context.Context) error {
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
		sendDelay := (rec.Timestamp - cfg.StartOffset) / cfg.Speedup
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
		go func(idx int, rec alibabaRecord, scheduledSec float64) {
			defer wg.Done()
			defer func() { <-sem }()

			firedAt := time.Since(t0).Seconds()
			reqBody := alibabaCompletionRequest{
				Model:        cfg.Model,
				Prompt:       rec.TokenIDs,
				MaxTokens:    rec.OutputLength,
				MinTokens:    rec.OutputLength,
				Temperature:  0,
				IgnoreEOS:    true,
				Stop:         []string{},
				StopTokenIDs: []int{},
			}
			payload, err := json.Marshal(&reqBody)
			if err != nil {
				failCnt.Add(1)
				g.writer.Write(alibabaResultRecord{
					RequestIndex: idx, ChatID: rec.ChatID, SessionID: rec.SessionID, Type: rec.Type,
					TraceTimestamp: rec.Timestamp, ScheduledAt: scheduledSec, FiredAt: firedAt,
					PromptLen: len(rec.TokenIDs), MaxTokens: rec.OutputLength,
					Error: "marshal: " + err.Error(),
				})
				return
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url, bytes.NewReader(payload))
			if err != nil {
				failCnt.Add(1)
				g.writer.Write(alibabaResultRecord{
					RequestIndex: idx, ChatID: rec.ChatID, SessionID: rec.SessionID, Type: rec.Type,
					TraceTimestamp: rec.Timestamp, ScheduledAt: scheduledSec, FiredAt: firedAt,
					PromptLen: len(rec.TokenIDs), MaxTokens: rec.OutputLength,
					Error: "newreq: " + err.Error(),
				})
				return
			}
			for k, v := range g.headers {
				req.Header[k] = v
			}

			start := time.Now()
			resp, err := g.http.Do(req)
			latency := time.Since(start).Seconds()

			if err != nil {
				failCnt.Add(1)
				g.writer.Write(alibabaResultRecord{
					RequestIndex: idx, ChatID: rec.ChatID, SessionID: rec.SessionID, Type: rec.Type,
					TraceTimestamp: rec.Timestamp, ScheduledAt: scheduledSec, FiredAt: firedAt,
					PromptLen: len(rec.TokenIDs), MaxTokens: rec.OutputLength,
					LatencySec: latency, Error: err.Error(),
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

			g.writer.Write(alibabaResultRecord{
				RequestIndex: idx, ChatID: rec.ChatID, SessionID: rec.SessionID, Type: rec.Type,
				TraceTimestamp: rec.Timestamp, ScheduledAt: scheduledSec, FiredAt: firedAt,
				PromptLen: len(rec.TokenIDs), MaxTokens: rec.OutputLength,
				OK: ok, Status: resp.StatusCode, LatencySec: latency,
				Response: bodyRaw,
			})

			n := fired.Add(1)
			if cfg.LogEvery > 0 && (n%int64(cfg.LogEvery) == 0 || !ok) {
				log.Printf("fired=%d ok=%v status=%d latency=%.2fs scheduled=%.3fs fired_at=%.3fs prompt_len=%d max_tokens=%d chat=%d session=%d",
					n, ok, resp.StatusCode, latency, scheduledSec, firedAt, len(rec.TokenIDs), rec.OutputLength, rec.ChatID, rec.SessionID)
			}
		}(idx, rec, sendDelay)
	}

	log.Printf("alibaba: all records dispatched, waiting for %d in-flight...", cfg.MaxInflight)
	wg.Wait()

	elapsed := time.Duration(0)
	if started {
		elapsed = time.Since(t0)
	}
	log.Printf("alibaba: done elapsed=%s scheduled=%d sent=%d failed=%d",
		elapsed.Round(time.Millisecond), scheduled, sentCnt.Load(), failCnt.Load())

	if g.writer != nil {
		log.Printf("alibaba: wrote %s (%d records)", cfg.OutputPath, g.writer.Count())
	}
	if failCnt.Load() > 0 {
		return fmt.Errorf("alibaba: %d failed requests", failCnt.Load())
	}
	return nil
}

// loadFilteredAlibabaRecords streams the JSONL file into memory, applying
// start-offset / duration / type / validity filters. MaxRequests is NOT
// applied here — we want it to cap sessions-kept records, not pre-sampling
// ones.
func loadFilteredAlibabaRecords(r io.Reader, cfg *CommonConfig, typeSet map[string]struct{}) ([]alibabaRecord, int) {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 1<<20)
	scanner.Buffer(buf, 32<<20)

	var (
		out     []alibabaRecord
		skipped int
	)
	endDuration := cfg.Duration

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var rec alibabaRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			skipped++
			if !cfg.ContinueOnError {
				log.Fatalf("parse line: %v", err)
			}
			continue
		}
		if rec.Timestamp < cfg.StartOffset {
			continue
		}
		if endDuration > 0 && rec.Timestamp > cfg.StartOffset+endDuration {
			continue
		}
		if len(typeSet) > 0 {
			if _, ok := typeSet[rec.Type]; !ok {
				continue
			}
		}
		if rec.OutputLength <= 0 || len(rec.TokenIDs) == 0 {
			skipped++
			continue
		}
		out = append(out, rec)
	}
	if err := scanner.Err(); err != nil {
		log.Printf("scanner error: %v", err)
	}
	return out, skipped
}

// resolveAlibabaSessions fills in SessionID for any record missing one, by
// walking chat_id -> parent_chat_id chains inside the loaded slice. Records
// whose parent is not in the slice (or parent == -1) become their own
// session root.
func resolveAlibabaSessions(records []alibabaRecord) []alibabaRecord {
	anyMissing := false
	for _, r := range records {
		if r.SessionID == 0 {
			anyMissing = true
			break
		}
	}
	if !anyMissing {
		return records
	}
	parentOf := make(map[int64]int64, len(records))
	present := make(map[int64]struct{}, len(records))
	for _, r := range records {
		parentOf[r.ChatID] = r.ParentChatID
		present[r.ChatID] = struct{}{}
	}
	rootOf := make(map[int64]int64, len(records))
	var walk func(id int64) int64
	walk = func(id int64) int64 {
		if r, ok := rootOf[id]; ok {
			return r
		}
		pid, hasParent := parentOf[id]
		if !hasParent || pid == -1 {
			rootOf[id] = id
			return id
		}
		if _, inFile := present[pid]; !inFile {
			rootOf[id] = id
			return id
		}
		rootOf[id] = id
		root := walk(pid)
		rootOf[id] = root
		return root
	}
	for i := range records {
		if records[i].SessionID == 0 {
			records[i].SessionID = walk(records[i].ChatID)
		}
	}
	return records
}

// sampleAlibabaSessions keeps records belonging to a deterministic subset of
// session ids. rate=1.0 is a no-op.
func sampleAlibabaSessions(records []alibabaRecord, rate float64, seed int64) []alibabaRecord {
	if rate >= 1.0 {
		return records
	}
	if rate <= 0.0 {
		return records[:0]
	}
	seen := make(map[int64]struct{})
	var sessionIDs []int64
	for _, r := range records {
		if _, ok := seen[r.SessionID]; ok {
			continue
		}
		seen[r.SessionID] = struct{}{}
		sessionIDs = append(sessionIDs, r.SessionID)
	}
	sort.Slice(sessionIDs, func(i, j int) bool { return sessionIDs[i] < sessionIDs[j] })

	rng := rand.New(rand.NewSource(seed))
	kept := make(map[int64]struct{}, len(sessionIDs))
	for _, sid := range sessionIDs {
		if rng.Float64() < rate {
			kept[sid] = struct{}{}
		}
	}
	log.Printf("alibaba session sampling: kept %d / %d sessions (rate=%.3f, seed=%d)", len(kept), len(sessionIDs), rate, seed)

	filtered := records[:0]
	for _, r := range records {
		if _, ok := kept[r.SessionID]; ok {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func countAlibabaSessions(records []alibabaRecord) int {
	seen := make(map[int64]struct{})
	for _, r := range records {
		seen[r.SessionID] = struct{}{}
	}
	return len(seen)
}
