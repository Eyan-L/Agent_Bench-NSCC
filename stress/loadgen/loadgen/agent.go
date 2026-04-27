package loadgen

import (
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
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"loadgen/client"
	"loadgen/result"
)

// AgentLoadGen replays agent-trace plans against an OpenAI-compatible
// /v1/chat/completions endpoint. Port of
// scripts/python/trace_load_generator_v2.py.
//
// Per agent, the generator reads cache/replay_plan.json from the agent
// directory, sends each row's payload with a staged system message
// ("I am agent <name>") prepended, and maintains an incremental
// message-history state in agentState (agent_state.go) to mirror how the
// original trace was produced at runtime.
//
// Two send modes:
//
//   - "sync"    — send back-to-back, no artificial pause.
//   - "attempt" — within an attempt (no new user message), send back-to-back;
//     between attempts insert a random delay (80% U[0,10]s, 20% U[10,300]s).
type AgentLoadGen struct {
	Common CommonConfig

	AgentDirsJSON     string
	Mode              string // "sync" | "attempt"
	ConcurrentAgents  int
	MaxAgents         int
	ResponseOutputDir string
	StartStagger      string // "", "poisson:<mean>" or "uniform:<lo>,<hi>"
	StaggerSeed       int64

	// populated by Prepare
	agents  []agentSpec
	delays  []float64
	url     string
	headers http.Header
	writer  *result.Writer
	http    *http.Client
}

type agentSpec struct {
	Dir  string
	Tag  string
	Name string
}

type replayPlanRow struct {
	RequestIndex int             `json:"requestIndex"`
	MessageCount int             `json:"messageCount"`
	MessageRoles []string        `json:"messageRoles"`
	MaxTokens    *int            `json:"max_tokens,omitempty"`
	MinTokens    *int            `json:"min_tokens,omitempty"`
	Payload      json.RawMessage `json:"payload"`
}

type agentResultRecord struct {
	Agent        string          `json:"agent"`
	RequestIndex int             `json:"requestIndex"`
	FiredAtSec   float64         `json:"fired_sec"`
	AttemptNum   int             `json:"attempt"`
	ResetReason  string          `json:"reset_reason,omitempty"`
	MessagesLen  int             `json:"messages_len"`
	OK           bool            `json:"ok"`
	Status       int             `json:"status"`
	LatencySec   float64         `json:"latency_sec"`
	MaxTokens    int             `json:"max_tokens,omitempty"`
	Error        string          `json:"error,omitempty"`
	Response     json.RawMessage `json:"response,omitempty"`
}

// perAgentResponseRow mirrors the record shape emitted by the original
// Python script, so the per-agent {dir}/cache/load_responses.json file stays
// a drop-in replacement for downstream tooling.
type perAgentResponseRow struct {
	Agent        string          `json:"agent"`
	RequestIndex int             `json:"requestIndex"`
	OK           bool            `json:"ok"`
	Status       int             `json:"status,omitempty"`
	LatencySec   float64         `json:"latency_sec,omitempty"`
	MaxTokens    int             `json:"max_tokens,omitempty"`
	Error        string          `json:"error,omitempty"`
	Response     json.RawMessage `json:"response,omitempty"`
}

func (g *AgentLoadGen) RegisterFlags(fs *flag.FlagSet) {
	g.Common.RegisterFlags(fs)
	fs.StringVar(&g.AgentDirsJSON, "agent-dirs-json", "", "JSON file with agent directories (array of {dir, tag, name}); required")
	fs.StringVar(&g.Mode, "mode", "sync", "Send mode: sync | attempt")
	fs.IntVar(&g.ConcurrentAgents, "concurrent-agents", 2, "Max agents sending concurrently (ThreadPoolExecutor equivalent)")
	fs.IntVar(&g.MaxAgents, "max-agents", 0, "Only run first N agents (0 = all)")
	fs.StringVar(&g.ResponseOutputDir, "response-output-dir", "cache", "Per-agent response log dir (relative lives under each agent dir)")
	fs.StringVar(&g.StartStagger, "agent-start-stagger", "", "Stagger spec: '' | poisson:<mean> | uniform:<lo>,<hi>")
	fs.Int64Var(&g.StaggerSeed, "stagger-seed", 0, "Seed for the stagger RNG (0 = non-deterministic via time)")
}

func (g *AgentLoadGen) Validate() error {
	g.Common.Finalize()
	if g.AgentDirsJSON == "" {
		return fmt.Errorf("-agent-dirs-json is required")
	}
	if g.Mode != "sync" && g.Mode != "attempt" {
		return fmt.Errorf("-mode must be 'sync' or 'attempt'")
	}
	if g.ConcurrentAgents <= 0 {
		g.ConcurrentAgents = 1
	}
	return nil
}

func (g *AgentLoadGen) Prepare(ctx context.Context) error {
	_ = ctx
	if err := g.Validate(); err != nil {
		return err
	}

	dirsPath, err := filepath.Abs(expandUser(g.AgentDirsJSON))
	if err != nil {
		return fmt.Errorf("resolve agent-dirs-json: %w", err)
	}
	raw, err := os.ReadFile(dirsPath)
	if err != nil {
		return fmt.Errorf("read agent-dirs-json: %w", err)
	}
	agents, err := parseAgentDirs(raw)
	if err != nil {
		return fmt.Errorf("parse agent-dirs-json: %w", err)
	}
	if len(agents) == 0 {
		return fmt.Errorf("no agent directories in %s", dirsPath)
	}
	if g.MaxAgents > 0 && g.MaxAgents < len(agents) {
		agents = agents[:g.MaxAgents]
	}
	g.agents = agents

	// Compute per-agent start delays up front so logging is deterministic.
	delays, err := computeStartDelays(len(agents), g.StartStagger, g.StaggerSeed)
	if err != nil {
		return fmt.Errorf("compute start delays: %w", err)
	}
	g.delays = delays

	log.Printf("agent: mode=%s agents=%d concurrent=%d stagger=%q", g.Mode, len(agents), g.ConcurrentAgents, g.StartStagger)
	for i, a := range agents {
		log.Printf("  agent[%d]: name=%s tag=%s start_delay=%.1fs dir=%s", i, a.Name, a.Tag, delays[i], a.Dir)
	}

	g.url = strings.TrimRight(g.Common.APIBase, "/") + "/chat/completions"
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

func (g *AgentLoadGen) Run(ctx context.Context) error {
	defer func() {
		if g.writer != nil {
			_ = g.writer.Close()
		}
	}()

	sem := make(chan struct{}, g.ConcurrentAgents)
	var wg sync.WaitGroup

	start := time.Now()
	type agentSummary struct {
		Name   string
		Sent   int
		Failed int
		Total  int
		Err    string
	}
	summaries := make([]agentSummary, len(g.agents))

	for i, agent := range g.agents {
		sem <- struct{}{}
		wg.Add(1)
		go func(i int, agent agentSpec, delay float64) {
			defer wg.Done()
			defer func() { <-sem }()
			sent, failed, total, err := g.runOneAgent(ctx, agent, delay, start)
			sum := agentSummary{Name: agent.Name, Sent: sent, Failed: failed, Total: total}
			if err != nil {
				sum.Err = err.Error()
				log.Printf("[%s] fatal: %v", agent.Name, err)
			}
			summaries[i] = sum
		}(i, agent, g.delays[i])
	}
	wg.Wait()

	totalSent, totalFailed := 0, 0
	log.Printf("===== AGENT SUMMARY =====")
	for _, s := range summaries {
		buf, _ := json.Marshal(s)
		log.Printf("%s", buf)
		totalSent += s.Sent
		totalFailed += s.Failed
	}
	log.Printf("agent: done elapsed=%s agents=%d sent=%d failed=%d",
		time.Since(start).Round(time.Millisecond), len(summaries), totalSent, totalFailed)

	if totalFailed > 0 && !g.Common.ContinueOnError {
		return fmt.Errorf("agent: %d failed requests", totalFailed)
	}
	return nil
}

func (g *AgentLoadGen) runOneAgent(ctx context.Context, agent agentSpec, startDelay float64, runStart time.Time) (sent, failed, total int, fatal error) {
	if startDelay > 0 {
		log.Printf("[%s] staggered start: sleeping %.1fs", agent.Name, startDelay)
		select {
		case <-time.After(time.Duration(startDelay * float64(time.Second))):
		case <-ctx.Done():
			return 0, 0, 0, ctx.Err()
		}
	}

	agentDir, err := filepath.Abs(expandUser(agent.Dir))
	if err != nil {
		return 0, 1, 0, fmt.Errorf("resolve agent dir: %w", err)
	}
	planPath := filepath.Join(agentDir, "cache", "replay_plan.json")
	responsePath := agentResponsePath(agentDir, agent.Name, g.ResponseOutputDir)

	planBytes, err := os.ReadFile(planPath)
	if err != nil {
		errStr := fmt.Sprintf("plan file not found: %s (%v)", planPath, err)
		_ = writePerAgentResponses(responsePath, []perAgentResponseRow{{
			Agent: agent.Name, OK: false, Error: errStr,
		}})
		return 0, 1, 0, nil
	}

	var planRows []replayPlanRow
	if err := json.Unmarshal(planBytes, &planRows); err != nil {
		return 0, 1, 0, fmt.Errorf("parse plan: %w", err)
	}

	totalRows := len(planRows)
	if g.Common.MaxRequests > 0 && g.Common.MaxRequests < totalRows {
		totalRows = g.Common.MaxRequests
	}

	systemMsg := json.RawMessage(mustMarshal(map[string]any{
		"role":    "system",
		"content": "I am agent " + agent.Name,
	}))

	state := &agentState{}
	responses := make([]perAgentResponseRow, 0, totalRows)
	attemptRng := rand.New(rand.NewSource(seedFor(agent.Name, g.StaggerSeed)))

	for i := 0; i < totalRows; i++ {
		if ctx.Err() != nil {
			break
		}
		row := planRows[i]
		requestIndex := row.RequestIndex
		if requestIndex == 0 {
			requestIndex = i + 1
		}

		// Parse the row payload so we can read and rewrite "messages" while
		// keeping every other field intact.
		var payloadMap map[string]json.RawMessage
		if err := json.Unmarshal(row.Payload, &payloadMap); err != nil {
			failed++
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Error: "invalid payload: " + err.Error(),
			})
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}
		rawMessages, ok := payloadMap["messages"]
		if !ok {
			failed++
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Error: "payload has no messages",
			})
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}
		var curTraceMsgs []json.RawMessage
		if err := json.Unmarshal(rawMessages, &curTraceMsgs); err != nil {
			failed++
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Error: "invalid messages: " + err.Error(),
			})
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}

		// Attempt-boundary detection & attempt-mode delay. Uses the
		// tail-user-message rule instead of v2.py's count-comparison so that
		// attempts opened right after a compaction (where user-message count
		// can drop) still register. See isNewAttemptByTail for details.
		isNewAttempt := isNewAttemptByTail(curTraceMsgs, state.prevTraceMessages)
		if isNewAttempt {
			state.currentAttempt++
		}
		if g.Mode == "attempt" && isNewAttempt && state.currentAttempt > 1 {
			delay := randomAttemptDelay(attemptRng)
			log.Printf("[%s %d/%d] new attempt #%d, sleeping %.1fs", agent.Name, requestIndex, totalRows, state.currentAttempt, delay)
			select {
			case <-time.After(time.Duration(delay * float64(time.Second))):
			case <-ctx.Done():
				return sent, failed, i, nil
			}
		}

		reason := state.advance(curTraceMsgs, i == 0)

		outbound := append([]json.RawMessage{systemMsg}, state.clientMessages...)
		outboundBytes, err := encodeMessagesPayload(payloadMap, outbound)
		if err != nil {
			failed++
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Error: "encode payload: " + err.Error(),
			})
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}

		// Apply default completion fields (ignore_eos / stop / stop_token_ids
		// and min_tokens := max_tokens if absent). This mirrors
		// _prepare_payload in trace_load_generator_v2.py.
		outboundBytes, maxTokens := injectAgentDefaults(outboundBytes)

		if i == 0 {
			log.Printf("[%s %d/%d] mode=%s first payload reset=%v reason=%s messages_len=%d",
				agent.Name, requestIndex, totalRows, g.Mode, reason != resetNone, reason, len(outbound))
		}

		firedAt := time.Since(runStart).Seconds()
		startReq := time.Now()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.url, bytes.NewReader(outboundBytes))
		if err != nil {
			failed++
			state.lastRequestOK = false
			state.prevTraceMessages = curTraceMsgs
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Error: "newreq: " + err.Error(),
			})
			g.writeGlobal(agentResultRecord{
				Agent: agent.Name, RequestIndex: requestIndex, FiredAtSec: firedAt,
				AttemptNum: state.currentAttempt, ResetReason: string(reason),
				MessagesLen: len(outbound), MaxTokens: maxTokens,
				Error: "newreq: " + err.Error(),
			})
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}
		for k, v := range g.headers {
			req.Header[k] = v
		}

		resp, err := g.http.Do(req)
		latency := time.Since(startReq).Seconds()

		if err != nil {
			failed++
			state.lastRequestOK = false
			state.prevTraceMessages = curTraceMsgs
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				LatencySec: latency, MaxTokens: maxTokens, Error: err.Error(),
			})
			g.writeGlobal(agentResultRecord{
				Agent: agent.Name, RequestIndex: requestIndex, FiredAtSec: firedAt,
				AttemptNum: state.currentAttempt, ResetReason: string(reason),
				MessagesLen: len(outbound), MaxTokens: maxTokens,
				LatencySec: latency, Error: err.Error(),
			})
			log.Printf("[%s %d/%d] request error: %v", agent.Name, requestIndex, totalRows, err)
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}

		var bodyRaw json.RawMessage
		var bodyBytes []byte
		if g.Common.RecordBody {
			bodyBytes, err = io.ReadAll(resp.Body)
			resp.Body.Close()
			if err == nil && json.Valid(bodyBytes) {
				bodyRaw = json.RawMessage(bodyBytes)
			} else if err == nil {
				wrapped, _ := json.Marshal(map[string]string{"raw_text": string(bodyBytes)})
				bodyRaw = json.RawMessage(wrapped)
			}
		} else {
			bodyBytes, _ = io.ReadAll(resp.Body)
			resp.Body.Close()
			if json.Valid(bodyBytes) {
				bodyRaw = json.RawMessage(bodyBytes)
			}
		}

		reqOK := resp.StatusCode < 400
		if !reqOK {
			failed++
			state.lastRequestOK = false
			state.prevTraceMessages = curTraceMsgs
			responses = append(responses, perAgentResponseRow{
				Agent: agent.Name, RequestIndex: requestIndex, OK: false,
				Status: resp.StatusCode, LatencySec: latency, MaxTokens: maxTokens,
				Response: bodyRaw,
			})
			g.writeGlobal(agentResultRecord{
				Agent: agent.Name, RequestIndex: requestIndex, FiredAtSec: firedAt,
				AttemptNum: state.currentAttempt, ResetReason: string(reason),
				MessagesLen: len(outbound), MaxTokens: maxTokens,
				OK: false, Status: resp.StatusCode, LatencySec: latency,
				Response: bodyRaw,
			})
			log.Printf("[%s %d/%d] HTTP %d max_tokens=%d", agent.Name, requestIndex, totalRows, resp.StatusCode, maxTokens)
			if !g.Common.ContinueOnError {
				break
			}
			continue
		}

		sent++
		state.lastRequestOK = true
		state.prevTraceMessages = curTraceMsgs
		assistantContent := extractAssistantContent(bodyRaw)
		state.appendAssistant(assistantContent)

		responses = append(responses, perAgentResponseRow{
			Agent: agent.Name, RequestIndex: requestIndex, OK: true,
			Status: resp.StatusCode, LatencySec: latency, MaxTokens: maxTokens,
			Response: bodyRaw,
		})
		g.writeGlobal(agentResultRecord{
			Agent: agent.Name, RequestIndex: requestIndex, FiredAtSec: firedAt,
			AttemptNum: state.currentAttempt, ResetReason: string(reason),
			MessagesLen: len(outbound), MaxTokens: maxTokens,
			OK: true, Status: resp.StatusCode, LatencySec: latency,
			Response: bodyRaw,
		})
		if g.Common.LogEvery > 0 && (sent%g.Common.LogEvery == 0 || i == totalRows-1) {
			log.Printf("[%s %d/%d] ok status=%d max_tokens=%d messages_len=%d attempt=%d %s",
				agent.Name, requestIndex, totalRows, resp.StatusCode, maxTokens, len(outbound),
				state.currentAttempt, ternary(reason != resetNone, "reset", "delta"))
		}
	}

	if err := writePerAgentResponses(responsePath, responses); err != nil {
		log.Printf("[%s] write responses: %v", agent.Name, err)
	}
	return sent, failed, totalRows, nil
}

func (g *AgentLoadGen) writeGlobal(r agentResultRecord) {
	if g.writer == nil {
		return
	}
	g.writer.Write(r)
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

func parseAgentDirs(raw []byte) ([]agentSpec, error) {
	// Accept either a bare array of {dir, tag, name} or {"agent_dirs": [...]}.
	var asArr []map[string]any
	if err := json.Unmarshal(raw, &asArr); err == nil {
		return agentSpecsFromMap(asArr)
	}
	var asObj map[string]any
	if err := json.Unmarshal(raw, &asObj); err != nil {
		return nil, fmt.Errorf("not a JSON array or object")
	}
	inner, ok := asObj["agent_dirs"].([]any)
	if !ok {
		return nil, fmt.Errorf("'agent_dirs' field missing or not an array")
	}
	conv := make([]map[string]any, 0, len(inner))
	for _, e := range inner {
		m, ok := e.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("entry is not an object")
		}
		conv = append(conv, m)
	}
	return agentSpecsFromMap(conv)
}

func agentSpecsFromMap(items []map[string]any) ([]agentSpec, error) {
	out := make([]agentSpec, 0, len(items))
	for i, item := range items {
		dir, _ := item["dir"].(string)
		tag, _ := item["tag"].(string)
		name, _ := item["name"].(string)
		if strings.TrimSpace(dir) == "" {
			return nil, fmt.Errorf("entry[%d]: missing 'dir'", i)
		}
		if strings.TrimSpace(tag) == "" {
			return nil, fmt.Errorf("entry[%d]: missing 'tag'", i)
		}
		if name == "" {
			name = filepath.Base(strings.TrimRight(dir, "/"))
		}
		out = append(out, agentSpec{
			Dir:  strings.TrimSpace(dir),
			Tag:  tag,
			Name: name,
		})
	}

	// Reject duplicate agent names. Each name ends up as the system prompt
	// content ("I am agent <name>") whose whole purpose is to give every
	// agent a unique prefix; a collision defeats prefix-cache isolation
	// silently, so fail fast instead.
	seen := make(map[string]int, len(out))
	for i, a := range out {
		if prev, dup := seen[a.Name]; dup {
			return nil, fmt.Errorf("entry[%d] and entry[%d] both have name=%q "+
				"(explicit or fallback from dir basename) — add an explicit unique \"name\" to at least one; "+
				"duplicate names collapse the per-agent system prompt and break prefix-cache isolation",
				prev, i, a.Name)
		}
		seen[a.Name] = i
	}
	return out, nil
}

func computeStartDelays(n int, spec string, seed int64) ([]float64, error) {
	out := make([]float64, n)
	if spec == "" {
		return out, nil
	}
	parts := strings.SplitN(spec, ":", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("stagger must be 'poisson:<mean>' or 'uniform:<lo>,<hi>', got %q", spec)
	}
	kind := strings.ToLower(strings.TrimSpace(parts[0]))
	rest := strings.TrimSpace(parts[1])

	var rng *rand.Rand
	if seed == 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	switch kind {
	case "poisson":
		mean, err := strconv.ParseFloat(rest, 64)
		if err != nil || mean <= 0 {
			return nil, fmt.Errorf("poisson mean must be > 0: %v", err)
		}
		cum := 0.0
		for i := 1; i < n; i++ {
			cum += rng.ExpFloat64() * mean
			out[i] = cum
		}
		return out, nil
	case "uniform":
		pair := strings.SplitN(rest, ",", 2)
		if len(pair) != 2 {
			return nil, fmt.Errorf("uniform requires 'lo,hi'")
		}
		lo, err1 := strconv.ParseFloat(strings.TrimSpace(pair[0]), 64)
		hi, err2 := strconv.ParseFloat(strings.TrimSpace(pair[1]), 64)
		if err1 != nil || err2 != nil || lo < 0 || hi < lo {
			return nil, fmt.Errorf("uniform requires 0 <= lo <= hi")
		}
		for i := 0; i < n; i++ {
			out[i] = lo + rng.Float64()*(hi-lo)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown stagger kind: %q", kind)
}

func randomAttemptDelay(rng *rand.Rand) float64 {
	if rng.Float64() < 0.8 {
		return rng.Float64() * 10
	}
	return 10 + rng.Float64()*290
}

func expandUser(p string) string {
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

func agentResponsePath(agentDir, agentName, responseOutputDir string) string {
	dir := expandUser(responseOutputDir)
	if filepath.IsAbs(dir) {
		return filepath.Join(dir, agentName, "load_responses.json")
	}
	return filepath.Join(agentDir, dir, "load_responses.json")
}

func writePerAgentResponses(path string, rows []perAgentResponseRow) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	buf, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, buf, 0o644)
}

// encodeMessagesPayload rebuilds the JSON payload bytes with "messages"
// replaced by outbound. All other fields of payloadMap are preserved
// verbatim via their RawMessage encodings.
func encodeMessagesPayload(payloadMap map[string]json.RawMessage, outbound []json.RawMessage) ([]byte, error) {
	msgsBytes, err := json.Marshal(outbound)
	if err != nil {
		return nil, err
	}
	payloadMap["messages"] = msgsBytes
	return json.Marshal(payloadMap)
}

// injectAgentDefaults adds ignore_eos / stop / stop_token_ids if absent and
// sets min_tokens := max_tokens when max_tokens is present but min_tokens is
// not. Returns the updated payload bytes and the extracted max_tokens (0 if
// absent). Mirrors _prepare_payload in v2.
func injectAgentDefaults(payload []byte) ([]byte, int) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(payload, &m); err != nil {
		return payload, 0
	}

	var maxTokens int
	if raw, ok := m["max_tokens"]; ok {
		_ = json.Unmarshal(raw, &maxTokens)
		if _, hasMin := m["min_tokens"]; !hasMin {
			m["min_tokens"] = raw
		}
	}
	if _, ok := m["ignore_eos"]; !ok {
		m["ignore_eos"] = json.RawMessage("true")
	}
	if _, ok := m["stop"]; !ok {
		m["stop"] = json.RawMessage("[]")
	}
	if _, ok := m["stop_token_ids"]; !ok {
		m["stop_token_ids"] = json.RawMessage("[]")
	}

	out, err := json.Marshal(m)
	if err != nil {
		return payload, maxTokens
	}
	return out, maxTokens
}

// extractAssistantContent pulls choices[0].message.content from a chat
// completion response body. Returns "" if the shape doesn't match.
func extractAssistantContent(body json.RawMessage) string {
	if len(body) == 0 {
		return ""
	}
	var shape struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &shape); err != nil {
		return ""
	}
	if len(shape.Choices) == 0 {
		return ""
	}
	return shape.Choices[0].Message.Content
}

func mustMarshal(v any) []byte {
	buf, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return buf
}

func ternary(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

// seedFor derives a deterministic per-agent RNG seed from the agent name
// and a base seed; used for attempt-mode delay so replays are reproducible.
func seedFor(name string, base int64) int64 {
	h := int64(0)
	for _, c := range name {
		h = h*131 + int64(c)
	}
	return base ^ h
}
