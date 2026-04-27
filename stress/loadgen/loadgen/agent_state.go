package loadgen

import (
	"bytes"
	"encoding/json"
)

// agentState tracks the incremental message history for a single virtual
// agent, mirroring the algorithm in scripts/python/trace_load_generator_v2.py
// (_send_for_one_agent). The state is owned by exactly one goroutine (the
// agent's worker), so no locking is needed.
type agentState struct {
	clientMessages    []json.RawMessage
	prevTraceMessages []json.RawMessage
	lastRequestOK     bool
	currentAttempt    int
}

// resetReason enumerates why we had to rebase the local history on the trace
// messages wholesale. Empty string = normal delta extension.
type resetReason string

const (
	resetNone                 resetReason = ""
	resetInitial              resetReason = "initial"
	resetPrevFailed           resetReason = "prev_failed"
	resetCompactionOrRebase   resetReason = "compaction_or_rebase"
	resetEmptyDelta           resetReason = "empty_delta"
	resetDeltaHeadNotAssistant resetReason = "delta_head_not_assistant"
)

// advance updates the state given the current trace messages (from this
// request's plan row) and whether this is the very first request for the
// agent. It returns the reason for any forced reset; an empty string means
// the history was extended incrementally.
//
// Equivalent Python lines (v2): 229-255.
func (s *agentState) advance(curTraceMsgs []json.RawMessage, isFirst bool) resetReason {
	reason := resetNone

	switch {
	case isFirst:
		reason = resetInitial
	case !s.lastRequestOK:
		reason = resetPrevFailed
	case len(curTraceMsgs) < len(s.prevTraceMessages) || !prefixEqual(curTraceMsgs, s.prevTraceMessages):
		reason = resetCompactionOrRebase
	default:
		delta := curTraceMsgs[len(s.prevTraceMessages):]
		switch {
		case len(delta) == 0:
			reason = resetEmptyDelta
		case !isAssistantMessage(delta[0]):
			reason = resetDeltaHeadNotAssistant
		default:
			// Normal extension: append delta[1:] (skip the assistant head,
			// keep tool results + new user messages).
			for _, m := range delta[1:] {
				s.clientMessages = append(s.clientMessages, cloneRaw(m))
			}
		}
	}

	if reason != resetNone {
		s.clientMessages = make([]json.RawMessage, len(curTraceMsgs))
		for i, m := range curTraceMsgs {
			s.clientMessages[i] = cloneRaw(m)
		}
	}
	return reason
}

// appendAssistant tacks the server-returned assistant content onto the local
// history. Called after a successful request; equivalent to python v2 line 319.
func (s *agentState) appendAssistant(content string) {
	msg := struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{Role: "assistant", Content: content}
	buf, err := json.Marshal(msg)
	if err != nil {
		// Marshal of such a simple struct cannot fail in practice; on the
		// off chance it does, skip the append rather than panicking.
		return
	}
	s.clientMessages = append(s.clientMessages, json.RawMessage(buf))
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

// prefixEqual reports whether b is a prefix of a (byte-wise on the RawMessage
// encodings). Since both slices come from the same JSON plan file loaded
// twice, identical prior messages will have identical byte encodings.
func prefixEqual(a, b []json.RawMessage) bool {
	if len(a) < len(b) {
		return false
	}
	for i := range b {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

// isAssistantMessage reports whether raw is a JSON object with "role" ==
// "assistant". It tolerates extra fields.
func isAssistantMessage(raw json.RawMessage) bool {
	var probe struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	return probe.Role == "assistant"
}

// cloneRaw returns a deep copy of a RawMessage so that later appends to one
// slice can't accidentally share backing storage with another.
func cloneRaw(raw json.RawMessage) json.RawMessage {
	out := make(json.RawMessage, len(raw))
	copy(out, raw)
	return out
}

// countUserMessages returns how many of the raw messages have role == "user".
func countUserMessages(msgs []json.RawMessage) int {
	n := 0
	for _, m := range msgs {
		if isUserMessage(m) {
			n++
		}
	}
	return n
}

// isUserMessage reports whether raw is a JSON object whose role field equals
// "user". Tolerates extra fields.
func isUserMessage(raw json.RawMessage) bool {
	var probe struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	return probe.Role == "user"
}

// lastUserMessage returns the last element of msgs with role == "user", or
// nil if there is none.
func lastUserMessage(msgs []json.RawMessage) json.RawMessage {
	for i := len(msgs) - 1; i >= 0; i-- {
		if isUserMessage(msgs[i]) {
			return msgs[i]
		}
	}
	return nil
}

// isNewAttemptByTail reports whether cur introduces a fresh user-initiated
// attempt vs prev. True when the LAST user message in cur is not
// byte-identical to the last user message in prev (nil prev => true iff cur
// has any user message at all).
//
// This supersedes v2.py's `count(user in cur) > count(user in prev)` rule,
// which silently misses attempts whose openings coincide with a compaction:
// compaction replaces a tail of old user messages with a single "summary"
// pseudo-user message, so the count can drop (e.g. 4 → 2) even when a
// genuine new user question has been added on top.
func isNewAttemptByTail(cur, prev []json.RawMessage) bool {
	curLast := lastUserMessage(cur)
	if curLast == nil {
		return false
	}
	prevLast := lastUserMessage(prev)
	return !bytes.Equal(curLast, prevLast)
}
