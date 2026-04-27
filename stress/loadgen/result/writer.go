// Package result provides a thread-safe JSONL writer shared by every
// LoadGenerator implementation. Each record is flushed to the OS as soon as
// it's written so that a mid-run SIGTERM (e.g. from the outer `timeout`
// wrapper) still leaves a complete file of already-finished records on disk.
package result

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sync"
)

// Writer serializes Record writes to a JSONL file.
type Writer struct {
	mu    sync.Mutex
	f     *os.File
	count int64
}

// New creates the parent directory (if missing) and opens path for writing,
// truncating any existing file.
func New(path string) (*Writer, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Writer{f: f}, nil
}

// Write serializes v as a single JSON line. v is typically a struct with
// the per-request metrics defined by each LoadGenerator subclass.
//
// Errors during marshal or write are logged but not returned; we treat
// partial output as the operative failure mode and prefer to keep replaying.
func (w *Writer) Write(v any) {
	if w == nil {
		return
	}
	line, err := json.Marshal(v)
	if err != nil {
		log.Printf("result: marshal: %v", err)
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, err := w.f.Write(line); err != nil {
		log.Printf("result: write: %v", err)
		return
	}
	if _, err := w.f.Write([]byte{'\n'}); err != nil {
		log.Printf("result: newline: %v", err)
		return
	}
	w.count++
}

// Count returns how many records have been written so far.
func (w *Writer) Count() int64 {
	if w == nil {
		return 0
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.count
}

// Close flushes and closes the underlying file.
func (w *Writer) Close() error {
	if w == nil || w.f == nil {
		return nil
	}
	return w.f.Close()
}
