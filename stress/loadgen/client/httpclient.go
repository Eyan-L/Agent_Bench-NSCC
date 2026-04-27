// Package client builds the shared *http.Client used by every LoadGenerator.
// Keeping the construction in one place means MaxInflight + Timeout get
// applied uniformly across subcommands.
package client

import (
	"net/http"
	"time"
)

// New returns an *http.Client tuned for high-concurrency request bursts.
// maxInflight is used to size the connection pool; timeout caps each
// request end-to-end.
func New(timeout time.Duration, maxInflight int) *http.Client {
	if maxInflight <= 0 {
		maxInflight = 1
	}
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			MaxIdleConns:        maxInflight * 2,
			MaxIdleConnsPerHost: maxInflight * 2,
			MaxConnsPerHost:     maxInflight * 2,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}

// Headers returns the common Content-Type / Authorization header pair used
// by every subclass. apiKey == "" means no Authorization header is sent.
func Headers(apiKey string) http.Header {
	h := http.Header{"Content-Type": []string{"application/json"}}
	if apiKey != "" {
		h.Set("Authorization", "Bearer "+apiKey)
	}
	return h
}
