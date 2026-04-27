package experiment

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// SglangHandle represents an owned sglang process launched via the bash
// launcher. Stop() tears it down, matching cleanup on SIGTERM/panic.
type SglangHandle struct {
	scriptPath  string // backend/sglang/serve.sh or backend/sglang/serve_multi.sh
	logFile     string
	pidFile     string
	env         []string
	projectRoot string
}

// StartSglang launches sglang via the matching bash launcher and waits until
// /health becomes reachable (or returns an error on timeout).
func StartSglang(ctx context.Context, cfg *Config, projectRoot, outDir string) (*SglangHandle, error) {
	script := filepath.Join(projectRoot, "backend", "sglang", "serve.sh")
	switch cfg.Sglang.Launcher {
	case "multiple":
		script = filepath.Join(projectRoot, "backend", "sglang", "serve_multi.sh")
	case "singularity":
		script = filepath.Join(projectRoot, "backend", "sglang", "serve_singularity.sh")
	}
	if _, err := os.Stat(script); err != nil {
		return nil, fmt.Errorf("sglang launcher not found: %s", script)
	}

	h := &SglangHandle{
		scriptPath:  script,
		logFile:     filepath.Join(outDir, "sglang.log"),
		pidFile:     filepath.Join(outDir, "sglang.pid"),
		projectRoot: projectRoot,
	}
	h.env = buildSglangEnv(cfg, projectRoot, h.logFile, h.pidFile)

	// If sglang is already up at the configured port, skip the launcher —
	// useful for "keep sglang running between runs" workflows.
	primaryHealth := healthURL(cfg.Sglang.Port)
	if probeHealth(primaryHealth) {
		log.Printf("sglang already responding at %s; skipping launcher", primaryHealth)
		h.pidFile = "" // mark as externally owned; Stop becomes a no-op
		h.logFile = ""
		return h, nil
	}

	log.Printf("starting sglang via %s", script)
	cmd := exec.CommandContext(ctx, script, "start")
	cmd.Env = h.env
	cmd.Stdout = os.Stderr // launcher backgrounds sglang itself; its own stdout is short
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("launcher start: %w", err)
	}

	// Wait for /health (single-instance) or worker nodes (multiple).
	timeout := time.Duration(cfg.Sglang.StartupTimeoutSec) * time.Second
	if cfg.Sglang.Launcher == "multiple" {
		// In multiple mode the router port is cfg.Sglang.Port while the
		// workers sit on 30001 / 30002 per sglang_launch_multiple.sh.
		for _, workerPort := range []int{30001, 30002} {
			if err := waitHealthy(ctx, healthURL(workerPort), timeout); err != nil {
				return h, err
			}
		}
	}
	if err := waitHealthy(ctx, primaryHealth, timeout); err != nil {
		return h, err
	}
	log.Printf("sglang ready at %s", primaryHealth)
	return h, nil
}

// Stop terminates the sglang process tree launched via Start. Safe to call
// even if Start short-circuited because sglang was already running.
func (h *SglangHandle) Stop(ctx context.Context) {
	if h == nil || h.pidFile == "" {
		return
	}
	log.Printf("stopping sglang")
	cmd := exec.CommandContext(ctx, h.scriptPath, "stop")
	cmd.Env = h.env
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Printf("sglang stop: %v (continuing)", err)
	}
}

// buildSglangEnv materializes the env vars the launcher script reads.
// Every field has a default in sglang_launcher.sh, so we only set those that
// the caller explicitly provided via the config.
func buildSglangEnv(cfg *Config, projectRoot, logFile, pidFile string) []string {
	env := os.Environ()
	add := func(k, v string) { env = append(env, k+"="+v) }
	add("SGLANG_LOG_FILE", logFile)
	add("SGLANG_PID_FILE", pidFile)
	add("SGLANG_PORT", strconv.Itoa(cfg.Sglang.Port))
	add("SGLANG_HOST", cfg.Sglang.Host)
	add("SGLANG_MODEL_PATH", cfg.Sglang.ModelPath)
	if cfg.Sglang.CudaVisibleDevices != "" {
		add("SGLANG_CUDA_VISIBLE_DEVICES", cfg.Sglang.CudaVisibleDevices)
		// CUDA_VISIBLE_DEVICES is what sglang actually looks at; also forward
		// to keep behavior identical to the legacy shell invocation path.
		add("CUDA_VISIBLE_DEVICES", cfg.Sglang.CudaVisibleDevices)
	}
	add("SGLANG_MEM_FRACTION_STATIC", strconv.FormatFloat(cfg.Sglang.MemFractionStatic, 'f', -1, 64))
	add("SGLANG_PAGE_SIZE", strconv.Itoa(cfg.Sglang.PageSize))
	add("SGLANG_HICACHE_SIZE", strconv.Itoa(cfg.Sglang.HicacheSize))
	add("SGLANG_TOOL_CALL_PARSER", cfg.Sglang.ToolCallParser)
	add("SGLANG_EXTRA_ARGS", cfg.Sglang.ExtraArgs)
	add("SGLANG_HEALTH_URL", healthURL(cfg.Sglang.Port))
	if cfg.Sglang.EnableHierarchicalCache != nil && !*cfg.Sglang.EnableHierarchicalCache {
		add("SGLANG_ENABLE_HIERARCHICAL_CACHE", "0")
	} else {
		add("SGLANG_ENABLE_HIERARCHICAL_CACHE", "1")
	}

	// Forward the extra knobs the singularity launcher expects. Passing them
	// unconditionally is harmless in the non-singularity scripts (they just
	// ignore these env vars).
	if sg := cfg.Sglang.Singularity; sg != nil {
		add("SGLANG_SIF_IMAGE", sg.SIFImage)
		add("SGLANG_MOUNTS", strings.Join(sg.Mounts, " "))
		add("SGLANG_MODULES", strings.Join(sg.Modules, " "))

		// Patch overlay: when enabled, resolve the patch dir to an absolute
		// path (defaulting to <project_root>/sglang_patch) and forward it;
		// the launcher script takes care of probing the SIF for the
		// in-container site-packages path.
		if sg.ApplyPatch {
			patchDir := sg.PatchDir
			if patchDir == "" {
				patchDir = filepath.Join(projectRoot, "sglang_patch")
			}
			add("SGLANG_PATCH_DIR", patchDir)
		}
	}
	return env
}

func healthURL(port int) string {
	return fmt.Sprintf("http://127.0.0.1:%d/health", port)
}

func probeHealth(url string) bool {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode < 500
}

func waitHealthy(ctx context.Context, url string, timeout time.Duration) error {
	log.Printf("waiting up to %s for %s", timeout, url)
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		if probeHealth(url) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("sglang did not become ready within %s at %s", timeout, url)
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
