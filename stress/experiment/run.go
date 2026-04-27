package experiment

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Run is the end-to-end orchestration: resolve output dir, snapshot config,
// launch sglang, (optionally) start SM trace, run loadgen, stop everything,
// then fire the plotting scripts.
func Run(ctx context.Context, cfg *Config, projectRoot string) (string, error) {
	now := time.Now()
	outDir, err := cfg.ResolveOutputDir(projectRoot, now)
	if err != nil {
		return "", err
	}

	// Mirror the standard logger to <outDir>/runner.log so the experiment
	// directory is self-contained, matching what the legacy shell script
	// produced. Everything routed through log.Print* lands in both places.
	runnerLog, err := os.Create(filepath.Join(outDir, "runner.log"))
	if err != nil {
		return outDir, fmt.Errorf("create runner.log: %w", err)
	}
	defer runnerLog.Close()
	prevOut := log.Writer()
	log.SetOutput(io.MultiWriter(prevOut, runnerLog))
	defer log.SetOutput(prevOut)

	log.Printf("experiment: %s (trace_type=%s)", cfg.Name, cfg.Loadgen.TraceType)
	log.Printf("output_dir=%s", outDir)
	if err := cfg.WriteSnapshot(outDir); err != nil {
		return outDir, fmt.Errorf("write run_config.json: %w", err)
	}

	sglang, err := StartSglang(ctx, cfg, projectRoot, outDir)
	// Stop must run even if StartSglang reported a partial failure — the
	// launcher may have left a worker mid-init that we still own.
	if sglang != nil {
		defer sglang.Stop(context.Background())
	}
	if err != nil {
		return outDir, fmt.Errorf("start sglang: %w", err)
	}

	var smTrace *smTraceHandle
	if cfg.Plots.SMTrace {
		smTrace, err = startSMTrace(ctx, outDir)
		if err != nil {
			log.Printf("sm_trace: %v (continuing without it)", err)
		}
	}

	resultFile, loadgenErr := RunLoadgen(ctx, cfg, projectRoot, outDir)

	if smTrace != nil {
		smTrace.Stop()
	}

	// Plots: run even if loadgen hit a non-zero exit, since partial results
	// are often still analyzable.
	RunPlots(context.Background(), cfg, projectRoot, outDir, resultFile)

	if loadgenErr != nil {
		return outDir, loadgenErr
	}
	return outDir, nil
}

// smTraceHandle wraps the dcgmi dmon background process started when
// cfg.Plots.SMTrace=true. Stop() sends SIGTERM and waits for the child.
type smTraceHandle struct {
	cmd     *exec.Cmd
	logFile *os.File
}

func startSMTrace(ctx context.Context, outDir string) (*smTraceHandle, error) {
	if _, err := exec.LookPath("dcgmi"); err != nil {
		return nil, fmt.Errorf("dcgmi not available: %w", err)
	}
	logPath := filepath.Join(outDir, "sm.log")
	f, err := os.Create(logPath)
	if err != nil {
		return nil, err
	}
	// Matches run_alibaba_trace_experiment.sh: SMACT, SMOCC, DRAMA, PCITX, PCIRX.
	cmd := exec.CommandContext(ctx, "dcgmi", "dmon", "-e", "1002,1003,1005,1009,1010", "-d", "1000")
	cmd.Stdout = f
	cmd.Stderr = f
	if err := cmd.Start(); err != nil {
		f.Close()
		return nil, err
	}
	log.Printf("sm_trace: dcgmi dmon -> %s (pid=%d)", logPath, cmd.Process.Pid)
	return &smTraceHandle{cmd: cmd, logFile: f}, nil
}

func (h *smTraceHandle) Stop() {
	if h == nil {
		return
	}
	if h.cmd.Process != nil {
		_ = h.cmd.Process.Signal(os.Interrupt)
	}
	_ = h.cmd.Wait()
	if h.logFile != nil {
		h.logFile.Close()
	}
}
