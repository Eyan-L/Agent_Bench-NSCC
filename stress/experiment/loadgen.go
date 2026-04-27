package experiment

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

// RunLoadgen builds the loadgen binary if needed and invokes the correct
// subcommand, writing results to <outDir>/<result-name> and stdout/stderr to
// <outDir>/loadgen.log. Returns the result-file path on success.
func RunLoadgen(ctx context.Context, cfg *Config, projectRoot, outDir string) (string, error) {
	bin, err := buildLoadgen(ctx, projectRoot, outDir)
	if err != nil {
		return "", err
	}

	apiBase := fmt.Sprintf("http://127.0.0.1:%d/v1", cfg.Sglang.Port)
	resultFile, args := buildLoadgenArgs(cfg, outDir, apiBase)

	log.Printf("running %s %v", filepath.Base(bin), args[:1])
	logPath := filepath.Join(outDir, "loadgen.log")
	logF, err := os.Create(logPath)
	if err != nil {
		return "", fmt.Errorf("create loadgen.log: %w", err)
	}
	defer logF.Close()

	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdout = logF
	cmd.Stderr = logF
	if err := cmd.Run(); err != nil {
		return resultFile, fmt.Errorf("loadgen exited with error (see %s): %w", logPath, err)
	}
	log.Printf("loadgen finished; result -> %s", resultFile)
	return resultFile, nil
}

// buildLoadgenArgs assembles CLI flags from the config, returning the result
// file path plus the args slice (subcommand included).
func buildLoadgenArgs(cfg *Config, outDir, apiBase string) (resultFile string, args []string) {
	resultFile = filepath.Join(outDir, defaultResultName(cfg.Loadgen.TraceType))
	lg := cfg.Loadgen

	// Common flags apply to every subcommand.
	common := []string{
		"-api-base", apiBase,
		"-api-key", defaultStr(lg.APIKey, "EMPTY"),
		"-model", lg.Model,
		"-timeout", strconv.FormatFloat(lg.TimeoutSec, 'f', -1, 64),
		"-speedup", strconv.FormatFloat(lg.Speedup, 'f', -1, 64),
		"-start-offset", strconv.FormatFloat(lg.StartOffset, 'f', -1, 64),
		"-max-requests", strconv.Itoa(lg.MaxRequests),
		"-max-inflight", strconv.Itoa(lg.MaxInflight),
		"-log-every", strconv.Itoa(lg.LogEvery),
		"-output", resultFile,
	}
	if lg.Duration > 0 {
		common = append(common, "-duration", strconv.FormatFloat(lg.Duration, 'f', -1, 64))
	}
	if lg.RecordBody {
		common = append(common, "-record-body=true")
	} else {
		common = append(common, "-record-body=false")
	}

	switch lg.TraceType {
	case "alibaba":
		a := lg.Alibaba
		args = append([]string{"alibaba"}, common...)
		args = append(args,
			"-trace", a.Trace,
			"-session-sample-rate", strconv.FormatFloat(a.SessionSampleRate, 'f', -1, 64),
			"-session-sample-seed", strconv.FormatInt(a.SessionSampleSeed, 10),
		)
		if a.TypeFilter != "" {
			args = append(args, "-type-filter", a.TypeFilter)
		}
	case "azure":
		z := lg.Azure
		args = append([]string{"azure"}, common...)
		args = append(args,
			"-trace", z.Trace,
			"-sampling-rate", strconv.FormatFloat(z.SamplingRate, 'f', -1, 64),
			"-prompt-name", z.PromptName,
			"-recur-step", strconv.Itoa(z.RecurStep),
			"-seed", strconv.FormatInt(z.Seed, 10),
			"-max-drift-ms", strconv.Itoa(z.MaxDriftMs),
		)
		if z.SaveSampled != "" {
			args = append(args, "-save-sampled", z.SaveSampled)
		}
	case "agent":
		g := lg.Agent
		args = append([]string{"agent"}, common...)
		args = append(args,
			"-agent-dirs-json", g.AgentDirsJSON,
			"-mode", g.Mode,
			"-concurrent-agents", strconv.Itoa(g.ConcurrentAgents),
			"-max-agents", strconv.Itoa(g.MaxAgents),
			"-response-output-dir", g.ResponseOutputDir,
			"-stagger-seed", strconv.FormatInt(g.StaggerSeed, 10),
		)
		if g.StartStagger != "" {
			args = append(args, "-agent-start-stagger", g.StartStagger)
		}
	}
	return resultFile, args
}

// defaultResultName keeps the file name aligned with what the legacy shell
// scripts wrote, so existing plot scripts / slo_attainment.py pick it up
// without special-casing.
func defaultResultName(traceType string) string {
	switch traceType {
	case "alibaba":
		return "responses.json"
	case "azure":
		return "result.json"
	case "agent":
		return "loadgen.jsonl"
	}
	return "result.json"
}

func defaultStr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

// buildLoadgen compiles /loadgen/cmd/loadgen to <projectRoot>/loadgen/loadgen-bin
// if the binary is missing or older than any .go file in the module. Writes
// build output to <outDir>/loadgen_build.log on failure.
func buildLoadgen(ctx context.Context, projectRoot, outDir string) (string, error) {
	modDir := filepath.Join(projectRoot, "stress", "loadgen")
	bin := filepath.Join(modDir, "loadgen-bin")

	if needsBuild, err := binOlderThanSources(bin, modDir); err != nil {
		return "", err
	} else if !needsBuild {
		return bin, nil
	}

	goBin := os.Getenv("GO_BIN")
	if goBin == "" {
		goBin = "/mnt/vdb/hongrui/opt/go/bin/go"
		if _, err := os.Stat(goBin); err != nil {
			goBin = "go"
		}
	}
	log.Printf("building loadgen with %s", goBin)
	cmd := exec.CommandContext(ctx, goBin, "build", "-o", bin, "./cmd/loadgen")
	cmd.Dir = modDir
	cmd.Env = append(os.Environ(),
		"GOPATH="+defaultStr(os.Getenv("GOPATH"), os.Getenv("HOME")+"/go"),
		"GOCACHE="+defaultStr(os.Getenv("GOCACHE"), os.Getenv("HOME")+"/.cache/go-build"),
	)
	buildLog := filepath.Join(outDir, "loadgen_build.log")
	f, err := os.Create(buildLog)
	if err != nil {
		return "", fmt.Errorf("create build log: %w", err)
	}
	defer f.Close()
	cmd.Stdout = f
	cmd.Stderr = f
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("go build failed (see %s): %w", buildLog, err)
	}
	return bin, nil
}

func binOlderThanSources(bin, srcDir string) (bool, error) {
	binInfo, err := os.Stat(bin)
	if err != nil {
		if os.IsNotExist(err) {
			return true, nil
		}
		return false, err
	}
	var newest time.Time
	err = filepath.Walk(srcDir, func(path string, info os.FileInfo, werr error) error {
		if werr != nil {
			return werr
		}
		if info.IsDir() {
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		return nil
	})
	if err != nil {
		return false, err
	}
	return newest.After(binInfo.ModTime()), nil
}
