package experiment

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// RunPlots invokes the post-run analysis scripts configured under
// cfg.Plots. Each script failure is logged but does NOT fail the whole
// experiment — plotting is best-effort.
func RunPlots(ctx context.Context, cfg *Config, projectRoot, outDir, resultFile string) {
	if !cfg.Plots.Enabled {
		return
	}
	drawDir := filepath.Join(projectRoot, "analysis_tools", "draw")
	sglangLog := filepath.Join(outDir, "sglang.log")
	smLog := filepath.Join(outDir, "sm.log")

	pyBin := cfg.Plots.PythonBin
	if pyBin == "" {
		pyBin = "python3"
	}
	prelude := strings.TrimSpace(cfg.Plots.EnvPrelude)
	if prelude != "" {
		log.Printf("[plots] prelude=%q python=%s", prelude, pyBin)
	}

	type plot struct {
		name   string
		enable bool
		cmd    []string // args[0] is the script, args[1:] are flags
		stdout string   // optional: capture stdout to this file
	}
	plots := []plot{
		{
			name:   "experiment-result",
			enable: cfg.Plots.Result,
			cmd: []string{
				filepath.Join(drawDir, "plot_experiment_result.py"),
				outDir,
			},
		},
		{
			name:   "sglang-tflops",
			enable: cfg.Plots.SglangTflops && fileExists(sglangLog),
			cmd: []string{
				filepath.Join(drawDir, "plot_sglang_tflops_over_time.py"),
				sglangLog,
				"--output-plot", filepath.Join(outDir, "sglang_tflops_over_time.png"),
			},
		},
		{
			name:   "sglang-tflops-per-gb",
			enable: cfg.Plots.SglangTflopsPerGB && fileExists(sglangLog),
			cmd: []string{
				filepath.Join(drawDir, "plot_sglang_tflops_per_gb_over_time.py"),
				sglangLog,
				"--output-plot", filepath.Join(outDir, "sglang_tflops_per_gb_over_time.png"),
			},
		},
		{
			name:   "sglang-token-usage",
			enable: cfg.Plots.SglangTokenUsage && fileExists(sglangLog),
			cmd: []string{
				filepath.Join(drawDir, "plot_sglang_token_usage_over_time.py"),
				sglangLog,
				"--output-plot", filepath.Join(outDir, "sglang_token_usage_over_time.png"),
			},
		},
		{
			name:   "sm-metrics",
			enable: cfg.Plots.SMMetrics && fileExists(smLog),
			cmd: []string{
				filepath.Join(drawDir, "plot_sm_metrics_over_time.py"),
				smLog,
				"--output-plot", filepath.Join(outDir, "sm_metrics_over_time.png"),
			},
		},
		{
			name:   "slo-attainment",
			enable: cfg.Plots.SLOAttainment && fileExists(resultFile),
			cmd: []string{
				filepath.Join(drawDir, "slo_attainment.py"),
				resultFile,
				"--summary-out", filepath.Join(outDir, "slo_summary.json"),
			},
			stdout: filepath.Join(outDir, "slo_summary.txt"),
		},
		// Alibaba-only: needs token_ids + parent_chat_id from the raw trace,
		// which only the qwen-bailian JSONL traces carry.
		{
			name:   "cache-hit-vs-theoretical",
			enable: cfg.Plots.CacheHit && cfg.Loadgen.TraceType == "alibaba" && fileExists(resultFile),
			cmd: []string{
				filepath.Join(drawDir, "plot_cache_hit_vs_theoretical.py"),
				"--experiment-dir", outDir,
			},
		},
	}

	for _, p := range plots {
		if !p.enable {
			continue
		}
		runPlot(ctx, p.name, pyBin, prelude, p.cmd, p.stdout, outDir)
	}
}

// runPlot invokes one plotting script. When prelude is non-empty the whole
// thing is executed through `bash -c "<prelude> && <pyBin> <script> <args>"`
// so that e.g. NSCC's `module load miniforge3` runs in the same shell that
// then calls python. Without a prelude we exec pyBin directly — one less
// shell in the pipeline.
func runPlot(ctx context.Context, name, pyBin, prelude string, scriptArgs []string, stdoutFile, outDir string) {
	var cmd *exec.Cmd
	if prelude == "" {
		full := append([]string{pyBin}, scriptArgs...)
		log.Printf("[plot:%s] %v", name, full)
		cmd = exec.CommandContext(ctx, full[0], full[1:]...)
	} else {
		// Quote each arg for safe interpolation into bash -c "..."
		quoted := make([]string, len(scriptArgs))
		for i, a := range scriptArgs {
			quoted[i] = shellQuote(a)
		}
		shellCmd := fmt.Sprintf("%s && %s %s", prelude, shellQuote(pyBin), strings.Join(quoted, " "))
		log.Printf("[plot:%s] bash -c %q", name, shellCmd)
		cmd = exec.CommandContext(ctx, "bash", "-c", shellCmd)
	}
	logPath := filepath.Join(outDir, "plot_"+name+".log")
	logF, err := os.Create(logPath)
	if err != nil {
		log.Printf("[plot:%s] create log: %v", name, err)
		return
	}
	defer logF.Close()

	cmd.Stderr = logF
	if stdoutFile != "" {
		out, err := os.Create(stdoutFile)
		if err != nil {
			log.Printf("[plot:%s] create stdout file: %v", name, err)
			cmd.Stdout = logF
		} else {
			defer out.Close()
			cmd.Stdout = out
		}
	} else {
		cmd.Stdout = logF
	}
	if err := cmd.Run(); err != nil {
		log.Printf("[plot:%s] failed (see %s): %v", name, logPath, err)
		return
	}
	log.Printf("[plot:%s] ok", name)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// shellQuote single-quotes s for safe interpolation inside a bash -c string.
// Any embedded ' is escaped as '\'' per the standard shell pattern.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// EnsureProjectRoot resolves --project-root if supplied; otherwise walks up
// from the current binary to find a marker file (stress/loadgen/go.mod + run/
// directory). Returns an absolute path.
func EnsureProjectRoot(explicit string) (string, error) {
	if explicit != "" {
		abs, err := filepath.Abs(explicit)
		if err != nil {
			return "", err
		}
		return abs, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	start, _ := filepath.EvalSymlinks(exe)
	if start == "" {
		start = exe
	}
	dir := filepath.Dir(start)
	for i := 0; i < 6; i++ {
		if fileExists(filepath.Join(dir, "stress", "loadgen", "go.mod")) &&
			fileExists(filepath.Join(dir, "backend", "sglang", "serve.sh")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	// Fallback: caller's CWD if it looks like the project root.
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if fileExists(filepath.Join(cwd, "stress", "loadgen", "go.mod")) {
		return cwd, nil
	}
	return "", fmt.Errorf("could not locate project root (no stress/loadgen/go.mod + backend/sglang/serve.sh found); pass -project-root explicitly")
}
