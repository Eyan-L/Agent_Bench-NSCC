// Package experiment orchestrates end-to-end trace-replay experiments:
// launch sglang, run loadgen, collect results, and drive the draw/*.py plots.
//
// Config is a single JSON file; see config/stress/ for examples.
package experiment

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Config is the top-level JSON document consumed by `experiment -config <path>`.
type Config struct {
	Name      string        `json:"name"`
	OutputDir string        `json:"output_dir"` // {timestamp} and {trace_type} tokens are expanded
	Timezone  string        `json:"timezone"`   // IANA tz name for {timestamp}; default "Asia/Singapore"
	Sglang    SglangConfig  `json:"sglang"`
	Loadgen   LoadgenConfig `json:"loadgen"`
	Plots     PlotsConfig   `json:"plots"`
}

type SglangConfig struct {
	Launcher           string  `json:"launcher"` // "single" (default) | "multiple" | "singularity"
	Port               int     `json:"port"`
	Host               string  `json:"host"`
	CudaVisibleDevices string  `json:"cuda_visible_devices"`
	ModelPath          string  `json:"model_path"`
	MemFractionStatic  float64 `json:"mem_fraction_static"`
	PageSize           int     `json:"page_size"`
	HicacheSize        int     `json:"hicache_size"`
	ToolCallParser     string  `json:"tool_call_parser"`
	ExtraArgs          string  `json:"extra_args"`
	StartupTimeoutSec  int     `json:"startup_timeout_sec"`

	// Whether to pass --enable-hierarchical-cache to sglang. Pointer so the
	// three-valued distinction "unset / true / false" is preserved across
	// JSON round-tripping; applyDefaults fills in true when unset.
	EnableHierarchicalCache *bool `json:"enable_hierarchical_cache,omitempty"`

	// Singularity carries the extra knobs needed when launcher=="singularity"
	// (NSCC 2A pattern). Ignored otherwise.
	Singularity *SingularityConfig `json:"singularity,omitempty"`
}

// SingularityConfig describes how to wrap the sglang.launch_server call
// inside `singularity run`. Paths inside ModelPath (at the parent level)
// should be the in-container paths once you've added the matching bind.
type SingularityConfig struct {
	SIFImage string   `json:"sif_image"`        // absolute path to the .sif image, required
	Mounts   []string `json:"mounts"`           // list of "host:container[:opts]" bind pairs
	Modules  []string `json:"modules,omitempty"` // `module load` names (default: ["singularity"])

	// ApplyPatch, when true, overlays every *.py under PatchDir/sglang on
	// top of the container's sglang site-packages. Before launch we probe
	// the SIF to discover the site-packages path, then add one `--bind`
	// flag per patched file. See sglang_patch/README.md for details.
	ApplyPatch bool `json:"apply_patch,omitempty"`

	// PatchDir is the host path to the sglang_patch tree (must contain a
	// `sglang/` subdir). Empty ⇒ default to <project_root>/sglang_patch.
	PatchDir string `json:"patch_dir,omitempty"`
}

// LoadgenConfig discriminates on TraceType; only the matching sub-struct is
// read. Fields common to all three subcommands live at the top level.
type LoadgenConfig struct {
	TraceType   string  `json:"trace_type"` // alibaba | azure | agent
	Model       string  `json:"model"`
	TimeoutSec  float64 `json:"timeout_sec"`
	Speedup     float64 `json:"speedup"`
	StartOffset float64 `json:"start_offset"`
	Duration    float64 `json:"duration"`
	MaxRequests int     `json:"max_requests"`
	MaxInflight int     `json:"max_inflight"`
	APIKey      string  `json:"api_key"`
	LogEvery    int     `json:"log_every"`
	RecordBody  bool    `json:"record_body"`

	Alibaba *AlibabaArgs `json:"alibaba,omitempty"`
	Azure   *AzureArgs   `json:"azure,omitempty"`
	Agent   *AgentArgs   `json:"agent,omitempty"`
}

type AlibabaArgs struct {
	Trace             string  `json:"trace"`
	TypeFilter        string  `json:"type_filter"`
	SessionSampleRate float64 `json:"session_sample_rate"`
	SessionSampleSeed int64   `json:"session_sample_seed"`
}

type AzureArgs struct {
	Trace        string  `json:"trace"`
	SamplingRate float64 `json:"sampling_rate"`
	PromptName   string  `json:"prompt_name"`
	RecurStep    int     `json:"recur_step"`
	Seed         int64   `json:"seed"`
	MaxDriftMs   int     `json:"max_drift_ms"`
	SaveSampled  string  `json:"save_sampled"`
}

type AgentArgs struct {
	AgentDirsJSON     string `json:"agent_dirs_json"`
	Mode              string `json:"mode"`
	ConcurrentAgents  int    `json:"concurrent_agents"`
	MaxAgents         int    `json:"max_agents"`
	ResponseOutputDir string `json:"response_output_dir"`
	StartStagger      string `json:"start_stagger"`
	StaggerSeed       int64  `json:"stagger_seed"`
}

type PlotsConfig struct {
	Enabled           bool `json:"enabled"`
	SMTrace           bool `json:"sm_trace"` // enables dcgmi dmon capture during the run
	Result            bool `json:"result"`
	SglangTflops      bool `json:"sglang_tflops"`
	SglangTflopsPerGB bool `json:"sglang_tflops_per_gb"` // compute / KV-in-use intensity vs time
	SglangTokenUsage  bool `json:"sglang_token_usage"`
	SMMetrics         bool `json:"sm_metrics"`
	SLOAttainment     bool `json:"slo_attainment"`
	CacheHit          bool `json:"cache_hit"` // alibaba-only: actual vs theoretical cache hit tokens

	// Optional shell prefix executed before each plot invocation. When
	// non-empty, the plot command is run as `bash -c "<prelude> && <bin> ..."`.
	// Intended for environments that need `module load miniforge3` or
	// `conda activate <env>` before python3 is on PATH (e.g. NSCC).
	EnvPrelude string `json:"env_prelude"`

	// Override the Python binary used by plot scripts (default: "python3").
	// Useful if a conda env has a specific interpreter path.
	PythonBin string `json:"python_bin"`
}

// LoadConfig reads path, applies defaults, validates, and returns the Config.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	cfg.applyDefaults()
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return &cfg, nil
}

func (c *Config) applyDefaults() {
	if c.OutputDir == "" {
		c.OutputDir = "experiment/{timestamp}_{trace_type}"
	}
	if c.Timezone == "" {
		c.Timezone = "Asia/Singapore"
	}
	if c.Sglang.Launcher == "" {
		c.Sglang.Launcher = "single"
	}
	if c.Sglang.Port == 0 {
		c.Sglang.Port = 30000
	}
	if c.Sglang.Host == "" {
		c.Sglang.Host = "0.0.0.0"
	}
	if c.Sglang.ModelPath == "" {
		c.Sglang.ModelPath = "Qwen/Qwen3-30B-A3B-Instruct-2507"
	}
	if c.Sglang.MemFractionStatic == 0 {
		c.Sglang.MemFractionStatic = 0.85
	}
	if c.Sglang.PageSize == 0 {
		c.Sglang.PageSize = 64
	}
	if c.Sglang.HicacheSize == 0 {
		c.Sglang.HicacheSize = 50
	}
	if c.Sglang.ToolCallParser == "" {
		c.Sglang.ToolCallParser = "qwen"
	}
	if c.Sglang.StartupTimeoutSec == 0 {
		c.Sglang.StartupTimeoutSec = 900
	}
	if c.Sglang.EnableHierarchicalCache == nil {
		t := true
		c.Sglang.EnableHierarchicalCache = &t
	}

	if c.Loadgen.Model == "" {
		c.Loadgen.Model = "default"
	}
	if c.Loadgen.TimeoutSec == 0 {
		c.Loadgen.TimeoutSec = 3000
	}
	if c.Loadgen.Speedup == 0 {
		c.Loadgen.Speedup = 1.0
	}
	if c.Loadgen.MaxInflight == 0 {
		c.Loadgen.MaxInflight = 1024
	}
	if c.Loadgen.LogEvery == 0 {
		c.Loadgen.LogEvery = 500
	}

	switch c.Loadgen.TraceType {
	case "agent":
		if c.Loadgen.Agent != nil {
			if c.Loadgen.Agent.Mode == "" {
				c.Loadgen.Agent.Mode = "sync"
			}
			if c.Loadgen.Agent.ConcurrentAgents == 0 {
				c.Loadgen.Agent.ConcurrentAgents = 2
			}
			if c.Loadgen.Agent.ResponseOutputDir == "" {
				c.Loadgen.Agent.ResponseOutputDir = "cache"
			}
		}
	case "azure":
		if c.Loadgen.Azure != nil {
			if c.Loadgen.Azure.SamplingRate == 0 {
				c.Loadgen.Azure.SamplingRate = 1.0
			}
			if c.Loadgen.Azure.PromptName == "" {
				c.Loadgen.Azure.PromptName = "None"
			}
			if c.Loadgen.Azure.RecurStep == 0 {
				c.Loadgen.Azure.RecurStep = 10
			}
			if c.Loadgen.Azure.Seed == 0 {
				c.Loadgen.Azure.Seed = 42
			}
		}
	case "alibaba":
		if c.Loadgen.Alibaba != nil {
			if c.Loadgen.Alibaba.SessionSampleRate == 0 {
				c.Loadgen.Alibaba.SessionSampleRate = 1.0
			}
			if c.Loadgen.Alibaba.SessionSampleSeed == 0 {
				c.Loadgen.Alibaba.SessionSampleSeed = 42
			}
		}
	}

	// Plots: once Enabled is true, individual toggles default to true so the
	// common case of "just plot everything" needs only one flag. Users still
	// opt out field-by-field by explicitly setting them to false.
	if c.Plots.Enabled {
		if !c.Plots.Result && !c.Plots.SglangTflops && !c.Plots.SglangTflopsPerGB && !c.Plots.SglangTokenUsage && !c.Plots.SMMetrics && !c.Plots.SLOAttainment && !c.Plots.CacheHit {
			c.Plots.Result = true
			c.Plots.SglangTflops = true
			c.Plots.SglangTflopsPerGB = true
			c.Plots.SglangTokenUsage = true
			c.Plots.SMMetrics = c.Plots.SMTrace
			c.Plots.SLOAttainment = true
			// cache-hit plot is alibaba-specific; enable by default and
			// gated by trace_type in RunPlots so non-alibaba runs skip it.
			c.Plots.CacheHit = true
		}
	}
}

func (c *Config) validate() error {
	if c.Loadgen.TraceType == "" {
		return fmt.Errorf("loadgen.trace_type is required (alibaba|azure|agent)")
	}
	switch c.Loadgen.TraceType {
	case "alibaba":
		if c.Loadgen.Alibaba == nil {
			return fmt.Errorf("loadgen.alibaba is required for trace_type=alibaba")
		}
		if c.Loadgen.Alibaba.Trace == "" {
			return fmt.Errorf("loadgen.alibaba.trace is required")
		}
	case "azure":
		if c.Loadgen.Azure == nil {
			return fmt.Errorf("loadgen.azure is required for trace_type=azure")
		}
		if c.Loadgen.Azure.Trace == "" {
			return fmt.Errorf("loadgen.azure.trace is required")
		}
	case "agent":
		if c.Loadgen.Agent == nil {
			return fmt.Errorf("loadgen.agent is required for trace_type=agent")
		}
		if c.Loadgen.Agent.AgentDirsJSON == "" {
			return fmt.Errorf("loadgen.agent.agent_dirs_json is required")
		}
	default:
		return fmt.Errorf("unknown loadgen.trace_type: %q", c.Loadgen.TraceType)
	}
	switch c.Sglang.Launcher {
	case "single", "multiple":
		// OK; no extra fields required.
	case "singularity":
		if c.Sglang.Singularity == nil || c.Sglang.Singularity.SIFImage == "" {
			return fmt.Errorf("sglang.singularity.sif_image is required when launcher=\"singularity\"")
		}
		if len(c.Sglang.Singularity.Modules) == 0 {
			c.Sglang.Singularity.Modules = []string{"singularity"}
		}
	default:
		return fmt.Errorf("sglang.launcher must be 'single', 'multiple' or 'singularity' (got %q)", c.Sglang.Launcher)
	}
	return nil
}

// ResolveOutputDir substitutes {timestamp} and {trace_type} tokens in
// OutputDir and returns an absolute path rooted at projectRoot. Creates the
// directory (and parents) if missing. The timestamp is formatted in the
// configured timezone (default Asia/Singapore).
func (c *Config) ResolveOutputDir(projectRoot string, now time.Time) (string, error) {
	loc, err := time.LoadLocation(c.Timezone)
	if err != nil {
		// Fall back to local time rather than fail the experiment — warn via
		// the returned timestamp carrying the local offset.
		loc = time.Local
	}
	stamp := now.In(loc).Format("2006-01-02_15-04-05")
	expanded := strings.ReplaceAll(c.OutputDir, "{timestamp}", stamp)
	expanded = strings.ReplaceAll(expanded, "{trace_type}", c.Loadgen.TraceType)
	if !filepath.IsAbs(expanded) {
		expanded = filepath.Join(projectRoot, expanded)
	}
	if err := os.MkdirAll(expanded, 0o755); err != nil {
		return "", fmt.Errorf("mkdir output_dir: %w", err)
	}
	return expanded, nil
}

// WriteSnapshot dumps the fully-resolved config to <outDir>/run_config.json so
// each experiment directory is self-describing.
func (c *Config) WriteSnapshot(outDir string) error {
	buf, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outDir, "run_config.json"), buf, 0o644)
}
