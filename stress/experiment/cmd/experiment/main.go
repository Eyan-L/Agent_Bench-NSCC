// Command experiment runs an end-to-end trace-replay experiment from a single
// JSON config file, then fires the configured draw/*.py plots.
//
// Usage:
//
//	experiment -config run/configs/alibaba.json
//	experiment -config my.json -project-root /path/to/openclaw_bench_coding
//
// The trace_type field in the config selects the loadgen subcommand
// (alibaba|azure|agent); see run/configs/ for complete examples.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"experiment"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	configPath := flag.String("config", "", "Path to experiment JSON config (required)")
	projectRoot := flag.String("project-root", "", "Repo root (default: auto-detect by walking up for loadgen/go.mod)")
	flag.Parse()

	if *configPath == "" {
		fmt.Fprintln(os.Stderr, "error: -config is required")
		flag.Usage()
		os.Exit(2)
	}

	cfg, err := experiment.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	root, err := experiment.EnsureProjectRoot(*projectRoot)
	if err != nil {
		log.Fatalf("%v", err)
	}
	log.Printf("project_root=%s", root)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	outDir, runErr := experiment.Run(ctx, cfg, root)
	if runErr != nil {
		log.Printf("experiment failed: %v", runErr)
		if outDir != "" {
			log.Printf("artifacts under %s", outDir)
		}
		os.Exit(1)
	}
	log.Printf("done. artifacts under %s", outDir)
}
