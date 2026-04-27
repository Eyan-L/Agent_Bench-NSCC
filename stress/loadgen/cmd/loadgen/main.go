// Command loadgen is the unified trace replayer.
//
// Usage:
//
//	loadgen alibaba  -trace ... [flags]      # qwen-bailian JSONL
//	loadgen azure    -trace ... [flags]      # Azure inference trace
//	loadgen agent    -agent-dirs-json ... [flags]
//
// Each subcommand shares a set of common flags (see loadgen.CommonConfig)
// and adds trace-specific ones on top. Run `loadgen <subcmd> -h` for the
// full list.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"loadgen/loadgen"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	sub := os.Args[1]
	args := os.Args[2:]

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var gen loadgen.LoadGenerator
	fs := flag.NewFlagSet(sub, flag.ExitOnError)

	switch sub {
	case "alibaba":
		g := &loadgen.AlibabaLoadGen{}
		g.RegisterFlags(fs)
		_ = fs.Parse(args)
		gen = g
	case "azure":
		g := &loadgen.AzureLoadGen{}
		g.RegisterFlags(fs)
		_ = fs.Parse(args)
		gen = g
	case "agent":
		g := &loadgen.AgentLoadGen{}
		g.RegisterFlags(fs)
		_ = fs.Parse(args)
		gen = g
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand: %s\n\n", sub)
		usage()
		os.Exit(2)
	}

	if err := gen.Prepare(ctx); err != nil {
		log.Fatalf("%s prepare: %v", sub, err)
	}
	if err := gen.Run(ctx); err != nil {
		log.Fatalf("%s run: %v", sub, err)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `loadgen — unified trace replayer

Usage:
  loadgen alibaba [flags]    replay a qwen-bailian JSONL trace
  loadgen azure   [flags]    replay an Azure inference trace
  loadgen agent   [flags]    replay an agent-trace replay plan

Run "loadgen <subcommand> -h" for the flags of each subcommand.
`)
}
