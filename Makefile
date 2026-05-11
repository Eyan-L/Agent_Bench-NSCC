# bench-platform top-level orchestration.
#
# Composed-config run model. A run is the cartesian product of three JSON configs:
#   APP      config/application/<app>/<exp>.json   what to run (instances, repos, budgets)
#   HARNESS  config/harness/<name>.json             how to drive it + which model
#                                                   (model selection lives here)
#   BACKEND  config/backend/<engine>/<name>.json   how to serve it (TP/GPU/port/...
#                                                   serving presets; no model)
# Override any of them on the command line, e.g.
#   make run_phase0 BACKEND=vllm/tp2 HARNESS=openclaw_35b_a3b
# SERVING_BACKEND={vllm|sglang} still controls which backend `make serve`
# launches; it is derived from BACKEND's first path segment by default.

SHELL := /usr/bin/env bash

# --- composed-config defaults --------------------------------------------
APP     ?= swebench/phase0
HARNESS ?= openclaw_27b
BACKEND ?= vllm/tp1
export APP HARNESS BACKEND

# SERVING_BACKEND := first path segment of BACKEND (vllm/... -> vllm).
SERVING_BACKEND ?= $(firstword $(subst /, ,$(BACKEND)))
export SERVING_BACKEND

# Eval inputs — set via env or CLI:
#   make run_phase0 SWEBENCH_ROOT=/path/to/SWE-bench INSTANCE_DATA=/path/to/verified.json
SWEBENCH_ROOT  ?=
INSTANCE_DATA  ?=
KIT_ROOT       ?= $(CURDIR)
export KIT_ROOT

# Serving auto-launch. _ensure_serve polls SERVE_URL; if it can't reach it,
# it forks `make serve` into the background, writes the engine log to
# SERVE_LOG, and waits up to SERVE_READY_TIMEOUT seconds for /v1/models to
# come up. Set SERVE_AUTO=0 to skip (useful when you launch the backend
# yourself or want to fail fast on missing serving).
SERVE_URL            ?= http://localhost:8000/v1/models
SERVE_LOG            ?= $(KIT_ROOT)/data/serving/vllm.log
SERVE_READY_TIMEOUT  ?= 600
SERVE_AUTO           ?= 1

_check_eval_inputs:
	@if [ -z "$(SWEBENCH_ROOT)" ] || [ -z "$(INSTANCE_DATA)" ]; then \
	    echo "ERROR: set SWEBENCH_ROOT and INSTANCE_DATA (env or make var)." >&2; \
	    exit 2; \
	fi

_ensure_serve:
	@if [ "$(SERVE_AUTO)" = "0" ]; then \
	    echo "[serve] SERVE_AUTO=0 — skipping auto-launch"; \
	elif curl -s -f -m 2 "$(SERVE_URL)" >/dev/null 2>&1; then \
	    echo "[serve] $(SERVE_URL) already up — reusing"; \
	else \
	    mkdir -p "$$(dirname "$(SERVE_LOG)")"; \
	    echo "[serve] launching $(SERVING_BACKEND) → $(SERVE_LOG)"; \
	    nohup bash backend/common/serve.sh >>"$(SERVE_LOG)" 2>&1 </dev/null & \
	    pid=$$!; \
	    echo "[serve] PID $$pid — polling $(SERVE_URL) (timeout $(SERVE_READY_TIMEOUT)s)"; \
	    deadline=$$(( $$(date +%s) + $(SERVE_READY_TIMEOUT) )); \
	    while ! curl -s -f -m 2 "$(SERVE_URL)" >/dev/null 2>&1; do \
	        if ! kill -0 $$pid 2>/dev/null; then \
	            echo "[serve] PID $$pid died — see $(SERVE_LOG)" >&2; \
	            tail -n 30 "$(SERVE_LOG)" >&2 2>/dev/null; exit 1; \
	        fi; \
	        if [ $$(date +%s) -gt $$deadline ]; then \
	            echo "[serve] timed out after $(SERVE_READY_TIMEOUT)s — see $(SERVE_LOG)" >&2; \
	            exit 1; \
	        fi; \
	        sleep 10; \
	    done; \
	    echo "[serve] ready"; \
	fi

.PHONY: help serve eval stress analyze report test clean \
        run_phase0 run_phase1 run_phase2 run_tb tb-tasks \
        _check_eval_inputs _ensure_serve \
        otel-up-host otel-down-host scoring-server

help:
	@echo "Targets:"
	@echo "  serve       Launch backend (SERVING_BACKEND=$(SERVING_BACKEND))"
	@echo "  eval        Run driver with current APP/HARNESS/BACKEND"
	@echo "  run_phase0  SWE-bench phase0 sanity (APP=swebench/phase0)"
	@echo "  run_phase1  SWE-bench phase1 mini-verified (APP=swebench/phase1)"
	@echo "  run_phase2  SWE-bench phase2 serving sweep (APP=swebench/phase2_serving)"
	@echo "  run_tb      Terminal-Bench kv-store-grpc (APP=terminal_bench/kv_store, HARNESS=tb_react)"
	@echo "  stress      Run simulated stress workload (stress/loadgen-bin)"
	@echo "  analyze     Aggregate results (orchestration/analysis/aggregate.py)"
	@echo "  report      Build report (orchestration/report/generate.py)"
	@echo "  test        Trace pipeline self-tests"
	@echo "  otel-up-host    Start native otelcol-contrib (no docker)"
	@echo "  otel-down-host  Stop the native collector"
	@echo ""
	@echo "Composed config (override any at the command line):"
	@echo "  APP=$(APP)"
	@echo "  HARNESS=$(HARNESS)"
	@echo "  BACKEND=$(BACKEND)  -> SERVING_BACKEND=$(SERVING_BACKEND)"
	@echo "  SWEBENCH_ROOT=$(SWEBENCH_ROOT)"
	@echo "  INSTANCE_DATA=$(INSTANCE_DATA)"
	@echo ""
	@echo "Serving auto-launch (run_phase*):"
	@echo "  SERVE_AUTO=1                 auto-start backend if down (0 to disable)"
	@echo "  SERVE_URL=$(SERVE_URL)"
	@echo "  SERVE_LOG=$(SERVE_LOG)"
	@echo "  SERVE_READY_TIMEOUT=$(SERVE_READY_TIMEOUT)s"

serve:
	bash backend/common/serve.sh

eval: _check_eval_inputs _ensure_serve
	python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --backend config/backend/$(BACKEND).json \
	  --swebench-root $(SWEBENCH_ROOT) \
	  --instance-data $(INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# Phase shortcuts — same as `eval`, but pin APP so the user can stay on
# `make run_phase0` muscle memory. Override HARNESS / BACKEND as needed.
run_phase0: APP := swebench/phase0
run_phase0: eval

run_phase1: APP := swebench/phase1
run_phase1: eval

run_phase2: APP := swebench/phase2_serving
run_phase2: HARNESS := openclaw_35b_a3b
run_phase2: BACKEND := vllm/tp2_rq3
run_phase2: eval

# Terminal-Bench shortcut: defaults to the kv_store task and tb_react harness.
# TB doesn't use SWE-bench inputs, so it bypasses `_check_eval_inputs` and
# materializes its own instance_data via the loader. Override:
#   make run_tb HARNESS=openclaw_27b   # OC-driven TB (needs `openclaw` block)
#   make run_tb HARNESS_BACKEND=singularity   # HPC fallback
TB_INSTANCE_DATA ?= $(KIT_ROOT)/data/terminal_bench/tasks.json

# Materialize the TB tasks.json from applications/terminal_bench/tasks/.
# Re-runs are idempotent — the loader overwrites the file each time.
tb-tasks:
	@mkdir -p "$$(dirname "$(TB_INSTANCE_DATA)")"
	python -m applications.terminal_bench.loader "$(TB_INSTANCE_DATA)"
	@echo "[tb] instance_data -> $(TB_INSTANCE_DATA)"

run_tb: APP := terminal_bench/kv_store
run_tb: HARNESS := tb_react_27b
run_tb: tb-tasks _ensure_serve
	python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --backend config/backend/$(BACKEND).json \
	  --instance-data $(TB_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# Native OTel collector (no docker). Downloads otelcol-contrib on first run.
otel-up-host:
	bash orchestration/otel/host_up.sh

otel-down-host:
	bash orchestration/otel/host_down.sh

# Self-hosted SWE-bench scoring server (run on a docker host).
# Clients set OC_SCORING_URL=http://<this-host>:8765 to use it.
SCORING_PORT ?= 8765
scoring-server: _check_eval_inputs
	python -m orchestration.scoring_server.main \
	  --swebench-root $(SWEBENCH_ROOT) \
	  --host 0.0.0.0 --port $(SCORING_PORT)

stress:
	./stress/loadgen-bin $(STRESS_ARGS)

analyze:
	python -m orchestration.analysis.aggregate $(ANALYZE_ARGS)

report:
	python -m orchestration.report.generate $(REPORT_ARGS)

test:
	python -m tests.trace.test_trace_pipeline

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
