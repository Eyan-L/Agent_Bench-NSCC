# bench-platform top-level orchestration.
# SERVING_BACKEND={vllm|sglang} controls which backend `make serve` launches.

SHELL := /usr/bin/env bash
SERVING_BACKEND ?= sglang
export SERVING_BACKEND

# Eval inputs — set via env or CLI:
#   make run_phase0 SWEBENCH_ROOT=/path/to/SWE-bench INSTANCE_DATA=/path/to/verified.json
SWEBENCH_ROOT  ?=
INSTANCE_DATA  ?=
KIT_ROOT       ?= $(CURDIR)
export KIT_ROOT

_check_eval_inputs:
	@if [ -z "$(SWEBENCH_ROOT)" ] || [ -z "$(INSTANCE_DATA)" ]; then \
	    echo "ERROR: set SWEBENCH_ROOT and INSTANCE_DATA (env or make var)." >&2; \
	    exit 2; \
	fi

.PHONY: help serve eval stress analyze report test clean \
        run_phase0 run_phase1 run_phase2 _check_eval_inputs \
        otel-up-host otel-down-host scoring-server

help:
	@echo "Targets:"
	@echo "  serve       Launch backend (SERVING_BACKEND=$(SERVING_BACKEND))"
	@echo "  eval        Run real-eval pipeline (eval/runner/driver.py)"
	@echo "  run_phase0  Run phase0 sanity sweep (1 instance)"
	@echo "  run_phase1  Run phase1 sweep"
	@echo "  run_phase2  Run phase2 serving sweep (RQ3)"
	@echo "  stress      Run simulated stress workload (stress/loadgen-bin)"
	@echo "  analyze     Aggregate results (eval/analysis/aggregate.py)"
	@echo "  report      Build report (eval/report/generate.py)"
	@echo "  test        Trace pipeline self-tests"
	@echo "  otel-up-host    Start native otelcol-contrib (no docker)"
	@echo "  otel-down-host  Stop the native collector"
	@echo ""
	@echo "Env: SERVING_BACKEND={vllm|sglang} (current: $(SERVING_BACKEND))"
	@echo "     SWEBENCH_ROOT=$(SWEBENCH_ROOT)"
	@echo "     INSTANCE_DATA=$(INSTANCE_DATA)"

serve:
	bash backend/common/serve.sh

eval:
	python -m eval.runner.driver $(EVAL_ARGS)

# Phase shortcuts — wrap `eval` with the canonical sweep yaml.
# Override SWEBENCH_ROOT / INSTANCE_DATA / EXTRA_ARGS as needed.
define _RUN_SWEEP
python -m eval.runner.driver \
  --sweep config/sweeps/$(1).yaml \
  --swebench-root $(SWEBENCH_ROOT) \
  --instance-data $(INSTANCE_DATA) \
  $(EXTRA_ARGS)
endef

run_phase0: _check_eval_inputs
	$(call _RUN_SWEEP,phase0)

run_phase1: _check_eval_inputs
	$(call _RUN_SWEEP,phase1)

run_phase2: _check_eval_inputs
	$(call _RUN_SWEEP,phase2_serving)

# Native OTel collector (no docker). Downloads otelcol-contrib on first run.
otel-up-host:
	bash eval/otel/host_up.sh

otel-down-host:
	bash eval/otel/host_down.sh

# Self-hosted SWE-bench scoring server (run on a docker host).
# Clients set OC_SCORING_URL=http://<this-host>:8765 to use it.
SCORING_PORT ?= 8765
scoring-server: _check_eval_inputs
	python -m eval.scoring_server.main \
	  --swebench-root $(SWEBENCH_ROOT) \
	  --host 0.0.0.0 --port $(SCORING_PORT)

stress:
	./stress/loadgen-bin $(STRESS_ARGS)

analyze:
	python -m eval.analysis.aggregate $(ANALYZE_ARGS)

report:
	python -m eval.report.generate $(REPORT_ARGS)

test:
	python -m tests.trace.test_trace_pipeline

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
