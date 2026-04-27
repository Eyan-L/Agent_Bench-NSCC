# bench-platform top-level orchestration.
# SERVING_BACKEND={vllm|sglang} controls which backend `make serve` launches.

SHELL := /usr/bin/env bash
SERVING_BACKEND ?= sglang
export SERVING_BACKEND

.PHONY: help serve eval stress analyze report test clean

help:
	@echo "Targets:"
	@echo "  serve       Launch backend (SERVING_BACKEND=$(SERVING_BACKEND))"
	@echo "  eval        Run real-eval pipeline (eval/runner/driver.py)"
	@echo "  stress      Run simulated stress workload (stress/loadgen-bin)"
	@echo "  analyze     Aggregate results (eval/analysis/aggregate.py)"
	@echo "  report      Build report (eval/report/generate.py)"
	@echo "  test        Trace pipeline self-tests"
	@echo ""
	@echo "Env: SERVING_BACKEND={vllm|sglang} (current: $(SERVING_BACKEND))"

serve:
	bash backend/common/serve.sh

eval:
	python -m eval.runner.driver $(EVAL_ARGS)

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
