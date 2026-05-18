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
PYTHON ?= python3

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

# Per-sweep timestamp (Singapore time) + experiment dir. Both Make and the
# driver use OC_RUN_STAMP so vllm.log + per-run dirs end up siblings under
# <app_output_dir>/<stamp>/. Override OC_RUN_STAMP to pin (resume an existing
# dir) — `?=` honors a value passed on the command line or env.
OC_RUN_STAMP ?= $(shell TZ=Asia/Singapore date '+%Y-%m-%d_%H-%M-%S')
export OC_RUN_STAMP

# Deferred (=) so target-specific `APP := <x>` overrides flow through. Reads
# output_dir from the active app JSON; falls back to data/runs/_unknown when
# the JSON is missing (e.g. `make help` with the default APP).
APP_OUTPUT_DIR = $(shell $(PYTHON) -c 'import json; print(json.load(open("config/application/$(APP).json"))["output_dir"])' 2>/dev/null)
SWEEP_DIR      = $(APP_OUTPUT_DIR)/$(OC_RUN_STAMP)

# Serving auto-launch. _ensure_serve polls SERVE_URL; if it can't reach it,
# it forks `make serve` into the background, writes the engine log to
# SERVE_LOG, and waits up to SERVE_READY_TIMEOUT seconds for /v1/models to
# come up. Set SERVE_AUTO=0 to skip (useful when you launch the backend
# yourself or want to fail fast on missing serving).
SERVE_URL            ?= http://localhost:8000/v1/models
SERVE_LOG            ?= $(SWEEP_DIR)/vllm.log
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
	    mkdir -p "$$(dirname "$(SERVE_LOG)")"; \
	    pid=$$(pgrep -f "$(SERVING_BACKEND).*serve|$(SERVING_BACKEND) serve" | head -1); \
	    if [ -n "$$pid" ] && [ -r /proc/$$pid/fd/1 ]; then \
	        actual=$$(readlink /proc/$$pid/fd/1); \
	        if [ -n "$$actual" ] && [ "$$actual" != "$(SERVE_LOG)" ]; then \
	            ln -sfn "$$actual" "$(SERVE_LOG)"; \
	            echo "[serve] linked $(SERVE_LOG) -> $$actual (reused PID $$pid)"; \
	        fi; \
	    fi; \
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

.PHONY: help serve serve-docker serve-stop serve-logs serve-status \
        eval stress analyze report test clean \
        run_phase0 run_phase1 run_phase2 run_tb tb-tasks \
        _check_eval_inputs _ensure_serve \
        otel-up-host otel-down-host scoring-server \
        openhands-image \
        mig-status mig-prep-test mig-teardown mig-smoke run_tb_mig \
        _check_mig_uuid _ensure_mig_serve

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
	@echo "  serve-docker    Launch vLLM as a docker container (MIG-aware via VLLM_MIG_UUID)"
	@echo "  serve-stop/logs/status  Lifecycle for the docker-launched vLLM"
	@echo "  openhands-image  Build agent-bench/openhands:<ver> harness image"
	@echo "  mig-status / mig-prep-test / mig-teardown  GPU MIG management"
	@echo "  mig-smoke   Dry-run MIG vLLM binding using data/last_mig_uuid"
	@echo "  run_tb_mig  One-command TB smoke against MIG-bound vLLM"
	@echo ""
	@echo "Composed config (override any at the command line):"
	@echo "  APP=$(APP)"
	@echo "  HARNESS=$(HARNESS)"
	@echo "  BACKEND=$(BACKEND)  -> SERVING_BACKEND=$(SERVING_BACKEND)"
	@echo "  SWEBENCH_ROOT=$(SWEBENCH_ROOT)"
	@echo "  INSTANCE_DATA=$(INSTANCE_DATA)"
	@echo ""
	@echo "Experiment folder (Singapore time):"
	@echo "  OC_RUN_STAMP=$(OC_RUN_STAMP)"
	@echo "  SWEEP_DIR=$(SWEEP_DIR)"
	@echo ""
	@echo "Serving auto-launch (run_phase*):"
	@echo "  SERVE_AUTO=1                 auto-start backend if down (0 to disable)"
	@echo "  SERVE_URL=$(SERVE_URL)"
	@echo "  SERVE_LOG=$(SERVE_LOG)              (lives inside SWEEP_DIR by default)"
	@echo "  SERVE_READY_TIMEOUT=$(SERVE_READY_TIMEOUT)s"
	@echo ""
	@echo "Docker / GPU helpers:"
	@echo "  VLLM_MIG_UUID                bind serve-docker to a MIG slice (see mig-status)"
	@echo "  MIG_GPU=$(MIG_GPU) / MIG_PROFILE=$(MIG_PROFILE) (mig-prep-test args)"
	@echo "  MIG_UUID=$(MIG_UUID) (defaults to data/last_mig_uuid)"

serve:
	bash backend/common/serve.sh

eval: _check_eval_inputs _ensure_serve
	$(PYTHON) -m orchestration.runner.driver \
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
	$(PYTHON) -m applications.terminal_bench.loader "$(TB_INSTANCE_DATA)"
	@echo "[tb] instance_data -> $(TB_INSTANCE_DATA)"

run_tb: APP := terminal_bench/kv_store
run_tb: HARNESS := tb_react_27b
run_tb: tb-tasks _ensure_serve
	$(PYTHON) -m orchestration.runner.driver \
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
	$(PYTHON) -m orchestration.scoring_server.main \
	  --swebench-root $(SWEBENCH_ROOT) \
	  --host 0.0.0.0 --port $(SCORING_PORT)

stress:
	./stress/loadgen-bin $(STRESS_ARGS)

analyze:
	$(PYTHON) -m orchestration.analysis.aggregate $(ANALYZE_ARGS)

report:
	$(PYTHON) -m orchestration.report.generate $(REPORT_ARGS)

test:
	$(PYTHON) -m tests.trace.test_trace_pipeline

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +

# ---- Containerized vLLM (MIG-aware) --------------------------------------
# Sibling of `make serve`. `make serve` runs vLLM natively (JSON-config
# driven); `make serve-docker` runs it as a docker container so it can bind
# to a MIG slice via CDI. See backend/vllm/serve_docker.sh for the env
# contract (VLLM_IMAGE / VLLM_MIG_UUID / VLLM_DETACH / ...).
VLLM_CONTAINER_NAME ?= oc-vllm-serve

serve-docker:
	bash backend/vllm/serve_docker.sh $(VLLM_ARGS)

serve-stop:
	@if docker ps -a --format '{{.Names}}' | grep -qx "$(VLLM_CONTAINER_NAME)"; then \
	    docker stop -t 30 "$(VLLM_CONTAINER_NAME)" >/dev/null && \
	    docker rm  -f    "$(VLLM_CONTAINER_NAME)" >/dev/null && \
	    echo "stopped $(VLLM_CONTAINER_NAME)"; \
	else echo "$(VLLM_CONTAINER_NAME) not running"; fi

serve-logs:
	docker logs -f --tail 200 $(VLLM_CONTAINER_NAME)

serve-status:
	@docker ps -a --filter "name=$(VLLM_CONTAINER_NAME)" \
	  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
	@echo "---"
	@cat data/serving_metadata.env 2>/dev/null || echo "(no serving_metadata.env yet)"

# ---- OpenHands harness image --------------------------------------------
OPENHANDS_IMAGE       ?= agent-bench/openhands:1.21.0
OPENHANDS_SDK_VERSION ?= 1.21.0

openhands-image:
	OH_IMAGE_TAG=$(OPENHANDS_IMAGE) \
	bash harnesses/openhands/docker/build.sh

# ---- MIG management ------------------------------------------------------
# nvidia-smi MIG commands require root; the scripts enforce + remind.
MIG_GPU     ?= 1
MIG_PROFILE ?= 1g.24gb
MIG_UUID    ?= $(shell cat data/last_mig_uuid 2>/dev/null)
MIG_SERVE_LOG ?= $(SWEEP_DIR)/vllm-mig.log
MIG_SERVE_READY_TIMEOUT ?= $(SERVE_READY_TIMEOUT)

mig-status:
	@bash scripts/mig/mig_status.sh

_check_mig_uuid:
	@if [ -z "$(MIG_UUID)" ]; then \
	    echo "ERROR: MIG_UUID is empty. Run 'make mig-prep-test' or pass MIG_UUID=MIG-..." >&2; \
	    exit 2; \
	fi
	@if command -v nvidia-smi >/dev/null 2>&1 && ! nvidia-smi -L 2>/dev/null | grep -qF "$(MIG_UUID)"; then \
	    echo "ERROR: MIG slice $(MIG_UUID) not found. Run 'make mig-status'." >&2; \
	    exit 2; \
	fi

mig-smoke: _check_mig_uuid
	@VLLM_MIG_UUID="$(MIG_UUID)" DRY_RUN=1 bash backend/vllm/serve_docker.sh --preset latency

_ensure_mig_serve: _check_mig_uuid
	@if curl -s -f -m 2 "$(SERVE_URL)" >/dev/null 2>&1; then \
	    echo "[mig-serve] $(SERVE_URL) already up — reusing"; \
	else \
	    mkdir -p "$$(dirname "$(MIG_SERVE_LOG)")"; \
	    echo "[mig-serve] launching vLLM on $(MIG_UUID) → $(MIG_SERVE_LOG)"; \
	    VLLM_MIG_UUID="$(MIG_UUID)" VLLM_TAIL_LOGS=0 VLLM_DETACH=1 \
	      bash backend/vllm/serve_docker.sh >>"$(MIG_SERVE_LOG)" 2>&1; \
	    echo "[mig-serve] polling $(SERVE_URL) (timeout $(MIG_SERVE_READY_TIMEOUT)s)"; \
	    deadline=$$(( $$(date +%s) + $(MIG_SERVE_READY_TIMEOUT) )); \
	    while ! curl -s -f -m 2 "$(SERVE_URL)" >/dev/null 2>&1; do \
	        if ! docker ps --format '{{.Names}}' | grep -qx "$(VLLM_CONTAINER_NAME)"; then \
	            echo "[mig-serve] container $(VLLM_CONTAINER_NAME) is not running — see $(MIG_SERVE_LOG)" >&2; \
	            tail -n 50 "$(MIG_SERVE_LOG)" >&2 2>/dev/null; exit 1; \
	        fi; \
	        if [ $$(date +%s) -gt $$deadline ]; then \
	            echo "[mig-serve] timed out after $(MIG_SERVE_READY_TIMEOUT)s — see $(MIG_SERVE_LOG)" >&2; \
	            exit 1; \
	        fi; \
	        sleep 10; \
	    done; \
	    echo "[mig-serve] ready"; \
	fi

run_tb_mig: APP := terminal_bench/kv_store
run_tb_mig: HARNESS := tb_react_27b
run_tb_mig: tb-tasks _ensure_mig_serve
	$(PYTHON) -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --backend config/backend/$(BACKEND).json \
	  --instance-data $(TB_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

mig-prep-test:
	@command -v sudo >/dev/null || { echo "ERROR: sudo not available" >&2; exit 1; }
	sudo bash scripts/mig/mig_enable.sh $(MIG_GPU)
	@mkdir -p data
	@sudo bash scripts/mig/mig_create.sh $(MIG_GPU) $(MIG_PROFILE) | tee data/last_mig_uuid
	@echo
	@echo "Slice ready. Launch vLLM bound to it:"
	@echo "    VLLM_MIG_UUID=\$$(cat data/last_mig_uuid) make serve-docker"

mig-teardown:
	sudo bash scripts/mig/mig_destroy_all.sh $(MIG_GPU)
	sudo bash scripts/mig/mig_disable.sh $(MIG_GPU)
	@rm -f data/last_mig_uuid
