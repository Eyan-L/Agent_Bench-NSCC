# bench-platform top-level orchestration.
#
# Composed-config run model. A run is the cartesian product of four JSON configs:
#   APP      config/application/<app>/<exp>.json   what to run (instances, repos, budgets)
#   HARNESS  config/harness/<name>.json             how to drive the agent framework
#   MODEL    config/model/<name>.json               which model + model-level limits
#   SERVING  config/serving/<engine>/<name>.json    serving endpoint / engine deployment
# Override any of them on the command line, e.g.
#   make run_phase0 SERVING=vllm/debug MODEL_PROFILE=qwen35_9b HARNESS=openclaw_swebench VLLM_TP=1
# SERVING_BACKEND is derived from SERVING's first path segment by default.
# `make serve` currently has local launchers for vLLM and SGLang; PBS also
# supports LiteLLM provider proxy profiles.

SHELL := /usr/bin/env bash
PYTHON ?= python3

# --- composed-config defaults --------------------------------------------
APP     ?= swebench/phase0
HARNESS ?= openclaw_swebench
MODEL_PROFILE ?= qwen36_27b
SERVING ?= vllm/default
export APP HARNESS MODEL_PROFILE SERVING

# SERVING_BACKEND := first path segment of SERVING (vllm/... -> vllm).
SERVING_BACKEND ?= $(firstword $(subst /, ,$(SERVING)))
export SERVING_BACKEND

# Eval inputs — set via env or CLI:
#   make run_phase0 SWEBENCH_ROOT=/path/to/SWE-bench INSTANCE_DATA=/path/to/verified.json
SWEBENCH_ROOT  ?=
INSTANCE_DATA  ?=
KIT_ROOT       ?= $(CURDIR)
export KIT_ROOT

# Per-sweep timestamp (Singapore time) + experiment dir. Both Make and the
# driver use AB_RUN_STAMP so engine logs and per-run dirs end up under one
# experiment root. Override AB_RUN_STAMP to pin/resume an existing run.
# Directory layout:
#   data/experiments/<AB_RUN_STAMP>/logs/
#   data/experiments/<AB_RUN_STAMP>/runs/<app-output-leaf>/
# `?=` honors a value passed on the command line or env.
# Lock AB_RUN_STAMP to a single timestamp for the whole make run. The old
# `?=` form left the right-hand side deferred — every $(SERVE_LOG)
# reference re-invoked `date`, so mkdir and nohup-redirect could land in
# different per-second dirs and the redirect would fail silently.
AB_RUN_STAMP := $(or $(AB_RUN_STAMP),$(shell TZ=Asia/Singapore date '+%Y-%m-%d_%H-%M-%S'))
export AB_RUN_STAMP
AB_EXPERIMENT_DIR ?= $(KIT_ROOT)/data/experiments/$(AB_RUN_STAMP)
export AB_EXPERIMENT_DIR

# Deferred (=) so target-specific `APP := <x>` overrides flow through. Reads
# output_dir from the active app JSON is now a run-set leaf (e.g. phase0).
# Fall back to _unknown when the JSON is missing (e.g. `make help` with a
# half-edited config).
APP_OUTPUT_DIR = $(shell $(PYTHON) -c 'import json; print(json.load(open("config/application/$(APP).json"))["output_dir"])' 2>/dev/null)
APP_OUTPUT_LEAF = $(notdir $(APP_OUTPUT_DIR))
SWEEP_DIR      = $(AB_EXPERIMENT_DIR)/runs/$(APP_OUTPUT_LEAF)
EXPERIMENT_LOG_DIR = $(AB_EXPERIMENT_DIR)/logs

# Serving auto-launch. _ensure_serve polls SERVE_URL; if it can't reach it,
# it forks `make serve` into the background, writes the engine log to
# SERVE_LOG, and waits up to SERVE_READY_TIMEOUT seconds for /v1/models to
# come up. Set SERVE_AUTO=0 to skip (useful when you launch the backend
# yourself or want to fail fast on missing serving).
SERVE_PORT           = $(shell $(PYTHON) -c 'import json; print((json.load(open("config/serving/$(SERVING).json"))["serving"].get("endpoint") or {}).get("port", 8000))' 2>/dev/null || echo 8000)
SERVE_URL            ?= http://localhost:$(SERVE_PORT)/v1/models
SERVE_LOG            ?= $(EXPERIMENT_LOG_DIR)/$(SERVING_BACKEND).log
SERVE_READY_TIMEOUT  ?= 600
SERVE_AUTO           ?= 1

# Shell snippet to register an EXIT trap that stops the active backend after
# the recipe's driver finishes (or on Ctrl-C). Skipped when SERVE_AUTO=0
# (user manages backend lifecycle themselves). One sglang/vllm process per
# experiment — never reused across runs.
SERVE_CLEANUP_HOOK = if [ "$(SERVE_AUTO)" != "0" ]; then trap 'echo "[serve] stopping $(SERVING_BACKEND) after experiment"; PORT="$(SERVE_PORT)" bash backend/common/serve.sh stop 2>&1 | sed "s|^|[serve] |" || true' EXIT INT TERM; fi

_check_eval_inputs:
	@if [ -z "$(SWEBENCH_ROOT)" ] || [ -z "$(INSTANCE_DATA)" ]; then \
	    echo "ERROR: set SWEBENCH_ROOT and INSTANCE_DATA (env or make var)." >&2; \
	    exit 2; \
	fi

_ensure_serve:
	@if [ "$(SERVE_AUTO)" = "0" ]; then \
	    echo "[serve] SERVE_AUTO=0 — skipping auto-launch"; \
	else \
	    if curl -s -f -m 2 "$(SERVE_URL)" >/dev/null 2>&1; then \
	        echo "[serve] $(SERVE_URL) already up — stopping for fresh per-experiment launch"; \
	        PORT="$(SERVE_PORT)" bash backend/common/serve.sh stop 2>&1 | sed 's/^/[serve] /'; \
	        for i in $$(seq 1 20); do \
	            sleep 1; \
	            curl -s -f -m 1 "$(SERVE_URL)" >/dev/null 2>&1 || break; \
	        done; \
	    fi; \
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

.PHONY: help serve serve-docker serve-stop serve-logs serve-status validate-configs \
        eval stress analyze report test clean \
        run_phase0 run_phase1 run_phase2 run_tb tb-tasks \
        run_dr_smoke1 run_dr_smoke run_dr_full dr-tasks \
        run_dr_smoke1_agent run_dr_smoke_agent run_dr_full_agent \
        run_dr_smoke1_eval  run_dr_smoke_eval  run_dr_full_eval \
        _check_eval_inputs _ensure_serve \
        otel-up-host otel-down-host scoring-server \
        openhands-image \
        mig-status mig-prep-test mig-teardown mig-smoke run_tb_mig \
        _check_mig_uuid _ensure_mig_serve

help:
	@echo "Targets:"
	@echo "  serve       Launch backend (SERVING_BACKEND=$(SERVING_BACKEND))"
	@echo "  eval        Run driver with current APP/HARNESS/MODEL_PROFILE/SERVING"
	@echo "  validate-configs  Static checks for config/application+harness+model+serving"
	@echo "  run_phase0  SWE-bench phase0 sanity (APP=swebench/phase0)"
	@echo "  run_phase1  SWE-bench phase1 mini-verified (APP=swebench/phase1)"
	@echo "  run_phase2  SWE-bench phase2 serving sweep (APP=swebench/phase2_serving)"
	@echo "  run_tb      Terminal-Bench kv-store-grpc (APP=terminal_bench/kv_store, HARNESS=tb_react_terminal_bench)"
	@echo "  run_dr_smoke1[_agent|_eval]  DeepResearch-Bench 1 task (full / agent only / re-judge persisted; ~28/3/20 min)"
	@echo "  run_dr_smoke[_agent|_eval]   DeepResearch-Bench 3 tasks (same split)"
	@echo "  run_dr_full[_agent|_eval]    DeepResearch-Bench full 100 tasks (same split)"
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
	@echo "  MODEL_PROFILE=$(MODEL_PROFILE)"
	@echo "  SERVING=$(SERVING)  -> SERVING_BACKEND=$(SERVING_BACKEND)"
	@echo "  SWEBENCH_ROOT=$(SWEBENCH_ROOT)"
	@echo "  INSTANCE_DATA=$(INSTANCE_DATA)"
	@echo ""
	@echo "Experiment folder (Singapore time):"
	@echo "  AB_RUN_STAMP=$(AB_RUN_STAMP)"
	@echo "  AB_EXPERIMENT_DIR=$(AB_EXPERIMENT_DIR)"
	@echo "  SWEEP_DIR=$(SWEEP_DIR)"
	@echo ""
	@echo "Serving auto-launch (run_phase*):"
	@echo "  SERVE_AUTO=1                 auto-start backend if down (0 to disable)"
	@echo "  SERVE_URL=$(SERVE_URL)"
	@echo "  SERVE_LOG=$(SERVE_LOG)              (lives inside AB_EXPERIMENT_DIR/logs by default)"
	@echo "  SERVE_READY_TIMEOUT=$(SERVE_READY_TIMEOUT)s"
	@echo ""
	@echo "Docker / GPU helpers:"
	@echo "  VLLM_MIG_UUID                bind serve-docker to a MIG slice (see mig-status)"
	@echo "  MIG_GPU=$(MIG_GPU) / MIG_PROFILE=$(MIG_PROFILE) (mig-prep-test args)"
	@echo "  MIG_UUID=$(MIG_UUID) (defaults to data/last_mig_uuid)"

serve:
	bash backend/common/serve.sh

validate-configs:
	$(PYTHON) scripts/validate_configs.py \
	  --app config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json

eval: _check_eval_inputs _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(PYTHON) -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --swebench-root $(SWEBENCH_ROOT) \
	  --instance-data $(INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# Phase shortcuts — same as `eval`, but pin APP so the user can stay on
# `make run_phase0` muscle memory. Override HARNESS / MODEL_PROFILE / SERVING as needed.
run_phase0: APP := swebench/phase0
run_phase0: eval

run_phase1: APP := swebench/phase1
run_phase1: eval

run_phase2: APP := swebench/phase2_serving
run_phase2: HARNESS := openclaw_swebench
run_phase2: MODEL_PROFILE := qwen36_35b_a3b_local
run_phase2: SERVING := vllm/default
run_phase2: eval

# Terminal-Bench shortcut: defaults to the kv_store task and tb_react_terminal_bench harness.
# TB doesn't use SWE-bench inputs, so it bypasses `_check_eval_inputs` and
# materializes its own instance_data via the loader. Override:
#   make run_tb HARNESS=openclaw_terminal_bench   # OC-driven TB (needs `openclaw` block)
#   make run_tb HARNESS_BACKEND=singularity   # HPC fallback
TB_INSTANCE_DATA ?= $(KIT_ROOT)/data/terminal_bench/tasks.json

# Materialize the TB tasks.json from applications/terminal_bench/tasks/.
# Re-runs are idempotent — the loader overwrites the file each time.
tb-tasks:
	@mkdir -p "$$(dirname "$(TB_INSTANCE_DATA)")"
	$(PYTHON) -m applications.terminal_bench.loader "$(TB_INSTANCE_DATA)"
	@echo "[tb] instance_data -> $(TB_INSTANCE_DATA)"

run_tb: APP := terminal_bench/kv_store
run_tb: HARNESS := tb_react_terminal_bench
run_tb: tb-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(PYTHON) -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(TB_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# ----------------------------------------------------------------------
# DeepResearch-Bench shortcuts.
#
# Three forms per sweep size (smoke1=1 instance, smoke=3, full=100):
#   run_dr_<sweep>        agent run + judge in one shot   (~28 min for smoke1)
#   run_dr_<sweep>_agent  agent run only (DR_SKIP_EVAL=1) (~3 min)
#   run_dr_<sweep>_eval   re-judge persisted report.md    (~20 min)
#
# The _agent / _eval split exists because the judge (RACE + FACT, ~25 min)
# is far slower than the agent (~3 min). Run agent first, look at the
# report, then judge — or rerun the judge with different settings without
# re-burning the agent budget.
#
# Requires four env vars at run time (auto-sourced from .env):
#   BRAVE_API_KEY      OC native web_search (Brave Search API)
#   JINA_READER_API    FACT citation scraping (aliased to JINA_API_KEY by
#                      applications/deepresearch_bench/eval/judge_runner.py)
#   DEEPSEEK_API_KEY   RACE + FACT judge (DeepSeek-V4-Pro)
#
# Override HARNESS to swap the web_search provider, and MODEL_PROFILE to swap
# the agent model. Default is Qwen3.6-35B-A3B local + Brave Search.
# Use openclaw_deepresearch_bench_ddg for DuckDuckGo (key-free; rate-limited).
# ----------------------------------------------------------------------
DR_INSTANCE_DATA ?= $(KIT_ROOT)/data/deepresearch_bench/instances.json
DR_LOAD_ENV      := set -a; [ -f $(KIT_ROOT)/.env ] && . $(KIT_ROOT)/.env; set +a;
DR_HARNESS       ?= openclaw_deepresearch_bench_brave
DR_MODEL_PROFILE ?= qwen36_35b_a3b_local_research
DR_SERVING       ?= vllm/nscc

# Materialize DR instances.json from eval/vendor/.../query.jsonl. Idempotent.
dr-tasks:
	@mkdir -p "$$(dirname "$(DR_INSTANCE_DATA)")"
	python -m applications.deepresearch_bench.loader "$(DR_INSTANCE_DATA)"
	@echo "[dr] instance_data -> $(DR_INSTANCE_DATA)"

# --- combined (agent + eval) -----------------------------------------
run_dr_smoke1: APP := deepresearch_bench/smoke1
run_dr_smoke1: HARNESS := $(DR_HARNESS)
run_dr_smoke1: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_smoke1: SERVING := $(DR_SERVING)
run_dr_smoke1: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_smoke: APP := deepresearch_bench/smoke3
run_dr_smoke: HARNESS := $(DR_HARNESS)
run_dr_smoke: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_smoke: SERVING := $(DR_SERVING)
run_dr_smoke: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_full: APP := deepresearch_bench/full100
run_dr_full: HARNESS := $(DR_HARNESS)
run_dr_full: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_full: SERVING := $(DR_SERVING)
run_dr_full: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# --- agent only (DR_SKIP_EVAL=1) -------------------------------------
run_dr_smoke1_agent: APP := deepresearch_bench/smoke1
run_dr_smoke1_agent: HARNESS := $(DR_HARNESS)
run_dr_smoke1_agent: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_smoke1_agent: SERVING := $(DR_SERVING)
run_dr_smoke1_agent: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) DR_SKIP_EVAL=1 python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_smoke_agent: APP := deepresearch_bench/smoke3
run_dr_smoke_agent: HARNESS := $(DR_HARNESS)
run_dr_smoke_agent: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_smoke_agent: SERVING := $(DR_SERVING)
run_dr_smoke_agent: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) DR_SKIP_EVAL=1 python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_full_agent: APP := deepresearch_bench/full100
run_dr_full_agent: HARNESS := $(DR_HARNESS)
run_dr_full_agent: MODEL_PROFILE := $(DR_MODEL_PROFILE)
run_dr_full_agent: SERVING := $(DR_SERVING)
run_dr_full_agent: dr-tasks _ensure_serve
	@$(SERVE_CLEANUP_HOOK); \
	$(DR_LOAD_ENV) DR_SKIP_EVAL=1 python -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

# --- eval only (re-judges latest stamped sweep) ----------------------
# Picks the most recent experiment under data/experiments that contains the
# current app's runs/<app-output-leaf> directory.
# Add EXTRA_ARGS="--force" to re-grade a sweep that already has verdicts.
DR_EVAL_ROOT ?= $(KIT_ROOT)/data/experiments
run_dr_smoke1_eval: APP := deepresearch_bench/smoke1
run_dr_smoke1_eval: dr-tasks
	$(DR_LOAD_ENV) python -m applications.deepresearch_bench.eval.runner \
	  --sweep         $(DR_EVAL_ROOT) \
	  --app-leaf      $(APP_OUTPUT_LEAF) \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_smoke_eval: APP := deepresearch_bench/smoke3
run_dr_smoke_eval: dr-tasks
	$(DR_LOAD_ENV) python -m applications.deepresearch_bench.eval.runner \
	  --sweep         $(DR_EVAL_ROOT) \
	  --app-leaf      $(APP_OUTPUT_LEAF) \
	  --instance-data $(DR_INSTANCE_DATA) \
	  $(EXTRA_ARGS)

run_dr_full_eval: APP := deepresearch_bench/full100
run_dr_full_eval: dr-tasks
	$(DR_LOAD_ENV) python -m applications.deepresearch_bench.eval.runner \
	  --sweep         $(DR_EVAL_ROOT) \
	  --app-leaf      $(APP_OUTPUT_LEAF) \
	  --instance-data $(DR_INSTANCE_DATA) \
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
MIG_SERVE_LOG ?= $(EXPERIMENT_LOG_DIR)/vllm-mig.log
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
run_tb_mig: HARNESS := tb_react_terminal_bench
run_tb_mig: tb-tasks _ensure_mig_serve
	$(PYTHON) -m orchestration.runner.driver \
	  --app     config/application/$(APP).json \
	  --harness config/harness/$(HARNESS).json \
	  --model   config/model/$(MODEL_PROFILE).json \
	  --serving config/serving/$(SERVING).json \
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
