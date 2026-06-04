"""Generate mock run directories so the RQ plot scripts have something to
render against. Useful for paper-figure iteration before real benchmarks
finish on the cluster.

Trace shapes are workload-class-specific — not just a single template
parameterized by AR/Agent — because the figures are meaningless if every
run looks the same:

  code_agent      Plan with the LLM, read source files (TOOL_READ), edit
                  them (TOOL_WRITE), run pytest (TOOL_EXEC, slow). Moderate
                  prompts (~3-10k), short completions, mid-length
                  trajectories. AR is "single-shot patch from issue text",
                  Agent iterates with test feedback.
  doc_analysis    Heavy prefill — load entire documents into the prompt,
                  emit a smaller summary. Tool calls are mostly TOOL_READ
                  for chunk loading. Few but long LLM turns. AR collapses
                  this to one giant LLM call; Agent splits into per-section
                  rounds.
  web_search      Many short LLM turns directing tool queries; lots of
                  TOOL_EXEC for HTTP fetches and TOOL_READ for parsing
                  responses. Network jitter shows up as WAIT_STALL.
                  Prompts/completions both small.

Schema produced per mock run:
    <out>/<sweep>/<run_id>/
      steps.jsonl       — LLM_INFER + TOOL_{READ,WRITE,EXEC} + WAIT_STALL
      util.jsonl        — _meta header (with ncpu + container_groups), then
                          per-second samples carrying cpu.by_group
      spans.jsonl       — synthetic openclaw.model.usage one per LLM step
      verdict.json      — status PASS/FAIL/TIMEOUT
      manifest.json     — condition_id, instance_id, workload_class

Run:
    python -m agent_bench.results.analysis.plots.mock_data --out-dir data/runs_mock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# ---- Conditions and instances --------------------------------------------

# (condition_id, family). Mock data uses a single rolled-up "ar" bucket
# rather than splitting into ar_issue_only / ar_static_context, since the
# paper-side narrative contrasts AR-vs-Agent at the family level. tb_react
# is a harness wrapper, not a separate condition for the figures.
CONDITIONS: List[Tuple[str, str]] = [
    ("ar",             "AR"),
    ("openclaw_agent", "Agent"),
]

WORKLOAD_INSTANCES: Dict[str, List[str]] = {
    "code_agent":   [
        "sympy__sympy-12345",
        "django__django-67890",
        "scikit__scikit-22222",
        "matplotlib__matplotlib-9876",
    ],
    "doc_analysis": [
        "doc-policy-eu-ai-act",
        "doc-medical-handbook-22",
        "doc-financial-10k-aapl",
    ],
    "web_search":   [
        "search-arxiv-llm-papers",
        "search-news-tech-2026",
        "search-product-reviews",
    ],
}

RUNS_PER_CELL = 4   # 4 seeds per (condition, instance, regime)


# Serving regimes — meant to evoke the lat-opt vs thru-opt tradeoff. The
# disagg system optimizes TTFT/TPOT (low queueing, separated prefill/decode);
# the agg system trades token-level latency for throughput via heavy batching.
#
# The interesting research point — encoded in `task_latency_mult` per family —
# is that AR (single-shot) inherits whatever the LLM costs, so lat-opt always
# wins. Agent (multi-turn) benefits from batched concurrency / KV-cache reuse
# under thru-opt, so even though per-token latency is worse, end-to-end task
# latency drops. That's the inversion the regime-comparison figure shows.
@dataclass
class ServingRegime:
    name: str
    label: str       # nice display name
    ttft_mult: float
    tpot_mult: float
    # Per-family wall-time multiplier on the *trajectory* (independent of
    # token-level metrics; captures concurrency / KV-cache effects).
    task_latency_mult: Dict[str, float]


SERVING_REGIMES: List[ServingRegime] = [
    ServingRegime(
        name="lat_opt_disagg",
        label="lat-opt (disagg)",
        ttft_mult=0.55,
        tpot_mult=0.70,
        task_latency_mult={"AR": 0.85, "Agent": 1.20},   # AR wins, Agent loses
    ),
    ServingRegime(
        name="thru_opt_agg",
        label="thru-opt (agg)",
        ttft_mult=1.45,
        tpot_mult=1.30,
        task_latency_mult={"AR": 1.10, "Agent": 0.80},   # AR loses, Agent wins
    ),
]


# ---- Per-workload trace recipes ------------------------------------------

@dataclass
class WorkloadProfile:
    # turns total per AR / Agent
    ar_turns: Tuple[int, int]
    agent_turns: Tuple[int, int]
    # LLM_INFER duration range (ms)
    llm_dur_ms: Tuple[int, int]
    # prompt / completion token ranges
    prompt_tokens:     Tuple[int, int]
    completion_tokens: Tuple[int, int]
    # number of tool calls per LLM turn (Agent only; AR is mostly 0-1)
    tools_per_turn_ar:    Tuple[int, int]
    tools_per_turn_agent: Tuple[int, int]
    # tool type distribution (sum=1.0): {step_subtype -> weight}
    tool_mix: Dict[str, float]   # keys: "TOOL_READ", "TOOL_WRITE", "TOOL_EXEC"
    # tool duration ranges keyed by step_type
    tool_dur_ms: Dict[str, Tuple[int, int]]
    # WAIT_STALL frequency per LLM turn (0..1) and duration range
    stall_prob: float
    stall_dur_ms: Tuple[int, int]
    # latency budgets shape success curves; not the only signal
    target_total_s_ar:    Tuple[int, int]
    target_total_s_agent: Tuple[int, int]
    # TTFT / TPOT magnitudes (ms) per family
    ttft_ar:    Tuple[int, int]
    ttft_agent: Tuple[int, int]
    tpot_ar:    Tuple[int, int]   # ms per output token
    tpot_agent: Tuple[int, int]
    # pass probability
    pass_ar:    float
    pass_agent: float
    # per-component CPU budget (cores) for each family
    cpu_budget_ar:    Dict[str, float]
    cpu_budget_agent: Dict[str, float]


# Numbers chosen to be plausibly representative — not measured. The point
# is to give plots non-trivial structure so layout/sizing/legends are
# usable before real runs land.
WORKLOADS: Dict[str, WorkloadProfile] = {
    "code_agent": WorkloadProfile(
        ar_turns=(1, 1),
        agent_turns=(8, 24),
        llm_dur_ms=(1500, 5000),         # LLM remains the dominant cost
        prompt_tokens=(3000, 9000),
        completion_tokens=(80, 320),
        tools_per_turn_ar=(0, 0),   # AR is single-shot LLM; no tools
        tools_per_turn_agent=(1, 3),
        tool_mix={"TOOL_READ": 0.60, "TOOL_WRITE": 0.25, "TOOL_EXEC": 0.15},
        tool_dur_ms={
            "TOOL_READ":  (60, 350),
            "TOOL_WRITE": (40, 200),
            # pytest is slow but rarely the per-turn norm; agents typically
            # run small targeted subsets, not the full suite.
            "TOOL_EXEC":  (600, 2500),
        },
        stall_prob=0.05,
        stall_dur_ms=(150, 800),
        target_total_s_ar=(20, 45),
        target_total_s_agent=(110, 260),
        ttft_ar=(60, 110),
        ttft_agent=(80, 140),
        tpot_ar=(20, 28),
        tpot_agent=(24, 34),
        pass_ar=0.32,
        pass_agent=0.58,
        cpu_budget_ar    = {"llm": 1.6, "client": 0.4, "tools": 0.0, "other": 0.5},
        cpu_budget_agent = {"llm": 1.5, "client": 0.5, "tools": 1.4, "other": 0.6},
    ),
    "doc_analysis": WorkloadProfile(
        ar_turns=(1, 2),
        agent_turns=(4, 10),
        llm_dur_ms=(4000, 14000),         # long because prompts are huge
        prompt_tokens=(20000, 90000),     # whole-doc prompt
        completion_tokens=(300, 1800),    # summary / answer
        tools_per_turn_ar=(0, 0),   # AR is single-shot LLM; no tools
        tools_per_turn_agent=(0, 2),
        tool_mix={"TOOL_READ": 0.85, "TOOL_WRITE": 0.10, "TOOL_EXEC": 0.05},
        tool_dur_ms={
            "TOOL_READ":  (200, 1200),    # chunk loaders
            "TOOL_WRITE": (40, 150),
            "TOOL_EXEC":  (200, 1500),
        },
        stall_prob=0.02,
        stall_dur_ms=(100, 400),
        target_total_s_ar=(35, 70),
        target_total_s_agent=(120, 320),
        ttft_ar=(180, 320),               # prefill heavy → long TTFT
        ttft_agent=(220, 380),
        tpot_ar=(30, 48),
        tpot_agent=(36, 60),
        pass_ar=0.18,
        pass_agent=0.42,
        cpu_budget_ar    = {"llm": 1.9, "client": 0.4, "tools": 0.0, "other": 0.5},
        cpu_budget_agent = {"llm": 1.8, "client": 0.5, "tools": 0.7, "other": 0.6},
    ),
    "web_search": WorkloadProfile(
        ar_turns=(1, 2),
        agent_turns=(10, 28),
        llm_dur_ms=(400, 1500),
        prompt_tokens=(700, 2800),
        completion_tokens=(40, 200),
        tools_per_turn_ar=(0, 0),   # AR is single-shot LLM; no tools
        tools_per_turn_agent=(2, 5),
        tool_mix={"TOOL_EXEC": 0.40, "TOOL_READ": 0.55, "TOOL_WRITE": 0.05},
        tool_dur_ms={
            "TOOL_READ":  (80, 600),
            "TOOL_WRITE": (40, 150),
            # HTTP/search calls — usually sub-second; pathological 2.5s is rare.
            "TOOL_EXEC":  (200, 1200),
        },
        stall_prob=0.18,                  # rate-limit waits
        stall_dur_ms=(400, 2500),
        target_total_s_ar=(15, 40),
        target_total_s_agent=(60, 180),
        ttft_ar=(50, 95),
        ttft_agent=(60, 120),
        tpot_ar=(20, 30),
        tpot_agent=(22, 32),
        pass_ar=0.24,
        pass_agent=0.56,
        cpu_budget_ar    = {"llm": 1.4, "client": 0.4, "tools": 0.0, "other": 0.5},
        cpu_budget_agent = {"llm": 1.2, "client": 0.6, "tools": 1.0, "other": 0.6},
    ),
}


# ---- Helpers --------------------------------------------------------------

def _stable_run_id(*parts) -> str:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


def _pick_tool_type(rng: random.Random, mix: Dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for k, w in mix.items():
        acc += w
        if r <= acc:
            return k
    return next(iter(mix))


def _gen_steps(run_id: str, instance_id: str, condition_id: str,
               family: str, profile: WorkloadProfile,
               regime: ServingRegime,
               start_ns: int, rng: random.Random) -> Tuple[List[Dict], int]:
    """Returns (steps, target_total_s).

    Step durations come from regime-scaled TTFT/TPOT so RQ3's TTFT/TPOT
    metrics reflect the regime. After step generation, the trajectory's
    timestamps are linearly rescaled so its wall-clock span equals the
    regime-targeted wall (`base x task_latency_mult[family]`). That makes
    the AR-wins-on-disagg / Agent-wins-on-agg inversion show up as a
    direct property of the manifest's `duration_seconds`, independent of
    the early-exit interaction with per-turn LLM cost.
    """
    n_turns = rng.randint(*(profile.ar_turns if family == "AR"
                            else profile.agent_turns))
    base_target_s = rng.randint(*(profile.target_total_s_ar if family == "AR"
                                  else profile.target_total_s_agent))
    target_total_s = max(1, int(base_target_s * regime.task_latency_mult.get(family, 1.0)))
    target_total_ns = target_total_s * 1_000_000_000

    steps: List[Dict] = []
    cur_ns = start_ns
    step_id = 0
    attempt_id = 1

    for turn_idx in range(n_turns):
        # LLM_INFER — duration is roughly TTFT + TPOT x completion_tokens, and
        # the regime scales TTFT and TPOT independently. Working backwards
        # from the profile's llm_dur_ms range gives a coherent story for the
        # token-level metrics that evaluation.py later derives.
        step_id += 1
        prompt_tokens = rng.randint(*profile.prompt_tokens)
        completion_tokens = rng.randint(*profile.completion_tokens)
        ttft_base = rng.randint(*(profile.ttft_ar if family == "AR"
                                  else profile.ttft_agent))
        tpot_base = rng.randint(*(profile.tpot_ar if family == "AR"
                                  else profile.tpot_agent))
        ttft_ms = ttft_base * regime.ttft_mult
        tpot_ms = tpot_base * regime.tpot_mult
        llm_dur_ms = int(ttft_ms + tpot_ms * completion_tokens)
        s = cur_ns
        e = s + llm_dur_ms * 1_000_000
        steps.append({
            "trajectory_id": run_id,
            "attempt_id": attempt_id,
            "step_id": step_id,
            "parent_step_id": None,
            "step_type": "LLM_INFER",
            "step_subtype": None,
            "tool_name": None,
            "command": None,
            "file_path": None,
            "start_ts": s,
            "end_ts": e,
            "duration_ms": llm_dur_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "exit_code": None,
            "error_type": None,
            "repo_id": None,
            "instance_id": instance_id,
            "config_id": condition_id,
            "model_name": "vllm/Qwen3.6-35B-A3B",
            "response_preview": None,
        })
        cur_ns = e

        # Tools per LLM turn
        n_tools = rng.randint(*(profile.tools_per_turn_ar if family == "AR"
                                else profile.tools_per_turn_agent))
        for _ in range(n_tools):
            step_id += 1
            ttype = _pick_tool_type(rng, profile.tool_mix)
            tdur_ms = rng.randint(*profile.tool_dur_ms[ttype])
            s = cur_ns
            e = s + tdur_ms * 1_000_000
            steps.append({
                "trajectory_id": run_id,
                "attempt_id": attempt_id,
                "step_id": step_id,
                "parent_step_id": None,
                "step_type": ttype,
                "step_subtype": None,
                "tool_name": "container_bash" if ttype == "TOOL_EXEC" else "fs",
                "command": "<mock>" if ttype == "TOOL_EXEC" else None,
                "file_path": "<mock-path>" if ttype != "TOOL_EXEC" else None,
                "start_ts": s,
                "end_ts": e,
                "duration_ms": tdur_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "exit_code": (0 if rng.random() > 0.12 else 1) if ttype == "TOOL_EXEC" else None,
                "error_type": None,
                "repo_id": None,
                "instance_id": instance_id,
                "config_id": condition_id,
                "model_name": None,
                "response_preview": None,
            })
            cur_ns = e

        # Optional WAIT_STALL
        if rng.random() < profile.stall_prob:
            step_id += 1
            wdur_ms = rng.randint(*profile.stall_dur_ms)
            s = cur_ns
            e = s + wdur_ms * 1_000_000
            steps.append({
                "trajectory_id": run_id,
                "attempt_id": attempt_id,
                "step_id": step_id,
                "parent_step_id": None,
                "step_type": "WAIT_STALL",
                "step_subtype": None,
                "tool_name": None,
                "command": None,
                "file_path": None,
                "start_ts": s,
                "end_ts": e,
                "duration_ms": wdur_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "exit_code": None,
                "error_type": None,
                "repo_id": None,
                "instance_id": instance_id,
                "config_id": condition_id,
                "model_name": None,
                "response_preview": None,
            })
            cur_ns = e

        # Agent: occasional retry attempt mid-trajectory
        if family == "Agent" and rng.random() < 0.07:
            attempt_id += 1

        # Bail if we've blown past the target wall budget — keeps durations
        # close to the per-workload latency target for the latency plots.
        if cur_ns - start_ns > target_total_ns:
            break

    # Pad or trim the trajectory's wall-clock span to match target_total_s
    # without touching individual step durations (so TTFT/TPOT computed
    # from steps stay regime-scaled).
    target_total_ns_actual = target_total_s * 1_000_000_000
    actual_span = cur_ns - start_ns
    if actual_span < target_total_ns_actual:
        # Pad with a final WAIT_STALL so steady-state task latency reflects
        # the regime's task_latency_mult.
        pad_ns = target_total_ns_actual - actual_span
        if pad_ns > 0:
            step_id += 1
            steps.append({
                "trajectory_id": run_id,
                "attempt_id": attempt_id,
                "step_id": step_id,
                "parent_step_id": None,
                "step_type": "WAIT_STALL",
                "step_subtype": "regime_pad",
                "tool_name": None,
                "command": None,
                "file_path": None,
                "start_ts": cur_ns,
                "end_ts":   cur_ns + pad_ns,
                "duration_ms": pad_ns / 1e6,
                "prompt_tokens": None,
                "completion_tokens": None,
                "exit_code": None,
                "error_type": None,
                "repo_id": None,
                "instance_id": instance_id,
                "config_id": condition_id,
                "model_name": None,
                "response_preview": None,
            })
            cur_ns += pad_ns
    return steps, target_total_s


def _gen_util_jsonl(family: str, profile: WorkloadProfile, duration_s: int,
                    start_ns: int, ncpu: int, rng: random.Random) -> List[Dict]:
    out: List[Dict] = []
    out.append({
        "_meta":         True,
        "started_iso":   datetime.now(tz=timezone.utc).isoformat(),
        "started_ts_ns": start_ns,
        "interval":      1.0,
        "pid":           100000 + rng.randint(0, 1000),
        "host":          "mock-host",
        "nvidia_smi":    True,
        "ncpu":          ncpu,
        "container_groups": {"llm_engine": ["oc-vllm-serve"], "client": ["oc-gateway"], "tools": ["task-sandbox"]},
    })
    budget = profile.cpu_budget_ar if family == "AR" else profile.cpu_budget_agent

    def _pct(target_cores: float, t_phase: float) -> float:
        # Slight ramp-up + jitter so timeline isn't dead-flat.
        ramp = 1.0 - 0.4 * (1.0 - min(1.0, t_phase / 6.0))
        jitter = rng.uniform(-0.18, 0.22)
        return max(0.0, round(100.0 * (target_cores * ramp + jitter), 2))

    for s in range(duration_s):
        ts = start_ns + s * 1_000_000_000
        llm_pct    = _pct(budget["llm"],    s)
        client_pct = _pct(budget["client"], s)
        tools_pct  = _pct(budget["tools"],  s)
        other_pct  = _pct(budget["other"],  s)
        host_pct = round(min(100.0, (llm_pct + client_pct + tools_pct + other_pct) / max(1, ncpu)), 2)
        # GPU stays busy across all conditions because vLLM is shared.
        gpu_util = round(min(100.0, max(0.0, rng.gauss(82, 7))), 2)
        out.append({
            "ts_ns": ts,
            "iso":   datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).isoformat(),
            "cpu": {
                "pct":          host_pct,
                "per_cpu_pct":  [round(rng.uniform(0, 100), 2) for _ in range(ncpu)],
                "loadavg":      [round(rng.gauss(2.0, 0.4), 2)] * 3,
                "mem_total_mb": 256000,
                "mem_used_mb":  rng.randint(50000, 110000),
                "mem_pct":      round(rng.uniform(20, 45), 2),
                "by_group": {
                    "llm_engine": {"pct": llm_pct,    "rss_mb": rng.randint(28000, 36000), "n_proc": 4},
                    "client":     {"pct": client_pct, "rss_mb": rng.randint(400, 1500),    "n_proc": 2},
                    "tools":      {"pct": tools_pct,  "rss_mb": rng.randint(80, 800),      "n_proc": rng.randint(1, 6)},
                    "other":      {"pct": other_pct},
                },
            },
            "gpu": [{
                "index": 0, "name": "NVIDIA A100 80G",
                "util_gpu_pct": gpu_util,
                "util_mem_pct": round(rng.uniform(60, 85), 2),
                "mem_used_mb":  rng.randint(40000, 70000),
                "mem_total_mb": 81920,
                "power_w":      round(rng.gauss(290, 18), 1),
                "temp_c":       round(rng.gauss(58, 4), 1),
            }],
        })
    return out


def _make_run(out_root: Path, sweep_name: str, condition_id: str, family: str,
              instance_id: str, workload_class: str, regime: ServingRegime,
              seed: int) -> Path:
    rng = random.Random(
        seed * 31 + hash(condition_id) % 100003 + hash(instance_id) % 100003
        + hash(regime.name) % 100003
    )
    profile = WORKLOADS[workload_class]
    run_id = _stable_run_id(condition_id, instance_id, seed, workload_class, regime.name)
    run_dir = out_root / sweep_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    start_ns = int(time.time_ns()) - rng.randint(60, 600) * 1_000_000_000

    steps, duration_s = _gen_steps(run_id, instance_id, condition_id,
                                   family, profile, regime, start_ns, rng)
    with (run_dir / "steps.jsonl").open("w") as fh:
        for s in steps:
            fh.write(json.dumps(s) + "\n")

    ncpu = 80
    util = _gen_util_jsonl(family, profile, duration_s, start_ns, ncpu, rng)
    with (run_dir / "util.jsonl").open("w") as fh:
        for r in util:
            fh.write(json.dumps(r) + "\n")

    # Verdict
    p_pass = profile.pass_ar if family == "AR" else profile.pass_agent
    passed = rng.random() < p_pass
    # Long doc_analysis runs sometimes time out; web_search rarely does.
    if not passed and workload_class == "doc_analysis" and rng.random() < 0.30:
        status = "TIMEOUT"
    elif not passed and rng.random() < 0.10:
        status = "TIMEOUT"
    else:
        status = "PASS" if passed else "FAIL"
    verdict = {
        "status": status,
        "test_pass_rate": 1.0 if passed else 0.0,
        "passed": int(passed),
        "failed": 0 if passed else 1,
        "instance_id": instance_id,
        "config_id": condition_id,
        "duration_seconds": duration_s,
    }
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    # Manifest
    end_ns = steps[-1]["end_ts"] if steps else (start_ns + duration_s * 1_000_000_000)
    manifest = {
        "run_id": run_id,
        "condition_id": condition_id,
        "config_id": condition_id,
        "instance_id": instance_id,
        "seed": seed,
        "serving_config_id": regime.name,
        "serving_regime":    regime.name,
        "serving_regime_label": regime.label,
        "model_name": "vllm/Qwen3.6-35B-A3B",
        "workload_class": workload_class,
        "status": status,
        "step_trace_path": str(run_dir / "steps.jsonl"),
        "verdict_path":    str(run_dir / "verdict.json"),
        "telemetry_path":  str(run_dir / "util.jsonl"),
        "wall_start_ns":   start_ns,
        "wall_end_ns":     end_ns,
        "duration_seconds": duration_s,
        "exit_code": 0 if passed else 1,
        "total_prompt_tokens":     sum(s.get("prompt_tokens") or 0     for s in steps),
        "total_completion_tokens": sum(s.get("completion_tokens") or 0 for s in steps),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # spans.jsonl — synthetic openclaw.model.usage per LLM turn, plus a
    # tool.call per tool step, plus a workload-level session span. TTFT /
    # TPOT magnitudes come from the workload profile so RQ3 scatter shows
    # workload-class clustering even though the figures derive these from
    # the step-trace fallback today.
    spans = []
    for st in steps:
        if st["step_type"] == "LLM_INFER":
            spans.append({
                "name":            "openclaw.model.usage",
                "start_unix_nano": st["start_ts"],
                "end_unix_nano":   st["end_ts"],
                "duration_ms":     st["duration_ms"],
                "attrs": {
                    "openclaw.prompt_tokens":     st["prompt_tokens"],
                    "openclaw.completion_tokens": st["completion_tokens"],
                },
                "resource": {
                    "oc.run_id":     run_id,
                    "service.name":  condition_id,
                    "tb.workload_class": workload_class,
                },
            })
    with (run_dir / "spans.jsonl").open("w") as fh:
        for sp in spans:
            fh.write(json.dumps(sp) + "\n")
    return run_dir


def generate(out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_name = "mock_sweep"
    produced: List[Path] = []
    for cid, family in CONDITIONS:
        for wl, instances in WORKLOAD_INSTANCES.items():
            for inst in instances:
                for regime in SERVING_REGIMES:
                    for seed in range(RUNS_PER_CELL):
                        p = _make_run(out_dir, sweep_name, cid, family,
                                      inst, wl, regime, seed)
                        produced.append(p)
    return produced


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/runs_mock"))
    args = ap.parse_args(argv)
    runs = generate(args.out_dir)
    print(f"wrote {len(runs)} mock run dirs under {args.out_dir}/mock_sweep/")
    print(f"  conditions: {[c for c, _ in CONDITIONS]}")
    print(f"  workloads:  {list(WORKLOAD_INSTANCES.keys())}")
    print(f"  regimes:    {[r.name for r in SERVING_REGIMES]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
