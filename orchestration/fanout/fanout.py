#!/usr/bin/env python3
"""Fan benchmark experiments out across every GPU we can reach.

A thin, experiment-agnostic orchestrator on top of the scripts/serve_only
wrappers (run_ccds.sh = Slurm, run_nscc.sh = PBS, run_baremetal.sh = fixed
hosts). It expands a TOML sweep into cells = arms x task-shards, schedules them
across the enabled targets (respecting per-target concurrency + GPU packing),
records WHERE each subjob landed and the GPU it ran on (the placement manifest),
and optionally runs a configurable analysis hook over the results at the end.

"Arms" are just variant labels. An experiment differentiates them via an optional
per-arm environment overlay in the sweep TOML ([arm_env.<arm>] tables) — the
orchestrator merges those vars into each cell's environment and otherwise knows
nothing about what they mean. A single-arm sweep (arms = ["default"]) is a plain
sharded fan-out; a multi-arm sweep with [arm_env] is an A/B (or A/B/C/...).

The wrappers already solve per-run free ports, atomic configs, safe
backgrounding and trap cleanup; this only does placement + bookkeeping.

Usage:
  python -m orchestration.fanout.fanout experiments/fanout.toml
  python -m orchestration.fanout.fanout <sweep.toml> --dry-run
  python -m orchestration.fanout.fanout <sweep.toml> --accounts <userA>,<userB>
  python -m orchestration.fanout.fanout <sweep.toml> --targets ccds,h100

Each cell launches ONE wrapper. The wrapper serves a vLLM endpoint on a GPU, runs
the benchmark client over it, and cleans the allocation up on exit. GPU packing
(several servers on one card) is requested per-target via gpu_mem_util.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.core.profile_io import load_profile

KIT_ROOT = Path(__file__).resolve().parents[2]
SERVE_ONLY = KIT_ROOT / "scripts" / "serve_only"
WRAPPERS = {
    "slurm": SERVE_ONLY / "run_ccds.sh",
    "pbs": SERVE_ONLY / "run_nscc.sh",
    "baremetal": SERVE_ONLY / "run_baremetal.sh",
}
# Substrings in a cell log that mean "the GPU wasn't available" rather than a
# real benchmark failure -> requeue the cell instead of recording it failed.
REQUEUE_MARKERS = (
    "no completely-free GPUs",
    "no node satisfies",
    "free-GPU discovery failed",
    "timed out waiting for job",
    "node not allocatable",        # ccds serve job stayed PENDING (gres reserved elsewhere)
    "reallocate: server not ready",  # serve didn't answer /v1/models within the readiness timeout
    "no runs completed (all errored)",  # driver: every run threw before scoring (transient client-side infra) — retry
)
NSCC_HOSTNAME = "aspire2antu.nscc.sg"


# --------------------------------------------------------------------------- #
# Sweep expansion
# --------------------------------------------------------------------------- #
def _abs(p: str) -> str:
    """Resolve a profile path against the repo root (sweep paths are repo-relative)."""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (KIT_ROOT / pp))


def shard_instances(instances: list[str], n: int) -> list[list[str]]:
    """Split task ids into n balanced, contiguous shards (drops empty trailing shards)."""
    n = max(1, min(n, len(instances)))
    base, extra = divmod(len(instances), n)
    out, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        out.append(instances[i : i + size])
        i += size
    return [s for s in out if s]


def write_shard_apps(benchmark_path: str, run_dir: Path, shards: int) -> list[tuple[int, str, list[str]]]:
    """Write one APP json per shard (benchmark profile with `instances` narrowed)."""
    profile = load_profile(benchmark_path)
    instances = profile.get("instances") or []
    if not instances:
        raise SystemExit(f"benchmark profile has no `instances`: {benchmark_path}")
    apps_dir = run_dir / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for k, chunk in enumerate(shard_instances(list(instances), shards)):
        shard_profile = copy.deepcopy(profile)
        shard_profile["instances"] = chunk
        path = apps_dir / f"shard{k}.json"
        path.write_text(json.dumps(shard_profile, indent=2) + "\n")
        result.append((k, str(path), chunk))
    return result


@dataclass
class Cell:
    cell_id: str
    arm: str            # experiment-variant label (see [arm_env] in the sweep TOML)
    shard: int
    app_path: str
    task_ids: list[str]
    attempts: int = 0
    status: str = "pending"
    rc: int | None = None
    target: str | None = None
    run_id: str | None = None
    results_dir: str | None = None


def build_cells(arms: list[str], shards_info: list[tuple[int, str, list[str]]]) -> list[Cell]:
    cells = []
    for arm in arms:
        for k, app_path, task_ids in shards_info:
            cells.append(Cell(cell_id=f"{arm}_s{k}", arm=arm, shard=k, app_path=app_path, task_ids=task_ids))
    return cells


# --------------------------------------------------------------------------- #
# Targets / lanes (a lane = one concurrency pool with a launcher)
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    name: str           # ccds | nscc:<user> | a6000
    kind: str           # slurm | pbs | baremetal
    capacity: int
    base_env: dict[str, str] = field(default_factory=dict)
    gpu_mem_util: float | None = None
    in_use: int = 0
    packed: bool = False      # several servers share ONE physical GPU (servers_per_gpu>1)
    last_launch: float = 0.0  # ts of the most recent launch on this lane (boot-stagger gate)
    last_stagger_log: float = 0.0


def _nscc_pass_file(user: str) -> str:
    """Per-account password file (nscc disabled key auth — only passwords)."""
    return os.path.expanduser(f"~/.ssh/nscc_{user}.pass")


def discover_nscc_accounts() -> list[str]:
    """Account = the suffix of each ~/.ssh/nscc_<user> credential — a password
    file (nscc_<user>.pass; nscc now requires password auth) or a legacy private
    key (nscc_<user>)."""
    users: set[str] = set()
    for p in glob.glob(os.path.expanduser("~/.ssh/nscc_*")):
        name = Path(p).name[len("nscc_"):]
        if name.endswith(".pub"):
            continue
        users.add(name[:-len(".pass")] if name.endswith(".pass") else name)
    return sorted(users)


# Shared ControlMaster socket dir + persistence. One master per account
# (ControlPath %C hashes host+user, so it's stable across the generated config
# and nscc_login.sh): the password authenticates ONCE and every later qstat/
# qdel/tunnel connection multiplexes over it. ControlPersist must outlast the
# longest idle gap on an account (and ideally the whole sweep) so a no-sshpass,
# pre-opened master stays up — keep it well above limits.time_budget_min.
SSH_CM_DIR = Path.home() / ".ssh" / "cm"
SSH_CM_PERSIST = "4h"


def gen_ssh_config(accounts: list[str], hostname: str, out_path: Path) -> Path:
    """Write a scoped ssh config (one Host nscc-<user> block per account) so the
    wrappers reach nscc under any account via `ssh -F <this>` without touching
    ~/.ssh/config. Each block multiplexes over a shared ControlMaster socket;
    accounts with a password file (nscc_<user>.pass) authenticate by password
    (sshpass / a pre-opened master), legacy key accounts keep IdentityFile."""
    SSH_CM_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for user in accounts:
        key = os.path.expanduser(f"~/.ssh/nscc_{user}")
        lines = [
            f"Host nscc-{user}",
            f"    HostName {hostname}",
            f"    User {user}",
            "    ControlMaster auto",
            f'    ControlPath "{SSH_CM_DIR}/%C"',
            f"    ControlPersist {SSH_CM_PERSIST}",
        ]
        if os.path.exists(_nscc_pass_file(user)):
            # nscc disabled key auth: go straight to password. No IdentityFile /
            # BatchMode — both suppress the password prompt sshpass needs.
            lines += [
                "    PreferredAuthentications password",
                "    PubkeyAuthentication no",
                "    NumberOfPasswordPrompts 1",
            ]
        else:
            lines += [f'    IdentityFile "{key}"', "    BatchMode yes"]
        blocks.append("\n".join(lines) + "\n")
    out_path.write_text("\n".join(blocks))
    return out_path


def build_lanes(sweep: dict, args, run_dir: Path) -> tuple[list[Lane], str | None]:
    targets = sweep.get("targets") or {}
    only = set(args.targets.split(",")) if args.targets else None
    lanes: list[Lane] = []
    ssh_config_path: str | None = None

    for tname, tcfg in targets.items():
        if not tcfg.get("enabled", False):
            continue
        if only and tname not in only:
            continue
        kind = tcfg.get("kind")
        if kind == "slurm":
            # A single lane with capacity>1 lets several run_ccds.sh race the same
            # non-atomic free-GPU probe and double-book one node's cards. The shared
            # claims file (flock'd in run_ccds.sh) keeps their picks disjoint.
            slurm_env = {"CCDS_CLAIMS": str(run_dir / f"{tname}.claims")}
            # Per-target serving venv (e.g. the vllm fork) — run_ccds.sh reads
            # VLLM_VENV; without it the wrapper defaults to $HOME/vllm/.venv.
            if tcfg.get("vllm_venv"):
                slurm_env["VLLM_VENV"] = str(tcfg["vllm_venv"])
            # Slurm calls it --time; run_ccds.sh reads SRUN_TIME (default 02:00:00).
            # Accept the same `walltime` key the PBS targets use so the TOML is uniform.
            if tcfg.get("walltime"):
                slurm_env["SRUN_TIME"] = str(tcfg["walltime"]).strip()
            lanes.append(Lane(
                name=tname, kind=kind,
                capacity=int(tcfg.get("max_parallel", 1)),
                gpu_mem_util=tcfg.get("gpu_mem_util"),
                base_env=slurm_env,
            ))
        elif kind == "pbs" and tcfg.get("remote"):
            # Single-account PBS host already in ~/.ssh/config (e.g. hpcc, an H200
            # cluster): one lane, plain `ssh <remote>` (no generated multi-account
            # config). run_nscc.sh is the generic PBS wrapper — pass project/queue/
            # ncpus/walltime through to its qsub. The serve_only README lists the
            # exact directives this maps to.
            env = {"REMOTE": str(tcfg["remote"]), "NGPUS": str(tcfg.get("ngpus", 1))}
            for ekey, opt in (("PROJECT", "project"), ("QUEUE", "queue"),
                              ("NCPUS", "ncpus"), ("WALLTIME", "walltime"),
                              ("CLIENT_HOST", "client_host"),
                              # per-target serving venv (e.g. vllm fork);
                              # run_nscc.sh reads NSCC_VLLM_VENV.
                              ("NSCC_VLLM_VENV", "vllm_venv")):
                if tcfg.get(opt) is not None:
                    env[ekey] = str(tcfg[opt])
            lanes.append(Lane(
                name=tname, kind="pbs",
                capacity=int(tcfg.get("max_parallel", 1)),
                gpu_mem_util=tcfg.get("gpu_mem_util"),
                base_env=env,
            ))
        elif kind == "pbs":
            accounts = (
                discover_nscc_accounts() if args.accounts == "all"
                else args.accounts.split(",") if args.accounts
                else tcfg.get("accounts") or []
            )
            if not accounts:
                raise SystemExit(
                    f"target '{tname}': no nscc accounts — set accounts=[...] in the "
                    "sweep TOML, pass --accounts <user>[,<user>], or --accounts all "
                    "(every ~/.ssh/nscc_* key)."
                )
            cfg_path = gen_ssh_config(accounts, tcfg.get("hostname", NSCC_HOSTNAME), run_dir / "ssh_config")
            ssh_config_path = str(cfg_path)
            for user in accounts:
                lanes.append(Lane(
                    name=f"{tname}:{user}", kind=kind,
                    capacity=int(tcfg.get("max_parallel_per_account", 1)),
                    gpu_mem_util=tcfg.get("gpu_mem_util"),
                    base_env={
                        "REMOTE": f"nscc-{user}",
                        "SSH_CONFIG": str(cfg_path),
                        "NGPUS": str(tcfg.get("ngpus", 2)),
                        **({"PROJECT": str(tcfg["project"])} if tcfg.get("project") else {}),
                        # PBS walltime; run_nscc.sh reads WALLTIME (default 02:00:00).
                        **({"WALLTIME": str(tcfg["walltime"]).strip()} if tcfg.get("walltime") else {}),
                        # per-target serving venv (e.g. vllm fork); run_nscc.sh reads it.
                        **({"NSCC_VLLM_VENV": str(tcfg["vllm_venv"])} if tcfg.get("vllm_venv") else {}),
                        # nscc disabled key auth: point the wrappers (so_ssh, tunnel.sh) at
                        # this account's password file so sshpass can feed it. Absent =
                        # legacy key / a pre-opened ControlMaster (nscc_login.sh).
                        **({"SO_SSH_PASS_FILE": _nscc_pass_file(user)}
                           if os.path.exists(_nscc_pass_file(user)) else {}),
                    },
                ))
        elif kind == "baremetal":
            env = {"REMOTE": tcfg["remote"]}
            if tcfg.get("direct_route"):
                env["DIRECT_ROUTE"] = "1"
            for opt in ("REMOTE_KIT_ROOT", "VLLM_VENV", "ENDPOINT_HOST"):
                if tcfg.get(opt.lower()):
                    env[opt] = str(tcfg[opt.lower()])
            spg = int(tcfg.get("servers_per_gpu", 1))
            cuda_devices = tcfg.get("cuda_devices")
            if cuda_devices:
                # Pin one lane per physical card so cells SPREAD across the host's
                # GPUs (run_baremetal's free-VRAM auto-pick can't coordinate
                # concurrent launches); servers_per_gpu then packs within a card.
                for dev in cuda_devices:
                    lanes.append(Lane(
                        name=f"{tname}:gpu{dev}", kind=kind, capacity=spg,
                        gpu_mem_util=tcfg.get("gpu_mem_util"),
                        base_env={**env, "CUDA_VISIBLE_DEVICES": str(dev)},
                        packed=spg > 1,
                    ))
            else:
                lanes.append(Lane(
                    name=tname, kind=kind, capacity=spg,
                    gpu_mem_util=tcfg.get("gpu_mem_util"), base_env=env,
                    packed=spg > 1,
                ))
        else:
            raise SystemExit(f"target {tname}: unknown kind {kind!r}")
    return lanes, ssh_config_path


# --------------------------------------------------------------------------- #
# Launch + scheduling
# --------------------------------------------------------------------------- #
def cell_env(cell: Cell, lane: Lane, sweep: dict, run_dir: Path, manifest: Path, skip_pull: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(lane.base_env)
    # APP + HARNESS_CONFIG are consumed locally on ntu_cpu (run_tb_client) -> absolute
    # is fine. MODEL_TEMPLATE + SERVING_CONFIG are FORWARDED to the remote GPU host's
    # serve_vllm.sh, which resolves them against ITS own checkout -> must stay
    # repo-relative (an ntu_cpu-absolute path doesn't exist there).
    env["APP"] = cell.app_path
    env["HARNESS_CONFIG"] = _abs(sweep["harness"])
    env["MODEL_TEMPLATE"] = sweep["model_template"]
    env["SERVING_CONFIG"] = sweep["serving"]
    env["ARM"] = cell.arm
    # Per-arm environment overlay from the sweep TOML ([arm_env.<arm>] tables).
    # This is how an experiment injects its variant-specific knobs (e.g. a feature
    # toggle the harness reads) without the orchestrator knowing what they mean.
    for k, v in (sweep.get("arm_env", {}).get(cell.arm, {}) or {}).items():
        env[str(k)] = str(v)
    env["ALLOC_MANIFEST"] = str(manifest)
    env["RUN_ID"] = cell.run_id
    env["RESULTS"] = cell.results_dir
    if skip_pull:
        env["SKIP_PULL"] = "1"
    if lane.gpu_mem_util is not None:
        # Per-lane packing override. serve_only/serve_vllm.sh forwards the bare
        # GPU_MEM_UTIL to the Python launcher as VLLM_GPU_UTIL (which feeds
        # launch_plan._build_vllm_plan's --gpu-memory-utilization).
        env["GPU_MEM_UTIL"] = str(lane.gpu_mem_util)
    return env


def launch(cell: Cell, lane: Lane, sweep: dict, run_dir: Path, manifest: Path,
           skip_pull: bool) -> tuple[subprocess.Popen, Any]:
    log_path = run_dir / "logs" / f"{cell.cell_id}.{cell.attempts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    env = cell_env(cell, lane, sweep, run_dir, manifest, skip_pull)
    proc = subprocess.Popen(
        ["bash", str(WRAPPERS[lane.kind])],
        cwd=str(KIT_ROOT), env=env,
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    print(f"[fanout] launch {cell.cell_id} -> {lane.name} (pid={proc.pid}, log={log_path.name})")
    return proc, log_fh


def is_requeue_failure(log_path: Path) -> bool:
    # Only the tail carries requeue markers; seek to the last 4000 bytes instead
    # of reading the whole (potentially large) log. Read as bytes — these logs
    # aren't guaranteed clean UTF-8. (Markers are ASCII, so the byte vs. char
    # window and a possible split multibyte char at the seek point don't matter.)
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(-4000, os.SEEK_END)
            except OSError:
                pass  # file shorter than 4000 bytes: read from the start
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(m in tail for m in REQUEUE_MARKERS)


def schedule(cells: list[Cell], lanes: list[Lane], sweep: dict, run_dir: Path) -> Path:
    manifest = run_dir / "manifest.jsonl"
    limits = sweep.get("limits") or {}
    deadline = time.time() + float(limits.get("time_budget_min", 60)) * 60
    max_attempts = int(limits.get("max_attempts_per_cell", 3))
    skip_pull = bool(limits.get("skip_pull", True))
    # When several servers share one GPU (servers_per_gpu>1), two concurrent vLLM
    # boots profile the card at the same time and the one that profiles second
    # sees the other's reservation -> negative KV cache -> engine-core crash. Hold
    # each subsequent pack on a lane until the booting sibling has finished its GPU
    # memory profiling. Must exceed the model's profile+KV-alloc time (h200 35B FP8
    # ~110s); override per-sweep with limits.boot_stagger_sec.
    boot_stagger = float(limits.get("boot_stagger_sec", 120))

    def log_event(**rec):
        rec["src"] = "orchestrator"
        rec["ts"] = round(time.time(), 3)
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")

    pending = list(cells)
    running: list[dict] = []           # {cell, lane, proc, log_fh, log_path}
    first_launch_done = False          # so the first cell pulls images, the rest reuse
    li = 0                             # round-robin lane cursor

    # Catch Ctrl-C so we TERMINATE running cells (their traps run scancel /
    # remote_serve stop to free the GPU) instead of crashing out and leaking the
    # remote vllm servers. A second Ctrl-C restores the default handler to force-quit.
    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        if interrupted["flag"]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        else:
            print("\n[fanout] interrupt — terminating running cells so their traps free the "
                  "GPU (Ctrl-C again to force-quit) ...")
        interrupted["flag"] = True
    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    while pending or running:
        if interrupted["flag"] or time.time() > deadline:
            why = "interrupted" if interrupted["flag"] else "time budget exceeded"
            print(f"[fanout] {why} — terminating running cells (their traps free the GPU).")
            for r in running:
                r["proc"].terminate()
            break

        # Assign pending cells to any lane with free capacity (round-robin lanes).
        progressed = True
        while pending and progressed:
            progressed = False
            for _ in range(len(lanes)):
                lane = lanes[li % len(lanes)] if lanes else None
                li += 1
                if lane is None:
                    raise SystemExit("[fanout] no enabled targets to schedule onto.")
                if lane.in_use < lane.capacity:
                    # Boot-stagger: don't pack a 2nd server onto a card while the
                    # first is still profiling GPU memory (else it computes negative
                    # KV and crashes). Skip this lane for now; try others / wait.
                    if lane.packed and lane.in_use > 0:
                        wait = boot_stagger - (time.time() - lane.last_launch)
                        if wait > 0:
                            if time.time() - lane.last_stagger_log > 15:
                                print(f"[fanout] {lane.name}: holding next pack ~{int(wait)}s "
                                      f"(sibling still profiling GPU memory)")
                                lane.last_stagger_log = time.time()
                            continue
                    cell = pending.pop(0)
                    cell.attempts += 1
                    cell.target = lane.name
                    cell.run_id = f"{cell.cell_id}@{lane.name.replace(':', '-')}#{cell.attempts}"
                    cell.results_dir = str(run_dir / "cells" / f"{cell.cell_id}__{lane.name.replace(':', '-')}_{cell.attempts}")
                    proc, log_fh = launch(cell, lane, sweep, run_dir, manifest,
                                          skip_pull and first_launch_done)
                    first_launch_done = True
                    lane.in_use += 1
                    lane.last_launch = time.time()
                    running.append({"cell": cell, "lane": lane, "proc": proc,
                                    "log_fh": log_fh, "log_path": run_dir / "logs" / f"{cell.cell_id}.{cell.attempts}.log"})
                    log_event(kind="launch", cell=cell.cell_id, arm=cell.arm, shard=cell.shard,
                              target=lane.name, run_id=cell.run_id, attempt=cell.attempts,
                              pid=proc.pid, task_ids=cell.task_ids, results_dir=cell.results_dir)
                    progressed = True
                    break

        # Reap finished cells.
        still = []
        for r in running:
            rc = r["proc"].poll()
            if rc is None:
                still.append(r)
                continue
            r["log_fh"].close()
            cell, lane = r["cell"], r["lane"]
            lane.in_use -= 1
            if rc != 0 and is_requeue_failure(r["log_path"]) and cell.attempts < max_attempts:
                print(f"[fanout] {cell.cell_id} on {lane.name}: GPU unavailable (rc={rc}) — requeue "
                      f"(attempt {cell.attempts}/{max_attempts})")
                log_event(kind="requeue", cell=cell.cell_id, target=lane.name, run_id=cell.run_id, rc=rc)
                cell.status = "pending"
                pending.append(cell)
            else:
                cell.status = "done" if rc == 0 else "failed"
                cell.rc = rc
                print(f"[fanout] {cell.cell_id} on {lane.name}: {cell.status} (rc={rc})")
                log_event(kind="finish", cell=cell.cell_id, target=lane.name, run_id=cell.run_id,
                          rc=rc, status=cell.status)
        running = still
        if pending or running:
            time.sleep(2)

    # Drain anything still alive after a budget break or interrupt — give each
    # wrapper's trap time to run scancel / remote_serve stop before we give up.
    for r in running:
        try:
            r["proc"].wait(timeout=120)
        except subprocess.TimeoutExpired:
            r["proc"].kill()
        r["log_fh"].close()
    signal.signal(signal.SIGINT, prev_sigint)
    return manifest


# --------------------------------------------------------------------------- #
# Reporting + aggregation
# --------------------------------------------------------------------------- #
def load_placement(manifest: Path) -> dict[str, dict]:
    """Collapse the JSONL manifest into one record per run_id (placement + status)."""
    by_run: dict[str, dict] = {}
    if not manifest.exists():
        return by_run
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = rec.get("run_id")
        if not rid:
            continue
        by_run.setdefault(rid, {}).update({k: v for k, v in rec.items() if v is not None})
    return by_run


def print_placement(manifest: Path) -> None:
    placement = load_placement(manifest)
    print("\n=== placement ledger (where each subjob ran) ===")
    cols = ("run_id", "cluster", "node", "gpu_name", "gpu_mem_mib", "gpu_mem_util", "port", "arm", "status", "rc")
    print("  ".join(c.upper() for c in cols))
    for rid, rec in sorted(placement.items()):
        print("  ".join(str(rec.get(c, "")) for c in cols))


def _cell_runs_dir(results_dir: str) -> Path | None:
    """The driver writes manifests under <results_dir>/runs/<leaf>/*/manifest.json
    (run_tb_client.sh points AB_EXPERIMENT_DIR at the cell's RESULTS dir). Return
    that runs/<leaf> dir — what run_comparison globs — or None if the cell produced
    no driver output (e.g. it failed before the run)."""
    runs = Path(results_dir) / "runs"
    if runs.is_dir():
        for leaf in sorted(runs.iterdir()):
            if leaf.is_dir() and any(leaf.glob("*/manifest.json")):
                return leaf
    return None


def collect_arm_runs(cells: list[Cell]) -> dict[str, list[dict]]:
    """Group completed cells by arm. Each entry carries the output paths an
    analysis hook needs: the driver's runs/<leaf> dir and the per-cell
    engine_metrics.jsonl, plus the cell's results_dir + run_id."""
    out: dict[str, list[dict]] = {}
    for c in cells:
        if c.status != "done":
            continue
        rd = _cell_runs_dir(c.results_dir)
        if rd is None:
            continue
        out.setdefault(c.arm, []).append({
            "arm": c.arm,
            "run_id": c.run_id,
            "results_dir": c.results_dir,
            "runs_dir": str(rd),
            "engine_metrics": str(Path(c.results_dir) / "logs" / "engine_metrics.jsonl"),
        })
    return out


def aggregate(cells: list[Cell], run_dir: Path, sweep: dict) -> None:
    """Run the optional, experiment-supplied analysis hook over the results.

    The hook is configured in the sweep TOML as `analysis = "module:function"`
    and is called as `function(arm_runs=<dict>, out_dir=<str>)`, where arm_runs
    maps each arm to the list of per-cell output records from collect_arm_runs().
    No hook configured -> just report where the cells landed (never fails the run)."""
    arm_runs = collect_arm_runs(cells)
    completed = sum(len(v) for v in arm_runs.values())
    spec = sweep.get("analysis")
    if not spec:
        print(f"\n[fanout] no analysis hook configured (set analysis = \"module:function\" in the "
              f"sweep to run one). {completed} completed cell(s); results under {run_dir / 'cells'}.")
        return
    if not completed:
        print("\n[fanout] analysis skipped (no completed cell with driver output).")
        return
    out_dir = run_dir / "analysis"
    try:
        import importlib
        mod_name, _, fn_name = str(spec).partition(":")
        if not fn_name:
            raise ValueError(f"analysis must be 'module:function', got {spec!r}")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        fn(arm_runs=arm_runs, out_dir=str(out_dir))
        print(f"\n[fanout] analysis ({spec}) written to {out_dir}")
    except (Exception, SystemExit) as exc:  # hook is best-effort — never fail the run on it
        print(f"\n[fanout] analysis degraded ({type(exc).__name__}: {exc}). "
              f"Cell results are under {run_dir / 'cells'}.")


# --------------------------------------------------------------------------- #
# Dry run + main
# --------------------------------------------------------------------------- #
def do_dry_run(cells: list[Cell], lanes: list[Lane], sweep: dict, run_dir: Path, ssh_config: str | None) -> None:
    print(f"\n=== DRY RUN: {sweep.get('name')} ===")
    print(f"run_dir: {run_dir}")
    print(f"benchmark: {sweep['benchmark']}  harness: {sweep['harness']}")
    print(f"\ncells ({len(cells)} = {len(sweep['arms'])} arms x shards):")
    for c in cells:
        print(f"  {c.cell_id:14s} arm={c.arm:8s} shard={c.shard}  tasks={c.task_ids}")
    print(f"\nlanes ({sum(l.capacity for l in lanes)} concurrent slots):")
    for l in lanes:
        util = f"util={l.gpu_mem_util}" if l.gpu_mem_util is not None else "util=whole-card"
        pack = "  packed(staggered)" if l.packed else ""
        print(f"  {l.name:18s} kind={l.kind:9s} capacity={l.capacity}  {util}{pack}  env={l.base_env}")
    if ssh_config:
        print(f"\ngenerated nscc ssh config ({ssh_config}):\n")
        print(Path(ssh_config).read_text())
    print("[fanout] dry run only — nothing launched.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="path to the sweep TOML (e.g. experiments/fanout.toml)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; launch nothing")
    ap.add_argument("--accounts", default="", help="nscc accounts: comma list, or 'all' for every ~/.ssh/nscc_*")
    ap.add_argument("--targets", default="", help="only schedule onto these target names (comma list)")
    ap.add_argument("--run-id", default="", help="override the run id (default: <name>_<stamp>)")
    args = ap.parse_args()

    sweep = load_profile(args.sweep)
    for required in ("name", "benchmark", "harness", "model_template", "serving", "arms", "shards"):
        if required not in sweep:
            raise SystemExit(f"sweep missing required key: {required}")

    run_id = args.run_id or f"{sweep['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = KIT_ROOT / "data" / "fanout" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shards_info = write_shard_apps(_abs(sweep["benchmark"]), run_dir, int(sweep["shards"]))
    cells = build_cells(list(sweep["arms"]), shards_info)
    lanes, ssh_config = build_lanes(sweep, args, run_dir)
    if not lanes:
        raise SystemExit("[fanout] no enabled targets (check `enabled` flags / --targets).")

    if args.dry_run:
        do_dry_run(cells, lanes, sweep, run_dir, ssh_config)
        return 0

    print(f"[fanout] run_id={run_id}  cells={len(cells)}  slots={sum(l.capacity for l in lanes)}")
    manifest = schedule(cells, lanes, sweep, run_dir)
    print_placement(manifest)
    aggregate(cells, run_dir, sweep)

    failed = [c.cell_id for c in cells if c.status != "done"]
    print(f"\n[fanout] done: {len(cells) - len(failed)}/{len(cells)} cells succeeded.")
    if failed:
        print(f"[fanout] not completed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
