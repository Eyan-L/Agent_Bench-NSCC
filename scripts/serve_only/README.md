# Decoupled serving — GPU serves vLLM, a CPU node runs the agent

Instead of co-locating serving + agent + scoring on one GPU node, this splits
them:

- **GPU host** (a bare-metal box or scheduler node) runs **only** the vLLM
  server, bound on `0.0.0.0:8000`.
- **CPU node** runs the agent + scoring (docker) and sends API requests to that
  server.

The GPU is freed the moment generation finishes; the CPU node owns the slow,
serial agent/scoring work. The existing co-located Slurm/PBS paths are untouched.

This pairs with the `[serving]` model in the run config (see the wiki "Run
experiment" instructions): the GPU side serves a `backend: vllm` profile and the
client side points at it with a `backend: existing_endpoint` profile.

## Why this needs almost no code

- `serve_vllm.sh` (via the Python launcher) already binds `0.0.0.0`.
- TB's `tb_react` condition runs the ReAct loop in the **host driver process** and
  calls `model_config.endpoint` directly (`agent_bench/benchmarks/conditions/_llm_client.py`),
  so it just needs the endpoint to point at the tunnel / GPU host.
- `backend = existing_endpoint` makes the driver skip launching any local server
  and resolve the endpoint from the serving profile / `AB_SERVING_BASE_URL`.

## Pieces

| File | Runs on | Does |
|---|---|---|
| `scripts/serve_only/serve_vllm.sh` | GPU host | wraps `serve.sh`, pins the decoupled serving profile + vLLM venv, binds `0.0.0.0:8000`; **supervises vllm in its own process group for clean Ctrl-C/SIGTERM shutdown** |
| `scripts/serve_only/remote_serve.sh` | CPU node | **control plane**: `start`/`stop`/`restart`/`status`/`logs` the GPU server over SSH — no manual shell on the GPU host |
| `scripts/serve_only/tunnel.sh` | CPU node | `ssh -L 0.0.0.0:8000 -> GPU:8000` (only for non-routable hosts) |
| `scripts/execution_client/run_tb_client.sh` | CPU node | materialize TB tasks, pull task image, tailer, run driver |
| `profiles/serving/vllm/decoupled.yaml` | GPU host | `backend: vllm`, venv-launched server |
| `profiles/serving/vllm/existing_endpoint.yaml` | CPU node | `backend: existing_endpoint`, endpoint `127.0.0.1:8000` |

## One-time prerequisites

**CPU node** `~/.ssh/config` — add the GPU host (use a `ProxyJump` only if it sits
behind a bastion/workstation; drop that line for a directly reachable host):
```
Host <gpu-host>
    HostName <gpu-host.example.com>
    User <user>
    IdentityFile ~/.ssh/<your-key>
    ProxyJump <jump-host>
```

**CPU node** driver env (CPU-only; no vLLM/torch needed) — the repo uses `uv`:
```
cd /path/to/Agent_Bench
uv sync                       # creates .venv from pyproject.toml + uv.lock
source .venv/bin/activate     # so `python3` in run_tb_client.sh = the uv venv
```

**GPU host** — a vLLM venv present (`VLLM_VENV`), model weights reachable, and the
model-config `name` resolvable by vLLM there (see "Model name" below).

## Drive the GPU server from the CPU node (no manual GPU-host shell)

`remote_serve.sh` runs on the CPU node and operates `serve_vllm.sh` on the GPU
host over SSH, so you never open a shell on the GPU host by hand. It is the
*control* plane; `tunnel.sh` is still the *data* plane.

```bash
# start (detached on the GPU host; waits until /v1/models answers)
VLLM_VENV='$HOME/vllm/.venv' TP=2 CUDA_VISIBLE_DEVICES=0,1 \
    MODEL='$HOME/scratch/models/Qwen3.6-35B-A3B-FP8' \
    MODEL_CONFIG=profiles/models/qwen36_35b_a3b_a6000.json \
    bash scripts/serve_only/remote_serve.sh start

bash scripts/serve_only/tunnel.sh &                 # forward :8000
MODEL_CONFIG=$PWD/profiles/models/qwen36_35b_a3b_a6000.json \
HARNESS_CONFIG=$PWD/profiles/harnesses/hermes_terminal_bench.yaml \
    bash scripts/execution_client/run_tb_client.sh

bash scripts/serve_only/remote_serve.sh stop        # graceful; frees the GPU
bash scripts/serve_only/remote_serve.sh status      # supervisor + endpoint health
bash scripts/serve_only/remote_serve.sh logs        # tail the remote vllm.log
```

Notes:
- Quote any passthrough value that references the **GPU host's** environment
  (`MODEL='$HOME/...'`) — values are evaluated on the GPU host, not on the CPU node.
- `REMOTE` (your GPU host — required) and `REMOTE_KIT_ROOT` (default `~/Agent_Bench`)
  select the host and checkout path. `PORT` defaults to 8000.

### Clean shutdown (why Ctrl-C used to leak the server)

`serve_vllm.sh` no longer `exec`s vllm. It runs `serve.sh` in its **own process
group** (`set -m`) and traps `INT`/`TERM`: on a signal it SIGTERMs the whole
group — vllm **and** its V1 EngineCore/TP-worker subprocesses — then escalates to
SIGKILL after `SERVE_GRACE` (default 20s). Previously `exec vllm serve` replaced
the launcher (no trap left) and the engine's child processes survived a plain
Ctrl-C/`kill`, which is why only `kill -9` worked. The PGID is written to
`logs/serve_only/serve_vllm.pgid`; `remote_serve.sh stop` targets it. Manual stop
on the GPU host is now just Ctrl-C, or `kill -TERM -$(cat logs/serve_only/serve_vllm.pgid)`.

## Model name (the one gotcha)

The agent sends `model_config.name` as the request `model`, and vLLM serves under
the name it was launched with. **Use the same `MODEL_CONFIG` on both sides.** If
the weights live at a different path than the config `name`, either:
- set `MODEL=<weights path>` on the GPU side AND keep the config `name` as the id
  the agent requests (then the served id == `name` only if you also pass
  `--served-model-name`), or — simpler —
- pick/author a `profiles/models/*.json` whose `name` is the path vLLM loads on
  the GPU host, and pass it as `MODEL_CONFIG` to **both** scripts.

`run_tb_client.sh` preflights `/v1/models` against the config `name` and aborts
with a clear message on mismatch.

## Networking: tunnel vs direct route vs tailscale

- **cloudflared** in your SSH config is the **SSH control plane only** — it does
  not carry model API traffic. Keep it for SSH; don't route inference through it
  (every request would round-trip the Cloudflare edge).
- **Directly-routed GPU hosts** — when the CPU node can reach the GPU host's IP on
  the vLLM port directly (~2 ms), skip the tunnel:
  ```bash
  ENDPOINT_HOST=<gpu-host-ip> bash scripts/execution_client/run_tb_client.sh
  ```
- A GPU host reachable only through a **ProxyJump** (bastion/workstation) is **not**
  directly routable → use `tunnel.sh`.
- **tailscale** is only worth installing if you later want one flat mesh across
  many clusters. It beats cloudflared for the data plane (peer-to-peer, no edge
  hop) but needs `tailscaled` (root/TUN, often blocked on HPC). Not needed for the
  GPU hosts here.

## Scheduler-allocated GPU nodes (PBS, Slurm)

`remote_serve.sh` assumes a fixed, directly-SSHable GPU host. On scheduler
clusters the GPU node must be **allocated through a scheduler** and is reached
**through a login node** — a 2-hop tunnel:

```
CPU node ──ssh──► login node ──sched──► compute node (vLLM on 0.0.0.0:8000)
        tunnel.sh: ssh -L 0.0.0.0:8000:<compute>:8000 <login>   (FORWARD_HOST=<compute>)
```

Two one-command wrappers stand up the server, run the TB client, and release the
allocation on exit. Both serve `$HOME/scratch/models/Qwen3.6-35B-A3B-FP8` (FP8,
~35B-A3B MoE — needs **2×40 GB or 1×80 GB**) from the native vLLM venv, sizing
TP/`CUDA_VISIBLE_DEVICES` to the cards.

The served model id == the weights path, and `$HOME` differs per cluster, so the
wrappers resolve the remote `$HOME` at runtime: the **server** gets
`MODEL=<resolved path>` (params from the committed template
`profiles/models/qwen36_35b_a3b_fp8.json`), and the **client** gets a name-patched
copy written to `data/serve_only/`. No per-cluster model JSON to hand-edit.

### Slurm `srun` (ccds-ai style)
```bash
APP=profiles/benchmarks/terminal_bench/smoke1.json \
HARNESS_CONFIG=profiles/harnesses/tb_react_terminal_bench.yaml \
    bash scripts/serve_only/run_ccds.sh
```
It picks genuinely-free cards using `gpu-status`'s **own** logic (idle/mix nodes →
`srun nvidia-smi` VRAM probe, free = ≤10 MiB used), **pins**
`CUDA_VISIBLE_DEVICES` to those physical ids (not Slurm's choice), allocates with
`srun -p <PART> -w <NODE> --gres=gpu:<n>`, serves, tunnels, runs, then `scancel`s.
Force the pick with `NODE=… PARTITION=… CUDA_VISIBLE_DEVICES=…` or the count with
`GPUS=1|2`. PCIe nodes need `--disable-custom-all-reduce` — already in
`decoupled.yaml`.

### Batch PBS `qsub` (auto-discover)
```bash
# one-time: add the PBS login Host entry + copy the key (see below)
NGPUS=2 bash scripts/serve_only/run_nscc.sh
```
It `qsub`s `serve_nscc.pbs` (project **`-P $PROJECT`**, `select=1:ngpus=$NGPUS`),
polls `qstat` until the job is `R`, reads the exec node from the job's
`logs/serve_only/<jobid>.node` file, tunnels, runs the client, then `qdel`s on
exit. Default `NGPUS=2` (40 GB A100 → 2×40 GB); set `NGPUS=1` on an 80 GB/H100
node. PBS cgroup-isolates the allocation, so the job uses `CUDA_VISIBLE_DEVICES=0..N-1`.

**PBS prerequisites (CPU node):** add a `Host` entry for the login node. nscc now
**requires password auth** (keys disabled), so prefer connection multiplexing so
you only type the password once:
```
Host nscc
    HostName <login-host>
    User <your-account>
    PreferredAuthentications password
    PubkeyAuthentication no
    ControlMaster auto
    ControlPath ~/.ssh/cm/%C
    ControlPersist 4h
```
(`mkdir -p ~/.ssh/cm`. For a key host instead, use `IdentityFile ~/.ssh/<your-key>`.)
Try a direct `ssh nscc hostname` first; add a `ProxyJump` only if that fails. For
the multi-account fan-out the orchestrator generates this block per account — see
**Multiple PBS accounts** for the password handling. Also
ensure the vLLM venv exists on the login/compute side (`NSCC_VLLM_VENV`, default
`~/vllm/.venv`) and the weights are staged at `~/scratch/models/Qwen3.6-35B-A3B-FP8`.

### Single-account PBS host (e.g. an H200 in `~/.ssh/config`)
`run_nscc.sh` is the generic PBS wrapper — point it at any single-account PBS host
already in `~/.ssh/config`. A queue that needs a project, queue name, `ncpus`
count, and a walltime cap:
```bash
REMOTE=hpcc PROJECT=<your-project> QUEUE=gpu_free NCPUS=4 NGPUS=1 \
    WALLTIME=00:30:00 bash scripts/serve_only/run_nscc.sh
```
This expands to `qsub -P <project> -q gpu_free -l select=1:ncpus=4:ngpus=1 -l
walltime=00:30:00`. `QUEUE`/`NCPUS` are appended only when set, so the plain nscc
invocation above is unchanged. Logs + the placement manifest are tagged by cluster
(`[hpcc]`, from `REMOTE`'s first token).

**Two H200 gotchas (both handled):** the vLLM venv must pin **`fastapi==0.136.3`
`starlette==1.2.1`** — newer starlette makes the prometheus instrumentator 500
*every* request (incl. `/v1/models`), which reads as "server up but every call
fails". And `decoupled.yaml` serves under **`FLASH_ATTN`**, so flashinfer's Hopper
JIT kernels never compile — sidestepping the CUDA-toolkit wall on hosts whose nvcc
predates the CCCL `<cuda/ptx>` header.

### Running many experiments in parallel

Both wrappers (`run_ccds.sh`, `run_nscc.sh`) **auto-allocate a free local port**
per run when `PORT` is unset, so you can launch several at once — each gets its own
SSH tunnel (`0.0.0.0:<port>`), compute-node served port, client endpoint, and
`data/tb_client_*_p<port>` results dir. No more `bind 0.0.0.0:8000: Address already
in use`. The client model-config under `data/serve_only/` is keyed by model and
written atomically, so same-model runs safely share one file and different models
don't clobber each other.

```bash
# fan out — each picks its own port; send output to per-run logs
mkdir -p logs/runs
APP=profiles/benchmarks/terminal_bench/smoke1.json \
HARNESS_CONFIG=profiles/harnesses/hermes_terminal_bench.yaml \
    bash scripts/serve_only/run_ccds.sh >logs/runs/ccds.log 2>&1 &
NGPUS=2 APP=profiles/benchmarks/terminal_bench/smoke1.json \
HARNESS_CONFIG=profiles/harnesses/hermes_terminal_bench.yaml \
    bash scripts/serve_only/run_nscc.sh >logs/runs/nscc.log 2>&1 &
wait                 # cleanup traps (qdel/scancel/kill tunnel) fire on Ctrl-C
```

The wrappers `exec </dev/null`, so backgrounding with `&` is safe — they won't
read the terminal and get suspended (`[N]+ Stopped` = the classic `SIGTTIN` a
backgrounded job earns when an `ssh` inside it tries to read the tty). Redirect
**output** to logs too, or the concurrent runs interleave on your terminal.
For an attachable view, run each in its own `tmux` window instead of `&`.

Pin `PORT=<n>` explicitly only if you need a fixed, known endpoint (e.g. to
`curl` it by hand) — then it's your job to keep concurrent runs on distinct
ports. Each run logs its chosen port: `[ccds] using port 44027 (...)`.

## Fan out across ALL GPUs in one command (`orchestration/fanout`)

The hand-written `&` fan-out above works but you compose every line yourself. The
orchestrator expands a TOML sweep into cells (`arm × shard`), schedules them across
the enabled targets respecting per-target concurrency + GPU packing, records
**where each subjob landed and the GPU it ran on**, and optionally runs an analysis
hook at the end. It is a thin, experiment-agnostic layer over the same three
wrappers — `run_ccds.sh` (Slurm), `run_nscc.sh` (PBS), `run_baremetal.sh` (fixed
hosts) — so all the port/cleanup/discovery logic above still applies per cell.

```bash
# 1) see the plan (cells, target lanes, computed packing util, generated ssh config)
python3 -m orchestration.fanout.fanout experiments/fanout.toml --dry-run

# 2) run it (launches nothing until you drop --dry-run)
python3 -m orchestration.fanout.fanout experiments/fanout.toml

# narrow scope while iterating
python3 -m orchestration.fanout.fanout experiments/fanout.toml --targets gpu_host
python3 -m orchestration.fanout.fanout experiments/fanout.toml --accounts <userA>,<userB>
```

Copy `experiments/fanout.example.toml` to `experiments/fanout.toml` (gitignored)
and fill in your accounts/projects/hosts. Profiles are repo-relative and the
benchmark's `instances` get sharded. `--accounts`/`--targets` override the file;
`--dry-run` writes the per-shard `APP` jsons + scoped ssh config and prints the
plan without launching.

### Arms and A/B variants

Arms are just variant **labels**. A single-arm sweep (`arms = ["default"]`) is a
plain sharded fan-out. For an A/B, add more arms and give each an environment
overlay in `[arm_env.<arm>]` — the orchestrator merges those vars into that arm's
cell environment (and sets `ARM=<arm>`) and otherwise knows nothing about what
they mean. The harness on the GPU/client side reads them. This is how an
experiment (e.g. a feature toggle on vs off) layers on top of the generic fan-out
without changing the orchestrator.

### Analysis hook

Set `analysis = "module:function"` in the sweep to post-process results. It is
called as `function(arm_runs=<dict>, out_dir=<str>)`, where `arm_runs` maps each
arm to a list of per-cell output records (`runs_dir`, `engine_metrics`,
`results_dir`, `run_id`). Unset = no analysis; cells just land under
`data/fanout/<run>/cells/`.

### GPU packing — several servers on one card

vLLM's `--gpu-memory-utilization` is set per cell via `GPU_MEM_UTIL` (the
orchestrator derives it from `gpu_mem_util`). `serve_vllm.sh` forwards it to the
Python launcher as `VLLM_GPU_UTIL`, which sets `--gpu-memory-utilization`.
Each server gets its own port and its own PORT-scoped supervisor state under
`logs/serve_only/p<PORT>/`, so co-resident servers don't clobber each other.

**Whether a model 2-packs is a memory question.** `gpu_memory_utilization` is a
fraction of the card's TOTAL memory, and vLLM needs that budget to cover weights
**+ KV cache + cudagraph/compile overhead**:

- A **35 GB FP8 model + cudagraph overhead needs ~0.9 of a 150 GB H200**; below
  that there is no room for KV (it fails with *"No available memory for the cache
  blocks"* even as a single server at 0.45). So such a card runs **1 server/card**.
- The same model **fills a 48 GB A6000** → `servers_per_gpu=1`; use
  `cuda_devices=[0,1,2,3]` to spread one server onto each of the 4 cards.
- **Packing (`servers_per_gpu>1` + lower `gpu_mem_util`) pays off for SMALLER
  models** relative to the card. When you pack, pick a util that fits
  weights+KV+overhead within `total/N` (with headroom); co-resident servers boot
  concurrently and vLLM's startup free-memory check can race, so the orchestrator
  staggers each next pack by `limits.boot_stagger_sec`.
- `cuda_devices` pins one **lane per physical card** so cells spread across a
  multi-GPU host (the bare-metal free-VRAM auto-pick can't coordinate simultaneous
  launches); `servers_per_gpu` then packs *within* each pinned card.

### Multiple PBS accounts

Credentials live at `~/.ssh/nscc_<user>` (the user is the suffix). The
orchestrator generates a **scoped ssh config** (one `Host nscc-<user>` block per
selected account) under the run dir and drives `qsub` per account via `ssh -F` —
your `~/.ssh/config` is never touched. Default is the `accounts` list in the TOML;
`--accounts all` fans across every discovered account, `--accounts a,b` picks a
subset. An account is "discovered" from either a password file
(`~/.ssh/nscc_<user>.pass`) or a legacy private key (`~/.ssh/nscc_<user>`).

**nscc now requires password auth (keys disabled).** Every generated block sets
`ControlMaster auto` + a shared `ControlPath`, so the password authenticates
**once per account** and every later connection (qstat poll, qdel, the tunnel)
multiplexes over that one master socket — no re-auth, and far fewer login-node
connections (which also dodges the MaxStartups "Connection reset" dogpile). Two
ways to supply the password, auto-detected at runtime:

- **sshpass (hands-off):** put each password in `~/.ssh/nscc_<user>.pass`
  (`chmod 600`) and `sudo apt-get install sshpass`. The wrappers feed it via
  `sshpass -f`; the single fan-out command then runs unattended.
- **Pre-opened master (no plaintext, no package):** before launching, run
  `bash scripts/serve_only/nscc_login.sh` and type each password once. It opens a
  persistent `ControlMaster` per account (same `ControlPath` the orchestrator
  uses); the fan-out reuses them with zero re-auth. `ACTION=check`/`stop` inspect
  or tear the masters down. The masters persist `4h` after last use.

Legacy key accounts (a `~/.ssh/nscc_<user>` key, no `.pass`) keep `IdentityFile`
+ `BatchMode` and are unaffected.

### Where everything lands

```
data/fanout/<run_id>/
  apps/shard<k>.json        # per-shard benchmark profile (instances narrowed)
  ssh_config                # generated scoped PBS config (multi-account)
  cells/<cell>__<lane>_<n>/ # each cell's RESULTS (driver output + logs/engine_metrics.jsonl)
  logs/<cell>.<attempt>.log # each wrapper's stdout/stderr
  manifest.jsonl            # the placement ledger
  analysis/                 # analysis-hook output (if configured; best-effort)
```

`manifest.jsonl` is the **"where did each subjob go + what GPU"** record: every
wrapper appends a line at allocation (`status=serving`, with `cluster/account`,
`node`, **`gpu_name`** + **`gpu_mem_mib`**, `gpu_mem_util`, `port`, `jobid`, `arm`)
and again on exit (`status`, `rc`). At the end the orchestrator prints a placement
table collapsing it by run. GPU **type** capture is wired through all three
backends: `run_ccds.sh` queries `nvidia-smi … name`; `serve_nscc.pbs` publishes a
`<jobid>.gpu` file; `run_baremetal.sh` probes the card directly.

Backstops: cells whose GPU was unavailable (`no free GPUs`, queued PBS job timing
out) are **requeued** up to `max_attempts_per_cell`; a `time_budget_min` wall-clock
cap tears running cells down (their own `scancel`/`qdel`/`remote_serve stop` traps
fire) so nothing leaks.
