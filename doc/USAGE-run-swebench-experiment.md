# USAGE — 跑一组 SWE-bench 实验 (NSCC singularity + sglang)

> 怎么配一组实验、要改哪些东西、最终 qsub 命令长什么样。
> 背景/排错见 [RUNBOOK-nscc-singularity-swebench.md](RUNBOOK-nscc-singularity-swebench.md)
> 和 [HANDOFF-nscc-singularity-2026-05-20.md](HANDOFF-nscc-singularity-2026-05-20.md)。
> 仓库 `/home/users/ntu/n2505289/Agent_Bench`，分支 `feat/nscc`。

---

## 0. 一句话模型

一次实验 = **APP**(跑哪些题) × **HARNESS**(哪个模型 + agent 配置) × **BACKEND**(sglang serving 参数)
三个 JSON，再加一组 PBS `-v` 覆盖项，用 `qsub` 提交
[run_benchmark_singularity.pbs](../scripts/execution_pbs/run_benchmark_singularity.pbs)。

PBS 内部按 stage 跑：装 swebench + **物化 DATASET** → pull 题目 sif → 起 sglang →
查 OC sif → 起 OTel → 调 driver 跑 sweep → cleanup。

---

## 1. 你要决定 / 改动的 5 件事

| # | 维度 | 改什么 | 怎么传 |
|---|---|---|---|
| 1 | 跑哪些题 | APP json (`instances` + `repo_mapping`) | `-v APP=...` |
| 2 | 用哪个模型 | HARNESS json (`model.name`) **＋** 模型 snapshot 目录 | `-v HARNESS_CONFIG=...,SGLANG_MODEL_DIR=...` |
| 3 | serving 参数 | BACKEND json ＋ `SGLANG_*` 环境变量 | `-v BACKEND_CONFIG=...,SGLANG_TP=...,SGLANG_TOOL_CALL_PARSER=...` |
| 4 | 同时起几个 task | `EVAL_MAX_WORKERS` ＋ qsub 资源 | `-v EVAL_MAX_WORKERS=N` ＋ `-l select=...` |
| 5 | OpenClaw runtime | OC sif | `-v OC_RUNTIME_SIF=...` |

> ⚠️ 第 2 条的两处路径**必须一致**：sglang 用 `SGLANG_MODEL_DIR` 起服务，注册的
> served-model-name 默认就是这个路径；OC 按 HARNESS 的 `model.name` 去请求模型。
> 两者不一致 → OC 请求的 model 名 sglang 不认。

---

## 2. 逐项怎么改

### 2.1 换题集 (APP + DATASET)

**APP** 决定跑哪些 instance，放 `config/application/swebench/<name>.json`，含 `instances`
列表和 `repo_mapping`。用脚本生成最省事：

```bash
# 从已 pull 的 sif 全量生成
scripts/make_swebench_config.py --name my_set --from-sifs /scratch/users/ntu/n2505289/swebench
# 或显式给 id
scripts/make_swebench_config.py --name my_set --instance-ids astropy__astropy-12907 django__django-10097
# 或取 verified_full 的一段
scripts/make_swebench_config.py --name my_set --slice 0 20
```

**DATASET**（带 `FAIL_TO_PASS`/`PASS_TO_PASS`/`problem_statement` 的题目 metadata）**不用手写**：
PBS 的 STAGE 0 会从 APP 的 ids 自动从 HF `princeton-nlp/SWE-bench_Verified` 物化并写到
`$DATASET`（幂等增量——已覆盖全部 id 就 skip）。你只需：

- 把 `-v DATASET=` 指向一个 **`/scratch` 下的路径**，例如 `/scratch/users/ntu/n2505289/swebench/my_set.json`。
- ⚠️ **不要指 home NFS 上一个已存在的文件**——compute 节点重写它会 `EPERM`（见 memory）。scratch 安全。

题目 **sif** 要么已在 `$SIF_DIR`（`/scratch/.../swebench/<id>.sif`），要么让 `STAGE_PULL=1`
去 pull。已存在就 `-v STAGE_PULL=0` 跳过省时间。

### 2.2 换模型 (HARNESS + SGLANG_MODEL_DIR)

1. 克隆一份现成 harness（如 [openclaw_qwen3_14b_sglang.json](../config/harness/openclaw_qwen3_14b_sglang.json)），
   只改 `model.name` 为新模型 snapshot 的**绝对路径**。其余字段（gateway/otel/plugins/agents）照抄。
2. `-v SGLANG_MODEL_DIR=` 设成**同一个 snapshot 路径**。
3. 现成的：14B = `openclaw_qwen3_14b_sglang.json`；35B = `openclaw_qwen36_35b_sglang.json`；
   9B = [openclaw_qwen35_9b_sglang.json](../config/harness/openclaw_qwen35_9b_sglang.json)。
   本机模型都在 `/scratch/users/ntu/n2505289/model/hub/models--*/snapshots/<hash>`。

### 2.3 serving 参数 (BACKEND + SGLANG_*)

- 单卡用 [config/backend/sglang/tp1.json](../config/backend/sglang/tp1.json)，4 卡用 `tp4.json`。
- 真正起 sglang 用的是 `SGLANG_*` 环境变量（backend json 主要进 manifest 记录）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_TP` | 1 | tensor parallel = 用几张卡（要和 `-l ngpus` 一致） |
| `SGLANG_CONTEXT_LENGTH` | 32768 | ⚠️ **no-op**：serve 脚本没接 `--context-length`，sglang 永远用模型原生 max（9B = 262144） |
| `SGLANG_TOOL_CALL_PARSER` | `qwen` | 工具调用解析器（14B 用 `qwen25`，35B 用 `qwen3_coder`） |
| `SGLANG_REASONING_PARSER` | `qwen3` | reasoning 解析器 |
| `SGLANG_MEM_FRACTION_STATIC` | 0.85 | 显存静态占比 |
| `SGLANG_ENABLE_HIERARCHICAL_CACHE` | 1 | host-RAM KV offload。**几乎都要设 `0`**：35B 和混合线性注意力模型（如 Qwen3.5-9B，会开 KV+Mamba 双份各 60GB host）否则 init 时 host OOM SIGKILL（见 memory / 第 5 节） |

### 2.4 同时起几个 task（并发）

- `-v EVAL_MAX_WORKERS=N`：driver 用 N 个线程并发跑 N 个 run（各自独立 OC gateway + task 实例，
  共用同一个 sglang）。**默认 1 = 串行**。
- gateway 端口段 18800–18899 stride 4 → 上限 ~25 并发；warmup 串行化（避免 npm 插件竞争），
  agent loop 全并发。这些修复已在 [app.py](../applications/swebench/openclaw/app.py)。
- 单卡时 sglang 吞吐是瓶颈，不是 GPU 数；N 调大主要吃 **CPU/内存**，按 N 调 `-l select`。

### 2.5 OC runtime sif

`-v OC_RUNTIME_SIF=/scratch/users/ntu/n2505289/sif_image/oc_runtime_new.sif`。
改了 OC patch 后用 [build_oc_sif.sh](../scripts/execution_pbs/build_oc_sif.sh) 在登录节点重建。

### 2.6 跳过 stage

`-v STAGE_X=0`：`STAGE_SWEBENCH STAGE_PULL STAGE_SGLANG STAGE_OC_CHECK STAGE_OTEL STAGE_RUN STAGE_CLEANUP`。
sif 已 pull → `STAGE_PULL=0`；只想测 sglang 半段 → `STAGE_OC_CHECK=0,STAGE_OTEL=0,STAGE_RUN=0`。

---

## 3. 最终命令行

### 3.1 模板

```bash
cd /home/users/ntu/n2505289/Agent_Bench
KIT=/home/users/ntu/n2505289/Agent_Bench
SCR=/scratch/users/ntu/n2505289

qsub -q normal -l select=1:ncpus=16:mem=128gb:ngpus=1 -l walltime=04:00:00 \
     -o "$KIT/<name>_$(date +%s).log" -j oe \
     -v KIT_ROOT=$KIT,\
APP=$KIT/config/application/swebench/<APP>.json,\
HARNESS_CONFIG=$KIT/config/harness/<HARNESS>.json,\
BACKEND_CONFIG=$KIT/config/backend/sglang/tp1.json,\
DATASET=$SCR/swebench/<DATASET>.json,\
OC_RUNTIME_SIF=$SCR/sif_image/oc_runtime_new.sif,\
SGLANG_MODEL_DIR=$SCR/model/hub/models--<ORG>--<MODEL>/snapshots/<HASH>,\
EVAL_MAX_WORKERS=4,STAGE_PULL=0 \
     scripts/execution_pbs/run_benchmark_singularity.pbs
```

> `-v` 里所有路径必须是**完全展开的绝对路径**（`-v` 不会展开 `$KIT_ROOT`）。
> `qsub` **不要带 `-P`**（默认 project）；单卡自动进 g1 队列，4 卡进 g3。

### 3.2 实测例子 — Qwen3.5-9B × phase1 × 4 并发（本次提交，job 14599106）

```bash
cd /home/users/ntu/n2505289/Agent_Bench
KIT=/home/users/ntu/n2505289/Agent_Bench
SCR=/scratch/users/ntu/n2505289

qsub -q normal -l select=1:ncpus=16:mem=128gb:ngpus=1 -l walltime=04:00:00 \
     -o "$KIT/phase1_9b_concurrent_$(date +%s).log" -j oe \
     -v KIT_ROOT=$KIT,\
APP=$KIT/config/application/swebench/phase1.json,\
HARNESS_CONFIG=$KIT/config/harness/openclaw_qwen35_9b_sglang.json,\
BACKEND_CONFIG=$KIT/config/backend/sglang/tp1.json,\
DATASET=$SCR/swebench/verified_phase1.json,\
OC_RUNTIME_SIF=$SCR/sif_image/oc_runtime_new.sif,\
SGLANG_MODEL_DIR=$SCR/model/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a,\
SGLANG_ENABLE_HIERARCHICAL_CACHE=0,\
EVAL_MAX_WORKERS=4,STAGE_PULL=0 \
     scripts/execution_pbs/run_benchmark_singularity.pbs
```

> `SGLANG_ENABLE_HIERARCHICAL_CACHE=0` 是 9B **必须**的：它是混合线性注意力模型，hicache 开着会
> 在 host 上各开 60GB 的 KV + Mamba 缓存（~120GB），超 cgroup 上限被 OOM killer SIGKILL。
> 第一次没加这个 flag 的 job（14599106）就是这么挂的。

### 3.3 35B 4 卡变体（改这几处）

```
-l select=1:ncpus=16:mem=192gb:ngpus=4
HARNESS_CONFIG=.../openclaw_qwen36_35b_sglang.json
BACKEND_CONFIG=.../config/backend/sglang/tp4.json
SGLANG_MODEL_DIR=.../models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
SGLANG_TP=4, SGLANG_TOOL_CALL_PARSER=qwen3_coder
```

---

## 4. 监控 + 产物

```bash
qstat -u $USER                                    # Q→R；卡不够会停在 Q
qstat -f <jobid> | grep exec_host                 # 跑在哪个节点
SWEEP=$(ls -td data/runs/<sweep_name>/*/ | head -1)
tail -f $SWEEP/logs/run.log                        # driver 进度（并发时多条 run 交错）
grep -v "GET /metrics" $SWEEP/logs/sglang.log | tail
```

里程碑：sglang `server is fired up` → run.log 出现 N 条 `ready: oc=` → `Running OpenClaw agent`
→ `Sweep complete`。每个 `data/runs/<sweep>/<stamp>/<run_id>/` 下产出
`verdict.json` / `candidate.patch` / `steps.jsonl` / `spans.jsonl` / `conversation/` / `util.jsonl`。

PBS 的 `-o` stdout 只在 **job 结束后**才落盘；运行中看上面的 per-sweep 日志。

---

## 5. 常见坑（详见 RUNBOOK 第 7 节）

- `qsub -P <code>` 报 project not found → **不要带 `-P`**。
- DATASET 指 home 上预存文件 → compute 重写 `EPERM`；放 `/scratch`。
- 大模型 sglang host-RAM OOM → `-v SGLANG_ENABLE_HIERARCHICAL_CACHE=0`。
- `-v` 路径没展开（写了 `$KIT_ROOT`）→ 用绝对路径。
- 工具调用解析不出来 → 调 `SGLANG_TOOL_CALL_PARSER`（`qwen` / `qwen25` / `qwen3_coder`）。
- sglang SIGBUS → 确认 `module load singularity/4.3.1`（脚本里已固定）。
