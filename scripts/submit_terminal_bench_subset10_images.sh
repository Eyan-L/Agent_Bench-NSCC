#!/usr/bin/env bash
set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER}"

TASKS_ROOT="$SCRATCH/datasets/terminal-bench-2-1-7131e4375048/tasks"
SIF_DIR="$SCRATCH/task-images"
JOBDIR="$SCRATCH/agent-bench/tb21-subset10-image-jobs"

mkdir -p "$SIF_DIR" "$JOBDIR"

TASKS=(
    llm-inference-batching-scheduler
    dna-assembly
    torch-tensor-parallelism
    multi-source-data-merger
    financial-document-processor
    large-scale-text-editing
    fix-code-vulnerability
    cobol-modernization
    kv-store-grpc
    configure-git-webserver
)

i=0
ready=0
submitted=0

for TASK in "${TASKS[@]}"; do

    ROW="$(
        AB_BENCHMARK_TASKS_ROOT="$TASKS_ROOT" \
        python3 - "$TASK" "$TASKS_ROOT" "$SIF_DIR" <<'PY'
import sys
from pathlib import Path

from agent_bench.benchmarks.terminal_bench import loader

task_id = sys.argv[1]
tasks_root = Path(sys.argv[2])
sif_dir = Path(sys.argv[3])

instance = loader.load_task(
    task_id,
    tasks_root=tasks_root,
)

image = instance["docker_image"]

name = (
    image
    .replace("/", "__")
    .replace(":", "__")
    + ".sif"
)

print(f"{image}\t{sif_dir / name}")
PY
    )"

    IFS=$'\t' read -r IMAGE SIF <<< "$ROW"

    i=$((i + 1))

    echo "[$i/10] $TASK"
    echo "  image: $IMAGE"
    echo "  sif:   $SIF"

    if [[ -s "$SIF" ]]; then
        echo "  READY"
        ready=$((ready + 1))
        echo
        continue
    fi

    JOBFILE="$JOBDIR/tb21-subset10-${i}.pbs"

    cat > "$JOBFILE" <<EOF
#!/usr/bin/env bash
#PBS -N tb21-s10-${i}
#PBS -P personal-n2602695
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=06:00:00
#PBS -q normal
#PBS -j oe

set -euo pipefail

module load apptainer/1.5.0

export SCRATCH=/scratch/users/ntu/\$USER

export APPTAINER_CACHEDIR="\$SCRATCH/.sing_cache"
export SINGULARITY_CACHEDIR="\$APPTAINER_CACHEDIR"

JOB_TMP="\$SCRATCH/.sing_tmp/\${PBS_JOBID//[^A-Za-z0-9._-]/_}"

export APPTAINER_TMPDIR="\$JOB_TMP"
export SINGULARITY_TMPDIR="\$JOB_TMP"

mkdir -p \
    "\$APPTAINER_CACHEDIR" \
    "\$JOB_TMP" \
    "$(dirname "$SIF")"

cleanup() {
    rm -rf "\$JOB_TMP"
}
trap cleanup EXIT

IMAGE='$IMAGE'
SIF='$SIF'
PARTIAL="\${SIF}.partial"

echo "task:   $TASK"
echo "image:  \$IMAGE"
echo "target: \$SIF"

if [[ -s "\$SIF" ]]; then
    singularity inspect "\$SIF" >/dev/null
    echo "already ready"
    exit 0
fi

rm -f "\$PARTIAL"

singularity pull \
    "\$PARTIAL" \
    "docker://\$IMAGE"

singularity inspect "\$PARTIAL" >/dev/null

mv "\$PARTIAL" "\$SIF"

echo "READY"
ls -lh "\$SIF"
EOF

    qsub "$JOBFILE"

    submitted=$((submitted + 1))
    echo "  SUBMITTED"
    echo
done

echo "subset10 total: 10"
echo "already ready:  $ready"
echo "submitted:      $submitted"
