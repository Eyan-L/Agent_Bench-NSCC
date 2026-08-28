#!/usr/bin/env bash
set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER}"

TASKS_ROOT="${1:-${AB_BENCHMARK_TASKS_ROOT:-$SCRATCH/datasets/terminal-bench-2-1-7131e4375048/tasks}}"

if [[ ! -d "$TASKS_ROOT" ]]; then
    echo "Terminal-Bench tasks root not found:" >&2
    echo "  $TASKS_ROOT" >&2
    exit 2
fi

export AB_BENCHMARK_TASKS_ROOT="$TASKS_ROOT"
export AB_DEPLOYMENT_TASK_IMAGE_DIR="${AB_DEPLOYMENT_TASK_IMAGE_DIR:-$SCRATCH/task-images}"

JOBDIR="$SCRATCH/agent-bench/tb21-image-jobs"

mkdir -p \
    "$JOBDIR" \
    "$AB_DEPLOYMENT_TASK_IMAGE_DIR"

echo "Terminal-Bench tasks:"
echo "  $AB_BENCHMARK_TASKS_ROOT"
echo
echo "SIF directory:"
echo "  $AB_DEPLOYMENT_TASK_IMAGE_DIR"
echo

i=0

while IFS=$'\t' read -r IMAGE SIF; do
    [[ -z "${IMAGE:-}" ]] && continue

    i=$((i + 1))

    JOBFILE="$JOBDIR/tb21-image-${i}.pbs"

    cat > "$JOBFILE" <<EOF
#!/usr/bin/env bash
#PBS -N tb21-img-${i}
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

echo "=== Terminal-Bench image ==="
echo "image:  \$IMAGE"
echo "target: \$SIF"
echo

if [[ -s "\$SIF" ]]; then
    echo "already ready: \$SIF"

    singularity inspect "\$SIF" >/dev/null

    echo "VALID"
    exit 0
fi

rm -f "\$PARTIAL"

echo "pulling docker://\$IMAGE"

singularity pull \
    "\$PARTIAL" \
    "docker://\$IMAGE"

echo "validating \$PARTIAL"

singularity inspect "\$PARTIAL" >/dev/null

mv "\$PARTIAL" "\$SIF"

echo
echo "READY"
ls -lh "\$SIF"
EOF

    echo "Submitting:"
    echo "  $IMAGE"
    echo "  -> $SIF"

    qsub "$JOBFILE"

done < <(
    python3 scripts/terminal_bench_task_images.py \
        --tasks-root "$TASKS_ROOT" \
        --missing-only \
        --tsv
)

echo
echo "submitted $i missing Terminal-Bench image job(s)"
