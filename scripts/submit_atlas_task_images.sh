#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?usage: $0 <atlas-benchmark-profile>}"

export SCRATCH="${SCRATCH:-/scratch/users/ntu/$USER}"

JOBDIR="$SCRATCH/agent-bench/atlas-image-jobs"
mkdir -p "$JOBDIR" "$SCRATCH/task-images"

i=0

while IFS=$'\t' read -r IMAGE SIF; do
    [[ -z "${IMAGE:-}" ]] && continue

    i=$((i + 1))
    JOBFILE="$JOBDIR/atlas-image-${i}.pbs"

    cat > "$JOBFILE" <<EOF
#!/usr/bin/env bash
#PBS -N atlas-img-${i}
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

mkdir -p "\$APPTAINER_CACHEDIR" "\$JOB_TMP" "$(dirname "$SIF")"

cleanup() {
    rm -rf "\$JOB_TMP"
}
trap cleanup EXIT

IMAGE='$IMAGE'
SIF='$SIF'
PARTIAL="\${SIF}.partial"

echo "image: \$IMAGE"
echo "target: \$SIF"

if [[ -s "\$SIF" ]]; then
    echo "already ready: \$SIF"
    singularity inspect "\$SIF" >/dev/null
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

    echo "Submitting:"
    echo "  $IMAGE"
    echo "  -> $SIF"
    qsub "$JOBFILE"

done < <(
    python3 scripts/atlas_task_images.py \
        "$PROFILE" \
        --missing-only \
        --tsv
)

echo
echo "submitted $i missing image job(s)"
