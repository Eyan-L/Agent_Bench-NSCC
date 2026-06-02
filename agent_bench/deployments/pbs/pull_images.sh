#!/usr/bin/env bash
# Pull SWE-bench per-instance docker images as Singularity .sif files.
#
# Usage:
#   pull_images.sh --instances <file>              # one instance_id per line
#   pull_images.sh --dataset <verified.json>       # extract every instance_id
#   pull_images.sh --instance-ids id1 id2 ...      # explicit list
#
# Optional:
#   --dir <path>        output dir (default: $AB_SCORING_IMAGE_DIR or ~/.cache/swebench-sif)
#   --parallel N        concurrent pulls (default: 1; bump cautiously, Docker
#                       Hub rate-limits anonymous pulls at 100 req/6h)
#   --tag latest        image tag (default: latest)
#   --force             re-pull even if <id>.sif exists
#
# Image naming follows SWE-bench convention:
#   swebench/sweb.eval.x86_64.<owner>_1776_<repo>-<num>:<tag>

set -euo pipefail

DIR="${AB_SCORING_IMAGE_DIR:-$HOME/.cache/swebench-sif}"
PARALLEL=1
TAG="latest"
FORCE=0
INSTANCES_FILE=""
DATASET_JSON=""
EXPLICIT_IDS=()

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instances) INSTANCES_FILE="$2"; shift 2 ;;
        --dataset)   DATASET_JSON="$2"; shift 2 ;;
        --dir)       DIR="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --tag)       TAG="$2"; shift 2 ;;
        --force)     FORCE=1; shift ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPLICIT_IDS+=("$1"); shift
            done ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: singularity not on PATH" >&2
    exit 1
fi

mkdir -p "$DIR"

# Build the instance id list.
declare -a IDS
if [[ -n "$INSTANCES_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [[ -z "$line" ]] || IDS+=("$line")
    done < "$INSTANCES_FILE"
fi
if [[ -n "$DATASET_JSON" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 needed for --dataset" >&2; exit 1
    fi
    while IFS= read -r line; do IDS+=("$line"); done < <(
        python3 -c "import json,sys
for d in json.load(open('$DATASET_JSON')): print(d['instance_id'])"
    )
fi
IDS+=("${EXPLICIT_IDS[@]}")
if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "ERROR: no instances provided. Use --instances/--dataset/--instance-ids." >&2
    exit 1
fi
# Dedupe, preserve order.
declare -A SEEN
declare -a UNIQ
for id in "${IDS[@]}"; do
    if [[ -z "${SEEN[$id]:-}" ]]; then SEEN[$id]=1; UNIQ+=("$id"); fi
done
IDS=("${UNIQ[@]}")

echo "[pull-sif] target dir : $DIR"
echo "[pull-sif] tag        : $TAG"
echo "[pull-sif] parallel   : $PARALLEL"
echo "[pull-sif] instances  : ${#IDS[@]}"

# instance_id -> docker image name
# astropy__astropy-12907 -> swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
image_for() {
    local id="$1"
    local owner="${id%%__*}"
    local rest="${id#*__}"
    echo "swebench/sweb.eval.x86_64.${owner}_1776_${rest}"
}

pull_one() {
    local id="$1"
    local img sif logf
    img="$(image_for "$id"):$TAG"
    sif="$DIR/$id.sif"
    logf="$DIR/$id.pull.log"

    if [[ -f "$sif" && "$FORCE" -ne 1 ]]; then
        echo "[skip] $id (sif exists, $(du -h "$sif" | awk '{print $1}'))"
        return 0
    fi
    rm -f "$sif" "$sif.tmp"
    echo "[pull] $id  <-  docker://$img"
    if singularity pull --dir "$DIR" --name "$id.sif" "docker://$img" \
            > "$logf" 2>&1; then
        echo "[ok]   $id  ($(du -h "$sif" | awk '{print $1}'))"
        rm -f "$logf"
        return 0
    else
        echo "[FAIL] $id  -- see $logf" >&2
        return 1
    fi
}

export -f pull_one image_for
export DIR TAG FORCE

if [[ "$PARALLEL" -le 1 ]]; then
    failures=0
    for id in "${IDS[@]}"; do
        pull_one "$id" || failures=$((failures + 1))
    done
else
    if command -v parallel >/dev/null 2>&1; then
        printf '%s\n' "${IDS[@]}" | parallel -j "$PARALLEL" pull_one
        failures=$?
    else
        printf '%s\n' "${IDS[@]}" \
            | xargs -P "$PARALLEL" -I{} bash -c 'pull_one "$@"' _ {}
        failures=$?
    fi
fi

echo "[pull-sif] done. ${#IDS[@]} requested; failures=$failures; dir=$DIR"
exit $failures
