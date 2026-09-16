#!/usr/bin/env bash
# Pull the ten candidate RoadmapBench task SIFs outside NSCC.

set -euo pipefail

IMAGE_DIR="${1:-$PWD/roadmapbench-subset10d-sifs}"

images=(
    znpt/roadmapbench-dsl-2.3.0-roadmap
    znpt/roadmapbench-fal-3.0.0-roadmap
    znpt/roadmapbench-fal-4.0.0-roadmap
    znpt/roadmapbench-fbr-2.42.0-roadmap
    znpt/roadmapbench-fyn-2.5.0-roadmap
    znpt/roadmapbench-fyn-2.7.0-roadmap
    znpt/roadmapbench-ktx-0.10.0-roadmap
    znpt/roadmapbench-ktx-0.15.1-roadmap
    znpt/roadmapbench-opt-4.0.0-roadmap
    znpt/roadmapbench-pyg-2.2.0-roadmap
)

if command -v apptainer >/dev/null 2>&1; then
    container_runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    container_runtime=singularity
else
    echo "ERROR: apptainer or singularity is required to pull task SIFs" >&2
    exit 1
fi

mkdir -p "$IMAGE_DIR"

for docker_image in "${images[@]}"; do
    sif_name="${docker_image//\//__}"
    sif_name="${sif_name//:/__}.sif"
    destination="$IMAGE_DIR/$sif_name"
    partial="$destination.partial"

    if [[ -s "$destination" ]]; then
        echo "$docker_image: existing non-empty $destination"
        continue
    fi

    pulled=0
    for attempt in 1 2 3; do
        rm -f "$partial"
        if (cd "$IMAGE_DIR" && "$container_runtime" pull \
                --name "$(basename "$partial")" "docker://$docker_image") \
                && [[ -s "$partial" ]]; then
            pulled=1
            break
        fi
        echo "$docker_image: pull attempt $attempt failed; retrying in 30 seconds" >&2
        sleep 30
    done
    [[ "$pulled" == 1 ]] || {
        echo "ERROR: pull produced an empty image for $docker_image" >&2
        exit 1
    }
    mv "$partial" "$destination"
done

(
    cd "$IMAGE_DIR"
    sha256sum ./*.sif > SHA256SUMS
)

echo
echo "Prepared task images: $IMAGE_DIR"
echo "Upload all .sif files into NSCC \$SCRATCH/task-images/."
