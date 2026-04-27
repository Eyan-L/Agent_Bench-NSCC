# sglang_patch/

Locally-modified sglang source files that we want to overlay on top of the
stock sglang shipped inside the `sglang.sif` container. Each file mirrors
the path it has inside sglang's python package, so the tree under
`sglang_patch/sglang/...` can be bind-mounted 1-for-1 onto the container's
`<site-packages>/sglang/...`.

## What's patched

See [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md). The manifest also tracks
the `.py.bak` fallbacks kept next to the host's site-packages originals.

## How the overlay works at run time

When `sglang.launcher == "singularity"` **and** `sglang.singularity.apply_patch == true`:

1. Before launching sglang, [`run/sglang_launcher_singularity.sh`](../run/sglang_launcher_singularity.sh)
   probes the container to discover sglang's site-packages path:

   ```bash
   singularity exec <sif> python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))'
   ```

2. For every `*.py` under `sglang_patch/sglang/`, a matching `--bind` flag is
   added:

   ```
   --bind <repo>/sglang_patch/sglang/srt/managers/scheduler.py:<site-pkgs>/sglang/srt/managers/scheduler.py
   ```

3. The container starts with the host's patched files "masking" the
   originals inside the image. No rebuild needed.

## Updating a patch

1. Edit the file under `sglang_patch/sglang/...` (NOT the site-packages
   copy in the container — that's read-only).
2. Commit. On the next experiment the updated file is overlaid automatically.

Removing a file: delete it from `sglang_patch/sglang/...` (and update
`SOURCE_MANIFEST.md`). The bind-generation only adds mounts for files that
currently exist under `sglang_patch/sglang/`.

## Manually resyncing from a dev checkout

```bash
# From /home/hongrui/sglang (or wherever your dev checkout lives)
for f in $(awk '/^- `/{gsub(/[`\-\s]/,""); print}' LOCAL_PATCHED_FILES.md); do
    # f looks like python/sglang/srt/managers/scheduler.py
    dst="<repo>/sglang_patch/${f#python/}"
    install -D "${f}" "${dst}"
done
```

(Or simply `rsync` the listed files — they're plain text.)
