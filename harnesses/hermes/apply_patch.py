#!/usr/bin/env python3
"""Apply Agent_Bench's Hermes patches declaratively from MANIFEST.json.

Modes (env HERMES_PATCH_MODE):
  runtime  apply patches in-place
  baked    assert markers already exist
  skip     no-op
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "patch" / "MANIFEST.json"


def load_hermes_root() -> Path:
    root = os.environ.get("HERMES_AGENT_ROOT")
    if root:
        p = Path(root)
    else:
        # The upstream Docker image uses /opt/hermes as WORKDIR. Keep this
        # path-based fast path first so patching does not depend on importing
        # optional runtime deps during image build.
        for candidate in (Path("/opt/hermes"), Path("/app"), Path.cwd()):
            if (
                (candidate / "tools/terminal_tool.py").is_file()
                and (candidate / "tools/environments/docker.py").is_file()
            ):
                p = candidate
                break
        else:
            # Fallback for non-Docker installs where Hermes is importable as a
            # package from site-packages.
            mod = importlib.import_module("tools.terminal_tool")
            p = Path(mod.__file__).resolve().parents[1]
    if not p.is_dir():
        sys.exit(f"HERMES_AGENT_ROOT={p} does not exist or is not a directory")
    return p


def is_applied(entry: dict, root: Path) -> bool:
    marker = entry["idempotent_marker"]
    targets = entry.get("target_paths") or (
        "tools/environments/docker.py",
        "tools/terminal_tool.py",
    )
    for rel in targets:
        path = root / rel
        if path.exists() and marker in path.read_text(errors="ignore"):
            return True
    return False


def apply_one(entry: dict, root: Path, patch_dir: Path) -> None:
    if is_applied(entry, root):
        print(f"[skip] {entry['file']}: already applied")
        return
    patch_file = patch_dir / entry["file"]
    rc = subprocess.run(
        ["patch", "-p1", "--forward", "--silent", "-i", str(patch_file)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        sys.exit(
            f"[fail] {entry['file']}: patch rc={rc.returncode}\n"
            f"{rc.stdout}\n{rc.stderr}"
        )
    check_targets = [root / rel for rel in (
        entry.get("compile_targets")
        or entry.get("target_paths")
        or ("tools/environments/docker.py", "tools/terminal_tool.py")
    )]
    pycheck = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, check_targets)],
        capture_output=True,
        text=True,
    )
    if pycheck.returncode != 0:
        sys.exit(
            f"[fail] {entry['file']}: py_compile failed\n"
            f"{pycheck.stdout}\n{pycheck.stderr}"
        )
    print(f"[ok]   {entry['file']}: applied")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("cmd", nargs="?", choices=["apply", "status"], default="apply")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    root = load_hermes_root()
    patch_dir = manifest_path.resolve().parent
    mode = os.environ.get("HERMES_PATCH_MODE", "runtime")

    print(f"HERMES_AGENT_ROOT = {root}")
    print(f"HERMES_PATCH_MODE  = {mode}")

    if mode == "skip":
        print("[mode=skip] no-op")
        return 0

    missing = []
    if args.cmd == "status" or mode == "baked":
        for entry in manifest["patches"]:
            applied = is_applied(entry, root)
            flag = "applied" if applied else "missing"
            print(f"  {flag:7s} {entry['file']}")
            if not applied:
                missing.append(entry["file"])
        if missing and mode == "baked":
            sys.exit(f"[mode=baked] markers missing for: {missing}")
        return 0

    for entry in manifest["patches"]:
        apply_one(entry, root, patch_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
