#!/usr/bin/env python3
"""Apply OpenClaw patches declaratively from MANIFEST.yaml.

Source of truth: the .patch unified diffs under patch/. This script is a thin
driver that resolves content-hashed bundle filenames, checks idempotency
markers, runs `patch -p0`, and validates the result with `node --check`.

Modes (env OC_PATCH_MODE):
  runtime  apply patches in-place (backup once at <file>.oc-orig)
  baked    assert markers already present (image was pre-patched at build)
  skip     no-op (used to run unpatched OpenClaw for ablations)

Subcommands:
  apply      default; apply per MANIFEST
  status     print per-patch state (applied / missing / mode)
  restore    restore each target from its .oc-orig backup
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "patch" / "MANIFEST.yaml"


def load_oc_root() -> Path:
    root = os.environ.get("OC_HOST_OPENCLAW")
    if not root:
        env_file = SCRIPT_DIR / "defaults.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith(": \"${OC_HOST_OPENCLAW:="):
                    root = line.split("=", 1)[1].rstrip("}\"")
                    break
    if not root:
        sys.exit("OC_HOST_OPENCLAW not set; export it or set in defaults.env")
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"OC_HOST_OPENCLAW={root} does not exist or is not a directory")
    return p


def resolve_target(entry: dict, oc_root: Path) -> Path:
    if "target_path" in entry:
        return oc_root / entry["target_path"]
    matches = sorted(glob.glob(str(oc_root / entry["target_glob"])))
    matches = [m for m in matches if not m.endswith(".oc-orig")]
    if not matches:
        sys.exit(f"no match for {entry['target_glob']} under {oc_root}")
    if len(matches) > 1:
        # OpenClaw ships short (~2-line) re-export stubs alongside the real
        # content-hashed bundle. Filter those out by size; if a content hint
        # is provided, also require it to be present.
        hint = entry.get("target_content_hint")
        substantive = []
        for m in matches:
            try:
                text = Path(m).read_text(errors="ignore")
            except OSError:
                continue
            if text.count("\n") < 50:
                continue
            if hint and hint not in text:
                continue
            substantive.append(m)
        if len(substantive) == 1:
            return Path(substantive[0])
        sys.exit(
            f"ambiguous glob {entry['target_glob']}: {matches} "
            f"(after filter: {substantive}); add target_content_hint to MANIFEST"
        )
    return Path(matches[0])


def materialize_patch(entry: dict, target: Path, patch_dir: Path) -> Path:
    src = patch_dir / entry["file"]
    text = src.read_text()
    placeholder = entry.get("chunk_placeholder")
    if placeholder:
        suffix = target.name.split("-", 1)[-1] if "-" in target.name else target.name
        chunk = suffix.rsplit(".", 1)[0]
        text = text.replace(placeholder, chunk)
    tmp = Path(tempfile.mkstemp(suffix=".patch")[1])
    tmp.write_text(text)
    return tmp


def is_applied(entry: dict, target: Path) -> bool:
    if not target.exists():
        return False
    return entry["idempotent_marker"] in target.read_text(errors="ignore")


def apply_one(entry: dict, oc_root: Path, patch_dir: Path) -> None:
    target = resolve_target(entry, oc_root)
    name = entry["file"]

    if is_applied(entry, target):
        print(f"[skip] {name}: already applied")
        return

    backup = target.with_suffix(target.suffix + ".oc-orig")
    if not backup.exists():
        shutil.copy2(target, backup)

    tmp = materialize_patch(entry, target, patch_dir)
    try:
        rc = subprocess.run(
            ["patch", "-p1", "--forward", "--silent", str(target), str(tmp)],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            shutil.copy2(backup, target)
            sys.exit(f"[fail] {name}: patch rc={rc.returncode}\n{rc.stderr}")
    finally:
        tmp.unlink(missing_ok=True)

    check = subprocess.run(
        ["node", "--check", str(target)],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        shutil.copy2(backup, target)
        sys.exit(f"[fail] {name}: node --check failed after apply\n{check.stderr}")

    print(f"[ok]   {name}: applied to {target.relative_to(oc_root)}")


def cmd_apply(args, manifest, oc_root, patch_dir, mode):
    if mode == "skip":
        print("[mode=skip] no-op")
        return
    if mode == "baked":
        bad = []
        for entry in manifest["patches"]:
            target = resolve_target(entry, oc_root)
            if not is_applied(entry, target):
                bad.append(entry["file"])
        if bad:
            sys.exit(f"[mode=baked] markers missing for: {bad}")
        print("[mode=baked] all markers present")
        return
    for entry in manifest["patches"]:
        apply_one(entry, oc_root, patch_dir)


def cmd_status(args, manifest, oc_root, patch_dir, mode):
    print(f"OC_HOST_OPENCLAW = {oc_root}")
    print(f"OC_PATCH_MODE    = {mode}")
    for entry in manifest["patches"]:
        try:
            target = resolve_target(entry, oc_root)
            applied = is_applied(entry, target)
        except SystemExit as e:
            print(f"  ?  {entry['file']}: {e}")
            continue
        flag = "applied" if applied else "missing"
        print(f"  {flag:7s} {entry['file']} -> {target.relative_to(oc_root)}")


def cmd_restore(args, manifest, oc_root, patch_dir, mode):
    for entry in manifest["patches"]:
        target = resolve_target(entry, oc_root)
        backup = target.with_suffix(target.suffix + ".oc-orig")
        if backup.exists():
            shutil.copy2(backup, target)
            print(f"[restored] {target.relative_to(oc_root)}")
        else:
            print(f"[no-backup] {target.relative_to(oc_root)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("apply")
    sub.add_parser("status")
    sub.add_parser("restore")
    args = ap.parse_args()
    cmd = args.cmd or "apply"

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    oc_root = load_oc_root()
    patch_dir = Path(args.manifest).resolve().parent
    mode = os.environ.get("OC_PATCH_MODE", "runtime")

    {"apply": cmd_apply, "status": cmd_status, "restore": cmd_restore}[cmd](
        args, manifest, oc_root, patch_dir, mode
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
