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
import re
import shutil
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
    if shutil.which("patch"):
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
    else:
        try:
            apply_unified_patch_native(root, patch_file)
        except Exception as exc:  # noqa: BLE001 - surface patch context cleanly.
            sys.exit(f"[fail] {entry['file']}: native patch failed: {exc}")
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


def _parse_range(raw: str) -> tuple[int, int]:
    if "," in raw:
        start, count = raw.split(",", 1)
        return int(start), int(count)
    return int(raw), 1


def _strip_patch_prefix(path: str) -> str:
    path = path.strip()
    if path == "/dev/null":
        return path
    path = path.split("\t", 1)[0].split(" ", 1)[0]
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def _apply_hunks(target: Path, hunks: list[tuple[int, list[str]]]) -> None:
    original = target.read_text().splitlines(keepends=True)
    out: list[str] = []
    cursor = 0

    for hunk_cursor, lines in hunks:
        if hunk_cursor < cursor:
            raise RuntimeError(f"hunk overlaps already-applied region in {target}")
        out.extend(original[cursor:hunk_cursor])
        cursor = hunk_cursor

        for line in lines:
            if line.startswith("\\ No newline"):
                continue
            if not line:
                raise RuntimeError(f"malformed empty patch line for {target}")
            op, payload = line[0], line[1:]
            if op == " ":
                if cursor >= len(original) or original[cursor] != payload:
                    got = original[cursor] if cursor < len(original) else "<EOF>"
                    raise RuntimeError(
                        f"context mismatch in {target} at source line {cursor + 1}: "
                        f"expected {payload!r}, got {got!r}"
                    )
                out.append(original[cursor])
                cursor += 1
            elif op == "-":
                if cursor >= len(original) or original[cursor] != payload:
                    got = original[cursor] if cursor < len(original) else "<EOF>"
                    raise RuntimeError(
                        f"delete mismatch in {target} at source line {cursor + 1}: "
                        f"expected {payload!r}, got {got!r}"
                    )
                cursor += 1
            elif op == "+":
                out.append(payload)
            else:
                raise RuntimeError(f"unsupported patch op {op!r} for {target}")

    out.extend(original[cursor:])
    target.write_text("".join(out))


def apply_unified_patch_native(root: Path, patch_file: Path) -> None:
    """Small unified-diff applier used when the container lacks `patch`.

    It intentionally supports only the patch shape we vendor here: git-style
    text diffs against existing files. If a future Hermes patch needs renames,
    binary hunks, or fuzzy offsets, install `patch` in the base image instead.
    """
    lines = patch_file.read_text().splitlines(keepends=True)
    i = 0
    current_target: Path | None = None
    current_hunks: list[tuple[int, list[str]]] = []

    def flush() -> None:
        nonlocal current_target, current_hunks
        if current_target is not None:
            _apply_hunks(current_target, current_hunks)
        current_target = None
        current_hunks = []

    hunk_re = re.compile(r"^@@ -([0-9]+(?:,[0-9]+)?) \+([0-9]+(?:,[0-9]+)?) @@")
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise RuntimeError(f"malformed patch header in {patch_file}")
            flush()
            target_rel = _strip_patch_prefix(lines[i + 1][4:])
            if target_rel == "/dev/null":
                raise RuntimeError(f"new-file patches are not supported: {patch_file}")
            current_target = root / target_rel
            if not current_target.is_file():
                raise RuntimeError(f"patch target not found: {current_target}")
            i += 2
            continue

        m = hunk_re.match(line)
        if m:
            if current_target is None:
                raise RuntimeError(f"hunk before target header in {patch_file}")
            old_start, old_count = _parse_range(m.group(1))
            hunk_cursor = old_start if old_count == 0 else max(old_start - 1, 0)
            hunk_lines: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (
                    next_line.startswith("diff --git ")
                    or next_line.startswith("--- ")
                    or hunk_re.match(next_line)
                ):
                    break
                if next_line.startswith((" ", "-", "+", "\\")):
                    hunk_lines.append(next_line)
                    i += 1
                    continue
                raise RuntimeError(f"malformed hunk line in {patch_file}: {next_line!r}")
            current_hunks.append((hunk_cursor, hunk_lines))
            continue

        i += 1

    flush()


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
