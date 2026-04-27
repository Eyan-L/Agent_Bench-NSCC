#!/usr/bin/env python3
"""Extract multiple files from a Claude response.

Claude is instructed to emit each file as a fenced code block whose first
interior line is a marker of the form:

    # file: relative/path.py
    // file: relative/path.js
    -- file: relative/path.sql

This script walks the response, finds fenced blocks with such markers, and
writes each file under --out-dir.

If no file markers are found, the raw response is copied to --raw-fallback
so the user can hand-extract.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

FILE_MARKER = re.compile(r"^\s*(?:#|//|--|;)\s*file\s*:\s*(.+?)\s*$")


def extract(text: str) -> dict[str, str]:
    """Return {relative_path: content} for each fenced block with a file marker."""
    files: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # Detect opening fence: ``` optionally followed by a language id.
        if line.lstrip().startswith("```"):
            # Peek at the next line for a file marker.
            j = i + 1
            if j < n:
                m = FILE_MARKER.match(lines[j])
                if m:
                    fname = m.group(1).strip()
                    # Collect body until closing fence.
                    body: list[str] = []
                    k = j + 1
                    while k < n and not lines[k].lstrip().startswith("```"):
                        body.append(lines[k])
                        k += 1
                    files[fname] = "".join(body)
                    i = k + 1  # skip past closing fence
                    continue
            # No file marker: skip this fenced block entirely.
            k = i + 1
            while k < n and not lines[k].lstrip().startswith("```"):
                k += 1
            i = k + 1
            continue
        i += 1

    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="path to Claude's raw response")
    p.add_argument("--out-dir", required=True, help="directory to write extracted files")
    p.add_argument(
        "--raw-fallback",
        help="if no file markers found, copy raw response here",
    )
    args = p.parse_args()

    text = pathlib.Path(args.input).read_text(encoding="utf-8")
    files = extract(text)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        msg = f"[extract_files] No `# file:` markers found in {args.input}."
        print(msg, file=sys.stderr)
        if args.raw_fallback:
            fallback = pathlib.Path(args.raw_fallback)
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(text, encoding="utf-8")
            print(
                f"[extract_files] Raw response copied to {fallback}. "
                "Hand-extract files and then `touch {out_dir}/.generated`.",
                file=sys.stderr,
            )
        return 1

    skipped: list[str] = []
    for fname, content in files.items():
        # Guard against path traversal.
        target = (out_dir / fname).resolve()
        if not str(target).startswith(str(out_dir.resolve())):
            print(f"[extract_files] WARN: skipping out-of-tree path: {fname}", file=sys.stderr)
            skipped.append(fname)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"[extract_files] Wrote {target.relative_to(out_dir.parent)}")

    # Also copy the raw response for audit.
    raw_copy = out_dir / "_raw_output.md"
    raw_copy.write_text(text, encoding="utf-8")

    written = len(files) - len(skipped)
    print(f"[extract_files] {written} file(s) extracted to {out_dir}")
    if skipped:
        print(
            f"[extract_files] {len(skipped)} file(s) skipped for being out of tree: "
            f"{', '.join(skipped)}. Extract them manually from _raw_output.md if needed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())