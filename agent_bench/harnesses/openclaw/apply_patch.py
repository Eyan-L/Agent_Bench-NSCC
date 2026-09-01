#!/usr/bin/env python3
"""Apply OpenClaw patches declaratively from MANIFEST.json.

Source of truth: the .patch unified diffs and named transforms under patch/.
The driver resolves content-hashed bundles, checks idempotency markers, applies
the changes, and validates every modified JavaScript target with `node --check`.

Modes (env AB_HARNESS_OPENCLAW_PATCH_MODE):
  runtime  apply patches/transforms in-place (backup once at <file>.oc-orig)
  baked    assert markers already present (image/overlay was pre-patched)
  skip     no-op (used to run unpatched OpenClaw for ablations)

Subcommands:
  apply      default; apply per MANIFEST
  status     print per-patch/transform state
  restore    restore each target from its .oc-orig backup
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "patch" / "MANIFEST.json"

_POST_COMPACTION_GUARD_TRANSFORM = "post_compaction_loop_guard"
_POST_COMPACTION_GUARD_RUNNER_MARKER = (
    "AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD_RUNNER_BEGIN"
)
_POST_COMPACTION_GUARD_OUTCOME_MARKER = (
    "AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD_OUTCOME_BEGIN"
)


def load_oc_root() -> Path:
    root = os.environ.get("AB_HARNESS_OPENCLAW_SOURCE_ROOT")
    if not root:
        env_file = SCRIPT_DIR / "defaults.env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith(": \"${AB_HARNESS_OPENCLAW_SOURCE_ROOT:="):
                    root = line.split("=", 1)[1].rstrip("}\"")
                    break
    if not root:
        sys.exit("AB_HARNESS_OPENCLAW_SOURCE_ROOT not set; export it or set in defaults.env")
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"AB_HARNESS_OPENCLAW_SOURCE_ROOT={root} does not exist or is not a directory")
    return p


def resolve_targets(entry: dict, oc_root: Path) -> list[Path]:
    """Multi-target variant. Returns a list of Path; most entries have one."""
    if "target_glob_all" in entry:
        matches = sorted(glob.glob(str(oc_root / entry["target_glob_all"]), recursive=True))
        matches = [m for m in matches if not m.endswith(".oc-orig")]
        if not matches:
            sys.exit(f"no match for {entry['target_glob_all']} under {oc_root}")
        return [Path(m) for m in matches]
    return [resolve_target(entry, oc_root)]


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


def _validate_js(target: Path, before_text: str) -> None:
    check = subprocess.run(
        ["node", "--check", str(target)],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        target.write_text(before_text)
        sys.exit(
            f"[fail] node --check failed after transform on {target}\n{check.stderr}"
        )


def apply_one(entry: dict, oc_root: Path, patch_dir: Path) -> None:
    targets = resolve_targets(entry, oc_root)
    name = entry["file"]

    for target in targets:
        if is_applied(entry, target):
            print(f"[skip] {name}: already applied -> {target.relative_to(oc_root)}")
            continue

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
                sys.exit(f"[fail] {name}: patch rc={rc.returncode} on {target}\n{rc.stderr}")
        finally:
            tmp.unlink(missing_ok=True)

        check = subprocess.run(
            ["node", "--check", str(target)],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            shutil.copy2(backup, target)
            sys.exit(f"[fail] {name}: node --check failed after apply on {target}\n{check.stderr}")

        print(f"[ok]   {name}: applied to {target.relative_to(oc_root)}")


def _iter_substantive_dist_js(oc_root: Path):
    dist = oc_root / "dist"
    if not dist.is_dir():
        return
    for target in sorted(dist.glob("*.js")):
        if target.name.endswith(".oc-orig"):
            continue
        try:
            text = target.read_text(errors="ignore")
        except OSError:
            continue
        if text.count("\n") < 50:
            continue
        yield target, text


def _resolve_post_compaction_guard_targets(oc_root: Path) -> list[Path]:
    """Locate the compiled v2026.4.22 runner/tool-hook bundles by stable content."""
    runner_needles = (
        "runEmbeddedAttemptWithBackend",
        "auto-compaction succeeded for",
        "[timeout-compaction] compaction succeeded for",
    )
    outcome_needles = (
        "async function recordLoopOutcome",
        "recordToolCallOutcome(sessionState",
        "toolCallHistory",
    )
    targets: list[Path] = []
    for target, text in _iter_substantive_dist_js(oc_root) or []:
        if (
            _POST_COMPACTION_GUARD_RUNNER_MARKER in text
            or _POST_COMPACTION_GUARD_OUTCOME_MARKER in text
            or all(needle in text for needle in runner_needles)
            or all(needle in text for needle in outcome_needles)
        ):
            targets.append(target)
    unique = list(dict.fromkeys(targets))
    if not unique:
        sys.exit(
            "[fail] post_compaction_loop_guard: could not locate the OpenClaw "
            "v2026.4.22 runner/tool-hook bundles under dist/"
        )
    return unique


def _line_indent(text: str, pos: int) -> str:
    line_start = text.rfind("\n", 0, pos) + 1
    line = text[line_start:pos]
    return line[: len(line) - len(line.lstrip())]


def _insert_before_following_continue(
    text: str,
    needle: str,
    statement: str,
) -> tuple[str, bool]:
    idx = text.find(needle)
    if idx < 0:
        return text, False
    continue_idx = text.find("continue;", idx)
    if continue_idx < 0 or continue_idx - idx > 800:
        return text, False
    indent = _line_indent(text, continue_idx)
    return (
        text[:continue_idx] + f"{indent}{statement}\n" + text[continue_idx:],
        True,
    )


def _transform_post_compaction_runner(text: str) -> tuple[str, bool]:
    if _POST_COMPACTION_GUARD_RUNNER_MARKER in text:
        return text, False

    state_match = re.search(r"(?m)^(\s*)let autoCompactionCount = 0;\s*$", text)
    if not state_match:
        return text, False

    indent = state_match.group(1)
    child = indent + "\t"
    state_block = (
        f"\n{indent}/* {_POST_COMPACTION_GUARD_RUNNER_MARKER} */\n"
        f"{indent}// Backport of openclaw/openclaw#77555 for pinned v2026.4.22.\n"
        f"{indent}// It is dormant until a successful auto-compaction retry.\n"
        f"{indent}const __abPostCompactionGuardKey = `${{params.runId ?? \"\"}}:${{params.sessionId ?? \"\"}}`;\n"
        f"{indent}const __abPostCompactionGuards = globalThis.__AB_HARNESS_POST_COMPACTION_GUARDS ?? "
        f"(globalThis.__AB_HARNESS_POST_COMPACTION_GUARDS = new Map());\n"
        f"{indent}const __abPostCompactionAbortController = new AbortController();\n"
        f"{indent}const __abPostCompactionGuard = {{\n"
        f"{child}enabled: params.config?.tools?.loopDetection?.enabled !== false,\n"
        f"{child}windowSize: 3,\n"
        f"{child}remainingAttempts: 0,\n"
        f"{child}history: [],\n"
        f"{child}abortController: __abPostCompactionAbortController,\n"
        f"{child}abortError: void 0\n"
        f"{indent}}};\n"
        f"{indent}const __abArmPostCompactionGuard = () => {{\n"
        f"{child}if (!__abPostCompactionGuard.enabled) return;\n"
        f"{child}__abPostCompactionGuard.remainingAttempts = __abPostCompactionGuard.windowSize;\n"
        f"{child}__abPostCompactionGuard.history = [];\n"
        f"{child}__abPostCompactionGuard.abortError = void 0;\n"
        f"{child}__abPostCompactionGuards.set(__abPostCompactionGuardKey, __abPostCompactionGuard);\n"
        f"{child}log.info(`[post-compaction-guard] armed for ${{__abPostCompactionGuard.windowSize}} tool outcomes`);\n"
        f"{indent}}};\n"
        f"{indent}if (params.abortSignal) {{\n"
        f"{child}const __abRelayParentAbort = () => "
        f"__abPostCompactionAbortController.abort(params.abortSignal?.reason);\n"
        f"{child}if (params.abortSignal.aborted) __abRelayParentAbort();\n"
        f"{child}else params.abortSignal.addEventListener(\"abort\", __abRelayParentAbort, {{ once: true }});\n"
        f"{indent}}}\n"
        f"{indent}/* AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD_RUNNER_END */"
    )
    insert_at = state_match.end()
    text = text[:insert_at] + state_block + text[insert_at:]

    call_idx = text.find("runEmbeddedAttemptWithBackend({", insert_at)
    if call_idx < 0:
        raise ValueError("runner anchor runEmbeddedAttemptWithBackend({ not found")
    abort_idx = text.find("abortSignal: params.abortSignal,", call_idx)
    if abort_idx < 0 or abort_idx - call_idx > 40000:
        raise ValueError("runner abortSignal anchor not found near embedded attempt")
    text = (
        text[:abort_idx]
        + "abortSignal: __abPostCompactionAbortController.signal,"
        + text[abort_idx + len("abortSignal: params.abortSignal,") :]
    )

    text, timeout_armed = _insert_before_following_continue(
        text,
        "[timeout-compaction] compaction succeeded for ${provider}/${modelId}; retrying prompt",
        "__abArmPostCompactionGuard();",
    )
    text, overflow_armed = _insert_before_following_continue(
        text,
        "auto-compaction succeeded for ${provider}/${modelId}; retrying prompt",
        "__abArmPostCompactionGuard();",
    )
    if not timeout_armed or not overflow_armed:
        raise ValueError(
            "runner compaction-success anchors changed "
            f"(timeout={timeout_armed}, overflow={overflow_armed})"
        )

    # This path was added after v2026.4.22. If a downstream 4.22 build happens
    # to contain it, arm the guard there too, matching upstream #77555.
    text, _ = _insert_before_following_continue(
        text,
        "compaction interrupted visible final answer",
        "__abArmPostCompactionGuard();",
    )
    return text, True


def _transform_post_compaction_outcome(text: str) -> tuple[str, bool]:
    if _POST_COMPACTION_GUARD_OUTCOME_MARKER in text:
        return text, False
    function_idx = text.find("async function recordLoopOutcome")
    if function_idx < 0:
        return text, False
    call_idx = text.find("recordToolCallOutcome(sessionState, {", function_idx)
    if call_idx < 0:
        raise ValueError("tool-outcome recordToolCallOutcome anchor not found")
    config_idx = text.find("config: args.ctx.loopDetection", call_idx)
    if config_idx < 0 or config_idx - call_idx > 8000:
        raise ValueError("tool-outcome loopDetection config anchor not found")
    call_end = text.find("});", config_idx)
    if call_end < 0 or call_end - config_idx > 1000:
        raise ValueError("tool-outcome recordToolCallOutcome closing anchor not found")
    call_end += len("});")
    indent = _line_indent(text, call_idx)
    child = indent + "\t"
    block = (
        f"\n{indent}/* {_POST_COMPACTION_GUARD_OUTCOME_MARKER} */\n"
        f"{indent}// Observe only completed outcomes while the runner's post-compaction window is armed.\n"
        f"{indent}const __abPostCompactionGuards = globalThis.__AB_HARNESS_POST_COMPACTION_GUARDS;\n"
        f"{indent}const __abPostCompactionGuardKey = `${{args.ctx?.runId ?? \"\"}}:${{args.ctx?.sessionId ?? \"\"}}`;\n"
        f"{indent}const __abPostCompactionGuard = __abPostCompactionGuards?.get(__abPostCompactionGuardKey);\n"
        f"{indent}if (__abPostCompactionGuard?.enabled && __abPostCompactionGuard.remainingAttempts > 0) {{\n"
        f"{child}const __abHistory = sessionState.toolCallHistory ?? [];\n"
        f"{child}let __abRecord;\n"
        f"{child}for (let __abI = __abHistory.length - 1; __abI >= 0; __abI -= 1) {{\n"
        f"{child}\tconst __abCandidate = __abHistory[__abI];\n"
        f"{child}\tif (!__abCandidate || __abCandidate.toolName !== args.toolName) continue;\n"
        f"{child}\tif (args.toolCallId && __abCandidate.toolCallId !== args.toolCallId) continue;\n"
        f"{child}\tif (typeof __abCandidate.resultHash !== \"string\" || !__abCandidate.resultHash) continue;\n"
        f"{child}\t__abRecord = __abCandidate;\n"
        f"{child}\tbreak;\n"
        f"{child}}}\n"
        f"{child}if (__abRecord) {{\n"
        f"{child}\t__abPostCompactionGuard.remainingAttempts -= 1;\n"
        f"{child}\tconst __abObservation = {{\n"
        f"{child}\t\ttoolName: __abRecord.toolName,\n"
        f"{child}\t\targsHash: __abRecord.argsHash,\n"
        f"{child}\t\tresultHash: __abRecord.resultHash\n"
        f"{child}\t}};\n"
        f"{child}\t__abPostCompactionGuard.history.push(__abObservation);\n"
        f"{child}\tconst __abMatches = __abPostCompactionGuard.history.filter((entry) =>\n"
        f"{child}\t\tentry.toolName === __abObservation.toolName &&\n"
        f"{child}\t\tentry.argsHash === __abObservation.argsHash &&\n"
        f"{child}\t\tentry.resultHash === __abObservation.resultHash\n"
        f"{child}\t);\n"
        f"{child}\tif (__abMatches.length >= __abPostCompactionGuard.windowSize) {{\n"
        f"{child}\t\t__abPostCompactionGuard.remainingAttempts = 0;\n"
        f"{child}\t\tconst __abError = new Error(`CRITICAL: tool ${{__abObservation.toolName}} repeated "
        f"${{__abMatches.length}} times with identical arguments and identical results within "
        f"${{__abPostCompactionGuard.windowSize}} attempts after auto-compaction. "
        f"The compaction did not break the loop. Aborting to prevent runaway resource use. "
        f"detector=compaction_loop_persisted`);\n"
        f"{child}\t\t__abError.name = \"PostCompactionLoopPersistedError\";\n"
        f"{child}\t\t__abError.detector = \"compaction_loop_persisted\";\n"
        f"{child}\t\t__abError.count = __abMatches.length;\n"
        f"{child}\t\t__abError.toolName = __abObservation.toolName;\n"
        f"{child}\t\t__abPostCompactionGuard.abortError = __abError;\n"
        f"{child}\t\tlog.error(`[post-compaction-guard] compaction_loop_persisted: tool=${{__abObservation.toolName}} "
        f"count=${{__abMatches.length}}`);\n"
        f"{child}\t\t__abPostCompactionGuards?.delete(__abPostCompactionGuardKey);\n"
        f"{child}\t\t__abPostCompactionGuard.abortController.abort(__abError);\n"
        f"{child}\t}} else if (__abPostCompactionGuard.remainingAttempts <= 0) {{\n"
        f"{child}\t\t__abPostCompactionGuards?.delete(__abPostCompactionGuardKey);\n"
        f"{child}\t}}\n"
        f"{child}}}\n"
        f"{indent}}}\n"
        f"{indent}/* AB_HARNESS_OPENCLAW_PATCH_POST_COMPACTION_LOOP_GUARD_OUTCOME_END */"
    )
    return text[:call_end] + block + text[call_end:], True


def _apply_post_compaction_guard_to_target(target: Path, oc_root: Path) -> bool:
    before = target.read_text(errors="ignore")
    after = before
    changed = False
    if (
        "runEmbeddedAttemptWithBackend" in after
        and "auto-compaction succeeded for" in after
        and "[timeout-compaction] compaction succeeded for" in after
    ):
        after, runner_changed = _transform_post_compaction_runner(after)
        changed = changed or runner_changed
    if (
        "async function recordLoopOutcome" in after
        and "recordToolCallOutcome(sessionState" in after
    ):
        after, outcome_changed = _transform_post_compaction_outcome(after)
        changed = changed or outcome_changed
    if not changed:
        return False

    backup = target.with_suffix(target.suffix + ".oc-orig")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(after)
    _validate_js(target, before)
    print(
        f"[ok]   {_POST_COMPACTION_GUARD_TRANSFORM}: "
        f"applied to {target.relative_to(oc_root)}"
    )
    return True


def _post_compaction_target_applied(target: Path) -> bool:
    text = target.read_text(errors="ignore")
    runner_target = (
        "runEmbeddedAttemptWithBackend" in text
        and "auto-compaction succeeded for" in text
        and "[timeout-compaction] compaction succeeded for" in text
    )
    outcome_target = (
        "async function recordLoopOutcome" in text
        and "recordToolCallOutcome(sessionState" in text
    )
    if runner_target and _POST_COMPACTION_GUARD_RUNNER_MARKER not in text:
        return False
    if outcome_target and _POST_COMPACTION_GUARD_OUTCOME_MARKER not in text:
        return False
    return runner_target or outcome_target


def apply_transform(entry: dict, oc_root: Path) -> None:
    name = entry["name"]
    if name != _POST_COMPACTION_GUARD_TRANSFORM:
        sys.exit(f"unknown OpenClaw transform: {name}")
    targets = _resolve_post_compaction_guard_targets(oc_root)
    changed = 0
    for target in targets:
        if _post_compaction_target_applied(target):
            print(f"[skip] {name}: already applied -> {target.relative_to(oc_root)}")
            continue
        try:
            changed += int(_apply_post_compaction_guard_to_target(target, oc_root))
        except ValueError as exc:
            sys.exit(f"[fail] {name}: {target.relative_to(oc_root)}: {exc}")
    if changed == 0 and not all(_post_compaction_target_applied(target) for target in targets):
        sys.exit(f"[fail] {name}: targets found but no transform was applied")


def _transform_status(entry: dict, oc_root: Path) -> list[tuple[Path, bool]]:
    if entry["name"] != _POST_COMPACTION_GUARD_TRANSFORM:
        return []
    targets = _resolve_post_compaction_guard_targets(oc_root)
    return [(target, _post_compaction_target_applied(target)) for target in targets]


def cmd_apply(args, manifest, oc_root, patch_dir, mode):
    transforms = manifest.get("transforms", [])
    if mode == "skip":
        print("[mode=skip] no-op")
        return
    if mode == "baked":
        bad = []
        for entry in manifest["patches"]:
            for target in resolve_targets(entry, oc_root):
                if not is_applied(entry, target):
                    bad.append(f"{entry['file']} -> {target}")
        for entry in transforms:
            for target, applied in _transform_status(entry, oc_root):
                if not applied:
                    bad.append(f"{entry['name']} -> {target}")
        if bad:
            sys.exit(f"[mode=baked] markers missing for: {bad}")
        print("[mode=baked] all markers present")
        return
    for entry in manifest["patches"]:
        apply_one(entry, oc_root, patch_dir)
    for entry in transforms:
        apply_transform(entry, oc_root)


def cmd_status(args, manifest, oc_root, patch_dir, mode):
    print(f"AB_HARNESS_OPENCLAW_SOURCE_ROOT = {oc_root}")
    print(f"AB_HARNESS_OPENCLAW_PATCH_MODE    = {mode}")
    for entry in manifest["patches"]:
        try:
            targets = resolve_targets(entry, oc_root)
        except SystemExit as e:
            print(f"  ?  {entry['file']}: {e}")
            continue
        for target in targets:
            applied = is_applied(entry, target)
            flag = "applied" if applied else "missing"
            print(f"  {flag:7s} {entry['file']} -> {target.relative_to(oc_root)}")
    for entry in manifest.get("transforms", []):
        try:
            states = _transform_status(entry, oc_root)
        except SystemExit as e:
            print(f"  ?  {entry['name']}: {e}")
            continue
        for target, applied in states:
            flag = "applied" if applied else "missing"
            print(f"  {flag:7s} {entry['name']} -> {target.relative_to(oc_root)}")


def cmd_restore(args, manifest, oc_root, patch_dir, mode):
    targets: list[Path] = []
    for entry in manifest["patches"]:
        targets.extend(resolve_targets(entry, oc_root))
    for entry in manifest.get("transforms", []):
        targets.extend(target for target, _ in _transform_status(entry, oc_root))
    for target in dict.fromkeys(targets):
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

    manifest = json.loads(Path(args.manifest).read_text())
    oc_root = load_oc_root()
    patch_dir = Path(args.manifest).resolve().parent
    mode = os.environ.get("AB_HARNESS_OPENCLAW_PATCH_MODE", "runtime")

    {"apply": cmd_apply, "status": cmd_status, "restore": cmd_restore}[cmd](
        args, manifest, oc_root, patch_dir, mode
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
