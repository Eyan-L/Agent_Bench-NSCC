#!/usr/bin/env python3
"""Agent_Bench semantic patches for the pinned OpenClaw v2026.9.2 runtime.

Unlike the legacy v2026.4.22 MANIFEST patches, this patcher targets stable
runtime semantics rather than content-hashed bundle filenames.  It intentionally
keeps the v2026.9.2 customization surface small:

* route sandbox bash exec through OPENCLAW_EXEC_BRIDGE;
* bypass OpenClaw-owned Docker lifecycle when the harness owns the sandbox;
* route sandbox filesystem operations through the same exec bridge;
* replace legacy public model-pricing endpoints if they are present.

OpenClaw v2026.9.2 already provides native rolling/post-compaction loop
detection and native model/tool OpenTelemetry events, so those legacy
Agent_Bench patches are deliberately not re-applied here.

Every transform is fail-closed: a required semantic anchor must be found and
all required markers must be present before the status record is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable


EXPECTED_VERSION = "2026.9.2"

EXEC_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_EXEC_BRIDGE_BEGIN"
SANDBOX_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_SANDBOX_BYPASS_BEGIN"
FS_HELPER_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_HELPER_BEGIN"
FS_RUN_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_RUN_BEGIN"
FS_WRITE_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_WRITE_BEGIN"
FS_READ_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_READ_BEGIN"
FS_CHECK_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_CHECK_BEGIN"
SHARED_WORKSPACE_SPAWN_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SHARED_WORKSPACE_SPAWN_BEGIN"
)

OPENROUTER_PRICING_URL = "https://openrouter.ai/api/v1/models"
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OPENROUTER_OFFLINE_URL = "data:application/json,%7B%22data%22%3A%5B%5D%7D"
LITELLM_OFFLINE_URL = "data:application/json,%7B%7D"


def _read(path: Path) -> str:
    return path.read_text(errors="strict")


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _runtime_version(root: Path) -> str:
    package = root / "package.json"
    if not package.is_file():
        raise RuntimeError(f"missing OpenClaw package.json: {package}")
    try:
        value = json.loads(package.read_text()).get("version")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid OpenClaw package.json: {package}") from exc
    return str(value or "")


def _js_files(root: Path) -> Iterable[Path]:
    for suffix in ("*.js", "*.mjs", "*.cjs"):
        yield from root.rglob(suffix)


def _patch_exec_bridge(root: Path) -> list[str]:
    changed: list[str] = []
    candidates = sorted((root / "dist").glob("bash-tools.exec-runtime-*.js"))
    targets = [
        path
        for path in candidates
        if "sandbox backend does not provide buildExecSpec" in _read(path)
        and "prepareSpawnSpec" in _read(path)
    ]
    if not targets:
        raise RuntimeError("v2026.9.2 exec bridge anchor not found")

    insert = f'''/* {EXEC_MARKER} */
      const __abExecBridge = process.env.OPENCLAW_EXEC_BRIDGE;
      if (__abExecBridge) {{
        return {{
          mode: "child",
          argv: [
            __abExecBridge,
            "--workdir", opts.containerWorkdir ?? opts.sandbox.containerWorkdir,
            "--env-json", JSON.stringify(shellRuntimeEnv),
            "--command", execCommand,
            ...(timeoutMs ? ["--timeout-ms", String(timeoutMs)] : []),
          ],
          env: process.env,
          stdinMode: opts.usePty ? "pipe-open" : "pipe-closed",
        }};
      }}
      /* AB_HARNESS_OPENCLAW_V2026_9_2_EXEC_BRIDGE_END */
      '''
    pattern = re.compile(
        r"(const\s+prepareSpawnSpec\s*=\s*async\s*\(\)\s*=>\s*\{\s*"
        r"if\s*\(opts\.sandbox\)\s*\{\s*)"
    )

    for path in targets:
        text = _read(path)
        if EXEC_MARKER in text:
            continue
        updated, count = pattern.subn(lambda m: m.group(1) + insert, text, count=1)
        if count != 1:
            raise RuntimeError(f"exec bridge anchor was not unique in {path}")
        _write(path, updated)
        changed.append(str(path.relative_to(root)))
    return changed


def _patch_sandbox_bypass(root: Path) -> list[str]:
    changed: list[str] = []
    candidates = sorted((root / "dist").glob("docker-*.js"))
    targets = [
        path
        for path in candidates
        if "ensureSandboxContainer" in _read(path)
        and "buildSandboxContainerName(prefix, slug)" in _read(path)
    ]
    if not targets:
        raise RuntimeError("v2026.9.2 ensureSandboxContainer anchor not found")

    insert = f'''\n  /* {SANDBOX_MARKER} */
  if (process.env.OPENCLAW_BYPASS_ENSURE_SANDBOX) {{
    console.error("[agent-bench] bypass OpenClaw sandbox lifecycle; container=" + containerName);
    return containerName;
  }}
  /* AB_HARNESS_OPENCLAW_V2026_9_2_SANDBOX_BYPASS_END */'''
    pattern = re.compile(
        r"(const\s+containerName\s*=\s*buildSandboxContainerName\(prefix,\s*slug\);)"
    )

    for path in targets:
        text = _read(path)
        if SANDBOX_MARKER in text:
            continue
        # The target bundle may contain other helper copies. Constrain the
        # replacement to a file that also contains ensureSandboxContainer.
        updated, count = pattern.subn(lambda m: m.group(1) + insert, text, count=1)
        if count != 1:
            raise RuntimeError(f"sandbox bypass anchor was not unique in {path}")
        _write(path, updated)
        changed.append(str(path.relative_to(root)))
    return changed


def _patch_shared_workspace_spawns(root: Path) -> list[str]:
    """Keep visible Roadmap children in the harness-owned shared workspace."""
    changed: list[str] = []
    candidates = sorted((root / "dist").glob("sessions-spawn-tool-*.js"))
    anchor = re.compile(
        r"(?P<indent>[ \t]*)const worktree = params\.raw\.worktree === true;\n"
        r"(?P=indent)const worktreeName = readToolStringParam\(params\.raw, [\"']worktreeName[\"']\);\n"
        r"(?P=indent)const worktreeBaseRef = readToolStringParam\(params\.raw, [\"']worktreeBaseRef[\"']\);"
    )
    targets = [path for path in candidates if anchor.search(_read(path))]
    already_patched = [
        path for path in candidates if SHARED_WORKSPACE_SPAWN_MARKER in _read(path)
    ]
    if not targets and not already_patched:
        raise RuntimeError("v2026.9.2 visible sessions_spawn worktree anchor not found")

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}/* {SHARED_WORKSPACE_SPAWN_MARKER} */\n"
            f'{indent}const sharedWorkspaceSpawn = process.env.OPENCLAW_AGENT_BENCH_SHARED_WORKSPACE === "1";\n'
            f'{indent}if (sharedWorkspaceSpawn && params.raw.worktree === true) console.error("[agent-bench] ignoring sessions_spawn worktree=true; children share the benchmark workspace");\n'
            f"{indent}const worktree = params.raw.worktree === true && !sharedWorkspaceSpawn;\n"
            f'{indent}const worktreeName = sharedWorkspaceSpawn ? void 0 : readToolStringParam(params.raw, "worktreeName");\n'
            f'{indent}const worktreeBaseRef = sharedWorkspaceSpawn ? void 0 : readToolStringParam(params.raw, "worktreeBaseRef");\n'
            f"{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_SHARED_WORKSPACE_SPAWN_END */"
        )
    for path in targets:
        text = _read(path)
        updated, count = anchor.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"shared-workspace spawn anchor missing in {path}")
        _write(path, updated)
        changed.append(str(path.relative_to(root)))
    return changed


def _fs_helper_source() -> str:
    return f'''/* {FS_HELPER_MARKER} */
const __abFsBridgeRun = (params) => {{
  const bridgeBin = process.env.OPENCLAW_EXEC_BRIDGE;
  if (!bridgeBin) throw new Error("OPENCLAW_EXEC_BRIDGE not set");
  const workdir = params.sandbox?.containerWorkdir ?? "/";
  const argv = ["sh", "-c", params.script, "openclaw-sandbox-fs", ...(params.options?.args ?? [])];
  return new Promise((resolve, reject) => {{
    const child = __abSpawn(bridgeBin, [
      "--workdir", workdir,
      "--env-json", "{{}}",
      "--argv-json", JSON.stringify(argv),
    ], {{ stdio: ["pipe", "pipe", "pipe"], env: process.env }});
    const stdoutChunks = [];
    const stderrChunks = [];
    let aborted = false;
    const signal = params.options?.signal;
    const onAbort = () => {{
      aborted = true;
      try {{ child.kill("SIGTERM"); }} catch {{}}
    }};
    if (signal) {{
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, {{ once: true }});
    }}
    child.stdout.on("data", (data) => stdoutChunks.push(data));
    child.stderr.on("data", (data) => stderrChunks.push(data));
    child.on("error", reject);
    child.on("close", (code) => {{
      if (signal) signal.removeEventListener("abort", onAbort);
      resolve({{
        stdout: Buffer.concat(stdoutChunks),
        stderr: Buffer.concat(stderrChunks),
        code: code ?? (aborted ? 143 : 1),
      }});
    }});
    if (params.options?.stdin !== undefined) child.stdin.write(params.options.stdin);
    child.stdin.end();
  }});
}};
/* AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_HELPER_END */
'''


def _patch_fs_bridge(root: Path) -> list[str]:
    changed: list[str] = []
    candidates = sorted((root / "dist").glob("fs-bridge-*.js"))
    targets = [
        path
        for path in candidates
        if "readPinnedFile" in _read(path)
        and "runCheckedCommand" in _read(path)
        and "runCommand" in _read(path)
    ]
    if not targets:
        raise RuntimeError("v2026.9.2 fs bridge anchor not found")

    run_pattern = re.compile(r"(async\s+runCommand\(script,\s*options\s*=\s*\{\}\)\s*\{\s*)")
    write_pattern = re.compile(
        r"(async\s+writeFile\(params\)\s*\{\s*"
        r"const\s+target\s*=\s*this\.resolveResolvedPath\(params\);\s*"
        r"this\.ensureWriteAccess\(target,\s*[\"']write files[\"']\);\s*)"
    )
    read_pattern = re.compile(
        r"(async\s+readPinnedFile\(target,\s*maxBytes\)\s*\{\s*)"
    )
    checked_pattern = re.compile(r"(async\s+runCheckedCommand\(plan\)\s*\{\s*)")

    for path in targets:
        text = _read(path)
        original = text

        if FS_HELPER_MARKER not in text:
            text = 'import { spawn as __abSpawn } from "node:child_process";\n' + text
            helper_anchor = re.search(r"(?:function|const)\s+createSandboxFsBridge\b", text)
            if not helper_anchor:
                raise RuntimeError(f"fs helper insertion anchor missing in {path}")
            index = helper_anchor.start()
            text = text[:index] + _fs_helper_source() + "\n" + text[index:]

        if FS_RUN_MARKER not in text:
            insert = f'''/* {FS_RUN_MARKER} */
    if (process.env.OPENCLAW_EXEC_BRIDGE) {{
      return await __abFsBridgeRun({{ sandbox: this.sandbox, script, options }});
    }}
    /* AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_RUN_END */
    '''
            text, count = run_pattern.subn(lambda m: m.group(1) + insert, text, count=1)
            if count != 1:
                raise RuntimeError(f"fs runCommand anchor missing/ambiguous in {path}")

        if FS_WRITE_MARKER not in text:
            insert = f'''/* {FS_WRITE_MARKER} */
    if (process.env.OPENCLAW_EXEC_BRIDGE) {{
      const buffer = Buffer.isBuffer(params.data)
        ? params.data
        : Buffer.from(params.data, params.encoding ?? "utf8");
      const script = params.mkdir !== false
        ? 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        : 'cat > "$1"';
      const result = await __abFsBridgeRun({{
        sandbox: this.sandbox,
        script,
        options: {{ args: [target.containerPath], stdin: buffer, signal: params.signal }},
      }});
      if (result.code !== 0) {{
        throw new Error(`fs write via bridge failed (code ${{result.code}}): ${{result.stderr.toString("utf8")}}`);
      }}
      return;
    }}
    /* AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_WRITE_END */
    '''
            text, count = write_pattern.subn(lambda m: m.group(1) + insert, text, count=1)
            if count != 1:
                raise RuntimeError(f"fs writeFile anchor missing/ambiguous in {path}")

        if FS_READ_MARKER not in text:
            insert = f'''/* {FS_READ_MARKER} */
    if (process.env.OPENCLAW_EXEC_BRIDGE) {{
      const result = await __abFsBridgeRun({{
        sandbox: this.sandbox,
        script: 'cat "$1"',
        options: {{ args: [target.containerPath] }},
      }});
      if (result.code !== 0) {{
        throw new Error(`fs read via bridge failed (code ${{result.code}}): ${{result.stderr.toString("utf8")}}`);
      }}
      if (maxBytes !== undefined) {{
        if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {{
          throw new RangeError("maxBytes must be a non-negative safe integer");
        }}
        if (result.stdout.length > maxBytes) throw new RangeError(`File exceeds ${{maxBytes}} bytes`);
      }}
      return result.stdout;
    }}
    /* AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_READ_END */
    '''
            text, count = read_pattern.subn(lambda m: m.group(1) + insert, text, count=1)
            if count != 1:
                raise RuntimeError(f"fs readPinnedFile anchor missing/ambiguous in {path}")

        if FS_CHECK_MARKER not in text:
            insert = f'''/* {FS_CHECK_MARKER} */
    if (process.env.OPENCLAW_EXEC_BRIDGE) {{
      return await this.runCommand(plan.script, {{
        args: plan.args,
        stdin: plan.stdin,
        allowFailure: plan.allowFailure,
        signal: plan.signal,
      }});
    }}
    /* AB_HARNESS_OPENCLAW_V2026_9_2_FS_BRIDGE_CHECK_END */
    '''
            text, count = checked_pattern.subn(lambda m: m.group(1) + insert, text, count=1)
            if count != 1:
                raise RuntimeError(f"fs runCheckedCommand anchor missing/ambiguous in {path}")

        required = (FS_HELPER_MARKER, FS_RUN_MARKER, FS_WRITE_MARKER, FS_READ_MARKER, FS_CHECK_MARKER)
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise RuntimeError(f"incomplete fs bridge patch in {path}: {missing}")
        if text != original:
            _write(path, text)
            changed.append(str(path.relative_to(root)))
    return changed


def _patch_pricing(root: Path) -> dict[str, Any]:
    openrouter = 0
    litellm = 0
    changed: list[str] = []
    for path in sorted(set(_js_files(root))):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        updated = text
        count = updated.count(OPENROUTER_PRICING_URL)
        if count:
            openrouter += count
            updated = updated.replace(OPENROUTER_PRICING_URL, OPENROUTER_OFFLINE_URL)
        count = updated.count(LITELLM_PRICING_URL)
        if count:
            litellm += count
            updated = updated.replace(LITELLM_PRICING_URL, LITELLM_OFFLINE_URL)
        if updated != text:
            _write(path, updated)
            changed.append(str(path.relative_to(root)))

    remaining_openrouter = 0
    remaining_litellm = 0
    for path in sorted(set(_js_files(root))):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        remaining_openrouter += text.count(OPENROUTER_PRICING_URL)
        remaining_litellm += text.count(LITELLM_PRICING_URL)
    if remaining_openrouter or remaining_litellm:
        raise RuntimeError(
            "public pricing endpoints remain after v2026.9.2 hardening: "
            f"openrouter={remaining_openrouter} litellm={remaining_litellm}"
        )
    return {
        "files_changed": changed,
        "openrouter_pricing_url_replacements": openrouter,
        "litellm_pricing_url_replacements": litellm,
        "pricing_mode": "offline_data_urls_if_present",
    }


def patch_runtime_tree(root: Path) -> dict[str, Any]:
    root = Path(root)
    version = _runtime_version(root)
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"expected OpenClaw {EXPECTED_VERSION}, found {version or 'unknown'}")

    exec_files = _patch_exec_bridge(root)
    sandbox_files = _patch_sandbox_bypass(root)
    shared_workspace_spawn_files = _patch_shared_workspace_spawns(root)
    fs_files = _patch_fs_bridge(root)
    pricing = _patch_pricing(root)

    status = {
        "upstream_version": version,
        "patch_strategy": "v2026.9.2 semantic dist transforms",
        "native_loop_detection": True,
        "native_otel": True,
        "legacy_post_compaction_backport_applied": False,
        "legacy_model_usage_patches_applied": False,
        "legacy_tool_call_patches_applied": False,
        "legacy_sessions_patch_timeout_applied": False,
        "raw_response_patch_applied": False,
        "exec_bridge_files_changed": exec_files,
        "sandbox_bypass_files_changed": sandbox_files,
        "shared_workspace_spawn_files_changed": shared_workspace_spawn_files,
        "fs_bridge_files_changed": fs_files,
        **pricing,
    }
    return status


def verify_runtime_tree(root: Path) -> dict[str, Any]:
    root = Path(root)
    version = _runtime_version(root)
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"expected OpenClaw {EXPECTED_VERSION}, found {version or 'unknown'}")

    checks = {
        "exec_bridge": ("dist/bash-tools.exec-runtime-*.js", EXEC_MARKER),
        "sandbox_bypass": ("dist/docker-*.js", SANDBOX_MARKER),
        "shared_workspace_spawn": (
            "dist/sessions-spawn-tool-*.js",
            SHARED_WORKSPACE_SPAWN_MARKER,
        ),
        "fs_helper": ("dist/fs-bridge-*.js", FS_HELPER_MARKER),
        "fs_run": ("dist/fs-bridge-*.js", FS_RUN_MARKER),
        "fs_write": ("dist/fs-bridge-*.js", FS_WRITE_MARKER),
        "fs_read": ("dist/fs-bridge-*.js", FS_READ_MARKER),
        "fs_check": ("dist/fs-bridge-*.js", FS_CHECK_MARKER),
    }
    found: dict[str, list[str]] = {}
    for name, (glob, marker) in checks.items():
        paths = []
        for path in root.glob(glob):
            try:
                if marker in _read(path):
                    paths.append(str(path.relative_to(root)))
            except (OSError, UnicodeDecodeError):
                continue
        if not paths:
            raise RuntimeError(f"missing required v2026.9.2 marker: {name}")
        found[name] = paths

    return {
        "upstream_version": version,
        "markers": found,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/app")
    parser.add_argument("--status-out", default="")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    status = verify_runtime_tree(root) if args.verify_only else patch_runtime_tree(root)
    if not args.verify_only:
        status["verification"] = verify_runtime_tree(root)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    if args.status_out:
        out = Path(args.status_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
