#!/usr/bin/env python3
"""Agent_Bench semantic patches for the pinned OpenClaw v2026.9.2 runtime.

Unlike the legacy v2026.4.22 MANIFEST patches, this patcher targets stable
runtime semantics rather than content-hashed bundle filenames.  It intentionally
keeps the v2026.9.2 customization surface small:

* route sandbox bash exec through OPENCLAW_EXEC_BRIDGE;
* bypass OpenClaw-owned Docker lifecycle when the harness owns the sandbox;
* route sandbox filesystem operations through the same exec bridge;
* keep Roadmap incomplete post-tool turns in the tools-enabled outer loop;
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
SETTLED_CONTINUATION_POLICY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_POLICY_BEGIN"
)
SETTLED_CONTINUATION_INSTRUCTION_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_INSTRUCTION_BEGIN"
)
SETTLED_CONTINUATION_RETRY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_RETRY_BEGIN"
)
SETTLED_CONTINUATION_LENGTH_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_LENGTH_BEGIN"
)
NONEMPTY_LENGTH_ELIGIBILITY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_NONEMPTY_LENGTH_ELIGIBILITY_BEGIN"
)
CURRENT_ASSISTANT_EMPTY_STOP_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_V2_BEGIN"
)
LEGACY_CURRENT_ASSISTANT_EMPTY_STOP_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_BEGIN"
)
LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_TERMINAL_PAYLOAD_V2_BEGIN"
)
CONTINUATION_DECISION_DIAGNOSTIC_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTINUATION_DECISION_DIAGNOSTIC_V1_BEGIN"
)
LEGACY_RUNTIME_OWNER_CONTINUATION_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V1_BEGIN"
)
RUNTIME_OWNER_CONTINUATION_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_BEGIN"
)
IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_BEGIN"
)
BOUNDED_LENGTH_CONTINUATION_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_BOUNDED_NO_PROGRESS_LENGTH_V1_BEGIN"
)
LENGTH_SEMANTIC_PROGRESS_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_LENGTH_SEMANTIC_PROGRESS_V2_BEGIN"
)
SETTLED_CONTINUATION_TOOL_ERROR_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_TOOL_ERROR_BEGIN"
)
EMPTY_STOP_RETRY_LIMIT_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_EMPTY_STOP_RETRY_LIMIT_BEGIN"
)
LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_BEGIN"
)
LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_V2_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_BEGIN"
)
CONSECUTIVE_EMPTY_STOP_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_BEGIN"
)
LOOP_VETO_RECOVERY_PROBE_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_LENGTH_RECOVERY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_LENGTH_RECOVERY_V1_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V2_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_BEGIN"
)
LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_BEGIN"
)
CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_BEGIN"
)
LEGACY_COMPACTION_LENGTH_STREAK_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_COMPACTION_LENGTH_STREAK_RESET_V1_BEGIN"
)
LEGACY_COMPACTION_RETRY_STREAK_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_COMPACTION_RETRY_STREAK_RESET_V2_BEGIN"
)
LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V3_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V3_BEGIN"
)
LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V4_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V4_BEGIN"
)
CONTEXT_RECOVERY_STREAK_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN"
)
TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_TRUNCATE_ONLY_NATIVE_RECOVERY_V1_BEGIN"
)

SETTLED_TOOL_POLICY_ENV = "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY"
EMPTY_STOP_RETRIES_ENV = "OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES"
LENGTH_NO_PROGRESS_RETRIES_ENV = (
    "OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES"
)
LOOP_VETO_RECOVERY_PROBE_ENV = (
    "OPENCLAW_AGENT_BENCH_LOOP_VETO_RECOVERY_PROBE"
)
ROADMAP_CONTINUATION_INSTRUCTION = (
    "The previous assistant turn ended before the assigned task was complete. "
    "Continue the assigned task from the current state. "
    "Use tools for any remaining implementation or validation work. "
    "Only produce the final answer when the task is complete. "
    "Do not restart from scratch or repeat completed work."
)
_UPSTREAM_EMPTY_RESPONSE_RETRY_INSTRUCTION = (
    "The previous attempt did not produce a user-visible answer. Continue "
    "from the current state and produce the visible answer now. Do not "
    "restart from scratch."
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


def _has_js_lexical_binding(text: str, name: str) -> bool:
    escaped = re.escape(name)
    if re.search(rf"function\s+{escaped}\s*\(", text):
        return True
    if re.search(rf"(?:const|let|var)\s+{escaped}\s*=", text):
        return True
    if re.search(rf"import\s+\*\s+as\s+{escaped}\s+from", text):
        return True
    if re.search(rf"import\s+{escaped}\s*(?:,|from)", text):
        return True
    for body in re.findall(
        r"import\s*\{(?P<body>[^}]*)\}\s*from",
        text,
        flags=re.DOTALL,
    ):
        for item in body.split(","):
            declaration = re.sub(r"^\s*type\s+", "", item).strip()
            if not declaration:
                continue
            parts = re.split(r"\s+as\s+", declaration)
            local_name = parts[-1].strip()
            if local_name == name:
                return True
    return False


_SETTLED_CONTINUATION_RESOLVER_BINDINGS = (
    "classifyAssistantTurn",
    "resolveSettledToolBatchEvidence",
    "hasCompletedMessagingToolDeliveryEvidence",
    "hasAsyncActivity",
)


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


def _patch_implicit_child_handoff_ownership(root: Path) -> list[str]:
    """Persist native child handoff ownership on the settled attempt."""
    changed: list[str] = []
    error_text = "accepted continuation children were not durably registered"
    ident = r"[A-Za-z_$][\w$]*"
    anchor = re.compile(
        rf"(?P<prefix>"
        rf"if\s*\(\s*shouldContinueInteractiveAcceptedSessionSpawns\s*\(\s*\{{\s*"
        rf"attempt\s*:\s*(?P<result>{ident})\s*,\s*"
        rf"run\s*:\s*{ident}\s*,?\s*\}}\s*\)\s*\)\s*\{{\s*"
        rf"if\s*\(\s*markRequesterTurnYielded\s*\(\s*\{{.*?\}}\s*\)\s*"
        rf"===\s*0\s*\)\s*throw\s+(?:new\s+)?Error\s*\(\s*"
        rf"(?P<quote>[\"'`]){re.escape(error_text)}(?P=quote)\s*\)\s*;?"
        rf")",
        re.DOTALL,
    )
    patched_block = re.compile(
        re.escape(IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER)
        + r".*?[A-Za-z_$][\w$]*\.runtimeContinuationStarted\s*=\s*true;"
        + ".*?AB_HARNESS_OPENCLAW_V2026_9_2_"
        + "IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_END",
        re.DOTALL,
    )
    dist_files = sorted(_js_files(root / "dist"))
    targets = [
        path for path in dist_files
        if error_text in _read(path)
        or IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER in _read(path)
    ]
    if not targets:
        raise RuntimeError(
            "v2026.9.2 implicit child handoff ownership anchor not found"
        )

    def replacement(match: re.Match[str]) -> str:
        result = match.group("result")
        return (
            match.group("prefix")
            + f"\n\t\t\t/* {IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER} */"
            + f"\n\t\t\t{result}.runtimeContinuationStarted = true;"
            + "\n\t\t\t/* AB_HARNESS_OPENCLAW_V2026_9_2_"
            + "IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_END */"
        )

    for path in targets:
        text = _read(path)
        call_count = text.count(error_text)
        marker_count = text.count(IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER)
        if call_count <= 0:
            raise RuntimeError(
                f"implicit child handoff ownership call site missing in {path}"
            )
        if marker_count:
            block_count = len(patched_block.findall(text))
            if marker_count != call_count or block_count != call_count:
                raise RuntimeError(
                    "incomplete implicit child handoff ownership patch in "
                    f"{path}: calls={call_count} markers={marker_count} "
                    f"blocks={block_count}"
                )
            continue
        matches = list(anchor.finditer(text))
        if len(matches) != call_count:
            raise RuntimeError(
                "implicit child handoff ownership anchor missing/ambiguous in "
                f"{path}: calls={call_count} anchors={len(matches)}"
            )
        updated, count = anchor.subn(replacement, text)
        if count != call_count:
            raise RuntimeError(
                f"implicit child handoff ownership patch failed in {path}"
            )
        _write(path, updated)
        changed.append(str(path.relative_to(root)))
    return changed


def _patch_settled_tool_continuation(root: Path) -> dict[str, list[str]]:
    """Route Roadmap incomplete post-tool turns through the normal retry.

    Upstream v2026.9.2 runs a tools-disabled isolated finalization when a
    settled tool batch is followed by an empty assistant stop. It can also
    finalize a length-truncated assistant turn that still promises more work.
    Under the explicit Agent_Bench policy, the existing tools-enabled retry
    path remains eligible; ordinary length retries use an independent semantic
    no-progress budget, while context-pressure length stops use native recovery.
    """
    changed: set[Path] = set()
    dist_files = sorted(_js_files(root / "dist"))

    policy_anchor = re.compile(
        r"(function\s+resolveSettledTurnFinalizationRequest\(input\)\s*\{\s*)"
    )
    policy_targets = [
        path for path in dist_files
        if policy_anchor.search(_read(path))
        or SETTLED_CONTINUATION_POLICY_MARKER in _read(path)
    ]
    if not policy_targets:
        raise RuntimeError("v2026.9.2 settled-turn finalization anchor not found")

    policy_insert = f'''/* {SETTLED_CONTINUATION_POLICY_MARKER} */
\tif (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue") return null;
\t/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_POLICY_END */
\t'''
    for path in policy_targets:
        text = _read(path)
        if SETTLED_CONTINUATION_POLICY_MARKER not in text:
            if len(policy_anchor.findall(text)) != 1:
                raise RuntimeError(
                    f"settled-turn policy anchor missing/ambiguous in {path}"
                )
            text, count = policy_anchor.subn(
                lambda match: match.group(1) + policy_insert,
                text,
                count=1,
            )
            if count != 1:
                raise RuntimeError(
                    f"settled-turn policy anchor missing/ambiguous in {path}"
                )
            _write(path, text)
            changed.add(path)

    instruction_anchor = re.compile(
        r'const\s+EMPTY_RESPONSE_RETRY_INSTRUCTION\s*=\s*'
        + re.escape(json.dumps(_UPSTREAM_EMPTY_RESPONSE_RETRY_INSTRUCTION))
        + r";"
    )
    instruction_targets = [
        path for path in dist_files
        if instruction_anchor.search(_read(path))
        or SETTLED_CONTINUATION_INSTRUCTION_MARKER in _read(path)
    ]
    if not instruction_targets:
        raise RuntimeError("v2026.9.2 empty-response instruction anchor not found")

    instruction_replacement = f'''/* {SETTLED_CONTINUATION_INSTRUCTION_MARKER} */
const EMPTY_RESPONSE_RETRY_INSTRUCTION = process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
\t? {json.dumps(ROADMAP_CONTINUATION_INSTRUCTION)}
\t: {json.dumps(_UPSTREAM_EMPTY_RESPONSE_RETRY_INSTRUCTION)};
/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_INSTRUCTION_END */'''
    for path in instruction_targets:
        text = _read(path)
        if SETTLED_CONTINUATION_INSTRUCTION_MARKER not in text:
            if len(instruction_anchor.findall(text)) != 1:
                raise RuntimeError(
                    f"empty-response instruction anchor missing/ambiguous in {path}"
                )
            text, count = instruction_anchor.subn(
                instruction_replacement,
                text,
                count=1,
            )
            if count != 1:
                raise RuntimeError(
                    f"empty-response instruction anchor missing/ambiguous in {path}"
                )
            _write(path, text)
            changed.add(path)

    retry_eligibility_anchor = re.compile(
        r"(function\s+resolveEmptyResponseRetryInstruction\("
        r"(?P<param>[A-Za-z_$][\w$]*)\)\s*\{\s*)"
    )
    retry_eligibility_targets = [
        path for path in dist_files
        if retry_eligibility_anchor.search(_read(path))
        or SETTLED_CONTINUATION_RETRY_MARKER in _read(path)
        or SETTLED_CONTINUATION_LENGTH_MARKER in _read(path)
        or SETTLED_CONTINUATION_TOOL_ERROR_MARKER in _read(path)
    ]
    if not retry_eligibility_targets:
        raise RuntimeError(
            "v2026.9.2 empty-response retry eligibility anchor not found"
        )
    for path in retry_eligibility_targets:
        text = _read(path)
        missing = [
            name
            for name in _SETTLED_CONTINUATION_RESOLVER_BINDINGS
            if not _has_js_lexical_binding(text, name)
        ]
        if missing:
            raise RuntimeError(
                "settled-continuation resolver lexical bindings missing in "
                f"{path}: {', '.join(missing)}"
            )

    stale_current_empty_targets = [
        path for path in dist_files
        if LEGACY_CURRENT_ASSISTANT_EMPTY_STOP_MARKER in _read(path)
        or LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER in _read(path)
        or "hasCurrentTerminalPayload" in _read(path)
    ]
    if stale_current_empty_targets:
        raise RuntimeError(
            "current-empty patch uses the obsolete cross-bundle payload "
            "contract; "
            "rebuild from the clean v2026.9.2 base: "
            + ", ".join(str(path) for path in stale_current_empty_targets)
        )

    def retry_eligibility_replacement(match: re.Match[str]) -> str:
        param = match.group("param")
        return match.group(1) + f'''/* {SETTLED_CONTINUATION_RETRY_MARKER} */
	/* {SETTLED_CONTINUATION_LENGTH_MARKER} */
	/* {NONEMPTY_LENGTH_ELIGIBILITY_MARKER} */
	const __abAssistantState = classifyAssistantTurn({param});
	/* {CURRENT_ASSISTANT_EMPTY_STOP_MARKER} */
	const __abCurrentAssistantContent = __abAssistantState.assistant?.content;
	const __abCurrentAssistantVisibleText = (
		typeof __abCurrentAssistantContent === "string"
			? __abCurrentAssistantContent
			: Array.isArray(__abCurrentAssistantContent)
				? __abCurrentAssistantContent
					.filter((item) => item?.type === "text" && typeof item.text === "string")
					.map((item) => item.text)
					.join("")
				: ""
	).trim();
	const __abCurrentAssistantHasToolCall =
		Array.isArray(__abCurrentAssistantContent)
		&& __abCurrentAssistantContent.some(
			(item) => item?.type === "toolCall" || item?.type === "toolUse"
		);
	const __abCurrentAssistantHasPayload =
		Array.isArray(__abCurrentAssistantContent)
		&& __abCurrentAssistantContent.some((item) => {{
			if (item === null || item === void 0) return false;
			if (typeof item !== "object") return String(item).trim().length > 0;
			return item.type !== "text"
				&& item.type !== "thinking"
				&& item.type !== "reasoning"
				&& item.type !== "toolCall"
				&& item.type !== "toolUse";
		}});
	const __abCurrentAssistantEmptyShape =
		__abAssistantState.assistant?.stopReason === "stop"
		&& !__abAssistantState.reasoningOnly
		&& !__abCurrentAssistantVisibleText
		&& !__abCurrentAssistantHasToolCall;
	const __abCurrentAssistantEmptyStop =
		__abCurrentAssistantEmptyShape
		&& !__abCurrentAssistantHasPayload;
	/* AB_HARNESS_OPENCLAW_V2026_9_2_CURRENT_ASSISTANT_EMPTY_STOP_V2_END */
	const __abExplicitDone =
		/(?:^|\\n)\\s*DONE\\s*(?:\\n|$)/.test(__abAssistantState.visibleText);
	const __abIncompleteLength =
		__abAssistantState.assistant?.stopReason === "length"
		&& !__abAssistantState.reasoningOnly
		&& !__abExplicitDone;
	const __abRoadmapContinuationEligible =
		__abAssistantState.emptyResponse
		|| __abCurrentAssistantEmptyStop
		|| __abIncompleteLength;
	/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_LENGTH_END */
	const __abSettledToolEvidence = resolveSettledToolBatchEvidence({param}.attempt);
	const __abCompletedMessagingDelivery =
		hasCompletedMessagingToolDeliveryEvidence({param}.attempt);
	const __abAsyncActivity = hasAsyncActivity({param}.attempt.toolMetas);
	/* {RUNTIME_OWNER_CONTINUATION_MARKER} */
	const __abActiveToolWork =
		Number({param}.attempt?.itemLifecycle?.activeCount ?? 0) > 0;
	const __abRuntimeContinuationOwner =
		{param}.attempt?.runtimeContinuationStarted === true;
	const __abContinuationAllowed =
		process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
		&& !{param}.aborted
		&& !{param}.timedOut
		&& {param}.attempt?.terminal?.kind !== "failed"
		&& {param}.attempt?.terminal?.kind !== "timeout"
		&& !{param}.attempt.clientToolCalls
		&& !{param}.attempt.yieldDetected
		&& !{param}.attempt.didSendDeterministicApprovalPrompt
		&& !{param}.attempt.hasToolMediaBlockReply
		&& !__abCompletedMessagingDelivery
		&& !(__abCurrentAssistantEmptyShape && __abCurrentAssistantHasPayload)
		&& __abRoadmapContinuationEligible
		&& !__abActiveToolWork
		&& !__abRuntimeContinuationOwner
		/* {SETTLED_CONTINUATION_TOOL_ERROR_MARKER} */
		/* Settled tool errors are recoverable model feedback, not runtime fatality. */
		/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_TOOL_ERROR_END */
		&& !__abSettledToolEvidence.intentionalTermination
		&& !__abSettledToolEvidence.parkedCodeModeRun
		&& !__abAsyncActivity;
	/* {CONTINUATION_DECISION_DIAGNOSTIC_MARKER} */
	const __abContinuationCandidate = __abCurrentAssistantEmptyShape
		? "current-empty"
		: __abIncompleteLength
			? "length"
			: __abAssistantState.emptyResponse
				? "accumulated-empty"
				: "";
	if (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
		&& __abContinuationCandidate) {{
		const __abDenyReasons = [];
		if ({param}.aborted) __abDenyReasons.push("aborted");
		if ({param}.timedOut) __abDenyReasons.push("timed_out");
		if ({param}.attempt?.terminal?.kind === "failed"
			|| {param}.attempt?.terminal?.kind === "timeout") {{
			__abDenyReasons.push("terminal_" + {param}.attempt.terminal.kind);
		}}
		if ({param}.attempt.clientToolCalls) __abDenyReasons.push("client_tool_calls");
		if ({param}.attempt.yieldDetected) __abDenyReasons.push("yield_owner");
		if ({param}.attempt.didSendDeterministicApprovalPrompt) __abDenyReasons.push("approval_prompt");
		if ({param}.attempt.hasToolMediaBlockReply) __abDenyReasons.push("tool_media");
		if (__abCompletedMessagingDelivery) __abDenyReasons.push("messaging_delivery");
		if (__abContinuationCandidate === "current-empty"
			&& __abCurrentAssistantHasPayload) __abDenyReasons.push("current_payload");
		if (!__abRoadmapContinuationEligible) __abDenyReasons.push("not_eligible");
		if (__abActiveToolWork) __abDenyReasons.push("active_tool_work");
		if (__abRuntimeContinuationOwner) __abDenyReasons.push("runtime_continuation_owner");
		if (__abSettledToolEvidence.intentionalTermination) __abDenyReasons.push("intentional_termination");
		if (__abSettledToolEvidence.parkedCodeModeRun) __abDenyReasons.push("parked_code_mode");
		if (__abAsyncActivity) __abDenyReasons.push("async_activity");
		const __abDecision = {{
			candidate: __abContinuationCandidate,
			decision: __abContinuationAllowed ? "continue" : "deny",
			deny_reason: __abDenyReasons,
			current_stop_reason: String(__abAssistantState.assistant?.stopReason ?? ""),
			current_visible_chars: __abCurrentAssistantVisibleText.length,
			current_has_tool_call: __abCurrentAssistantHasToolCall,
			current_has_payload: __abCurrentAssistantHasPayload,
			cumulative_payload_count: Number({param}.payloadCount ?? 0),
			active_tool_work: __abActiveToolWork,
			all_tools_settled: Boolean(__abSettledToolEvidence.allToolsProvenSettled),
			terminal_kind: String({param}.attempt?.terminal?.kind ?? ""),
			last_tool_error: Boolean({param}.attempt?.lastToolError),
		}};
		try {{
			Object.defineProperty(
				{param}.attempt,
				"__agentBenchContinuationDecision",
				{{ value: __abDecision, configurable: true, writable: true }},
			);
		}} catch {{
			console.warn("[agent-bench-continuation-decision] " + JSON.stringify({{
				...__abDecision,
				empty_retry_attempt: null,
				length_retry_attempt: null,
			}}));
		}}
	}}
	if (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
		&& __abRuntimeContinuationOwner
		&& __abRoadmapContinuationEligible) return null;
	/* AB_HARNESS_OPENCLAW_V2026_9_2_RUNTIME_OWNER_CONTINUATION_V2_END */
	/* AB_HARNESS_OPENCLAW_V2026_9_2_CONTINUATION_DECISION_DIAGNOSTIC_V1_END */
	if (__abContinuationAllowed) {{
		return EMPTY_RESPONSE_RETRY_INSTRUCTION;
	}}
	/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_RETRY_END */
	'''

    stale_retry_targets = [
        path for path in retry_eligibility_targets
        if SETTLED_CONTINUATION_RETRY_MARKER in _read(path)
        and (
            SETTLED_CONTINUATION_LENGTH_MARKER not in _read(path)
            or SETTLED_CONTINUATION_TOOL_ERROR_MARKER not in _read(path)
            or NONEMPTY_LENGTH_ELIGIBILITY_MARKER not in _read(path)
            or CURRENT_ASSISTANT_EMPTY_STOP_MARKER not in _read(path)
            or CONTINUATION_DECISION_DIAGNOSTIC_MARKER not in _read(path)
            or RUNTIME_OWNER_CONTINUATION_MARKER not in _read(path)
            or LEGACY_RUNTIME_OWNER_CONTINUATION_MARKER in _read(path)
        )
    ]
    if stale_retry_targets:
        raise RuntimeError(
            "settled continuation patch predates current-empty-stop, "
            "empty-length, or recoverable-tool-error support; "
            "rebuild from the clean v2026.9.2 base: "
            + ", ".join(str(path) for path in stale_retry_targets)
        )

    for path in retry_eligibility_targets:
        text = _read(path)
        if SETTLED_CONTINUATION_TOOL_ERROR_MARKER not in text:
            if len(retry_eligibility_anchor.findall(text)) != 1:
                raise RuntimeError(
                    "empty-response retry eligibility anchor "
                    f"missing/ambiguous in {path}"
                )
            text, count = retry_eligibility_anchor.subn(
                retry_eligibility_replacement,
                text,
                count=1,
            )
            if count != 1:
                raise RuntimeError(
                    "empty-response retry eligibility anchor "
                    f"missing/ambiguous in {path}"
                )
            _write(path, text)
            changed.add(path)

    ident = r"[A-Za-z_$][\w$]*"
    length_budget_readable_anchor = re.compile(
        rf'''(?P<indent>[ \t]*)if \(!(?P<reason>{ident}) && (?P<instruction>{ident}) && (?P<state>{ident})\.emptyResponseAttempts < (?P<input>{ident})\.maxEmptyResponseRetryAttempts\) \{{\s*
(?P=indent)[ \t]+(?P=state)\.emptyResponseAttempts \+= 1;\s*
(?P=indent)[ \t]+(?P=input)\.activateInternalPrompt\((?P=instruction)\);\s*
(?P=indent)[ \t]+(?P<logger>{ident})\.warn\(`empty response detected:[^`]*with visible-answer continuation`\);\s*
(?P=indent)[ \t]+return \{{ action: ["']retry["'] \}};\s*
(?P=indent)\}}'''
    )
    length_budget_minified_anchor = re.compile(
        rf'''if\(!(?P<reason>{ident})&&(?P<instruction>{ident})&&(?P<state>{ident})\.emptyResponseAttempts<(?P<input>{ident})\.maxEmptyResponseRetryAttempts\)return (?P=state)\.emptyResponseAttempts\+=1,(?P=input)\.activateInternalPrompt\((?P=instruction)\),(?P<logger>{ident})\.warn\(`empty response detected:[^`]*with visible-answer continuation`\),\{{action:`retry`\}};'''
    )
    length_budget_targets = [
        path for path in dist_files
        if length_budget_readable_anchor.search(_read(path))
        or length_budget_minified_anchor.search(_read(path))
        or BOUNDED_LENGTH_CONTINUATION_MARKER in _read(path)
    ]
    if not length_budget_targets:
        raise RuntimeError(
            "v2026.9.2 empty-response retry counter anchor not found"
        )

    def consecutive_reset_source(
        indent: str,
        input_name: str,
        state: str,
    ) -> str:
        return f'''{indent}/* {CONSECUTIVE_EMPTY_STOP_RESET_MARKER} */
{indent}if (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
{indent}\t&& {state}.emptyResponseAttempts > 0
{indent}\t&& !isEmbeddedRunTerminalAbort({input_name}.terminalState.outcome)
{indent}\t&& !isEmbeddedRunTerminalTimeout({input_name}.terminalState.outcome)
{indent}\t&& {input_name}.attempt?.terminal?.kind !== "failed"
{indent}\t&& {input_name}.attempt?.terminal?.kind !== "timeout") {{
{indent}\tconst __abProgressToolEvidence = resolveSettledToolBatchEvidence({input_name}.attempt);
{indent}\tconst __abBlockedToolLoopVeto = ({input_name}.attempt.messagesSnapshot ?? []).some(
{indent}\t\t(message) => message?.role === "toolResult"
{indent}\t\t\t&& message?.details?.status === "blocked"
{indent}\t\t\t&& message?.details?.deniedReason === "tool-loop"
{indent}\t);
{indent}\tconst __abControlActivity = Boolean(
{indent}\t\t{input_name}.attempt.clientToolCalls
{indent}\t\t|| {input_name}.attempt.yieldDetected
{indent}\t\t|| {input_name}.attempt.didSendDeterministicApprovalPrompt
{indent}\t\t|| {input_name}.attempt.heartbeatToolResponse
{indent}\t\t|| {input_name}.attempt.hasToolMediaBlockReply
{indent}\t\t|| {input_name}.attempt.didDeliverSourceReplyViaMessageTool
{indent}\t\t|| ({input_name}.attempt.messagingToolSourceReplyPayloads?.length ?? 0) > 0
{indent}\t\t|| hasMessagingToolDeliveryEvidence({input_name}.attempt)
{indent}\t\t|| {input_name}.attempt.runtimeContinuationStarted === true
{indent}\t\t|| ({input_name}.attempt.successfulCronAdds ?? 0) > 0
{indent}\t\t|| hasAsyncActivity({input_name}.attempt.toolMetas)
{indent}\t);
{indent}\tconst __abVisibleProgressText = String(
{indent}\t\t{input_name}.finalAssistantVisibleText ?? ""
{indent}\t).trim();
{indent}\tconst __abMeaningfulAssistantProgress = Boolean(
{indent}\t\t__abVisibleProgressText
{indent}\t\t&& __abVisibleProgressText !== "NO_REPLY"
{indent}\t\t&& __abVisibleProgressText !== "HEARTBEAT_OK"
{indent}\t);
{indent}\tconst __abRealSettledToolProgress = Boolean(
{indent}\t\t__abProgressToolEvidence.allToolsProvenSettled
{indent}\t\t&& ({input_name}.attempt.toolMetas?.length ?? 0) > 0
{indent}\t);
{indent}\tif (!__abBlockedToolLoopVeto
{indent}\t\t&& !__abControlActivity
{indent}\t\t&& (__abRealSettledToolProgress || __abMeaningfulAssistantProgress)) {{
{indent}\t\t{state}.emptyResponseAttempts = 0;
{indent}\t}}
{indent}}}
{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V3_END */'''

    required_reset_bindings = (
        "resolveSettledToolBatchEvidence",
        "hasMessagingToolDeliveryEvidence",
        "hasAsyncActivity",
        "isEmbeddedRunTerminalAbort",
        "isEmbeddedRunTerminalTimeout",
    )

    def require_reset_bindings(text: str, path: Path) -> None:
        missing = [
            name for name in required_reset_bindings
            if not _has_js_lexical_binding(text, name)
        ]
        if missing:
            raise RuntimeError(
                "consecutive-empty reset lexical bindings missing in "
                f"{path}: {', '.join(missing)}"
            )

    def length_budget_replacement(match: re.Match[str]) -> str:
        indent = match.groupdict().get("indent") or ""
        reason = match.group("reason")
        instruction = match.group("instruction")
        state = match.group("state")
        input_name = match.group("input")
        logger = match.group("logger")
        reset_source = consecutive_reset_source(indent, input_name, state)
        return f'''{indent}/* {BOUNDED_LENGTH_CONTINUATION_MARKER} */
{reset_source}
{indent}const __abLengthRetryLimit = (() => {{
{indent}\tconst raw = process.env.{LENGTH_NO_PROGRESS_RETRIES_ENV};
{indent}\tif (raw === void 0 || raw === "") return 4;
{indent}\tconst parsed = Number.parseInt(raw, 10);
{indent}\treturn Number.isSafeInteger(parsed) && parsed >= 0
{indent}\t\t? Math.min(parsed, 16)
{indent}\t\t: 4;
{indent}}})();
{indent}{state}.lengthNoProgressAttempts ??= 0;
{indent}if (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
{indent}\t&& !isEmbeddedRunTerminalAbort({input_name}.terminalState.outcome)
{indent}\t&& !isEmbeddedRunTerminalTimeout({input_name}.terminalState.outcome)
{indent}\t&& {input_name}.attempt?.terminal?.kind !== "failed"
{indent}\t&& {input_name}.attempt?.terminal?.kind !== "timeout") {{
{indent}\t/* {LENGTH_SEMANTIC_PROGRESS_MARKER} */
{indent}\tconst __abLengthVolatileResultKeys = new Set([
{indent}\t\t"toolCallId", "runId", "sessionId", "timestamp", "timestampMs",
{indent}\t\t"startedAt", "completedAt", "endedAt", "durationMs",
{indent}\t\t"elapsedMs", "pid",
{indent}\t]);
{indent}\tconst __abNormalizeLengthFingerprintValue = (value, omitVolatile) => {{
{indent}\t\tif (typeof value === "string") return value.replace(/\\r\\n?/g, "\\n").trimEnd();
{indent}\t\tif (value === null || typeof value !== "object") return value;
{indent}\t\tif (Array.isArray(value)) return value.map(
{indent}\t\t\t(item) => __abNormalizeLengthFingerprintValue(item, omitVolatile)
{indent}\t\t);
{indent}\t\treturn Object.fromEntries(Object.keys(value).sort()
{indent}\t\t\t.filter((key) => !omitVolatile || !__abLengthVolatileResultKeys.has(key))
{indent}\t\t\t.map((key) => [
{indent}\t\t\t\tkey,
{indent}\t\t\t\t__abNormalizeLengthFingerprintValue(value[key], omitVolatile),
{indent}\t\t\t]));
{indent}\t}};
{indent}\tconst __abHashLengthFingerprint = (value) => {{
{indent}\t\tlet hash = 2166136261;
{indent}\t\tfor (let index = 0; index < value.length; index += 1) {{
{indent}\t\t\thash ^= value.charCodeAt(index);
{indent}\t\t\thash = Math.imul(hash, 16777619);
{indent}\t\t}}
{indent}\t\treturn `${{value.length}}:${{(hash >>> 0).toString(16).padStart(8, "0")}}`;
{indent}\t}};
{indent}\tconst __abLengthEvidence = resolveSettledToolBatchEvidence({input_name}.attempt);
{indent}\tconst __abLengthToolCalls = Array.isArray(__abLengthEvidence.assistant?.content)
{indent}\t\t? __abLengthEvidence.assistant.content.filter(
{indent}\t\t\t(item) => (item?.type === "toolCall" || item?.type === "toolUse")
{indent}\t\t\t\t&& typeof item?.id === "string" && item.id.length > 0
{indent}\t\t)
{indent}\t\t: [];
{indent}\tconst __abLengthResults = {input_name}.attempt.messagesSnapshot ?? [];
{indent}\tconst __abLengthProgressEntries = __abLengthToolCalls.map((call) => {{
{indent}\t\tconst result = __abLengthResults.findLast(
{indent}\t\t\t(message) => message?.role === "toolResult"
{indent}\t\t\t\t&& message?.toolCallId === call.id
{indent}\t\t);
{indent}\t\tconst meta = ({input_name}.attempt.toolMetas ?? []).findLast(
{indent}\t\t\t(entry) => entry?.toolCallId === call.id
{indent}\t\t);
{indent}\t\tconst blocked = result?.details?.status === "blocked";
{indent}\t\tif (!result || result?.isError === true || meta?.isError === true
{indent}\t\t\t|| meta?.asyncStarted === true || blocked) return null;
{indent}\t\tconst semanticArguments = call.arguments ?? call.args ?? call.input ?? call.params ?? null;
{indent}\t\tconst semanticResult = {{
{indent}\t\t\tcontent: result.content ?? null,
{indent}\t\t\tdetails: result.details ?? null,
{indent}\t\t}};
{indent}\t\tconst semanticRecord = JSON.stringify({{
{indent}\t\t\ttool: String(call.name ?? ""),
{indent}\t\t\targuments: __abNormalizeLengthFingerprintValue(semanticArguments, false),
{indent}\t\t\tresult: __abNormalizeLengthFingerprintValue(semanticResult, true),
{indent}\t\t}});
{indent}\t\treturn __abHashLengthFingerprint(semanticRecord);
{indent}\t}});
{indent}\tconst __abLengthProgressSignature =
{indent}\t\t__abLengthEvidence.allToolsProvenSettled
{indent}\t\t&& __abLengthProgressEntries.length > 0
{indent}\t\t&& __abLengthProgressEntries.every(Boolean)
{indent}\t\t\t? __abLengthProgressEntries.join("|")
{indent}\t\t\t: "";
{indent}\t{state}.lengthProgressSignatures ??= [];
{indent}\tif (__abLengthProgressSignature
{indent}\t\t&& !{state}.lengthProgressSignatures.includes(__abLengthProgressSignature)) {{
{indent}\t\t{state}.lengthProgressSignatures.push(__abLengthProgressSignature);
{indent}\t\t{state}.lengthNoProgressAttempts = 0;
{indent}\t}}
{indent}\t/* AB_HARNESS_OPENCLAW_V2026_9_2_LENGTH_SEMANTIC_PROGRESS_V2_END */
{indent}}}
{indent}const __abLengthContinuation =
{indent}\tprocess.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
{indent}\t&& {input_name}.attemptAssistant?.stopReason === "length"
{indent}\t&& {instruction};
{indent}const __abContinuationDecision =
{indent}\t{input_name}.attempt?.__agentBenchContinuationDecision;
{indent}if (__abContinuationDecision) {{
{indent}\tconst __abDiagnosticIsLength =
{indent}\t\t__abContinuationDecision.candidate === "length";
{indent}\tconst __abDiagnosticBudgetAvailable = __abDiagnosticIsLength
{indent}\t\t? {state}.lengthNoProgressAttempts < __abLengthRetryLimit
{indent}\t\t: {state}.emptyResponseAttempts < {input_name}.maxEmptyResponseRetryAttempts;
{indent}\tconst __abDiagnosticWillRetry = Boolean(
{indent}\t\t!{reason} && {instruction} && __abDiagnosticBudgetAvailable
{indent}\t);
{indent}\tconst __abDiagnosticDenyReasons = [
{indent}\t\t...(__abContinuationDecision.deny_reason ?? []),
{indent}\t];
{indent}\tif ({reason}) __abDiagnosticDenyReasons.push("reasoning_only_retry_precedence");
{indent}\tif (!{instruction} && __abDiagnosticDenyReasons.length === 0) {{
{indent}\t\t__abDiagnosticDenyReasons.push("resolver_rejected");
{indent}\t}}
{indent}\tif (!__abDiagnosticBudgetAvailable) {{
{indent}\t\t__abDiagnosticDenyReasons.push("retry_budget_exhausted");
{indent}\t}}
{indent}\tconsole.warn("[agent-bench-continuation-decision] " + JSON.stringify({{
{indent}\t\t...__abContinuationDecision,
{indent}\t\tdecision: __abDiagnosticWillRetry ? "continue" : "deny",
{indent}\t\tdeny_reason: __abDiagnosticWillRetry ? [] : __abDiagnosticDenyReasons,
{indent}\t\tempty_retry_attempt: {state}.emptyResponseAttempts
{indent}\t\t\t+ (__abDiagnosticIsLength ? 0 : 1),
{indent}\t\tlength_retry_attempt: {state}.lengthNoProgressAttempts
{indent}\t\t\t+ (__abDiagnosticIsLength ? 1 : 0),
{indent}\t}}));
{indent}\ttry {{
{indent}\t\tdelete {input_name}.attempt.__agentBenchContinuationDecision;
{indent}\t}} catch {{}}
{indent}}}
{indent}if (!{reason} && {instruction}
{indent}\t&& ((__abLengthContinuation
{indent}\t\t&& {state}.lengthNoProgressAttempts < __abLengthRetryLimit)
{indent}\t\t|| (!__abLengthContinuation
{indent}\t\t\t&& {state}.emptyResponseAttempts < {input_name}.maxEmptyResponseRetryAttempts))) {{
{indent}\t{input_name}.activateInternalPrompt({instruction});
{indent}\tif (__abLengthContinuation) {{
{indent}\t\t{state}.lengthNoProgressAttempts += 1;
{indent}\t\t{logger}.warn(`length-truncated response detected: runId=${{{input_name}.runParams.runId}} sessionId=${{{input_name}.runParams.sessionId}} provider=${{{input_name}.activeErrorContext.provider}}/${{{input_name}.activeErrorContext.model}} — retrying ${{{state}.lengthNoProgressAttempts}}/${{__abLengthRetryLimit}} without new settled-tool progress`);
{indent}\t}} else {{
{indent}\t\t{state}.emptyResponseAttempts += 1;
{indent}\t\t{logger}.warn(`empty response detected: runId=${{{input_name}.runParams.runId}} sessionId=${{{input_name}.runParams.sessionId}} provider=${{{input_name}.activeErrorContext.provider}}/${{{input_name}.activeErrorContext.model}} — retrying ${{{state}.emptyResponseAttempts}}/${{{input_name}.maxEmptyResponseRetryAttempts}} with visible-answer continuation`);
{indent}\t}}
{indent}\treturn {{ action: "retry" }};
{indent}}}
{indent}if (!{reason} && __abLengthContinuation
{indent}\t&& {state}.lengthNoProgressAttempts >= __abLengthRetryLimit) {{
{indent}\t{logger}.warn(`length-truncated response retry budget exhausted: runId=${{{input_name}.runParams.runId}} sessionId=${{{input_name}.runParams.sessionId}} attempts=${{{state}.lengthNoProgressAttempts}} limit=${{__abLengthRetryLimit}}`);
{indent}}}
{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_BOUNDED_NO_PROGRESS_LENGTH_V1_END */'''

    stale_unbounded_targets = [
        path for path in dist_files
        if "AB_HARNESS_OPENCLAW_V2026_9_2_UNBOUNDED_LENGTH_CONTINUATION_BEGIN"
        in _read(path)
    ]
    if stale_unbounded_targets:
        raise RuntimeError(
            "stale unbounded-length continuation patch detected; rebuild from "
            "the clean v2026.9.2 base: "
            + ", ".join(str(path) for path in stale_unbounded_targets)
        )

    stale_or_incomplete_length_targets = [
        path for path in dist_files
        if LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER in _read(path)
        or LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_V2_MARKER in _read(path)
        or (
            BOUNDED_LENGTH_CONTINUATION_MARKER in _read(path)
            and (
                CONSECUTIVE_EMPTY_STOP_RESET_MARKER not in _read(path)
                or LENGTH_SEMANTIC_PROGRESS_MARKER not in _read(path)
            )
        )
    ]
    if stale_or_incomplete_length_targets:
        raise RuntimeError(
            "stale or incomplete length-continuation patch detected; rebuild "
            "from the clean v2026.9.2 base: "
            + ", ".join(str(path) for path in stale_or_incomplete_length_targets)
        )

    for path in length_budget_targets:
        text = _read(path)
        if BOUNDED_LENGTH_CONTINUATION_MARKER not in text:
            require_reset_bindings(text, path)
            readable_count = len(length_budget_readable_anchor.findall(text))
            minified_count = len(length_budget_minified_anchor.findall(text))
            if readable_count + minified_count != 1:
                raise RuntimeError(
                    f"empty-response retry counter anchor missing/ambiguous in {path}"
                )
            anchor = (
                length_budget_readable_anchor
                if readable_count
                else length_budget_minified_anchor
            )
            text, count = anchor.subn(length_budget_replacement, text, count=1)
            if count != 1:
                raise RuntimeError(
                    f"empty-response retry counter anchor missing/ambiguous in {path}"
                )
            _write(path, text)
            changed.add(path)

    retry_anchor = re.compile(r"maxEmptyResponseRetryAttempts:\s*1,")
    retry_targets = [
        path for path in dist_files
        if (
            "prepareTerminalWithSettledTurnFinalization" in _read(path)
            and retry_anchor.search(_read(path))
        )
        or EMPTY_STOP_RETRY_LIMIT_MARKER in _read(path)
    ]
    if not retry_targets:
        raise RuntimeError("v2026.9.2 empty-response retry-limit anchor not found")

    retry_replacement = f'''/* {EMPTY_STOP_RETRY_LIMIT_MARKER} */
\t\t\t\tmaxEmptyResponseRetryAttempts: (() => {{
\t\t\t\t\tconst raw = process.env.{EMPTY_STOP_RETRIES_ENV};
\t\t\t\t\tif (raw === void 0 || raw === "") return 1;
\t\t\t\t\tconst parsed = Number.parseInt(raw, 10);
\t\t\t\t\treturn Number.isSafeInteger(parsed) && parsed >= 0
\t\t\t\t\t\t? Math.min(parsed, 16)
\t\t\t\t\t\t: 1;
\t\t\t\t}})(),
\t\t\t\t/* AB_HARNESS_OPENCLAW_V2026_9_2_EMPTY_STOP_RETRY_LIMIT_END */'''
    for path in retry_targets:
        text = _read(path)
        if EMPTY_STOP_RETRY_LIMIT_MARKER not in text:
            if len(retry_anchor.findall(text)) != 1:
                raise RuntimeError(
                    f"empty-response retry-limit anchor missing/ambiguous in {path}"
                )
            text, count = retry_anchor.subn(retry_replacement, text, count=1)
            if count != 1:
                raise RuntimeError(
                    f"empty-response retry-limit anchor missing/ambiguous in {path}"
                )
            _write(path, text)
            changed.add(path)

    return {
        "policy_files": [str(path.relative_to(root)) for path in policy_targets],
        "instruction_files": [
            str(path.relative_to(root)) for path in instruction_targets
        ],
        "retry_eligibility_files": [
            str(path.relative_to(root))
            for path in retry_eligibility_targets
        ],
        "length_budget_files": [
            str(path.relative_to(root)) for path in length_budget_targets
        ],
        "retry_limit_files": [
            str(path.relative_to(root)) for path in retry_targets
        ],
        "files_changed": [str(path.relative_to(root)) for path in sorted(changed)],
    }


def _patch_minified_context_pressure_terminal_recovery(
    text: str,
    *,
    path: Path,
    helper_source: str,
) -> str:
    """Patch the minified worker form of the native recovery pipeline."""
    ident = r"[A-Za-z_$][\w$]*"
    overflow_function = re.compile(
        rf"async function recoverEmbeddedRunOverflow\((?P<input>{ident})\)\{{"
    )
    attempt_function = re.compile(
        rf"async function recoverEmbeddedRunAttempt\((?P<input>{ident})\)\{{"
    )
    normalized_fields = re.compile(
        rf"\{{attempt:(?P<attempt>{ident}),sessionIdUsed:{ident},"
        rf"attemptAssistant:(?P<attempt_assistant>{ident}),"
        rf"currentAttemptAssistant:{ident},"
        rf"currentAttemptCompletedAssistant:(?P<completed>{ident}),"
    )
    terminal_fields = re.compile(
        rf"\{{aborted:(?P<aborted>{ident}),externalAbort:{ident},"
        rf"promptError:{ident},promptErrorSource:{ident},"
        rf"timedOut:(?P<timed_out>{ident}),[^}}]*\}}="
        rf"projectAgentRunAttemptTerminal\([^)]*\),"
        rf"(?P<interrupted>{ident})=isEmbeddedRunTerminalInterrupted\("
    )
    proceed_guard = re.compile(
        rf"if\(!(?P<replay>{ident})&&!(?P<mid>{ident})&&!"
        rf"(?P<transport>{ident})\)return\{{action:`proceed`\}};"
    )
    overflow_call = re.compile(
        rf"assistantErrorText:(?P<assistant_error>{ident}),"
        rf"assistantOverflowCandidate:(?P<overflow_candidate>{ident}),"
        r"attemptCompactionCount:"
    )
    budget = re.compile(
        rf"(?P<budget>{ident})=(?P<terminal>{ident})\.promptErrorSource"
        rf"===`precheck`&&(?P<preflight>{ident})\?\.source===`mid-turn`"
        rf"&&typeof (?P=preflight)\.promptBudgetBeforeReserve==`number`"
    )
    overflow_await = re.compile(
        rf"let (?P<recovery>{ident})=await recoverEmbeddedRunOverflow\("
    )
    retry_return = re.compile(
        rf"if\((?P<recovery>{ident})\.action===`retry`\)"
        rf"return (?P<retry>{ident})\(\);"
    )
    retry_state = re.compile(
        rf"(?P<state>{ident})=createEmbeddedRunTerminalRetryState\(\)"
    )
    outer_recovery_call = re.compile(
        r"recoverEmbeddedRunAttempt\(\{(?P<body>[^{}]+)\}\);"
        rf"if\((?P<recovery>{ident})\.action===`complete`\)"
    )
    compaction_branch = re.compile(
        rf"if\(!(?P<failure>{ident})&&(?P<input>{ident})\."
        rf"attemptCompactionCount(?P<operator>>0|===0)"
    )

    anchors = {
        "overflow helper": overflow_function,
        "attempt helper": attempt_function,
        "normalized attempt fields": normalized_fields,
        "terminal fields": terminal_fields,
        "attempt proceed guard": proceed_guard,
        "overflow call": overflow_call,
        "preflight budget": budget,
        "overflow await": overflow_await,
        "overflow retry": retry_return,
        "terminal retry state": retry_state,
        "outer recovery call": outer_recovery_call,
    }
    for name, anchor in anchors.items():
        if len(anchor.findall(text)) != 1:
            raise RuntimeError(
                f"context-pressure minified {name} anchor "
                f"missing/ambiguous in {path}"
            )
    if len(compaction_branch.findall(text)) != 2:
        raise RuntimeError(
            f"context-pressure minified compaction branch anchors "
            f"missing/ambiguous in {path}"
        )

    attempt_match = normalized_fields.search(text)
    terminal_match = terminal_fields.search(text)
    attempt_function_match = attempt_function.search(text)
    overflow_function_match = overflow_function.search(text)
    budget_match = budget.search(text)
    retry_state_match = retry_state.search(text)
    assert attempt_match and terminal_match and attempt_function_match
    assert overflow_function_match and budget_match and retry_state_match
    attempt = attempt_match.group("attempt")
    attempt_assistant = attempt_match.group("attempt_assistant")
    completed = attempt_match.group("completed")
    attempt_input = attempt_function_match.group("input")
    overflow_input = overflow_function_match.group("input")
    preflight_name = budget_match.group("preflight")
    retry_state_name = retry_state_match.group("state")

    text, count = overflow_function.subn(
        lambda match: helper_source + match.group(0), text, count=1
    )
    if count != 1:
        raise RuntimeError(f"context-pressure minified helper failed in {path}")

    def replace_guard(match: re.Match[str]) -> str:
        pressure = "__abContextPressureTerminal"
        return (
            f"const {pressure}=!{terminal_match.group('aborted')}"
            f"&&!{terminal_match.group('timed_out')}"
            f"&&!{terminal_match.group('interrupted')}"
            f"?__abResolveRoadmapContextPressureTerminal("
            f"{attempt},{completed}??{attempt_assistant}):null;"
            f"if({pressure}&&!{attempt}.preflightRecovery)"
            f"{attempt}.preflightRecovery={{route:{pressure}.route,"
            f"source:`agent-bench-terminal`,"
            f"estimatedPromptTokens:{pressure}.estimatedPromptTokens,"
            f"promptBudgetBeforeReserve:{pressure}.promptBudgetBeforeReserve}};"
            f"if({pressure})console.warn("
            f"`[agent-bench-continuation-decision] ${{JSON.stringify({{"
            f"candidate:{pressure}.candidate,decision:`recover`,"
            f"deny_reason:[],recovery_reason:`context_pressure_compaction`,"
            f"current_stop_reason:{pressure}.stopReason,"
            f"context_pressure_route:{pressure}.route,"
            f"estimated_prompt_tokens:{pressure}.estimatedPromptTokens,"
            f"prompt_budget_before_reserve:{pressure}.promptBudgetBeforeReserve"
            f"}})}}`);"
            f"if(!{match.group('replay')}&&!{match.group('mid')}"
            f"&&!{match.group('transport')}&&!{pressure})"
            f"return{{action:`proceed`}};"
        )

    text, count = proceed_guard.subn(replace_guard, text, count=1)
    if count != 1:
        raise RuntimeError(f"context-pressure minified guard failed in {path}")

    def replace_overflow_call(match: re.Match[str]) -> str:
        pressure = "__abContextPressureTerminal"
        return (
            f"assistantErrorText:{pressure}?`Context window exceeded under "
            f"preflight pressure`:{match.group('assistant_error')},"
            f"assistantOverflowCandidate:{pressure}"
            f"?({completed}??{attempt_assistant})"
            f":{match.group('overflow_candidate')},"
            f"agentBenchContextPressure:{pressure},attemptCompactionCount:"
        )

    text, count = overflow_call.subn(replace_overflow_call, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"context-pressure minified overflow call failed in {path}"
        )

    def replace_budget(match: re.Match[str]) -> str:
        preflight = match.group("preflight")
        return (
            f"{match.group('budget')}=(("
            f"{match.group('terminal')}.promptErrorSource===`precheck`"
            f"&&{preflight}?.source===`mid-turn`)"
            f"||{preflight}?.source===`agent-bench-terminal`"
            f"||{overflow_input}.agentBenchContextPressure)"
            f"&&typeof {preflight}.promptBudgetBeforeReserve==`number`"
        )

    text, count = budget.subn(replace_budget, text, count=1)
    if count != 1:
        raise RuntimeError(f"context-pressure minified budget failed in {path}")

    def replace_minified_compaction_branch(match: re.Match[str]) -> str:
        marker = (
            f"/* {TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER} */"
            if match.group("operator") == "===0"
            else ""
        )
        return (
            f"{marker}if({preflight_name}?.route"
            f"!==`truncate_tool_results_only`"
            f"&&!{match.group('failure')}&&{match.group('input')}."
            f"attemptCompactionCount{match.group('operator')}"
        )

    text, count = compaction_branch.subn(
        replace_minified_compaction_branch,
        text,
    )
    if count != 2:
        raise RuntimeError(
            f"context-pressure minified compaction routing failed in {path}"
        )

    def replace_overflow_await(match: re.Match[str]) -> str:
        return (
            f"let __abCompactionsBeforeAttemptRecovery="
            f"{attempt_input}.contextRecoveryState.autoCompactionCount,"
            f"__abToolResultTruncationBeforeAttemptRecovery=Boolean("
            f"{attempt_input}.contextRecoveryState.toolResultTruncationAttempted),"
            f"{match.group('recovery')}=await recoverEmbeddedRunOverflow("
        )

    text, count = overflow_await.subn(replace_overflow_await, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"context-pressure minified compaction snapshot failed in {path}"
        )

    def replace_retry_return(match: re.Match[str]) -> str:
        recovery = match.group("recovery")
        pressure = "__abContextPressureTerminal"
        return (
            f"if({recovery}.action===`retry`){{"
            f"const __abContextRecoveryChanged="
            f"{attempt_input}.contextRecoveryState.autoCompactionCount>"
            f"__abCompactionsBeforeAttemptRecovery||({pressure}?.route"
            f"===`truncate_tool_results_only`"
            f"&&!__abToolResultTruncationBeforeAttemptRecovery"
            f"&&Boolean({attempt_input}.contextRecoveryState."
            f"toolResultTruncationAttempted));"
            f"if(__abContextRecoveryChanged"
            f"&&{attempt_input}.terminalRetryState){{"
            f"if({pressure}?.candidate===`length`)"
            f"{attempt_input}.terminalRetryState.lengthNoProgressAttempts=0;"
            f"else if({pressure}?.candidate===`current-empty`"
            f"||{pressure}?.candidate===`accumulated-empty`)"
            f"{attempt_input}.terminalRetryState.emptyResponseAttempts=0;"
            f"}}/* {CONTEXT_RECOVERY_STREAK_RESET_MARKER} */"
            f"/* AB_HARNESS_OPENCLAW_V2026_9_2_"
            f"CONTEXT_RECOVERY_STREAK_RESET_V5_END */"
            f"return {match.group('retry')}();}}"
        )

    text, count = retry_return.subn(replace_retry_return, text, count=1)
    if count != 1:
        raise RuntimeError(f"context-pressure minified reset failed in {path}")

    def add_retry_state(match: re.Match[str]) -> str:
        body = match.group("body")
        if "terminalRetryState:" in body:
            raise RuntimeError(
                f"context-pressure minified retry state already present in {path}"
            )
        return (
            f"recoverEmbeddedRunAttempt({{{body},"
            f"terminalRetryState:{retry_state_name}}});"
            f"if({match.group('recovery')}.action===`complete`)"
        )

    text, count = outer_recovery_call.subn(add_retry_state, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"context-pressure minified retry-state wiring failed in {path}"
        )
    return text


def _patch_context_pressure_terminal_recovery(root: Path) -> list[str]:
    """Route pressure-induced empty/length stops through native recovery."""
    changed: list[str] = []
    dist_files = sorted(_js_files(root / "dist"))
    required_functions = (
        "compactEmbeddedRunForRecovery",
        "recoverEmbeddedRunOverflow",
        "recoverEmbeddedRunAttempt",
    )
    targets = [
        path
        for path in dist_files
        if all(f"function {name}(" in _read(path) for name in required_functions)
        or CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in _read(path)
        or CONTEXT_RECOVERY_STREAK_RESET_MARKER in _read(path)
        or TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_LENGTH_RECOVERY_MARKER in _read(path)
        or LEGACY_COMPACTION_LENGTH_STREAK_RESET_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_MARKER in _read(path)
        or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_MARKER in _read(path)
        or LEGACY_COMPACTION_RETRY_STREAK_RESET_MARKER in _read(path)
        or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V3_MARKER in _read(path)
        or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V4_MARKER in _read(path)
    ]
    if not targets:
        raise RuntimeError(
            "v2026.9.2 context-pressure terminal recovery anchors not found"
        )

    helper_anchor = re.compile(
        r"(?P<indent>[ \t]*)async function recoverEmbeddedRunOverflow\(input\) \{"
    )
    eligibility_anchor = re.compile(
        r'(?P<indent>[ \t]*)const canContinueSettledMidTurnOverflow = '
        r'promptErrorSource === "precheck" && '
        r'attempt\.preflightRecovery\?\.source === "mid-turn" && '
        r'midTurnBatchSettled && !hasAsyncActivity\(attempt\.toolMetas\);'
    )
    classifier_anchor = re.compile(
        r'(?P<indent>[ \t]*)if '
        r'\(isProviderRefusalAssistantError\(input\.assistantOverflowCandidate\)\) '
        r'return null;'
    )
    context_error_anchor = re.compile(
        r'(?P<indent>[ \t]*)const contextOverflowError\s*=\s*'
        r'!input\.aborted\s*&&\s*!input\.signalOwnedInterruption\s*'
        r'\?\s*\(\(\)\s*=>\s*\{'
    )
    preflight_metadata_anchor = re.compile(
        r'(?P<indent>[ \t]*)const preflightRecovery = '
        r'input\.attempt\.preflightRecovery;'
    )
    budget_anchor = re.compile(
        r'(?P<indent>[ \t]*)const preflightPromptBudget = '
        r'terminal\.promptErrorSource === "precheck" && '
        r'preflightRecovery\?\.source === "mid-turn" && '
    )
    compaction_retry_anchor = re.compile(
        r'(?P<indent>[ \t]*)if \(!isCompactionFailure && '
        r'input\.attemptCompactionCount > 0'
    )
    compaction_attempt_anchor = re.compile(
        r'(?P<indent>[ \t]*)if \('
        r'(?P<failure>!isCompactionFailure && )?'
        r'input\.attemptCompactionCount === 0'
    )
    recovery_call_anchor = re.compile(
        r'(?P<indent>^[ \t]*)const recovery = await '
        r'recoverEmbeddedRunAttempt\(\{'
        r'(?P<body_before>.*?)'
        r'^[ \t]*normalizedAttempt'
        r'(?:\s*:\s*(?P<normalized_attempt>'
        r'[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*)?,',
        flags=re.MULTILINE | re.DOTALL,
    )
    overflow_candidate_call_anchor = re.compile(
        r'(?P<indent>[ \t]*)assistantErrorText'
        r'(?:: (?P<assistant_error>[^,\n]+))?,\s*'
        r'(?P=indent)assistantOverflowCandidate,'
    )
    recovery_retry_anchor = re.compile(
        r'(?P<indent>[ \t]*)if \(recovery\.action === "complete"\) '
        r'return recovery\.result;\s*'
        r'(?P=indent)if \(recovery\.action === "retry"\) \{'
    )

    helper_source = f'''/* {CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER} */
function __abResolveRoadmapContextPressureTerminal(attempt, assistant) {{
\tif (process.env.{SETTLED_TOOL_POLICY_ENV} !== "continue") return null;
\tconst content = assistant?.content;
\tconst accumulatedVisibleText = (attempt?.assistantTexts ?? [])
\t\t.join("\\n\\n")
\t\t.trim();
\tconst currentVisibleText = (
\t\ttypeof content === "string"
\t\t\t? content
\t\t\t: Array.isArray(content)
\t\t\t\t? content
\t\t\t\t\t.filter((item) => item?.type === "text" && typeof item.text === "string")
\t\t\t\t\t.map((item) => item.text)
\t\t\t\t\t.join("")
\t\t\t\t: ""
\t).trim();
\tconst currentHasToolCall =
\t\tArray.isArray(content)
\t\t&& content.some((item) => item?.type === "toolCall" || item?.type === "toolUse");
\tconst currentHasPayload =
\t\tArray.isArray(content)
\t\t&& content.some((item) => {{
\t\t\tif (item === null || item === void 0) return false;
\t\t\tif (typeof item !== "object") return String(item).trim().length > 0;
\t\t\treturn item.type !== "text"
\t\t\t\t&& item.type !== "thinking"
\t\t\t\t&& item.type !== "reasoning"
\t\t\t\t&& item.type !== "toolCall"
\t\t\t\t&& item.type !== "toolUse";
\t\t}});
\tconst currentHasReasoning =
\t\tArray.isArray(content)
\t\t&& content.some((item) => item?.type === "thinking" || item?.type === "reasoning");
\tconst currentReasoningOnly = Boolean(
\t\tcurrentHasReasoning
\t\t&& !currentVisibleText
\t\t&& !currentHasToolCall
\t\t&& !currentHasPayload
\t);
\tconst explicitDone = /(?:^|\\n)\\s*DONE\\s*(?:\\n|$)/.test(
\t\t[accumulatedVisibleText, currentVisibleText].filter(Boolean).join("\\n")
\t);
\tconst nonSpawnTerminalOwner = hasNonToolTerminalState({{
\t\t...attempt,
\t\tacceptedSessionSpawns: [],
\t}});
\tconst runtimeContinuationStarted =
\t\tattempt?.runtimeContinuationStarted === true;
\tconst stopReason = String(assistant?.stopReason ?? "");
\tconst candidate = stopReason === "length" && !currentReasoningOnly && !explicitDone
\t\t? "length"
\t\t: stopReason === "stop"
\t\t\t&& !currentVisibleText
\t\t\t&& !currentHasToolCall
\t\t\t&& !currentHasPayload
\t\t\t&& !currentHasReasoning
\t\t\t\t? "current-empty"
\t\t\t\t: assistant === void 0
\t\t\t\t\t&& !accumulatedVisibleText
\t\t\t\t\t&& !nonSpawnTerminalOwner
\t\t\t\t\t? "accumulated-empty"
\t\t\t\t\t: "";
\tif (!candidate) return null;
\tconst preflight = attempt?.preflightRecovery ?? attempt?.contextBudgetStatus;
\tif (preflight?.route !== "compact_only"
\t\t&& preflight?.route !== "truncate_tool_results_only"
\t\t&& preflight?.route !== "compact_then_truncate") return null;
\tconst estimatedPromptTokens = Number(preflight.estimatedPromptTokens);
\tconst promptBudgetBeforeReserve = Number(preflight.promptBudgetBeforeReserve);
\tif (!Number.isFinite(estimatedPromptTokens)
\t\t|| !Number.isFinite(promptBudgetBeforeReserve)
\t\t|| promptBudgetBeforeReserve <= 0
\t\t|| estimatedPromptTokens <= promptBudgetBeforeReserve) return null;
\tconst evidence = resolveSettledToolBatchEvidence(attempt);
\tif (attempt?.terminal?.kind === "failed"
\t\t|| attempt?.terminal?.kind === "timeout"
\t\t|| Number(attempt?.itemLifecycle?.activeCount ?? 0) > 0
\t\t|| nonSpawnTerminalOwner
\t\t|| runtimeContinuationStarted
\t\t|| evidence.intentionalTermination
\t\t|| evidence.parkedCodeModeRun) return null;
\treturn {{
\t\tcandidate,
\t\tstopReason,
\t\troute: preflight.route,
\t\testimatedPromptTokens,
\t\tpromptBudgetBeforeReserve,
\t}};
}}
/* AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_END */
'''

    required_bindings = (
        "resolveSettledToolBatchEvidence",
        "hasNonToolTerminalState",
    )
    for path in targets:
        text = _read(path)
        markers = (
            CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in text,
            CONTEXT_RECOVERY_STREAK_RESET_MARKER in text,
            TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER in text,
        )
        if (
            LEGACY_CONTEXT_PRESSURE_LENGTH_RECOVERY_MARKER in text
            or LEGACY_COMPACTION_LENGTH_STREAK_RESET_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_MARKER in text
            or LEGACY_COMPACTION_RETRY_STREAK_RESET_MARKER in text
            or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V3_MARKER in text
            or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V4_MARKER in text
        ):
            raise RuntimeError(
                "context-pressure recovery patch predates current "
                "delivery/truncate-only semantics in "
                f"{path}; rebuild from the clean v2026.9.2 base"
            )
        if any(markers) and not all(markers):
            raise RuntimeError(
                "incomplete context-pressure terminal recovery patch in "
                f"{path}; rebuild from the clean v2026.9.2 base"
            )
        if all(markers):
            continue

        missing = [
            name for name in required_bindings
            if not _has_js_lexical_binding(text, name)
        ]
        if missing:
            raise RuntimeError(
                "context-pressure recovery lexical bindings missing in "
                f"{path}: {', '.join(missing)}"
            )
        anchors = {
            "overflow helper": helper_anchor,
            "attempt eligibility": eligibility_anchor,
            "overflow classifier": classifier_anchor,
            "overflow context-error": context_error_anchor,
            "preflight metadata": preflight_metadata_anchor,
            "preflight budget": budget_anchor,
            "overflow candidate call": overflow_candidate_call_anchor,
            "outer recovery call": recovery_call_anchor,
            "outer recovery retry": recovery_retry_anchor,
        }
        if not all(len(anchor.findall(text)) == 1 for anchor in anchors.values()):
            updated = _patch_minified_context_pressure_terminal_recovery(
                text,
                path=path,
                helper_source=helper_source,
            )
            _write(path, updated)
            changed.append(str(path.relative_to(root)))
            continue
        for name, anchor in anchors.items():
            if len(anchor.findall(text)) != 1:
                raise RuntimeError(
                    f"context-pressure {name} anchor missing/ambiguous in {path}"
                )

        if len(compaction_retry_anchor.findall(text)) > 1:
            raise RuntimeError(
                f"context-pressure compaction retry anchor ambiguous in {path}"
            )
        if len(compaction_attempt_anchor.findall(text)) != 1:
            raise RuntimeError(
                f"context-pressure compaction attempt anchor "
                f"missing/ambiguous in {path}"
            )

        text, count = helper_anchor.subn(
            lambda match: helper_source + match.group(0), text, count=1
        )
        if count != 1:
            raise RuntimeError(f"context-pressure helper insertion failed in {path}")

        def replace_eligibility(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f'''{indent}const __abContextPressureTerminal =
{indent}\t__abResolveRoadmapContextPressureTerminal(
{indent}\t\tattempt,
{indent}\t\tcurrentAttemptCompletedAssistant ?? attemptAssistant,
{indent}\t);
{indent}const canContinueSettledMidTurnOverflow = Boolean(
{indent}\t__abContextPressureTerminal
{indent}\t&& !aborted
{indent}\t&& !timedOut
{indent}\t&& !terminalInterrupted
{indent}) || (promptErrorSource === "precheck"
{indent}\t&& attempt.preflightRecovery?.source === "mid-turn"
{indent}\t&& midTurnBatchSettled
{indent}\t&& !hasAsyncActivity(attempt.toolMetas));'''

        text, count = eligibility_anchor.subn(replace_eligibility, text, count=1)
        if count != 1:
            raise RuntimeError(f"context-pressure eligibility patch failed in {path}")

        def replace_context_error(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f'''{indent}const __abContextPressureTerminal =
{indent}\t__abResolveRoadmapContextPressureTerminal(
{indent}\t\tinput.attempt,
{indent}\t\tinput.assistantOverflowCandidate,
{indent}\t);
{match.group(0)}'''

        text, count = context_error_anchor.subn(
            replace_context_error, text, count=1
        )
        if count != 1:
            raise RuntimeError(
                f"context-pressure context-error patch failed in {path}"
            )

        def replace_classifier(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return match.group(0) + f'''
{indent}if (__abContextPressureTerminal) {{
{indent}\tconsole.warn("[agent-bench-continuation-decision] " + JSON.stringify({{
{indent}\t\tcandidate: __abContextPressureTerminal.candidate,
{indent}\t\tdecision: "recover",
{indent}\t\tdeny_reason: [],
{indent}\t\trecovery_reason: "context_pressure_compaction",
{indent}\t\tcurrent_stop_reason: __abContextPressureTerminal.stopReason,
{indent}\t\tcontext_pressure_route: __abContextPressureTerminal.route,
{indent}\t\testimated_prompt_tokens: __abContextPressureTerminal.estimatedPromptTokens,
{indent}\t\tprompt_budget_before_reserve: __abContextPressureTerminal.promptBudgetBeforeReserve,
{indent}\t}}));
{indent}\treturn {{
{indent}\t\ttext: `Context-pressure ${{__abContextPressureTerminal.candidate}} stop: estimated prompt ${{__abContextPressureTerminal.estimatedPromptTokens}} exceeds budget ${{__abContextPressureTerminal.promptBudgetBeforeReserve}}`,
{indent}\t\tsource: "preflightPressure",
{indent}\t}};
{indent}}}'''

        text, count = classifier_anchor.subn(replace_classifier, text, count=1)
        if count != 1:
            raise RuntimeError(f"context-pressure classifier patch failed in {path}")

        def replace_preflight_metadata(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f'''{indent}const preflightRecovery =
{indent}\tinput.attempt.preflightRecovery
{indent}\t?? (__abContextPressureTerminal ? {{
{indent}\t\troute: __abContextPressureTerminal.route,
{indent}\t\tsource: "agent-bench-terminal",
{indent}\t\testimatedPromptTokens: __abContextPressureTerminal.estimatedPromptTokens,
{indent}\t\tpromptBudgetBeforeReserve: __abContextPressureTerminal.promptBudgetBeforeReserve,
{indent}\t}} : void 0);'''

        text, count = preflight_metadata_anchor.subn(
            replace_preflight_metadata, text, count=1
        )
        if count != 1:
            raise RuntimeError(
                f"context-pressure preflight metadata patch failed in {path}"
            )

        text, count = budget_anchor.subn(
            lambda match: (
                f'{match.group("indent")}const preflightPromptBudget = '
                '(terminal.promptErrorSource === "precheck" '
                '&& preflightRecovery?.source === "mid-turn" '
                '|| contextOverflowError.source === "preflightPressure") && '
            ),
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"context-pressure budget patch failed in {path}")

        text, count = compaction_retry_anchor.subn(
            lambda match: (
                f'{match.group("indent")}if ('
                'preflightRecovery?.route !== "truncate_tool_results_only" '
                '&& !isCompactionFailure && input.attemptCompactionCount > 0'
            ),
            text,
            count=1,
        )
        if count not in {0, 1}:
            raise RuntimeError(
                f"context-pressure compaction retry patch failed in {path}"
            )

        def replace_compaction_attempt(match: re.Match[str]) -> str:
            indent = match.group("indent")
            failure = match.group("failure") or ""
            return (
                f'{indent}/* {TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER} */\n'
                f'{indent}if (preflightRecovery?.route '
                '!== "truncate_tool_results_only" && '
                f'{failure}input.attemptCompactionCount === 0'
            )

        text, count = compaction_attempt_anchor.subn(
            replace_compaction_attempt,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"context-pressure compaction attempt patch failed in {path}"
            )

        def replace_overflow_candidate_call(match: re.Match[str]) -> str:
            indent = match.group("indent")
            assistant_error = match.group("assistant_error")
            assistant_error_property = (
                "assistantErrorText"
                if assistant_error is None
                else f"assistantErrorText: {assistant_error}"
            )
            return f'''{indent}{assistant_error_property},
{indent}assistantOverflowCandidate: __abContextPressureTerminal
{indent}\t? (currentAttemptCompletedAssistant ?? attemptAssistant)
{indent}\t: assistantOverflowCandidate,'''

        text, count = overflow_candidate_call_anchor.subn(
            replace_overflow_candidate_call,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"context-pressure overflow candidate patch failed in {path}"
            )

        def replace_recovery_call(match: re.Match[str]) -> str:
            indent = match.group("indent")
            normalized_attempt = (
                match.group("normalized_attempt") or "normalizedAttempt"
            )
            return f'''{indent}/* {CONTEXT_RECOVERY_STREAK_RESET_MARKER} */
{indent}const __abCompactionsBeforeAttemptRecovery =
{indent}\tcontextRecoveryState.autoCompactionCount;
{indent}const __abToolResultTruncationBeforeAttemptRecovery =
{indent}\tBoolean(contextRecoveryState.toolResultTruncationAttempted);
{indent}const __abContextPressureCandidate =
{indent}\t__abResolveRoadmapContextPressureTerminal(
{indent}\t\t{normalized_attempt}.attempt,
{indent}\t\t{normalized_attempt}.currentAttemptCompletedAssistant
{indent}\t\t\t?? {normalized_attempt}.attemptAssistant,
{indent}\t)?.candidate;
{indent}const __abContextPressureRoute =
{indent}\t{normalized_attempt}.attempt.preflightRecovery?.route
{indent}\t?? {normalized_attempt}.attempt.contextBudgetStatus?.route;
{match.group(0)}'''

        text, count = recovery_call_anchor.subn(
            replace_recovery_call, text, count=1
        )
        if count != 1:
            raise RuntimeError(f"compaction streak snapshot patch failed in {path}")

        def replace_recovery_retry(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f'{indent}if (recovery.action === "complete") return recovery.result;\n'
                f'{indent}if (recovery.action === "retry") {{\n'
                f'{indent}\tconst __abContextRecoveryChanged =\n'
                f'{indent}\t\tcontextRecoveryState.autoCompactionCount '
                '> __abCompactionsBeforeAttemptRecovery\n'
                f'{indent}\t\t|| (__abContextPressureRoute '
                '=== "truncate_tool_results_only"\n'
                f'{indent}\t\t\t&& !__abToolResultTruncationBeforeAttemptRecovery\n'
                f'{indent}\t\t\t&& Boolean('
                'contextRecoveryState.toolResultTruncationAttempted));\n'
                f'{indent}\tif (__abContextRecoveryChanged) {{\n'
                f'{indent}\t\tif (__abContextPressureCandidate === "length") {{\n'
                f'{indent}\t\t\tterminalRetryState.lengthNoProgressAttempts = 0;\n'
                f'{indent}\t\t}} else if (__abContextPressureCandidate === "current-empty"\n'
                f'{indent}\t\t\t|| __abContextPressureCandidate === "accumulated-empty") {{\n'
                f'{indent}\t\t\tterminalRetryState.emptyResponseAttempts = 0;\n'
                f'{indent}\t\t}}\n'
                f'{indent}\t}}\n'
                f'{indent}\t/* '
                'AB_HARNESS_OPENCLAW_V2026_9_2_CONTEXT_RECOVERY_STREAK_RESET_V5_END */'
            )

        text, count = recovery_retry_anchor.subn(
            replace_recovery_retry, text, count=1
        )
        if count != 1:
            raise RuntimeError(f"compaction streak reset patch failed in {path}")

        _write(path, text)
        changed.append(str(path.relative_to(root)))
    return changed


def _patch_loop_veto_recovery_probe(root: Path) -> list[str]:
    """Allow one real probe after a native critical loop veto.

    Native v9.2 records the blocked call as ``tool-loop-veto`` while retaining
    the outcomes that triggered the block. Without a probe, the same call is
    rejected again before it can observe workspace progress made during the
    recovery turn, making the detector state self-latching.
    """
    changed: list[str] = []
    dist_files = sorted(_js_files(root / "dist"))
    anchor = re.compile(
        r"(?P<indent>[ \t]*)const history = selectHistoryForScope\("
        r"state\.toolCallHistory \?\? \[\], scope\);\n"
        r"(?P=indent)const currentHash = hashToolCall\(toolName, params\);"
    )
    targets = [
        path
        for path in dist_files
        if anchor.search(_read(path))
        or LOOP_VETO_RECOVERY_PROBE_MARKER in _read(path)
    ]
    if not targets:
        raise RuntimeError("v2026.9.2 loop-veto recovery anchor not found")

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return match.group(0) + f'''\n{indent}/* {LOOP_VETO_RECOVERY_PROBE_MARKER} */
{indent}if (process.env.{LOOP_VETO_RECOVERY_PROBE_ENV} === "1") {{
{indent}\tfor (let i = history.length - 1; i >= 0; i -= 1) {{
{indent}\t\tconst record = history[i];
{indent}\t\tif (!record || record.toolName !== toolName || record.argsHash !== currentHash) continue;
{indent}\t\tif (record.outcomeKind === "tool-loop-veto") return {{ stuck: false }};
{indent}\t\tbreak;
{indent}\t}}
{indent}}}
{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_END */'''

    for path in targets:
        text = _read(path)
        if LOOP_VETO_RECOVERY_PROBE_MARKER in text:
            continue
        if len(anchor.findall(text)) != 1:
            raise RuntimeError(
                f"loop-veto recovery anchor missing/ambiguous in {path}"
            )
        updated, count = anchor.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(
                f"loop-veto recovery anchor missing/ambiguous in {path}"
            )
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
    implicit_child_handoff_ownership_files = (
        _patch_implicit_child_handoff_ownership(root)
    )
    settled_continuation = _patch_settled_tool_continuation(root)
    context_pressure_terminal_recovery_files = (
        _patch_context_pressure_terminal_recovery(root)
    )
    loop_veto_recovery_probe_files = _patch_loop_veto_recovery_probe(root)
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
        "implicit_child_handoff_ownership_files_changed": (
            implicit_child_handoff_ownership_files
        ),
        "settled_tool_continuation": settled_continuation,
        "context_pressure_terminal_recovery_files_changed": (
            context_pressure_terminal_recovery_files
        ),
        "loop_veto_recovery_probe_files_changed": (
            loop_veto_recovery_probe_files
        ),
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
        "settled_continuation_policy": (
            "dist/**/*.js",
            SETTLED_CONTINUATION_POLICY_MARKER,
        ),
        "settled_continuation_instruction": (
            "dist/**/*.js",
            SETTLED_CONTINUATION_INSTRUCTION_MARKER,
        ),
        "settled_continuation_retry": (
            "dist/**/*",
            SETTLED_CONTINUATION_LENGTH_MARKER,
        ),
        "nonempty_length_eligibility": (
            "dist/**/*",
            NONEMPTY_LENGTH_ELIGIBILITY_MARKER,
        ),
        "current_assistant_empty_stop": (
            "dist/**/*",
            CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
        ),
        "continuation_decision_diagnostic": (
            "dist/**/*",
            CONTINUATION_DECISION_DIAGNOSTIC_MARKER,
        ),
        "runtime_owner_continuation": (
            "dist/**/*",
            RUNTIME_OWNER_CONTINUATION_MARKER,
        ),
        "implicit_child_handoff_ownership": (
            "dist/**/*",
            IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER,
        ),
        "bounded_no_progress_length": (
            "dist/**/*",
            BOUNDED_LENGTH_CONTINUATION_MARKER,
        ),
        "length_semantic_progress": (
            "dist/**/*",
            LENGTH_SEMANTIC_PROGRESS_MARKER,
        ),
        "consecutive_empty_stop_reset": (
            "dist/**/*",
            CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
        ),
        "settled_continuation_tool_error": (
            "dist/**/*",
            SETTLED_CONTINUATION_TOOL_ERROR_MARKER,
        ),
        "empty_stop_retry_limit": (
            "dist/**/*.js",
            EMPTY_STOP_RETRY_LIMIT_MARKER,
        ),
        "loop_veto_recovery_probe": (
            "dist/**/*",
            LOOP_VETO_RECOVERY_PROBE_MARKER,
        ),
        "context_pressure_terminal_recovery": (
            "dist/**/*",
            CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER,
        ),
        "compaction_retry_streak_reset": (
            "dist/**/*",
            CONTEXT_RECOVERY_STREAK_RESET_MARKER,
        ),
        "truncate_only_native_recovery": (
            "dist/**/*",
            TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER,
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

    handoff_error = "accepted continuation children were not durably registered"
    handoff_files: list[str] = []
    handoff_block = re.compile(
        re.escape(IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER)
        + r".*?[A-Za-z_$][\w$]*\.runtimeContinuationStarted\s*=\s*true;"
        + ".*?AB_HARNESS_OPENCLAW_V2026_9_2_"
        + "IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_END",
        re.DOTALL,
    )
    for path in sorted(_js_files(root / "dist")):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        if (
            handoff_error not in text
            and IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER not in text
        ):
            continue
        calls = text.count(handoff_error)
        markers = text.count(IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER)
        blocks = len(handoff_block.findall(text))
        if calls <= 0 or markers != calls or blocks != calls:
            raise RuntimeError(
                "incomplete implicit child handoff ownership contract in "
                f"{path}: calls={calls} markers={markers} blocks={blocks}"
            )
        handoff_files.append(str(path.relative_to(root)))
    if not handoff_files:
        raise RuntimeError(
            "no v2026.9.2 implicit child handoff ownership call sites found"
        )

    resolver_anchor = re.compile(
        r"function\s+resolveEmptyResponseRetryInstruction\s*\("
    )
    resolver_files: list[str] = []
    resolver_bindings: dict[str, list[str]] = {}
    stale_contract_files: list[str] = []
    owner_block_pattern = re.compile(
        re.escape(RUNTIME_OWNER_CONTINUATION_MARKER)
        + r".*?AB_HARNESS_OPENCLAW_V2026_9_2_"
        + r"RUNTIME_OWNER_CONTINUATION_V2_END",
        re.DOTALL,
    )
    for path in sorted(_js_files(root / "dist")):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        definitions = len(resolver_anchor.findall(text))
        if definitions:
            markers = text.count(CURRENT_ASSISTANT_EMPTY_STOP_MARKER)
            owner_markers = text.count(RUNTIME_OWNER_CONTINUATION_MARKER)
            if markers != definitions or owner_markers != definitions:
                raise RuntimeError(
                    "not every v2026.9.2 empty-response resolver definition "
                    f"contains current-empty/runtime-owner logic: {path} "
                    f"definitions={definitions} current_empty={markers} "
                    f"runtime_owner={owner_markers}"
                )
            owner_blocks = owner_block_pattern.findall(text)
            if len(owner_blocks) != definitions or any(
                "runtimeContinuationStarted === true" not in block
                or "&& !__abRuntimeContinuationOwner" not in block
                or 'push("runtime_continuation_owner")' not in block
                or "&& __abRoadmapContinuationEligible) return null;" not in block
                or "acceptedSessionSpawns" in block
                for block in owner_blocks
            ):
                raise RuntimeError(
                    "ordinary continuation runtime-owner guard missing or "
                    f"stale in {path}"
                )
            if "tools_unsettled" in text:
                raise RuntimeError(
                    "settled-tool proof remains a continuation hard guard in "
                    f"{path}"
                )
            relative = str(path.relative_to(root))
            missing = [
                name
                for name in _SETTLED_CONTINUATION_RESOLVER_BINDINGS
                if not _has_js_lexical_binding(text, name)
            ]
            if missing:
                raise RuntimeError(
                    "settled-continuation resolver lexical bindings missing "
                    f"in {path}: {', '.join(missing)}"
                )
            resolver_files.append(relative)
            resolver_bindings[relative] = list(
                _SETTLED_CONTINUATION_RESOLVER_BINDINGS
            )
        if (
            LEGACY_CURRENT_ASSISTANT_EMPTY_STOP_MARKER in text
            or LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER in text
            or LEGACY_RUNTIME_OWNER_CONTINUATION_MARKER in text
            or "hasCurrentTerminalPayload" in text
        ):
            stale_contract_files.append(str(path.relative_to(root)))
    if not resolver_files:
        raise RuntimeError(
            "no v2026.9.2 empty-response resolver definitions found"
        )
    if stale_contract_files:
        raise RuntimeError(
            "obsolete cross-bundle current-empty contract remains in: "
            + ", ".join(stale_contract_files)
        )

    reset_block_pattern = re.compile(
        re.escape(CONSECUTIVE_EMPTY_STOP_RESET_MARKER)
        + r".*?AB_HARNESS_OPENCLAW_V2026_9_2_"
        + r"CONSECUTIVE_EMPTY_STOP_RESET_V3_END",
        re.DOTALL,
    )
    for path in sorted(_js_files(root / "dist")):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        if (
            LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER in text
            or LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_V2_MARKER in text
        ):
            raise RuntimeError(
                f"legacy consecutive-empty reset contract remains in {path}"
            )
        bounded_count = text.count(BOUNDED_LENGTH_CONTINUATION_MARKER)
        if not bounded_count:
            continue
        reset_markers = text.count(CONSECUTIVE_EMPTY_STOP_RESET_MARKER)
        reset_blocks = reset_block_pattern.findall(text)
        if (
            reset_markers != bounded_count
            or len(reset_blocks) != bounded_count
            or any(
                "runtimeContinuationStarted === true" not in block
                or "acceptedSessionSpawns" in block
                for block in reset_blocks
            )
        ):
            raise RuntimeError(
                "consecutive-empty reset ownership contract missing or "
                f"stale in {path}"
            )

    recovery_files: list[str] = []
    for path in sorted(_js_files(root / "dist")):
        try:
            text = _read(path)
        except (OSError, UnicodeDecodeError):
            continue
        if "function recoverEmbeddedRunOverflow(" not in text:
            continue
        overflow_definitions = text.count(
            "function recoverEmbeddedRunOverflow("
        )
        attempt_definitions = text.count(
            "function recoverEmbeddedRunAttempt("
        )
        pressure_markers = text.count(
            CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER
        )
        reset_markers = text.count(CONTEXT_RECOVERY_STREAK_RESET_MARKER)
        helper_start = text.find(CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER)
        helper_end = text.find(
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_END"
        )
        helper_block = (
            text[helper_start:helper_end]
            if helper_start >= 0 and helper_end > helper_start
            else ""
        )
        readable_pressure_contract = bool(
            re.search(r'source:\s*["`]preflightPressure["`]', text)
        )
        minified_pressure_contract = (
            text.count("agentBenchContextPressure") >= 2
            and "Context window exceeded under preflight pressure" in text
        )
        if (
            overflow_definitions != 1
            or attempt_definitions != 1
            or pressure_markers != overflow_definitions
            or reset_markers != attempt_definitions
            or "function compactEmbeddedRunForRecovery(" not in text
            or "function recoverEmbeddedRunAttempt(" not in text
            or CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER not in text
            or CONTEXT_RECOVERY_STREAK_RESET_MARKER not in text
            or TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER not in text
            or LEGACY_CONTEXT_PRESSURE_LENGTH_RECOVERY_MARKER in text
            or LEGACY_COMPACTION_LENGTH_STREAK_RESET_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_MARKER in text
            or LEGACY_CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_MARKER in text
            or LEGACY_COMPACTION_RETRY_STREAK_RESET_MARKER in text
            or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V3_MARKER in text
            or LEGACY_CONTEXT_RECOVERY_STREAK_RESET_V4_MARKER in text
            or "attempt?.preflightRecovery ?? attempt?.contextBudgetStatus"
            not in text
            or not (
                readable_pressure_contract or minified_pressure_contract
            )
            or not re.search(r'source:\s*["`]agent-bench-terminal["`]', text)
            or not re.search(r'lengthNoProgressAttempts\s*=\s*0', text)
            or not re.search(r'emptyResponseAttempts\s*=\s*0', text)
            or "hasNonToolTerminalState({" not in helper_block
            or "acceptedSessionSpawns: []" not in helper_block
            or "hasCompletionOwningSpawn" in helper_block
            or "runtimeContinuationStarted === true" not in helper_block
            or "hasOutboundDeliveryEvidence(attempt)" in helper_block
            or "(attempt?.assistantTexts ?? [])" not in helper_block
            or '"accumulated-empty"' not in helper_block
            or "currentReasoningOnly" not in helper_block
            or "explicitDone" not in helper_block
            or 'preflight?.route !== "truncate_tool_results_only"'
            not in helper_block
            or "input.normalizedAttempt" in text
            or not all(
                _has_js_lexical_binding(text, name)
                for name in (
                    "resolveSettledToolBatchEvidence",
                    "hasNonToolTerminalState",
                )
            )
        ):
            raise RuntimeError(
                "incomplete context-pressure native recovery contract in "
                f"{path}"
            )
        recovery_files.append(str(path.relative_to(root)))
    if not recovery_files:
        raise RuntimeError(
            "no v2026.9.2 context-pressure recovery definitions found"
        )

    return {
        "upstream_version": version,
        "markers": found,
        "current_empty_resolver_files": resolver_files,
        "implicit_child_handoff_ownership_files": handoff_files,
        "current_empty_resolver_bindings": resolver_bindings,
        "context_pressure_recovery_files": recovery_files,
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
