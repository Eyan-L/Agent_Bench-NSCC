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
UNBOUNDED_LENGTH_CONTINUATION_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_UNBOUNDED_LENGTH_CONTINUATION_BEGIN"
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
CONSECUTIVE_EMPTY_STOP_RESET_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_BEGIN"
)
LOOP_VETO_RECOVERY_PROBE_MARKER = (
    "AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_BEGIN"
)

SETTLED_TOOL_POLICY_ENV = "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY"
EMPTY_STOP_RETRIES_ENV = "OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES"
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


def _patch_settled_tool_continuation(root: Path) -> dict[str, list[str]]:
    """Route Roadmap incomplete post-tool turns through the normal retry.

    Upstream v2026.9.2 runs a tools-disabled isolated finalization when a
    settled tool batch is followed by an empty assistant stop. It can also
    finalize a length-truncated assistant turn that still promises more work.
    Under the explicit Agent_Bench policy, the existing tools-enabled retry
    path remains eligible; length retries are fenced by the benchmark deadline
    rather than the empty-stop retry count.
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

    def retry_eligibility_replacement(match: re.Match[str]) -> str:
        param = match.group("param")
        return match.group(1) + f'''/* {SETTLED_CONTINUATION_RETRY_MARKER} */
	/* {SETTLED_CONTINUATION_LENGTH_MARKER} */
	/* {NONEMPTY_LENGTH_ELIGIBILITY_MARKER} */
	const __abAssistantState = classifyAssistantTurn({param});
	const __abExplicitDone =
		/(?:^|\\n)\\s*DONE\\s*(?:\\n|$)/.test(__abAssistantState.visibleText);
	const __abIncompleteLength =
		__abAssistantState.assistant?.stopReason === "length"
		&& !__abAssistantState.reasoningOnly
		&& !__abExplicitDone;
	const __abRoadmapContinuationEligible =
		__abAssistantState.emptyResponse || __abIncompleteLength;
	/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_LENGTH_END */
	const __abSettledToolEvidence = resolveSettledToolBatchEvidence({param}.attempt);
	if (process.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
		&& !{param}.aborted
		&& !{param}.timedOut
		&& {param}.attempt?.terminal?.kind !== "failed"
		&& {param}.attempt?.terminal?.kind !== "timeout"
		&& !{param}.attempt.clientToolCalls
		&& !{param}.attempt.yieldDetected
		&& !{param}.attempt.didSendDeterministicApprovalPrompt
		&& !{param}.attempt.hasToolMediaBlockReply
		&& !hasCompletedMessagingToolDeliveryEvidence({param}.attempt)
		&& __abRoadmapContinuationEligible
		&& __abSettledToolEvidence.allToolsProvenSettled
		/* {SETTLED_CONTINUATION_TOOL_ERROR_MARKER} */
		/* Settled tool errors are recoverable model feedback, not runtime fatality. */
		/* AB_HARNESS_OPENCLAW_V2026_9_2_SETTLED_CONTINUATION_TOOL_ERROR_END */
		&& !__abSettledToolEvidence.intentionalTermination
		&& !__abSettledToolEvidence.parkedCodeModeRun
		&& !hasAsyncActivity({param}.attempt.toolMetas)) {{
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
        )
    ]
    if stale_retry_targets:
        raise RuntimeError(
            "settled continuation patch predates empty-length or "
            "recoverable-tool-error support; "
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
        or UNBOUNDED_LENGTH_CONTINUATION_MARKER in _read(path)
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
{indent}\t\t|| ({input_name}.attempt.acceptedSessionSpawns?.length ?? 0) > 0
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
{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_END */'''

    required_reset_bindings = (
        "resolveSettledToolBatchEvidence",
        "hasMessagingToolDeliveryEvidence",
        "hasAsyncActivity",
        "isEmbeddedRunTerminalAbort",
        "isEmbeddedRunTerminalTimeout",
    )

    def has_lexical_binding(text: str, name: str) -> bool:
        return bool(
            re.search(rf"function\s+{re.escape(name)}\s*\(", text)
            or re.search(
                rf"import\s*\{{[^}}]*\b{re.escape(name)}\b[^}}]*\}}\s*from",
                text,
                flags=re.DOTALL,
            )
        )

    def require_reset_bindings(text: str, path: Path) -> None:
        missing = [
            name for name in required_reset_bindings
            if not has_lexical_binding(text, name)
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
        return f'''{indent}/* {UNBOUNDED_LENGTH_CONTINUATION_MARKER} */
{reset_source}
{indent}const __abLengthContinuation =
{indent}\tprocess.env.{SETTLED_TOOL_POLICY_ENV} === "continue"
{indent}\t&& {input_name}.attemptAssistant?.stopReason === "length"
{indent}\t&& {instruction};
{indent}if (!{reason} && {instruction}
{indent}\t&& (__abLengthContinuation
{indent}\t\t|| {state}.emptyResponseAttempts < {input_name}.maxEmptyResponseRetryAttempts)) {{
{indent}\t{input_name}.activateInternalPrompt({instruction});
{indent}\tif (__abLengthContinuation) {{
{indent}\t\t{logger}.warn(`length-truncated response detected: runId=${{{input_name}.runParams.runId}} sessionId=${{{input_name}.runParams.sessionId}} provider=${{{input_name}.activeErrorContext.provider}}/${{{input_name}.activeErrorContext.model}} — continuing under the enclosing benchmark deadline`);
{indent}\t}} else {{
{indent}\t\t{state}.emptyResponseAttempts += 1;
{indent}\t\t{logger}.warn(`empty response detected: runId=${{{input_name}.runParams.runId}} sessionId=${{{input_name}.runParams.sessionId}} provider=${{{input_name}.activeErrorContext.provider}}/${{{input_name}.activeErrorContext.model}} — retrying ${{{state}.emptyResponseAttempts}}/${{{input_name}.maxEmptyResponseRetryAttempts}} with visible-answer continuation`);
{indent}\t}}
{indent}\treturn {{ action: "retry" }};
{indent}}}
{indent}/* AB_HARNESS_OPENCLAW_V2026_9_2_UNBOUNDED_LENGTH_CONTINUATION_END */'''

    for path in length_budget_targets:
        text = _read(path)
        if LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER in text:
            legacy_pattern = re.compile(
                rf"(?P<indent>^[ \t]*)/\* "
                rf"{re.escape(LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER)} "
                rf"\*/.*?^(?P=indent)/\* AB_HARNESS_OPENCLAW_V2026_9_2_"
                rf"CONSECUTIVE_EMPTY_STOP_RESET_END \*/\n?",
                flags=re.MULTILINE | re.DOTALL,
            )
            text, legacy_count = legacy_pattern.subn("", text, count=1)
            if legacy_count != 1:
                raise RuntimeError(
                    f"legacy consecutive-empty reset block is malformed in {path}"
                )
            _write(path, text)
            changed.add(path)
        if (
            UNBOUNDED_LENGTH_CONTINUATION_MARKER in text
            and CONSECUTIVE_EMPTY_STOP_RESET_MARKER not in text
        ):
            require_reset_bindings(text, path)
            marker_matches = list(re.finditer(
                rf"(?P<indent>^[ \t]*)/\* "
                rf"{re.escape(UNBOUNDED_LENGTH_CONTINUATION_MARKER)} \*/$",
                text,
                flags=re.MULTILINE,
            ))
            if len(marker_matches) != 1:
                raise RuntimeError(
                    "existing empty-response counter marker is ambiguous in "
                    f"{path}"
                )
            marker_match = marker_matches[0]
            end_marker = (
                "AB_HARNESS_OPENCLAW_V2026_9_2_"
                "UNBOUNDED_LENGTH_CONTINUATION_END"
            )
            end = text.find(end_marker, marker_match.end())
            if end < 0:
                raise RuntimeError(
                    f"existing empty-response counter end marker missing in {path}"
                )
            block = text[marker_match.end():end]
            counter_match = re.search(
                rf"\|\|\s*(?P<state>{ident})\.emptyResponseAttempts"
                rf"\s*<\s*(?P<input>{ident})"
                r"\.maxEmptyResponseRetryAttempts",
                block,
            )
            if counter_match is None or not re.search(
                rf"const\s+__abLengthContinuation\s*=.*?"
                rf"{re.escape(counter_match.group('input'))}"
                r"\.attemptAssistant\?\.stopReason",
                block,
                flags=re.DOTALL,
            ):
                raise RuntimeError(
                    "existing empty-response counter structure missing in "
                    f"{path}"
                )
            insert_at = marker_match.end()
            reset_source = consecutive_reset_source(
                marker_match.group("indent"),
                counter_match.group("input"),
                counter_match.group("state"),
            )
            text = text[:insert_at] + "\n" + reset_source + text[insert_at:]
            _write(path, text)
            changed.add(path)
        if UNBOUNDED_LENGTH_CONTINUATION_MARKER not in text:
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
    settled_continuation = _patch_settled_tool_continuation(root)
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
        "settled_tool_continuation": settled_continuation,
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
        "unbounded_length_continuation": (
            "dist/**/*",
            UNBOUNDED_LENGTH_CONTINUATION_MARKER,
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
