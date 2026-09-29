"""OpenClaw 2026.9.6 bridge integration; keep native child settlement.

Bundle extensions, delivery evidence and context recovery changed in this
release. Reuse the v9 bridge transforms, with version-specific verification.
The v9.2 handoff flag patch is intentionally absent: v9.6 settles the whole
logical requester run after its retries and guards duplicate settlement.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

if __package__:
    from . import v2026_9_2_patch as shared
else:
    import v2026_9_2_patch as shared

EXPECTED_VERSION = "2026.9.6"
CONTEXT_MARKER = "AB_HARNESS_OPENCLAW_V2026_9_6_CONTEXT_PRESSURE_BEGIN"
_IDENT = r"[A-Za-z_$][\w$]*"

CONTEXT_HELPER = '''
function __abV96ContextPressure(attempt, assistant) {
  assistant ??= resolveCurrentAttemptAssistant(attempt);
  if (process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY !== "continue"
      || attempt?.terminal?.kind === "failed" || attempt?.terminal?.kind === "timeout"
      || Number(attempt?.itemLifecycle?.activeCount ?? 0) > 0
      || attempt?.runtimeContinuationStarted === true
      || hasAttemptTerminalState(attempt) || hasAsyncActivity(attempt?.toolMetas)) return null;
  const content = assistant?.content;
  const visible = (typeof content === "string" ? content : Array.isArray(content)
    ? content.filter(item => item?.type === "text" && typeof item.text === "string")
      .map(item => item.text).join("") : "").trim();
  const hasNonText = Array.isArray(content) && content.some(item =>
    item?.type !== "text");
  const explicitDone = /(?:^|\\n)\\s*DONE\\s*(?:\\n|$)/.test(visible);
  const candidate = assistant?.stopReason === "length" && !hasNonText && !explicitDone
    || assistant?.stopReason === "stop" && !visible && !hasNonText;
  if (!candidate) return null;
  const evidence = resolveSettledToolBatchEvidence(attempt);
  if (evidence.intentionalTermination || evidence.parkedCodeModeRun) return null;
  const preflight = attempt?.preflightRecovery ?? attempt?.contextBudgetStatus;
  if (!["compact_only", "compact_then_truncate", "truncate_tool_results_only"]
      .includes(preflight?.route)) return null;
  const estimatedPromptTokens = Number(preflight.estimatedPromptTokens);
  const promptBudgetBeforeReserve = Number(preflight.promptBudgetBeforeReserve);
  if (!Number.isFinite(estimatedPromptTokens) || !Number.isFinite(promptBudgetBeforeReserve)
      || promptBudgetBeforeReserve <= 0 || estimatedPromptTokens <= promptBudgetBeforeReserve) return null;
  return { route: preflight.route, estimatedPromptTokens, promptBudgetBeforeReserve };
}
'''


def _patch_context_pressure(root: Path) -> list[str]:
    """Supply missing pressure evidence to v9.6's own bounded recovery.

    Upstream explicitly excludes length stops from overflow classification.
    Retain the Roadmap pressure guard, adapting only the input to native
    compaction/truncation; do not transplant v9.2's recovery implementation.
    """
    changed = []
    start_pattern = re.compile(rf"async function recoverEmbeddedRunOverflow\((?P<param>{_IDENT})\)\s*\{{")
    for path in shared._js_files(root / "dist"):
        text = shared._read(path)
        matches = list(start_pattern.finditer(text))
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous v9.6 overflow definition in {path}")
        if CONTEXT_MARKER in text:
            continue
        if not all(shared._has_js_lexical_binding(text, name) for name in (
            "hasAttemptTerminalState", "hasAsyncActivity", "resolveSettledToolBatchEvidence",
            "resolveCurrentAttemptAssistant",
        )):
            raise RuntimeError(f"unbound v9.6 pressure helper in {path}")
        start = matches[0]
        end_match = re.search(r"async function handleEmbeddedPromptFailure\(", text[start.end():])
        if not end_match:
            raise RuntimeError(f"v9.6 overflow boundary missing in {path}")
        end = start.end() + end_match.start()
        body = text[start.end():end]
        param = start.group("param")
        preflight = re.compile(rf"{_IDENT}\.promptErrorSource\s*===\s*[\"'`]precheck[\"'`]")
        if len(preflight.findall(body)) != 1:
            raise RuntimeError(f"v9.6 preflight classification anchor missing in {path}")
        body = preflight.sub(lambda m: f"({m.group(0)} || Boolean(__abPressure))", body, count=1)
        compaction = re.compile(rf"if\s*\(\s*!{_IDENT}\s*&&\s*{re.escape(param)}\.attemptCompactionCount\s*>\s*0")
        helpers = re.findall(rf"({_IDENT})\s*=\s*async\s*\(\s*\)\s*=>", body)
        if len(helpers) != 1 or len(compaction.findall(body)) != 1:
            raise RuntimeError(f"v9.6 truncation/compaction anchor missing in {path}")
        truncate = helpers[0]
        truncate_branch = f'''
  if (__abPressure?.route === "truncate_tool_results_only") {{
    if (!{param}.state.toolResultTruncationAttempted) {{
      {param}.state.toolResultTruncationAttempted = true;
      const result = await {truncate}();
      {param}.assertRecoveryActive();
      if (result.truncated) {{
        {param}.markOwnedTranscriptRetry();
        {param}.prepareCurrentTranscriptRetry();
        return {{ action: "retry" }};
      }}
    }}
    return {{ action: "surface", kind: "context_overflow",
      errorText: "Context pressure persists after tool-result truncation",
      userText: "Context pressure persists after tool-result truncation" }};
  }}
'''
        body = compaction.sub(lambda m: truncate_branch + m.group(0), body, count=1)
        insert = f'''
  /* {CONTEXT_MARKER} */
  const __abPressure = !{param}.aborted && !{param}.signalOwnedInterruption
    && !{param}.promptError
    ? __abV96ContextPressure({param}.attempt, {param}.assistantOverflowCandidate?.message)
    : null;
  if (__abPressure) {{
    console.warn("[agent-bench-continuation-decision] " + JSON.stringify({{
      decision: "recover", recovery_reason: "context_pressure", ...__abPressure }}));
    {param} = {{ ...{param},
      promptError: new Error("Context window exceeded under preflight pressure"),
      attempt: {{ ...{param}.attempt,
        preflightRecovery: {{ ...__abPressure, source: "mid-turn" }} }} }};
  }}
  /* AB_HARNESS_OPENCLAW_V2026_9_6_CONTEXT_PRESSURE_END */
'''
        updated = text[:start.start()] + CONTEXT_HELPER + start.group(0) + insert + body + text[end:]
        retry_states = re.findall(rf"\b({_IDENT})\s*=\s*createEmbeddedRunTerminalRetryState\(\)", updated)
        recovery_states = re.findall(rf"\b({_IDENT})\s*=\s*createEmbeddedRunContextRecoveryState\(\)", updated)
        retry_anchor = re.compile(
            rf"(?P<prefix>if\s*\(\s*(?P<recovery>{_IDENT})\.action\s*===\s*[\"'`]retry[\"'`]\s*\)\s*\{{)"
            rf"(?=\s*{_IDENT}\s*=\s*(?P=recovery)\.thinkLevel)"
        )
        if len(retry_states) != 1 or len(recovery_states) != 1 or len(retry_anchor.findall(updated)) != 1:
            raise RuntimeError(f"v9.6 retry/compaction state anchor missing in {path}")
        retry_state, recovery_state = retry_states[0], recovery_states[0]
        reset = f'''
  const __abCompactions = Number({recovery_state}.autoCompactionCount ?? 0);
  if (process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY === "continue"
      && __abCompactions > Number({retry_state}.__abCompactions ?? 0)) {{
    {retry_state}.emptyResponseAttempts = 0;
    {retry_state}.lengthNoProgressAttempts = 0;
    {retry_state}.__abCompactions = __abCompactions;
  }}
'''
        updated = retry_anchor.sub(lambda m: m.group("prefix") + reset, updated, count=1)
        shared._write(path, updated)
        changed.append(str(path.relative_to(root)))
    if not any(CONTEXT_MARKER in shared._read(path) for path in shared._js_files(root / "dist")):
        raise RuntimeError("no v9.6 pressure recovery definitions found")
    return sorted(changed)


def _require_version(root: Path) -> None:
    version = shared._runtime_version(root)
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"expected OpenClaw {EXPECTED_VERSION}, found {version}")


def _patch_loop_probe(root: Path) -> list[str]:
    """Cover the readable loop module and both minified workers."""
    changed = shared._patch_loop_veto_recovery_probe(root)
    definition = re.compile(
        rf"function detectToolCallLoop\((?P<state>{_IDENT}),(?P<tool>{_IDENT}),"
        rf"(?P<params>{_IDENT}),(?P<config>{_IDENT}),(?P<scope>{_IDENT})\)\s*\{{"
    )
    for path in shared._js_files(root / "dist"):
        text = shared._read(path)
        if "function detectToolCallLoop(" not in text or shared.LOOP_VETO_RECOVERY_PROBE_MARKER in text:
            continue
        matches = list(definition.finditer(text))
        if len(matches) != 1:
            raise RuntimeError(f"v9.6 worker loop definition missing/ambiguous in {path}")
        names = matches[0].groupdict()
        state, tool, params, config, scope = (names[name] for name in ("state", "tool", "params", "config", "scope"))
        anchor = f"{state} = __abFilterSessionsYieldHistoryAfterOwnership({state},{scope}?.runId);"
        if text.count(anchor) != 1 or not all(shared._has_js_lexical_binding(text, name) for name in (
            "selectHistoryForScope", "hashToolCall",
        )):
            raise RuntimeError(f"v9.6 worker loop ownership anchor missing in {path}")
        insert = f'''
  /* {shared.LOOP_VETO_RECOVERY_PROBE_MARKER} */
  if ({config}?.enabled && process.env.OPENCLAW_AGENT_BENCH_LOOP_VETO_RECOVERY_PROBE === "1") {{
    const history = selectHistoryForScope({state}.toolCallHistory ?? [], {scope});
    const currentHash = hashToolCall({tool}, {params});
    for (let i = history.length - 1; i >= 0; i -= 1) {{
      const record = history[i];
      if (!record || record.toolName !== {tool} || record.argsHash !== currentHash) continue;
      if (record.outcomeKind === "tool-loop-veto") return {{ stuck: false }};
      break;
    }}
  }}
  /* AB_HARNESS_OPENCLAW_V2026_9_2_LOOP_VETO_RECOVERY_PROBE_END */
'''
        shared._write(path, text.replace(anchor, anchor + insert, 1))
        changed.append(str(path.relative_to(root)))
    return sorted(changed)


def _native_contracts(root: Path) -> dict[str, list[str]]:
    """Check the new native owners rather than requiring obsolete markers."""
    files = {path: shared._read(path) for path in shared._js_files(root / "dist")}
    contracts = {
        "child_settlement": (
            "function settleRequesterRun(",
            ("meta.continuationPending", "markRequesterTurnYielded(",
             "requesterContinuationSettled", "settleRequesterAfterSessionSpawns("),
        ),
        "completion_ownership": (
            "function shouldSkipNonVisibleTurnRetry(",
            ("hasAcceptedSessionSpawn(", "itemLifecycle.activeCount", "yieldDetected"),
        ),
        "context_recovery": (
            "function recoverEmbeddedRunOverflow(",
            ("compactEmbeddedRunForRecovery(", "truncateOversizedToolResultsInSessionManager(",
             "prepareCurrentTranscriptRetry", "armPostCompactionGuard("),
        ),
        "loop_detection": (
            "function detectToolCallLoop(",
            ("getNoProgressStreak(", "tool-loop-veto", "GLOBAL_CIRCUIT_BREAKER_THRESHOLD"),
        ),
    }
    found: dict[str, list[str]] = {}
    for name, (definition, required) in contracts.items():
        targets = [(path, text) for path, text in files.items() if definition in text]
        if not targets:
            raise RuntimeError(f"missing v9.6 native {name} definition")
        for path, text in targets:
            missing = [token for token in required if token not in text]
            if missing:
                raise RuntimeError(f"incomplete native {name} in {path}: {missing}")
        found[name] = [str(path.relative_to(root)) for path, _ in targets]
    # Native OTel supplies model/tool timings, including non-blocking children.
    for name, token in (("model_otel", "openclaw.model.usage"),
                        ("tool_otel", "openclaw.tool.execution"),
                        ("post_compaction_guard", "compaction_loop_persisted"),
                        ("delegation_prefer", "function buildDelegationGuidanceSection(")):
        targets = [str(path.relative_to(root)) for path, text in files.items() if token in text]
        if not targets:
            raise RuntimeError(f"missing v9.6 native {name}")
        found[name] = targets
    return found


def patch_runtime_tree(root: Path) -> dict[str, Any]:
    root = Path(root)
    _require_version(root)
    _native_contracts(root)
    status: dict[str, Any] = {
        "upstream_version": EXPECTED_VERSION,
        "patch_strategy": "v2026.9.6 semantic dist transforms",
        "native_loop_detection": True,
        "native_otel": True,
        "native_child_settlement": True,
        "implicit_child_handoff_patch_applied": False,
        "legacy_post_compaction_backport_applied": False,
        "legacy_model_usage_patches_applied": False,
        "legacy_tool_call_patches_applied": False,
        "exec_bridge_files_changed": shared._patch_exec_bridge(root),
        "sandbox_bypass_files_changed": shared._patch_sandbox_bypass(root),
        "shared_workspace_spawn_files_changed": shared._patch_shared_workspace_spawns(root),
        "settled_tool_continuation": shared._patch_settled_tool_continuation(root),
        "context_pressure_terminal_recovery_files_changed": _patch_context_pressure(root),
        "sessions_yield_no_pending_loop_files_changed": shared._patch_sessions_yield_no_pending_loop(root),
        "loop_veto_recovery_probe_files_changed": _patch_loop_probe(root),
        "fs_bridge_files_changed": shared._patch_fs_bridge(root),
    }
    status.update(shared._patch_pricing(root))
    return status


def verify_runtime_tree(root: Path) -> dict[str, Any]:
    root = Path(root)
    _require_version(root)
    required = {
        "exec_bridge": shared.EXEC_MARKER,
        "sandbox_bypass": shared.SANDBOX_MARKER,
        "shared_workspace_spawn": shared.SHARED_WORKSPACE_SPAWN_MARKER,
        "settled_continuation_policy": shared.SETTLED_CONTINUATION_POLICY_MARKER,
        "settled_continuation_instruction": shared.SETTLED_CONTINUATION_INSTRUCTION_MARKER,
        "settled_continuation_retry": shared.SETTLED_CONTINUATION_RETRY_MARKER,
        "empty_stop_retry_limit": shared.EMPTY_STOP_RETRY_LIMIT_MARKER,
        "loop_veto_recovery_probe": shared.LOOP_VETO_RECOVERY_PROBE_MARKER,
        "sessions_yield_no_pending_loop": shared.SESSIONS_YIELD_NO_PENDING_LOOP_MARKER,
        "context_pressure_terminal_recovery": CONTEXT_MARKER,
        "fs_helper": shared.FS_HELPER_MARKER,
        "fs_run": shared.FS_RUN_MARKER,
        "fs_write": shared.FS_WRITE_MARKER,
        "fs_read": shared.FS_READ_MARKER,
        "fs_check": shared.FS_CHECK_MARKER,
    }
    files = {path: shared._read(path) for path in shared._js_files(root / "dist")}
    found = {}
    for name, marker in required.items():
        targets = [str(path.relative_to(root)) for path, text in files.items() if marker in text]
        if not targets:
            raise RuntimeError(f"missing required v9.6 marker: {name}")
        found[name] = targets
    for path, text in files.items():
        if shared.IMPLICIT_CHILD_HANDOFF_OWNERSHIP_MARKER in text:
            raise RuntimeError(f"obsolete v9.2 child settlement patch in {path}")
        if "function resolveEmptyResponseRetryInstruction(" in text:
            if not all(shared._has_js_lexical_binding(text, name) for name in (
                "classifyAssistantTurn", "resolveSettledToolBatchEvidence",
                "hasCommittedMessagingToolDeliveryEvidence", "hasAsyncActivity",
                "hasCompletionMessageSessionSpawn",
            )):
                raise RuntimeError(f"unbound v9.6 continuation helper in {path}")
            if not re.search(
                r"runtimeContinuationStarted === true\s*\|\| hasCompletionMessageSessionSpawn\(", text
            ):
                raise RuntimeError(f"missing native child ownership guard in {path}")
        for name, marker in (
            ("resolveSettledTurnFinalizationRequest", shared.SETTLED_CONTINUATION_POLICY_MARKER),
            ("resolveEmptyResponseRetryInstruction", shared.SETTLED_CONTINUATION_INSTRUCTION_MARKER),
            ("detectToolCallLoop", shared.LOOP_VETO_RECOVERY_PROBE_MARKER),
            ("recoverEmbeddedRunOverflow", CONTEXT_MARKER),
        ):
            definitions = len(re.findall(rf"function\s+{name}\(", text))
            if definitions and text.count(marker) != definitions:
                raise RuntimeError(f"not every v9.6 {name} copy is patched in {path}")
    return {
        "upstream_version": EXPECTED_VERSION,
        "markers": found,
        "native_contracts": _native_contracts(root),
        "verified": True,
    }
