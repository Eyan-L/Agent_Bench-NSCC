from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from agent_bench.harnesses.openclaw.v2026_9_2_patch import (
    CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
    CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
    CURRENT_TERMINAL_PAYLOAD_MARKER,
    EMPTY_STOP_RETRY_LIMIT_MARKER,
    EXEC_MARKER,
    FS_CHECK_MARKER,
    FS_HELPER_MARKER,
    FS_READ_MARKER,
    FS_RUN_MARKER,
    FS_WRITE_MARKER,
    LITELLM_PRICING_URL,
    LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
    LENGTH_SEMANTIC_PROGRESS_MARKER,
    LOOP_VETO_RECOVERY_PROBE_MARKER,
    NONEMPTY_LENGTH_ELIGIBILITY_MARKER,
    OPENROUTER_PRICING_URL,
    ROADMAP_CONTINUATION_INSTRUCTION,
    SANDBOX_MARKER,
    SETTLED_CONTINUATION_INSTRUCTION_MARKER,
    SETTLED_CONTINUATION_LENGTH_MARKER,
    SETTLED_CONTINUATION_POLICY_MARKER,
    SETTLED_CONTINUATION_RETRY_MARKER,
    SETTLED_CONTINUATION_TOOL_ERROR_MARKER,
    SHARED_WORKSPACE_SPAWN_MARKER,
    BOUNDED_LENGTH_CONTINUATION_MARKER,
    patch_runtime_tree,
    verify_runtime_tree,
)


def _runtime_tree(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    dist = root / "dist"
    dist.mkdir(parents=True)
    (root / "package.json").write_text(json.dumps({"version": "2026.9.2"}))

    (dist / "bash-tools.exec-runtime-test.js").write_text(
        "const prepareSpawnSpec = async () => {\n"
        "  if (opts.sandbox) {\n"
        "    if (!opts.sandbox.buildExecSpec) {\n"
        "      throw new Error(\"sandbox backend does not provide buildExecSpec\");\n"
        "    }\n"
        "    const backendExecSpec = await opts.sandbox.buildExecSpec({\n"
        "      command: execCommand, workdir: opts.containerWorkdir ?? opts.sandbox.containerWorkdir,\n"
        "      env: shellRuntimeEnv, usePty: opts.usePty,\n"
        "    });\n"
        "    return backendExecSpec;\n"
        "  }\n"
        "};\n"
    )

    (dist / "docker-test.js").write_text(
        "async function ensureSandboxContainer(params) {\n"
        "  const engine = params.engine ?? DOCKER_SANDBOX_ENGINE;\n"
        "  const slug = params.cfg.scope === \"shared\" ? \"shared\" : slugifySessionKey(params.scopeKey);\n"
        "  const prefix = params.cfg.docker.containerPrefix;\n"
        "  const containerName = buildSandboxContainerName(prefix, slug);\n"
        "  return await sandboxContainerLifecycleQueue.enqueue(containerName, async () => containerName);\n"
        "}\n"
    )

    (dist / "sessions-spawn-tool-test.js").write_text(
        "async function maybeSpawnVisibleSession(params) {\n"
        "  const worktree = params.raw.worktree === true;\n"
        "  const worktreeName = readToolStringParam(params.raw, 'worktreeName');\n"
        "  const worktreeBaseRef = readToolStringParam(params.raw, 'worktreeBaseRef');\n"
        "  return { worktree, worktreeName, worktreeBaseRef };\n"
        "}\n"
    )

    (dist / "fs-bridge-test.js").write_text(
        "import fs from \"node:fs\";\n"
        "function createSandboxFsBridge(params) { return new SandboxFsBridgeImpl(params.sandbox); }\n"
        "class SandboxFsBridgeImpl {\n"
        "  async writeFile(params) {\n"
        "    const target = this.resolveResolvedPath(params);\n"
        "    this.ensureWriteAccess(target, \"write files\");\n"
        "    const writeCheck = { target };\n"
        "    await this.pathGuard.assertPathSafety(target, writeCheck.options);\n"
        "  }\n"
        "  async runCommand(script, options = {}) {\n"
        "    const backend = this.sandbox.backend;\n"
        "    if (backend) return await backend.runShellCommand({ script });\n"
        "  }\n"
        "  async readPinnedFile(target, maxBytes) {\n"
        "    const opened = await this.pathGuard.openReadableFile(target);\n"
        "    return fs.readFileSync(opened.fd);\n"
        "  }\n"
        "  async runCheckedCommand(plan) {\n"
        "    await this.pathGuard.assertPathChecks(plan.checks);\n"
        "    return await this.runCommand(plan.script, {});\n"
        "  }\n"
        "}\n"
    )

    (dist / "pricing-test.js").write_text(
        f"const a = '{OPENROUTER_PRICING_URL}';\n"
        f"const b = '{LITELLM_PRICING_URL}';\n"
    )
    (dist / "builtin-openclaw-test.js").write_text(
        "function classifyAssistantTurn(params) {\n"
        "  const assistant = params.attempt.currentAttemptAssistant;\n"
        "  const visibleText = (params.attempt.assistantTexts ?? []).join('');\n"
        "  return {\n"
        "    assistant, visibleText, reasoningOnly: false,\n"
        "    emptyResponse: visibleText.length === 0,\n"
        "  };\n"
        "}\n"
        "function resolveSettledToolBatchEvidence(attempt) {\n"
        "  const messages = attempt.messagesSnapshot ?? [];\n"
        "  const assistant = messages.findLast((message) =>\n"
        "    message?.role === 'assistant'\n"
        "    && (message.content ?? []).some((item) =>\n"
        "      item?.type === 'toolCall' || item?.type === 'toolUse'));\n"
        "  const calls = (assistant?.content ?? []).filter((item) =>\n"
        "    item?.type === 'toolCall' || item?.type === 'toolUse');\n"
        "  const results = calls.map((call) => messages.findLast((message) =>\n"
        "    message?.role === 'toolResult' && message?.toolCallId === call.id\n"
        "    && message?.toolName === call.name));\n"
        "  return {\n"
        "    assistant,\n"
        "    allToolsProvenSettled: calls.length > 0 && results.every(Boolean)\n"
        "      && attempt.itemLifecycle.startedCount > 0\n"
        "      && attempt.itemLifecycle.completedCount\n"
        "        === attempt.itemLifecycle.startedCount\n"
        "      && attempt.itemLifecycle.activeCount === 0,\n"
        "    failedToolNames: new Set(calls.filter((_, index) =>\n"
        "      results[index]?.isError === true).map((call) => call.name)),\n"
        "    intentionalTermination: false, parkedCodeModeRun: false,\n"
        "  };\n"
        "}\n"
        "function hasCompletedMessagingToolDeliveryEvidence(attempt) {\n"
        "  return attempt.completedDelivery === true;\n"
        "}\n"
        "function hasAsyncActivity(toolMetas) {\n"
        "  return (toolMetas ?? []).some((entry) => entry.asyncStarted === true);\n"
        "}\n"
        "function shouldSkipNonVisibleTurnRetry() { return false; }\n"
        "function shouldApplyNonVisibleTurnRetryGuard() { return true; }\n"
        "function isOllamaIncompleteTurnProvider() { return false; }\n"
        "function hasPositiveOutputTokenUsage() { return false; }\n"
        "function isZeroUsageEmptyStopAssistantTurn() { return false; }\n"
        "const EMPTY_RESPONSE_RETRY_INSTRUCTION = "
        '"The previous attempt did not produce a user-visible answer. Continue '
        "from the current state and produce the visible answer now. Do not "
        'restart from scratch.";\n'
        "function resolveSettledTurnFinalizationRequest(input) {\n"
        "  const terminalAssistant = resolveCurrentAttemptAssistant(input.attempt);\n"
        "  return terminalAssistant ? 'isolated-finalization' : null;\n"
        "}\n"
        "function resolveEmptyResponseRetryInstruction(params) {\n"
        "  if (shouldSkipNonVisibleTurnRetry(params)) return null;\n"
        "  return classifyAssistantTurn(params).emptyResponse\n"
        "    ? EMPTY_RESPONSE_RETRY_INSTRUCTION : null;\n"
        "}\n"
        "function buildEmbeddedRunPayloads(params) {\n"
        "  return (params.assistantTexts ?? [])\n"
        "    .filter((text) => String(text).trim().length > 0)\n"
        "    .map((text) => ({ text }));\n"
        "}\n"
        "function resolveEmbeddedRunTerminal(input, retryState) {\n"
        "  const payloads = buildEmbeddedRunPayloads({\n"
        "    assistantTexts: input.attempt.assistantTexts,\n"
        "  });\n"
        "  const instruction = resolveEmptyResponseRetryInstruction({\n"
        "    provider: 'agent_model', modelId: 'roadmap',\n"
        "    modelApi: 'openai-completions', payloadCount: payloads.length,\n"
        "    aborted: input.aborted, timedOut: input.timedOut,\n"
        "    attempt: input.attempt\n"
        "  });\n"
        "  if (instruction && retryState.emptyResponseAttempts < 4) {\n"
        "    retryState.emptyResponseAttempts += 1;\n"
        "    return { action: 'retry', instruction, payloads };\n"
        "  }\n"
        "  return { action: 'complete', payloads };\n"
        "}\n"
    )
    (dist / "embedded-agent-test.js").write_text(
        "function resolveSettledToolBatchEvidence(attempt) {\n"
        "  const messages = attempt.messagesSnapshot ?? [];\n"
        "  const assistant = messages.findLast((message) =>\n"
        "    message?.role === 'assistant'\n"
        "    && (message.content ?? []).some((item) =>\n"
        "      item?.type === 'toolCall' || item?.type === 'toolUse'));\n"
        "  const calls = (assistant?.content ?? []).filter((item) =>\n"
        "    item?.type === 'toolCall' || item?.type === 'toolUse');\n"
        "  const settled = calls.length > 0 && calls.every((call) =>\n"
        "    messages.some((message) => message?.role === 'toolResult'\n"
        "      && message?.toolCallId === call.id\n"
        "      && message?.toolName === call.name))\n"
        "    && attempt.itemLifecycle.startedCount > 0\n"
        "    && attempt.itemLifecycle.completedCount\n"
        "      === attempt.itemLifecycle.startedCount\n"
        "    && attempt.itemLifecycle.activeCount === 0;\n"
        "  return { assistant, allToolsProvenSettled: settled };\n"
        "}\n"
        "function hasMessagingToolDeliveryEvidence(attempt) {\n"
        "  return attempt.didSendViaMessagingTool === true;\n"
        "}\n"
        "function hasAsyncActivity(toolMetas) {\n"
        "  return (toolMetas ?? []).some((entry) => entry.asyncStarted === true);\n"
        "}\n"
        "function isEmbeddedRunTerminalAbort(outcome) {\n"
        "  return outcome === 'abort';\n"
        "}\n"
        "function isEmbeddedRunTerminalTimeout(outcome) {\n"
        "  return outcome === 'timeout';\n"
        "}\n"
        "const log$3 = { warn() {} };\n"
        "async function prepareTerminalWithSettledTurnFinalization(input) {\n"
        "  return input;\n"
        "}\n"
        "function resolveEmbeddedRunTerminal(input) {\n"
        "  const nextEmptyResponseRetryInstruction = "
        "input.settledTurnFinalizationAttempted ? null : "
        "resolveEmptyResponseRetryInstruction(input);\n"
        "  if (nextEmptyResponseRetryInstruction) return { action: 'retry' };\n"
        "}\n"
        "const retryOptions = {\n"
        "  maxEmptyResponseRetryAttempts: 1,\n"
        "};\n"
        "function retryTerminal(input, retryState, nextReasoningOnlyRetryInstruction, "
        "nextEmptyResponseRetryInstruction) {\n"
        "  const runParams = input.runParams;\n"
        "  if (!nextReasoningOnlyRetryInstruction && nextEmptyResponseRetryInstruction "
        "&& retryState.emptyResponseAttempts < input.maxEmptyResponseRetryAttempts) {\n"
        "    retryState.emptyResponseAttempts += 1;\n"
        "    input.activateInternalPrompt(nextEmptyResponseRetryInstruction);\n"
        "    log$3.warn(`empty response detected: runId=${runParams.runId} "
        "sessionId=${runParams.sessionId} provider=${input.activeErrorContext.provider}/"
        "${input.activeErrorContext.model} — retrying "
        "${retryState.emptyResponseAttempts}/${input.maxEmptyResponseRetryAttempts} "
        "with visible-answer continuation`);\n"
        "    return { action: \"retry\" };\n"
        "  }\n"
        "}\n"
    )
    (dist / "worker-test.mjs").write_text(
        "function resolveSettledToolBatchEvidence(attempt){"
        "return attempt.settledToolEvidence??{allToolsProvenSettled:false}}"
        "function hasMessagingToolDeliveryEvidence(attempt){"
        "return attempt.didSendViaMessagingTool===true}"
        "function hasAsyncActivity(toolMetas){"
        "return(toolMetas??[]).some((entry)=>entry.asyncStarted===true)}"
        "function isEmbeddedRunTerminalAbort(outcome){return outcome===`abort`}"
        "function isEmbeddedRunTerminalTimeout(outcome){return outcome===`timeout`}"
        "function retryMinified(Ot,Dn,mi,Si){"
        "if(!mi&&Si&&Dn.emptyResponseAttempts<Ot.maxEmptyResponseRetryAttempts)"
        "return Dn.emptyResponseAttempts+=1,Ot.activateInternalPrompt(Si),"
        "log$190.warn(`empty response detected: runId=${Zt.runId} "
        "sessionId=${Zt.sessionId} provider=${Ot.activeErrorContext.provider}/"
        "${Ot.activeErrorContext.model} — retrying ${Dn.emptyResponseAttempts}/"
        "${Ot.maxEmptyResponseRetryAttempts} with visible-answer continuation`),"
        "{action:`retry`};}\n"
    )
    (dist / "tool-loop-detection-test.js").write_text(
        "function detectToolCallLoop(state, toolName, params, config, scope) {\n"
        "  if (!config?.enabled) return { stuck: false };\n"
        "  const history = selectHistoryForScope(state.toolCallHistory ?? [], scope);\n"
        "  const currentHash = hashToolCall(toolName, params);\n"
        "  const noProgress = getNoProgressStreak(history, toolName, currentHash);\n"
        "  return noProgress.count >= 20 ? { stuck: true } : { stuck: false };\n"
        "}\n"
    )
    return root


def test_v2026_9_2_semantic_patcher_is_complete_and_idempotent(tmp_path):
    root = _runtime_tree(tmp_path)

    first = patch_runtime_tree(root)
    verified = verify_runtime_tree(root)

    exec_text = (root / "dist/bash-tools.exec-runtime-test.js").read_text()
    docker_text = (root / "dist/docker-test.js").read_text()
    spawn_text = (root / "dist/sessions-spawn-tool-test.js").read_text()
    fs_text = (root / "dist/fs-bridge-test.js").read_text()
    pricing_text = (root / "dist/pricing-test.js").read_text()
    builtin_text = (root / "dist/builtin-openclaw-test.js").read_text()
    embedded_text = (root / "dist/embedded-agent-test.js").read_text()
    worker_text = (root / "dist/worker-test.mjs").read_text()
    loop_detection_text = (
        root / "dist/tool-loop-detection-test.js"
    ).read_text()

    assert EXEC_MARKER in exec_text
    assert SANDBOX_MARKER in docker_text
    assert SHARED_WORKSPACE_SPAWN_MARKER in spawn_text
    assert "OPENCLAW_AGENT_BENCH_SHARED_WORKSPACE" in spawn_text
    assert "params.raw.worktree === true && !sharedWorkspaceSpawn" in spawn_text
    assert "worktreeName = sharedWorkspaceSpawn ? void 0" in spawn_text
    assert "worktreeBaseRef = sharedWorkspaceSpawn ? void 0" in spawn_text
    assert SETTLED_CONTINUATION_POLICY_MARKER in builtin_text
    assert (
        'OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY === "continue") return null'
        in builtin_text
    )
    assert SETTLED_CONTINUATION_INSTRUCTION_MARKER in builtin_text
    assert ROADMAP_CONTINUATION_INSTRUCTION in builtin_text
    assert SETTLED_CONTINUATION_RETRY_MARKER in builtin_text
    assert SETTLED_CONTINUATION_LENGTH_MARKER in builtin_text
    assert NONEMPTY_LENGTH_ELIGIBILITY_MARKER in builtin_text
    assert CURRENT_ASSISTANT_EMPTY_STOP_MARKER in builtin_text
    assert CURRENT_TERMINAL_PAYLOAD_MARKER in builtin_text
    assert "hasCurrentTerminalPayload:" in builtin_text
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in builtin_text
    assert "__abSettledToolEvidence.allToolsProvenSettled" in builtin_text
    assert "!__abSettledToolEvidence.hasUnsettledToolError" not in builtin_text
    assert "const __abAssistantState = classifyAssistantTurn(params)" in builtin_text
    assert '__abAssistantState.assistant?.stopReason === "length"' in builtin_text
    assert "const __abExplicitDone" in builtin_text
    assert "!__abExplicitDone" in builtin_text
    assert "!__abAssistantState.reasoningOnly" in builtin_text
    assert "__abAssistantState.emptyResponse" in builtin_text
    assert "|| __abCurrentAssistantEmptyStop" in builtin_text
    assert "|| __abIncompleteLength" in builtin_text
    assert "&& __abRoadmapContinuationEligible" in builtin_text
    assert 'params.attempt?.terminal?.kind !== "timeout"' in builtin_text
    assert "!hasCompletedMessagingToolDeliveryEvidence(params.attempt)" in (
        builtin_text
    )
    assert "!hasAsyncActivity(params.attempt.toolMetas)" in builtin_text
    assert builtin_text.index(SETTLED_CONTINUATION_RETRY_MARKER) < (
        builtin_text.index("shouldSkipNonVisibleTurnRetry(params)")
    )
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in embedded_text
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in embedded_text
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in worker_text
    assert BOUNDED_LENGTH_CONTINUATION_MARKER in embedded_text
    assert BOUNDED_LENGTH_CONTINUATION_MARKER in worker_text
    assert LENGTH_SEMANTIC_PROGRESS_MARKER in embedded_text
    assert LENGTH_SEMANTIC_PROGRESS_MARKER in worker_text
    assert 'Ot.attemptAssistant?.stopReason === "length"' in worker_text
    assert 'input.attemptAssistant?.stopReason === "length"' in embedded_text
    assert "if (__abLengthContinuation)" in embedded_text
    assert "lengthNoProgressAttempts" in embedded_text
    assert "lengthProgressSignatures" in embedded_text
    assert "OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES" in embedded_text
    assert "if (raw === void 0 || raw === \"\") return 1" in embedded_text
    assert "Math.min(parsed, 16)" in embedded_text
    assert "retryState.emptyResponseAttempts = 0" in embedded_text
    assert 'message?.details?.deniedReason === "tool-loop"' in embedded_text
    assert "!__abBlockedToolLoopVeto" in embedded_text
    assert "!__abControlActivity" in embedded_text
    assert LOOP_VETO_RECOVERY_PROBE_MARKER in loop_detection_text
    assert "OPENCLAW_AGENT_BENCH_LOOP_VETO_RECOVERY_PROBE" in (
        loop_detection_text
    )
    assert 'record.outcomeKind === "tool-loop-veto"' in loop_detection_text
    assert "return { stuck: false }" in loop_detection_text
    # Skipping isolated finalization leaves the existing outer, tools-enabled
    # empty-response retry path intact.
    assert (
        "settledTurnFinalizationAttempted ? null : "
        "resolveEmptyResponseRetryInstruction"
    ) in embedded_text
    assert "return { action: 'retry' }" in embedded_text
    assert "disableTools" not in embedded_text
    for marker in (
        FS_HELPER_MARKER,
        FS_RUN_MARKER,
        FS_WRITE_MARKER,
        FS_READ_MARKER,
        FS_CHECK_MARKER,
    ):
        assert marker in fs_text
    assert OPENROUTER_PRICING_URL not in pricing_text
    assert LITELLM_PRICING_URL not in pricing_text
    assert first["openrouter_pricing_url_replacements"] == 1
    assert first["litellm_pricing_url_replacements"] == 1
    assert first["legacy_post_compaction_backport_applied"] is False
    assert first["legacy_sessions_patch_timeout_applied"] is False
    assert first["raw_response_patch_applied"] is False
    assert verified["verified"] is True

    snapshot = {
        path.name: path.read_text()
        for path in (root / "dist").iterdir()
        if path.suffix in {".js", ".mjs", ".cjs"}
    }
    second = patch_runtime_tree(root)
    assert second["exec_bridge_files_changed"] == []
    assert second["sandbox_bypass_files_changed"] == []
    assert second["shared_workspace_spawn_files_changed"] == []
    assert second["settled_tool_continuation"]["files_changed"] == []
    assert second["loop_veto_recovery_probe_files_changed"] == []
    assert second["fs_bridge_files_changed"] == []
    assert second["openrouter_pricing_url_replacements"] == 0
    assert second["litellm_pricing_url_replacements"] == 0
    assert snapshot == {
        path.name: path.read_text()
        for path in (root / "dist").iterdir()
        if path.suffix in {".js", ".mjs", ".cjs"}
    }


def test_v2026_9_2_transformed_retry_bundles_parse_with_real_node(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)

    for relative in ("dist/embedded-agent-test.js", "dist/worker-test.mjs"):
        completed = subprocess.run(
            [*_node_command(), "--check", str(root / relative)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, (
            relative + "\n" + completed.stdout + completed.stderr
        )


def test_v2026_9_2_length_has_independent_no_progress_budget(tmp_path):
    root = _runtime_tree(tmp_path)

    patch_runtime_tree(root)

    builtin = (root / "dist/builtin-openclaw-test.js").read_text()
    embedded = (root / "dist/embedded-agent-test.js").read_text()
    assert "__abAssistantState.emptyResponse" in builtin
    assert "|| __abCurrentAssistantEmptyStop" in builtin
    assert "|| __abIncompleteLength" in builtin
    assert '__abAssistantState.assistant?.stopReason === "length"' in builtin
    assert "!__abExplicitDone" in builtin
    assert builtin.count("return EMPTY_RESPONSE_RETRY_INSTRUCTION;") == 1
    assert "maxEmptyResponseRetryAttempts" in embedded
    assert f"process.env.OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES" in embedded
    assert "Math.min(parsed, 16)" in embedded
    assert BOUNDED_LENGTH_CONTINUATION_MARKER in embedded
    assert "__abLengthContinuation" in embedded
    assert "retryState.emptyResponseAttempts += 1" in embedded
    assert "retryState.lengthNoProgressAttempts += 1" in embedded
    assert "retryState.lengthProgressSignatures" in embedded
    assert "OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES" in embedded
    assert "__abLengthProgressEntries.length > 0" in embedded
    assert "__abLengthProgressEntries.every(Boolean)" in embedded

    safety_guards = (
        "!params.aborted",
        "!params.timedOut",
        'params.attempt?.terminal?.kind !== "failed"',
        'params.attempt?.terminal?.kind !== "timeout"',
        "!params.attempt.clientToolCalls",
        "!params.attempt.yieldDetected",
        "!hasCompletedMessagingToolDeliveryEvidence(params.attempt)",
        "__abSettledToolEvidence.allToolsProvenSettled",
        "!hasAsyncActivity(params.attempt.toolMetas)",
    )
    assert all(guard in builtin for guard in safety_guards)
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in builtin
    assert "!__abSettledToolEvidence.hasUnsettledToolError" not in builtin


def _advance_empty_stop_streak(attempts, event):
    """Model the generated patch's retry-state transition contract."""
    genuine_progress = event in {"settled_tool", "assistant_progress"}
    if genuine_progress:
        attempts = 0
    if event != "empty":
        return attempts, False
    if attempts >= 4:
        return attempts, True
    return attempts + 1, False


def test_v2026_9_2_empty_stop_budget_is_consecutive(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)

    embedded = (root / "dist/embedded-agent-test.js").read_text()
    reset_index = embedded.index("retryState.emptyResponseAttempts = 0")
    increment_index = embedded.index("retryState.emptyResponseAttempts += 1")
    assert reset_index < increment_index
    assert "__abProgressToolEvidence.allToolsProvenSettled" in embedded
    assert "__abMeaningfulAssistantProgress" in embedded
    assert "!isEmbeddedRunTerminalAbort(input.terminalState.outcome)" in embedded
    assert "!isEmbeddedRunTerminalTimeout(input.terminalState.outcome)" in embedded
    assert "classifyAssistantTurn" not in embedded
    assert "hasCompletedMessagingToolDeliveryEvidence" not in embedded
    assert "hasMessagingToolDeliveryEvidence(input.attempt)" in embedded

    attempts = 0
    exhausted = False
    for _ in range(5):
        attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
    assert attempts == 4
    assert exhausted is True

    attempts = 0
    for _ in range(4):
        attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
        assert exhausted is False
    attempts, exhausted = _advance_empty_stop_streak(attempts, "settled_tool")
    assert (attempts, exhausted) == (0, False)
    attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
    assert (attempts, exhausted) == (1, False)

    attempts = 3
    attempts, exhausted = _advance_empty_stop_streak(
        attempts, "assistant_progress"
    )
    assert (attempts, exhausted) == (0, False)
    attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
    assert (attempts, exhausted) == (1, False)


@pytest.mark.parametrize(
    "control_event",
    ["reasoning_only", "internal_prompt", "aborted", "timeout", "loop_veto"],
)
def test_v2026_9_2_control_activity_does_not_reset_empty_streak(
    control_event,
):
    attempts = 3
    attempts, exhausted = _advance_empty_stop_streak(attempts, control_event)
    assert (attempts, exhausted) == (3, False)
    attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
    assert (attempts, exhausted) == (4, False)
    attempts, exhausted = _advance_empty_stop_streak(attempts, "empty")
    assert (attempts, exhausted) == (4, True)


def _node_command() -> list[str]:
    node = shutil.which("node")
    if node:
        return [node]
    sif = os.environ.get("AB_TEST_OPENCLAW_NODE_SIF")
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime and sif and Path(sif).is_file():
        return [runtime, "exec", sif, "node"]
    pytest.skip(
        "Node runtime unavailable; set AB_TEST_OPENCLAW_NODE_SIF to run "
        "the transformed-bundle execution contract"
    )


def test_v2026_9_2_current_empty_stop_ignores_historical_visible_text(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    bundle = root / "dist/builtin-openclaw-test.js"
    bundle.write_text(bundle.read_text() + r'''
function terminalInput({
  content = [], aborted = false, settled = true, completedDelivery = false,
  asyncStarted = false, yielded = false, timedOut = false,
  toolFailed = false,
} = {}) {
  const toolCall = { type: "toolCall", id: "call-1", name: "edit" };
  const messagesSnapshot = [{ role: "assistant", content: [toolCall] }];
  if (settled) {
    messagesSnapshot.push({
      role: "toolResult", toolCallId: "call-1", toolName: "edit",
      isError: toolFailed,
      content: [{ type: "text", text: toolFailed ? "edit failed" : "edited" }],
    });
  }
  return {
    aborted,
    timedOut,
    attempt: {
      assistantTexts: ["Historical visible progress before tool calls."],
      currentAttemptAssistant: { content, stopReason: "stop" },
      terminal: { kind: "ok" },
      messagesSnapshot,
      itemLifecycle: {
        startedCount: 1,
        completedCount: settled ? 1 : 0,
        activeCount: settled ? 0 : 1,
      },
      toolMetas: [{ toolCallId: "call-1", asyncStarted }],
      yieldDetected: yielded,
      completedDelivery,
      lastToolError: toolFailed ? { toolName: "edit" } : undefined,
    },
  };
}
function assert(value, message) {
  if (!value) throw new Error(message);
}
const retryState = { emptyResponseAttempts: 0 };
const historicalThenEmpty = resolveEmbeddedRunTerminal(
  terminalInput(), retryState,
);
assert(
  historicalThenEmpty.action === "retry",
  "current empty stop must remain eligible despite historical visible text",
);
assert(
  historicalThenEmpty.payloads.length === 1,
  "the production-style path must retain cumulative historical payloads",
);
assert(
  retryState.emptyResponseAttempts === 1,
  "the production-style path must enter the retry counter",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    content: [{ type: "text", text: "Current final answer" }],
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "current visible text must not be treated as an empty stop",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    completedDelivery: true,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "a current terminal delivery must suppress current-empty-stop continuation",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    content: [{ type: "image", data: "current-terminal-payload" }],
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "a current non-text terminal payload must suppress continuation",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    settled: false,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "unsettled tools must preserve the existing safety guard",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    aborted: true,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "aborted runs must preserve the existing safety guard",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    timedOut: true,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "timed-out runs must preserve the existing safety guard",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    yielded: true,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "yielded runs must preserve the existing safety guard",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    asyncStarted: true,
  }), { emptyResponseAttempts: 0 }).action === "complete",
  "async tools must preserve the existing safety guard",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    toolFailed: true,
  }), { emptyResponseAttempts: 0 }).action === "retry",
  "a settled tool failure remains recoverable model feedback",
);
const consecutive = { emptyResponseAttempts: 0 };
for (let i = 0; i < 4; i += 1) {
  assert(
    resolveEmbeddedRunTerminal(terminalInput(), consecutive).action === "retry",
    "four consecutive current-empty stops remain retryable",
  );
}
assert(
  resolveEmbeddedRunTerminal(terminalInput(), consecutive).action === "complete",
  "the fifth consecutive current-empty stop exhausts the retry budget",
);
process.stdout.write("CURRENT_EMPTY_STOP_CONTRACT_OK\n");
''')
    env = dict(os.environ)
    env["OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY"] = "continue"
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CURRENT_EMPTY_STOP_CONTRACT_OK" in completed.stdout


def test_v2026_9_2_transformed_bundle_executes_retry_state_contract(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    bundle = root / "dist/embedded-agent-test.js"
    bundle.write_text(bundle.read_text() + r'''
function abInput({
  progress = "none", outcome = "ok", loopVeto = false,
  toolId = "call-1", toolError = false, stopReason = "stop",
  toolArgs = { query: "needle", path: "src" }, resultText = "same result",
  asyncStarted = false, includeResult = true,
} = {}) {
  const toolProgress = progress === "tool";
  const toolCall = {
    type: "toolCall", id: toolId, name: "exec", arguments: toolArgs,
  };
  const toolResult = {
    role: "toolResult", toolCallId: toolId, toolName: "exec",
    isError: toolError,
    content: [{ type: "text", text: resultText }],
    details: loopVeto
      ? { status: "blocked", deniedReason: "tool-loop" }
      : { status: "completed", durationMs: toolId.length, toolCallId: toolId },
  };
  const completed = toolProgress && includeResult;
  return {
    attempt: {
      terminal: { kind: outcome === "timeout" ? "timeout" : "ok" },
      messagesSnapshot: toolProgress
        ? [
            { role: "assistant", content: [toolCall] },
            ...(includeResult ? [toolResult] : []),
          ]
        : [],
      itemLifecycle: {
        startedCount: toolProgress ? 1 : 0,
        completedCount: completed ? 1 : 0,
        activeCount: toolProgress && !completed ? 1 : 0,
      },
      toolMetas: toolProgress ? [{
        toolCallId: toolId, isError: toolError, asyncStarted,
      }] : [],
    },
    terminalState: { outcome },
    finalAssistantVisibleText: progress === "assistant" ? "Implemented feature" : "",
    attemptAssistant: { stopReason },
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "probe", sessionId: "probe" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  };
}
function empty(state) {
  return retryTerminal(abInput(), state, null, "continue");
}
function length(state, options = {}) {
  return retryTerminal(
    abInput({ ...options, stopReason: "length" }), state, null, "continue",
  );
}
function assert(value, message) {
  if (!value) throw new Error(message);
}
process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY = "continue";
const consecutive = { emptyResponseAttempts: 0 };
for (let i = 0; i < 4; i += 1) {
  assert(empty(consecutive)?.action === "retry", "empty retry " + i);
}
assert(consecutive.emptyResponseAttempts === 4, "four empties consume budget");
assert(empty(consecutive) === undefined, "fifth consecutive empty exhausts budget");

const toolReset = { emptyResponseAttempts: 4 };
retryTerminal(abInput({ progress: "tool" }), toolReset, null, null);
assert(toolReset.emptyResponseAttempts === 0, "settled tool progress resets streak");
assert(empty(toolReset)?.action === "retry", "empty after tool progress is allowed");
assert(toolReset.emptyResponseAttempts === 1, "new tool-progress streak starts at one");

const assistantReset = { emptyResponseAttempts: 3 };
retryTerminal(abInput({ progress: "assistant" }), assistantReset, null, null);
assert(assistantReset.emptyResponseAttempts === 0, "assistant progress resets streak");

const veto = { emptyResponseAttempts: 3 };
retryTerminal(abInput({ progress: "tool", loopVeto: true }), veto, null, null);
assert(veto.emptyResponseAttempts === 3, "loop veto does not reset streak");

const timeout = { emptyResponseAttempts: 3 };
retryTerminal(abInput({ progress: "assistant", outcome: "timeout" }), timeout, null, null);
assert(timeout.emptyResponseAttempts === 3, "timeout does not reset streak");

const lengthStreak = { emptyResponseAttempts: 2, lengthNoProgressAttempts: 0 };
for (let i = 0; i < 4; i += 1) {
  assert(length(lengthStreak)?.action === "retry", "length retry " + i);
}
assert(
  lengthStreak.lengthNoProgressAttempts === 4,
  "four no-progress length turns consume the independent budget",
);
assert(
  lengthStreak.emptyResponseAttempts === 2,
  "length retries do not consume the empty-stop budget",
);
assert(length(lengthStreak) === undefined, "fifth no-progress length exhausts budget");

process.env.OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES = "1";
const configuredLength = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 0 };
assert(length(configuredLength)?.action === "retry", "configured first retry allowed");
assert(length(configuredLength) === undefined, "configured second retry is blocked");
delete process.env.OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES;

const repeatedSemanticTool = {
  emptyResponseAttempts: 4, lengthNoProgressAttempts: 4,
};
assert(
  length(repeatedSemanticTool, { progress: "tool", toolId: "call-A" })?.action
    === "retry",
  "the first successful semantic tool result is progress",
);
assert(
  repeatedSemanticTool.lengthNoProgressAttempts === 1,
  "length streak restarts at one after valid progress",
);
assert(
  repeatedSemanticTool.emptyResponseAttempts === 0,
  "the existing empty streak reset remains independent",
);
for (const toolId of ["call-B", "call-C", "call-D"]) {
  assert(
    length(repeatedSemanticTool, {
      progress: "tool", toolId,
      toolArgs: { path: "src", query: "needle" },
    })?.action === "retry",
    "same semantics with a fresh tool id consumes the no-progress budget",
  );
}
assert(
  repeatedSemanticTool.lengthNoProgressAttempts === 4,
  "fresh tool ids do not reset an unchanged semantic/result fingerprint",
);
assert(
  length(repeatedSemanticTool, { progress: "tool", toolId: "call-E" })
    === undefined,
  "repeated semantic progress eventually exhausts the length budget",
);

assert(
  length(repeatedSemanticTool, {
    progress: "tool",
    toolId: "call-F",
    toolArgs: { query: "different", path: "src" },
    resultText: "different result",
  })?.action === "retry",
  "a genuinely different successful tool execution resets the length streak",
);
assert(
  repeatedSemanticTool.lengthNoProgressAttempts === 1,
  "new semantic progress starts a fresh length streak",
);

const lengthText = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 4 };
assert(
  length(lengthText, { progress: "assistant" }) === undefined,
  "assistant text does not reset length no-progress streak",
);
const lengthVeto = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 4 };
assert(
  length(lengthVeto, { progress: "tool", loopVeto: true }) === undefined,
  "loop-veto activity does not reset length no-progress streak",
);
const lengthFailure = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 4 };
assert(
  length(lengthFailure, { progress: "tool", toolError: true }) === undefined,
  "failed settled tools do not reset length no-progress streak",
);
const lengthAsync = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 4 };
assert(
  length(lengthAsync, { progress: "tool", asyncStarted: true }) === undefined,
  "async tools do not reset length no-progress streak",
);
const lengthIncomplete = { emptyResponseAttempts: 0, lengthNoProgressAttempts: 4 };
assert(
  length(lengthIncomplete, { progress: "tool", includeResult: false })
    === undefined,
  "incomplete tools do not reset length no-progress streak",
);
process.stdout.write("TRANSFORMED_RETRY_CONTRACT_OK\n");
''')
    env = dict(os.environ)
    env["OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY"] = "continue"
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "TRANSFORMED_RETRY_CONTRACT_OK" in completed.stdout


def test_v2026_9_2_patcher_fails_closed_on_wrong_version(tmp_path):
    root = _runtime_tree(tmp_path)
    (root / "package.json").write_text(json.dumps({"version": "2026.4.22"}))

    with pytest.raises(RuntimeError, match="expected OpenClaw 2026.9.2"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_exec_anchor_moves(tmp_path):
    root = _runtime_tree(tmp_path)
    (root / "dist/bash-tools.exec-runtime-test.js").write_text("const moved = true;\n")

    with pytest.raises(RuntimeError, match="exec bridge anchor"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_continuation_anchor_moves(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    (root / "dist/builtin-openclaw-test.js").write_text(
        "const movedSettledFinalization = true;\n"
    )

    with pytest.raises(
        RuntimeError, match="settled-turn finalization anchor"
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_retry_eligibility_anchor_moves(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    path = root / "dist/builtin-openclaw-test.js"
    path.write_text(
        path.read_text().replace(
            "function resolveEmptyResponseRetryInstruction(params)",
            "function movedEmptyResponseRetryInstruction(params)",
        )
    )

    with pytest.raises(
        RuntimeError, match="empty-response retry eligibility anchor"
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_without_terminal_payload_call_anchor(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    path = root / "dist/builtin-openclaw-test.js"
    path.write_text(
        path.read_text().replace(
            "payloadCount: payloads.length",
            "movedPayloadCount: payloads.length",
        )
    )

    with pytest.raises(
        RuntimeError, match="current terminal payload call anchor"
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_stale_current_empty_payload_patch(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/builtin-openclaw-test.js"
    path.write_text(
        path.read_text().replace(
            CURRENT_TERMINAL_PAYLOAD_MARKER,
            "AB_HARNESS_OPENCLAW_V2026_9_2_OLD_PAYLOAD_PROVENANCE_BEGIN",
        )
    )

    with pytest.raises(RuntimeError, match="terminal-payload provenance"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_retry_counter_anchor_moves(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    for name in ("embedded-agent-test.js", "worker-test.mjs"):
        path = root / "dist" / name
        path.write_text(
            path.read_text().replace(
                "emptyResponseAttempts",
                "movedEmptyResponseAttempts",
            )
        )

    with pytest.raises(
        RuntimeError, match="empty-response retry counter anchor"
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_counter_patch_without_streak_reset(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/embedded-agent-test.js"
    old_patch, count = re.subn(
        rf"^[ \t]*/\* {CONSECUTIVE_EMPTY_STOP_RESET_MARKER} \*/.*?"
        r"^[ \t]*/\* AB_HARNESS_OPENCLAW_V2026_9_2_"
        r"CONSECUTIVE_EMPTY_STOP_RESET_V2_END \*/\n?",
        "",
        path.read_text(),
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert count == 1
    path.write_text(old_patch)

    with pytest.raises(RuntimeError, match="stale or incomplete"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_old_length_identity_patch(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/embedded-agent-test.js"
    path.write_text(
        path.read_text().replace(
            LENGTH_SEMANTIC_PROGRESS_MARKER,
            "REMOVED_LENGTH_SEMANTIC_PROGRESS_MARKER",
        )
    )

    with pytest.raises(RuntimeError, match="stale or incomplete"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_legacy_consecutive_reset(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/embedded-agent-test.js"
    text = path.read_text().replace(
        CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
        LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
    ).replace(
        "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_V2_END",
        "AB_HARNESS_OPENCLAW_V2026_9_2_CONSECUTIVE_EMPTY_STOP_RESET_END",
    )
    path.write_text(text)

    with pytest.raises(RuntimeError, match="stale or incomplete"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_unbounded_length_patch(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/embedded-agent-test.js"
    path.write_text(
        path.read_text().replace(
            BOUNDED_LENGTH_CONTINUATION_MARKER,
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "UNBOUNDED_LENGTH_CONTINUATION_BEGIN",
        )
    )

    with pytest.raises(RuntimeError, match="stale unbounded-length"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_on_missing_reset_binding(tmp_path):
    root = _runtime_tree(tmp_path)
    path = root / "dist/embedded-agent-test.js"
    path.write_text(
        path.read_text().replace(
            "function hasMessagingToolDeliveryEvidence(attempt) {\n"
            "  return attempt.didSendViaMessagingTool === true;\n"
            "}\n",
            "",
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "consecutive-empty reset lexical bindings missing.*"
            "hasMessagingToolDeliveryEvidence"
        ),
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_loop_recovery_anchor_moves(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    (root / "dist/tool-loop-detection-test.js").write_text(
        "function detectToolCallLoop() { return { stuck: false }; }\n"
    )

    with pytest.raises(RuntimeError, match="loop-veto recovery anchor"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_rejects_stale_stop_only_continuation(tmp_path):
    root = _runtime_tree(tmp_path)
    path = root / "dist/builtin-openclaw-test.js"
    path.write_text(
        path.read_text().replace(
            "function resolveEmptyResponseRetryInstruction(params) {",
            "function resolveEmptyResponseRetryInstruction(params) {\n"
            f"/* {SETTLED_CONTINUATION_RETRY_MARKER} */",
        )
    )

    with pytest.raises(RuntimeError, match="predates current-empty-stop"):
        patch_runtime_tree(root)


def test_v2026_9_2_builder_does_not_reapply_legacy_runtime_patches():
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "agent_bench/deployments/pbs/build_oc_v2026_9_2_sif.sh"
    ).read_text()

    assert "v2026_9_2_patch.py" in script
    assert "audit_oc_v2026_9_2.sh" in script
    assert ".agent-bench-v2026.9.2.json" in script
    assert "shared_workspace_spawn" in script
    assert "settled_tool_continuation" in script
    assert 'mkdir -p "$(dirname "$OUT_SIF")" "${HOME}/agent-bench-sif-build"' in script
    assert "apply_patch.py apply" not in script
    assert "runtime_compat_patch.py" not in script
    assert "--session-patch-timeout-ms" not in script

    audit = (
        repo_root / "agent_bench/deployments/pbs/audit_oc_v2026_9_2.sh"
    ).read_text()
    assert SETTLED_CONTINUATION_POLICY_MARKER in audit
    assert SETTLED_CONTINUATION_INSTRUCTION_MARKER in audit
    assert SETTLED_CONTINUATION_RETRY_MARKER in audit
    assert SETTLED_CONTINUATION_LENGTH_MARKER in audit
    assert NONEMPTY_LENGTH_ELIGIBILITY_MARKER in audit
    assert CURRENT_ASSISTANT_EMPTY_STOP_MARKER in audit
    assert CURRENT_TERMINAL_PAYLOAD_MARKER in audit
    assert BOUNDED_LENGTH_CONTINUATION_MARKER in audit
    assert LENGTH_SEMANTIC_PROGRESS_MARKER in audit
    assert "stale_unbounded_length_continuation_present" in audit
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in audit
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in audit
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in audit
    assert LOOP_VETO_RECOVERY_PROBE_MARKER in audit
