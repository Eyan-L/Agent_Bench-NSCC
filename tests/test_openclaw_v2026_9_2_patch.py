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
    CONTEXT_RECOVERY_STREAK_RESET_MARKER,
    CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER,
    CONTINUATION_DECISION_DIAGNOSTIC_MARKER,
    CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
    EMPTY_STOP_RETRY_LIMIT_MARKER,
    EXEC_MARKER,
    FS_CHECK_MARKER,
    FS_HELPER_MARKER,
    FS_READ_MARKER,
    FS_RUN_MARKER,
    FS_WRITE_MARKER,
    LITELLM_PRICING_URL,
    LEGACY_CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
    LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER,
    LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
    LENGTH_SEMANTIC_PROGRESS_MARKER,
    LOOP_VETO_RECOVERY_PROBE_MARKER,
    NONEMPTY_LENGTH_ELIGIBILITY_MARKER,
    OPENROUTER_PRICING_URL,
    ROADMAP_CONTINUATION_INSTRUCTION,
    RUNTIME_OWNER_CONTINUATION_MARKER,
    SANDBOX_MARKER,
    SETTLED_CONTINUATION_INSTRUCTION_MARKER,
    SETTLED_CONTINUATION_LENGTH_MARKER,
    SETTLED_CONTINUATION_POLICY_MARKER,
    SETTLED_CONTINUATION_RETRY_MARKER,
    SETTLED_CONTINUATION_TOOL_ERROR_MARKER,
    SHARED_WORKSPACE_SPAWN_MARKER,
    TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER,
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
        "  const assistant = params.attempt.currentAttemptAssistant\n"
        "    ?? params.attempt.currentAttemptCompletedAssistant;\n"
        "  const visibleText = (params.attempt.assistantTexts ?? []).join('');\n"
        "  const content = assistant?.content;\n"
        "  const reasoningOnly = Array.isArray(content) && content.length > 0\n"
        "    && content.every((item) => item?.type === 'reasoning'\n"
        "      || item?.type === 'thinking');\n"
        "  return {\n"
        "    assistant, visibleText, reasoningOnly,\n"
        "    emptyResponse: visibleText.length === 0 && !reasoningOnly,\n"
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
        "  const failedToolNames = new Set(calls.filter((_, index) =>\n"
        "    results[index]?.isError === true).map((call) => call.name));\n"
        "  return {\n"
        "    assistant,\n"
        "    allToolsProvenSettled: calls.length > 0 && results.every(Boolean)\n"
        "      && attempt.itemLifecycle.startedCount > 0\n"
        "      && attempt.itemLifecycle.completedCount\n"
        "        === attempt.itemLifecycle.startedCount\n"
        "      && attempt.itemLifecycle.activeCount === 0,\n"
        "    failedToolNames,\n"
        "    hasUnsettledToolError: Boolean(attempt.lastToolError\n"
        "      && !failedToolNames.has(attempt.lastToolError.toolName)),\n"
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
    )
    (dist / "embedded-terminal-call-test.js").write_text(
        "function buildEmbeddedRunPayloads(params) {\n"
        "  return (params.assistantTexts ?? [])\n"
        "    .filter((text) => String(text).trim().length > 0)\n"
        "    .map((text) => ({ text }));\n"
        "}\n"
        "function resolveEmbeddedRunTerminal(input, retryState) {\n"
        "  const payloads = buildEmbeddedRunPayloads({\n"
        "    assistantTexts: input.attempt.assistantTexts,\n"
        "  });\n"
        "  const payloadCount = payloads.length;\n"
        "  const aborted = input.aborted;\n"
        "  const timedOut = input.timedOut;\n"
        "  const attempt = input.attempt;\n"
        "  const instruction = resolveEmptyResponseRetryInstruction({\n"
        "    provider: 'agent_model', modelId: 'roadmap',\n"
        "    modelApi: 'openai-completions',\n"
        "    payloadCount, aborted, timedOut, attempt\n"
        "  });\n"
        "  if (instruction && retryState.emptyResponseAttempts < 4) {\n"
        "    retryState.emptyResponseAttempts += 1;\n"
        "    return { action: 'retry', instruction, payloads };\n"
        "  }\n"
        "  return { action: 'complete', payloads };\n"
        "}\n"
    )
    (dist / "resolver-worker-test.js").write_text(
        "function classifyAssistantTurn(params) {\n"
        "  const assistant = params.attempt.currentAttemptAssistant\n"
        "    ?? params.attempt.currentAttemptCompletedAssistant;\n"
        "  const visibleText = (params.attempt.assistantTexts ?? []).join('');\n"
        "  const content = assistant?.content;\n"
        "  const reasoningOnly = Array.isArray(content) && content.length > 0\n"
        "    && content.every((item) => item?.type === 'reasoning'\n"
        "      || item?.type === 'thinking');\n"
        "  return {\n"
        "    assistant, visibleText, reasoningOnly,\n"
        "    emptyResponse: params.payloadCount === 0\n"
        "      && visibleText.length === 0 && !reasoningOnly,\n"
        "  };\n"
        "}\n"
        "function resolveSettledToolBatchEvidence(attempt) {\n"
        "  return attempt.settledToolEvidence;\n"
        "}\n"
        "function hasCompletedMessagingToolDeliveryEvidence(attempt) {\n"
        "  return attempt.completedDelivery === true;\n"
        "}\n"
        "function hasAsyncActivity(toolMetas) {\n"
        "  return (toolMetas ?? []).some((entry) => entry.asyncStarted);\n"
        "}\n"
        "const EMPTY_RESPONSE_RETRY_INSTRUCTION = "
        '"The previous attempt did not produce a user-visible answer. Continue '
        "from the current state and produce the visible answer now. Do not "
        'restart from scratch.";\n'
        "function resolveEmptyResponseRetryInstruction(params) {\n"
        "  return classifyAssistantTurn(params).emptyResponse\n"
        "    ? EMPTY_RESPONSE_RETRY_INSTRUCTION : null;\n"
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
        "function resolveOuterRetryCandidate(input) {\n"
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
    (dist / "embedded-recovery-test.js").write_text(
        "function resolveSettledToolBatchEvidence(attempt) {\n"
        "  return attempt.settledToolEvidence ?? {\n"
        "    allToolsProvenSettled: false, intentionalTermination: false,\n"
        "    parkedCodeModeRun: false, failedToolNames: new Set(),\n"
        "  };\n"
        "}\n"
        "function hasMessagingToolDeliveryEvidence(attempt) {\n"
        "  return attempt.completedDelivery === true;\n"
        "}\n"
        "function hasNonToolTerminalState(attempt) {\n"
        "  return Boolean(\n"
        "    attempt.clientToolCalls || attempt.yieldDetected\n"
        "      || attempt.didSendDeterministicApprovalPrompt\n"
        "      || attempt.heartbeatToolResponse || attempt.hasToolMediaBlockReply\n"
        "      || attempt.didDeliverSourceReplyViaMessageTool\n"
        "      || (attempt.messagingToolSourceReplyPayloads?.length ?? 0) > 0\n"
        "      || hasMessagingToolDeliveryEvidence(attempt)\n"
        "      || (attempt.acceptedSessionSpawns?.length ?? 0) > 0\n"
        "      || (attempt.successfulCronAdds ?? 0) > 0\n"
        "      || hasAsyncActivity(attempt.toolMetas)\n"
        "  );\n"
        "}\n"
        "function hasAsyncActivity(toolMetas) {\n"
        "  return (toolMetas ?? []).some((entry) => entry.asyncStarted === true);\n"
        "}\n"
        "function projectAgentRunAttemptTerminal(terminal) {\n"
        "  return {\n"
        "    aborted: terminal.kind === 'aborted',\n"
        "    promptErrorSource: terminal.promptErrorSource,\n"
        "    timedOut: terminal.kind === 'timeout',\n"
        "  };\n"
        "}\n"
        "function isEmbeddedRunTerminalInterrupted(outcome) {\n"
        "  return outcome === 'aborted' || outcome === 'timeout';\n"
        "}\n"
        "function isCurrentAttemptReplaySafe() { return false; }\n"
        "function isSilentTransportDropAssistant() { return false; }\n"
        "function isProviderRefusalAssistantError() { return false; }\n"
        "function isLikelyContextOverflowError() { return false; }\n"
        "function isContextOverflow() { return false; }\n"
        "function formatErrorMessage(error) { return String(error); }\n"
        "async function compactEmbeddedRunForRecovery(input, recovery) {\n"
        "  input.events.push(['compact', recovery.tokenBudget]);\n"
        "  input.state.autoCompactionCount += 1;\n"
        "  return { result: { ok: true, compacted: true }, previousSessionId: null };\n"
        "}\n"
        "async function recoverEmbeddedRunOverflow(input) {\n"
        "  const contextOverflowError = !input.aborted\n"
        "    && !input.signalOwnedInterruption ? (() => {\n"
        "      if (input.promptError) {\n"
        "        const errorText = formatErrorMessage(input.promptError);\n"
        "        if (isLikelyContextOverflowError(errorText)) {\n"
        "          return { text: errorText, source: 'promptError' };\n"
        "        }\n"
        "        return null;\n"
        "      }\n"
        "      if (isProviderRefusalAssistantError(input.assistantOverflowCandidate)) return null;\n"
        "      if (input.assistantOverflowCandidate\n"
        "        && input.contextTokenBudget !== void 0\n"
        "        && isContextOverflow(input.assistantOverflowCandidate, input.contextTokenBudget)) {\n"
        "        return { text: 'native overflow', source: 'assistantError' };\n"
        "      }\n"
        "      if (input.assistantErrorText\n"
        "        && isLikelyContextOverflowError(input.assistantErrorText)) {\n"
        "        return { text: input.assistantErrorText, source: 'assistantError' };\n"
        "      }\n"
        "      return null;\n"
        "    })() : null;\n"
        "  if (!contextOverflowError || !input.genericCompactionRecoveryAllowed\n"
        "    || input.contextTokenBudget === void 0) return { action: 'none' };\n"
        "  const terminal = projectAgentRunAttemptTerminal(input.attempt.terminal);\n"
        "  const preflightRecovery = input.attempt.preflightRecovery;\n"
        "  const preflightPromptBudget = terminal.promptErrorSource === \"precheck\" && preflightRecovery?.source === \"mid-turn\" && typeof preflightRecovery.promptBudgetBeforeReserve === 'number' ? preflightRecovery.promptBudgetBeforeReserve : void 0;\n"
        "  if (input.attemptCompactionCount === 0\n"
        "    && input.state.overflowCompactionAttempts < 3) {\n"
        "    input.state.overflowCompactionAttempts += 1;\n"
        "    const compaction = await compactEmbeddedRunForRecovery(input, {\n"
        "      tokenBudget: preflightPromptBudget ?? input.contextTokenBudget,\n"
        "      trigger: 'overflow', attempt: input.state.overflowCompactionAttempts,\n"
        "    });\n"
        "    if (compaction.result.compacted) {\n"
        "      if (preflightRecovery?.route === 'compact_then_truncate') {\n"
        "        input.truncateToolResults();\n"
        "      }\n"
        "      input.runParams.onAutoCompactionSucceeded?.(input.state.autoCompactionCount);\n"
        "      input.armPostCompactionGuard();\n"
        "      input.prepareCurrentTranscriptRetry();\n"
        "      return { action: 'retry' };\n"
        "    }\n"
        "  }\n"
        "  if (!input.state.toolResultTruncationAttempted) {\n"
        "    input.state.toolResultTruncationAttempted = true;\n"
        "    const truncResult = input.truncateToolResults();\n"
        "    if (truncResult.truncated) {\n"
        "      input.prepareCurrentTranscriptRetry();\n"
        "      return { action: 'retry' };\n"
        "    }\n"
        "  }\n"
        "  return { action: 'surface', kind: 'context_overflow' };\n"
        "}\n"
        "async function recoverEmbeddedRunAttempt(input) {\n"
        "  const { runInput, normalizedAttempt } = input;\n"
        "  const { attempt, attemptAssistant, currentAttemptCompletedAssistant,\n"
        "    terminalState } = normalizedAttempt;\n"
        "  const { aborted, promptErrorSource, timedOut } =\n"
        "    projectAgentRunAttemptTerminal(attempt.terminal);\n"
        "  const terminalInterrupted =\n"
        "    isEmbeddedRunTerminalInterrupted(terminalState.outcome);\n"
        "  const currentAttemptReplaySafe = isCurrentAttemptReplaySafe(attempt);\n"
        "  const settledEvidence = resolveSettledToolBatchEvidence(attempt);\n"
        "  const midTurnBatchSettled = settledEvidence.allToolsProvenSettled\n"
        "    || settledEvidence.parkedCodeModeRun;\n"
        "  const transportBatchSettled = settledEvidence.allToolsProvenSettled\n"
        "    || settledEvidence.failedToolNames.size === 0\n"
        "      && settledEvidence.parkedCodeModeRun;\n"
        "  const canContinueSettledMidTurnOverflow = promptErrorSource === \"precheck\" && attempt.preflightRecovery?.source === \"mid-turn\" && midTurnBatchSettled && !hasAsyncActivity(attempt.toolMetas);\n"
        "  const settledTransportDropAssistant = false;\n"
        "  if (!currentAttemptReplaySafe && !canContinueSettledMidTurnOverflow\n"
        "    && !settledTransportDropAssistant) return { action: 'proceed' };\n"
        "  const assistantOverflowCandidate = currentAttemptCompletedAssistant\n"
        "    ?? attemptAssistant;\n"
        "  const overflowRecovery = await recoverEmbeddedRunOverflow({\n"
        "    ...input.commonRecoveryInput, aborted, signalOwnedInterruption: false,\n"
        "    promptError: null, assistantErrorText: null, assistantOverflowCandidate,\n"
        "    attempt, attemptCompactionCount: 0,\n"
        "  });\n"
        "  if (overflowRecovery.action === 'retry') return { action: 'retry' };\n"
        "  if (overflowRecovery.action === 'surface') {\n"
        "    return {\n"
        "      action: 'complete',\n"
        "      result: {\n"
        "        errorKind: overflowRecovery.kind,\n"
        "        livenessState: 'blocked',\n"
        "      },\n"
        "    };\n"
        "  }\n"
        "  return { action: 'proceed' };\n"
        "}\n"
        "async function normalizeEmbeddedRunAttempt(input) {\n"
        "  return input.attemptEnvelope;\n"
        "}\n"
        "async function runPreparedEmbeddedLoop(input) {\n"
        "  const { contextRecoveryState, terminalRetryState } = input;\n"
        "  const admittedRunInput = input.runInput;\n"
        "  const preparedRuntime = {};\n"
        "  const normalizedAttempt = await normalizeEmbeddedRunAttempt(input);\n"
        "  const recovery = await recoverEmbeddedRunAttempt({\n"
        "    runInput: admittedRunInput,\n"
        "    preparedRuntime,\n"
        "    normalizedAttempt,\n"
        "    commonRecoveryInput: input.commonRecoveryInput, contextRecoveryState,\n"
        "  });\n"
        "  if (recovery.action === \"complete\") return recovery.result;\n"
        "  if (recovery.action === \"retry\") {\n"
        "    return { action: 'retry', length: terminalRetryState.lengthNoProgressAttempts };\n"
        "  }\n"
        "  return recovery;\n"
        "}\n"
        "const executeRecoveryContract = runPreparedEmbeddedLoop;\n"
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
    embedded_call_text = (
        root / "dist/embedded-terminal-call-test.js"
    ).read_text()
    embedded_text = (root / "dist/embedded-agent-test.js").read_text()
    recovery_text = (root / "dist/embedded-recovery-test.js").read_text()
    resolver_worker_text = (
        root / "dist/resolver-worker-test.js"
    ).read_text()
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
    assert CONTINUATION_DECISION_DIAGNOSTIC_MARKER in builtin_text
    assert RUNTIME_OWNER_CONTINUATION_MARKER in builtin_text
    assert LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER not in builtin_text
    assert "hasCurrentTerminalPayload" not in builtin_text
    assert "payloadCount, aborted, timedOut, attempt" in embedded_call_text
    assert CURRENT_ASSISTANT_EMPTY_STOP_MARKER not in embedded_call_text
    assert "hasCurrentTerminalPayload" not in embedded_call_text
    assert CURRENT_ASSISTANT_EMPTY_STOP_MARKER in resolver_worker_text
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in builtin_text
    assert "__abSettledToolEvidence.allToolsProvenSettled" in builtin_text
    assert "&& !__abActiveToolWork" in builtin_text
    assert 'push("tools_unsettled")' not in builtin_text
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
    assert "current_visible_chars" in builtin_text
    assert "current_has_tool_call" in builtin_text
    assert "current_has_payload" in builtin_text
    assert "cumulative_payload_count" in builtin_text
    assert "all_tools_settled" in builtin_text
    assert "terminal_kind" in builtin_text
    assert "last_tool_error" in builtin_text
    assert 'params.attempt?.terminal?.kind !== "timeout"' in builtin_text
    assert (
        "hasCompletedMessagingToolDeliveryEvidence(params.attempt)"
        in builtin_text
    )
    assert "&& !__abCompletedMessagingDelivery" in builtin_text
    assert (
        "&& !(__abCurrentAssistantEmptyShape "
        "&& __abCurrentAssistantHasPayload)" in builtin_text
    )
    assert "hasAsyncActivity(params.attempt.toolMetas)" in builtin_text
    assert "&& !__abAsyncActivity" in builtin_text
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
    assert re.search(
        r"const raw = process\.env\."
        r"OPENCLAW_AGENT_BENCH_LENGTH_NO_PROGRESS_RETRIES;\s*"
        r'if \(raw === void 0 \|\| raw === ""\) return 4;',
        embedded_text,
    )
    assert re.search(
        r"const raw = process\.env\.OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES;"
        r"\s*if \(raw === void 0 \|\| raw === \"\"\) return 1;",
        embedded_text,
    )
    assert "Math.min(parsed, 16)" in embedded_text
    assert "retryState.emptyResponseAttempts = 0" in embedded_text
    assert 'message?.details?.deniedReason === "tool-loop"' in embedded_text
    assert "!__abBlockedToolLoopVeto" in embedded_text
    assert "!__abControlActivity" in embedded_text
    assert "empty_retry_attempt" in embedded_text
    assert "length_retry_attempt" in embedded_text
    assert CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER in recovery_text
    assert CONTEXT_RECOVERY_STREAK_RESET_MARKER in recovery_text
    assert TRUNCATE_ONLY_NATIVE_RECOVERY_MARKER in recovery_text
    recovery_helper = recovery_text.split(
        CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER, 1
    )[1].split(
        "AB_HARNESS_OPENCLAW_V2026_9_2_"
        "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V7_END",
        1,
    )[0]
    assert "hasNonToolTerminalState({" in recovery_helper
    assert "acceptedSessionSpawns: []" in recovery_helper
    assert "hasCompletionOwningSpawn" not in recovery_helper
    assert "runtimeContinuationStarted === true" in recovery_helper
    assert "hasOutboundDeliveryEvidence(attempt)" not in recovery_helper
    assert "currentReasoningOnly" in recovery_helper
    assert "explicitDone" in recovery_helper
    assert 'preflight?.route !== "truncate_tool_results_only"' in (
        recovery_helper
    )
    assert "attempt?.preflightRecovery ?? attempt?.contextBudgetStatus" in (
        recovery_text
    )
    assert 'source: "preflightPressure"' in recovery_text
    assert 'source: "agent-bench-terminal"' in recovery_text
    assert "terminalRetryState.lengthNoProgressAttempts = 0" in recovery_text
    assert "terminalRetryState.emptyResponseAttempts = 0" in recovery_text
    assert "input.normalizedAttempt" not in recovery_text
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
    assert set(verified["current_empty_resolver_files"]) == {
        "dist/builtin-openclaw-test.js",
        "dist/resolver-worker-test.js",
    }
    assert all(
        set(bindings) == {
            "classifyAssistantTurn",
            "resolveSettledToolBatchEvidence",
            "hasCompletedMessagingToolDeliveryEvidence",
            "hasAsyncActivity",
        }
        for bindings in verified["current_empty_resolver_bindings"].values()
    )

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
    assert second["context_pressure_terminal_recovery_files_changed"] == []
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

    for relative in (
        "dist/builtin-openclaw-test.js",
        "dist/resolver-worker-test.js",
        "dist/embedded-terminal-call-test.js",
        "dist/embedded-agent-test.js",
        "dist/embedded-recovery-test.js",
        "dist/worker-test.mjs",
    ):
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


def test_v2026_9_2_second_resolver_executes_with_lexical_bindings(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    bundle = root / "dist/resolver-worker-test.js"
    bundle.write_text(bundle.read_text() + r'''
const attempt = {
  assistantTexts: ["Historical visible activity."],
  currentAttemptCompletedAssistant: { content: [], stopReason: "stop" },
  terminal: { kind: "ok" },
  clientToolCalls: false,
  yieldDetected: false,
  didSendDeterministicApprovalPrompt: false,
  hasToolMediaBlockReply: false,
  completedDelivery: false,
  toolMetas: [],
  settledToolEvidence: {
    allToolsProvenSettled: true,
    intentionalTermination: false,
    parkedCodeModeRun: false,
  },
};
const instruction = resolveEmptyResponseRetryInstruction({
  provider: "agent_model",
  modelId: "roadmap",
  modelApi: "openai-completions",
  executionContract: "strict-agentic",
  payloadCount: 1,
  aborted: false,
  timedOut: false,
  attempt,
});
if (!instruction) throw new Error("second resolver did not continue");
const decision = attempt.__agentBenchContinuationDecision;
if (decision?.candidate !== "current-empty" || decision?.decision !== "continue") {
  throw new Error("second resolver did not materialize current-empty logic");
}
const reasoningAttempt = {
  ...attempt,
  assistantTexts: [],
  currentAttemptCompletedAssistant: {
    content: [{ type: "reasoning", text: "still working" }],
    stopReason: "length",
  },
};
const reasoningInstruction = resolveEmptyResponseRetryInstruction({
  provider: "agent_model",
  modelId: "roadmap",
  modelApi: "openai-completions",
  executionContract: "strict-agentic",
  payloadCount: 0,
  aborted: false,
  timedOut: false,
  attempt: reasoningAttempt,
});
if (reasoningInstruction !== null) {
  throw new Error("reasoning-only length entered ordinary continuation");
}
process.stdout.write("SECOND_RESOLVER_EXECUTION_OK\n");
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
    assert "SECOND_RESOLVER_EXECUTION_OK" in completed.stdout


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
        "hasCompletedMessagingToolDeliveryEvidence(params.attempt)",
        "&& !__abCompletedMessagingDelivery",
        "Number(params.attempt?.itemLifecycle?.activeCount ?? 0) > 0",
        "&& !__abActiveToolWork",
        "hasAsyncActivity(params.attempt.toolMetas)",
        "&& !__abAsyncActivity",
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
    bundle = root / "dist/current-empty-split-contract.js"
    bundle.write_text(
        (root / "dist/builtin-openclaw-test.js").read_text()
        + "\n"
        + (root / "dist/embedded-agent-test.js").read_text()
        + "\n"
        + (root / "dist/embedded-terminal-call-test.js").read_text()
        + r'''
function terminalInput({
  content = [], aborted = false, settled = true, completedDelivery = false,
  asyncStarted = false, yielded = false, timedOut = false,
  toolFailed = false, completedFallback = false,
  historicalText = "Historical visible progress before tool calls.",
  stopReason = "stop", staleHistoricalMutationError = false,
} = {}) {
  const toolCall = { type: "toolCall", id: "call-1", name: "edit" };
  let messagesSnapshot = [{ role: "assistant", content: [toolCall] }];
  let toolMetas = [{ toolCallId: "call-1", toolName: "edit", asyncStarted }];
  let startedCount = 1;
  let completedCount = settled ? 1 : 0;
  let activeCount = settled ? 0 : 1;
  let lastToolError = toolFailed ? { toolName: "edit" } : undefined;
  if (staleHistoricalMutationError) {
    const batches = [
      { id: "call-A", name: "apply_patch", isError: true },
      { id: "call-B", name: "edit", isError: false },
      { id: "call-C", name: "write", isError: false },
    ];
    messagesSnapshot = batches.flatMap((batch) => [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: batch.id, name: batch.name }],
      },
      {
        role: "toolResult", toolCallId: batch.id, toolName: batch.name,
        isError: batch.isError,
        content: [{
          type: "text",
          text: batch.isError ? "apply_patch failed" : `${batch.name} succeeded`,
        }],
      },
    ]);
    toolMetas = batches.map((batch) => ({
      toolCallId: batch.id, toolName: batch.name,
      isError: batch.isError, asyncStarted: false,
    }));
    startedCount = 3;
    completedCount = 3;
    activeCount = 0;
    lastToolError = { toolName: "apply_patch" };
  } else if (settled) {
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
      assistantTexts: [historicalText],
      ...(completedFallback
        ? { currentAttemptCompletedAssistant: { content, stopReason } }
        : { currentAttemptAssistant: { content, stopReason } }),
      terminal: { kind: "ok" },
      messagesSnapshot,
      itemLifecycle: {
        startedCount,
        completedCount,
        activeCount,
      },
      toolMetas,
      yieldDetected: yielded,
      completedDelivery,
      lastToolError,
    },
  };
}
function assert(value, message) {
  if (!value) throw new Error(message);
}
const retryState = { emptyResponseAttempts: 0 };
const historicalInput = terminalInput();
const historicalThenEmpty = resolveEmbeddedRunTerminal(
  historicalInput, retryState,
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
  historicalInput.attempt.__agentBenchContinuationDecision?.decision === "continue",
  "the resolver must record an allowed current-empty decision",
);
assert(
  historicalInput.attempt.__agentBenchContinuationDecision
    ?.cumulative_payload_count === 1,
  "diagnostics must preserve the cumulative historical payload count",
);
const outerRetryState = { emptyResponseAttempts: 0 };
assert(
  retryTerminal({
    ...historicalInput,
    terminalState: { outcome: "ok" },
    finalAssistantVisibleText: "",
    attemptAssistant: historicalInput.attempt.currentAttemptAssistant,
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "split-contract", sessionId: "split-contract" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  }, outerRetryState, null, historicalThenEmpty.instruction)?.action === "retry",
  "the split production path must reach the existing outer retry",
);
assert(
  outerRetryState.emptyResponseAttempts === 1,
  "the outer retry path must consume the first empty attempt",
);
const visibleHistoryWithoutTools = {
  aborted: false,
  timedOut: false,
  attempt: {
    assistantTexts: ["Historical visible activity without tool work."],
    currentAttemptAssistant: { content: [], stopReason: "stop" },
    terminal: { kind: "ok" },
    messagesSnapshot: [],
    itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
    toolMetas: [],
  },
};
const visibleHistoryWithoutToolsResult = resolveEmbeddedRunTerminal(
  visibleHistoryWithoutTools, { emptyResponseAttempts: 0 },
);
assert(
  visibleHistoryWithoutToolsResult.action === "retry",
  "historical visible text plus latest empty continues without an active owner",
);
assert(
  visibleHistoryWithoutTools.attempt.__agentBenchContinuationDecision
    ?.candidate === "current-empty"
    && visibleHistoryWithoutTools.attempt.__agentBenchContinuationDecision
      ?.decision === "continue"
    && visibleHistoryWithoutTools.attempt.__agentBenchContinuationDecision
      ?.all_tools_settled === false,
  "settled proof remains observable but is not a continuation hard guard",
);
const accumulatedEmptyInput = terminalInput({ historicalText: "" });
delete accumulatedEmptyInput.attempt.currentAttemptAssistant;
const accumulatedEmptyTerminal = resolveEmbeddedRunTerminal(
  accumulatedEmptyInput, { emptyResponseAttempts: 0 },
);
assert(
  accumulatedEmptyTerminal.action === "retry"
    && accumulatedEmptyInput.attempt.__agentBenchContinuationDecision
      ?.candidate === "accumulated-empty",
  "native accumulated empty must be identified without changing eligibility",
);
const accumulatedEmptyRetryState = {
  emptyResponseAttempts: 0, lengthNoProgressAttempts: 0,
};
assert(
  retryTerminal({
    ...accumulatedEmptyInput,
    terminalState: { outcome: "ok" },
    finalAssistantVisibleText: "",
    attemptAssistant: undefined,
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "accumulated-empty", sessionId: "accumulated-empty" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  }, accumulatedEmptyRetryState, null, accumulatedEmptyTerminal.instruction)
    ?.action === "retry",
  "accumulated empty must use the outer empty retry path",
);
assert(
  accumulatedEmptyRetryState.emptyResponseAttempts === 1
    && accumulatedEmptyRetryState.lengthNoProgressAttempts === 0,
  "accumulated empty consumes only the empty retry budget",
);
const lengthInput = terminalInput({
  content: [{ type: "text", text: "I still need to validate the change." }],
  stopReason: "length",
});
const lengthTerminal = resolveEmbeddedRunTerminal(
  lengthInput, { emptyResponseAttempts: 0 },
);
assert(lengthTerminal.action === "retry", "incomplete length remains eligible");
const outerLengthState = {
  emptyResponseAttempts: 0, lengthNoProgressAttempts: 0,
};
assert(
  retryTerminal({
    ...lengthInput,
    terminalState: { outcome: "ok" },
    finalAssistantVisibleText: "I still need to validate the change.",
    attemptAssistant: lengthInput.attempt.currentAttemptAssistant,
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "length-contract", sessionId: "length-contract" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  }, outerLengthState, null, lengthTerminal.instruction)?.action === "retry",
  "length candidates must reach the independent outer retry budget",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    completedFallback: true,
  }), { emptyResponseAttempts: 0 }).action === "retry",
  "completed-assistant fallback must use the same current-assistant semantics",
);
assert(
  resolveEmbeddedRunTerminal(terminalInput({
    toolFailed: true,
    historicalText: "Tool warning: edit failed before the latest empty stop.",
  }), { emptyResponseAttempts: 0 }).action === "retry",
  "historical tool-error payloads must not suppress the latest empty stop",
);
const staleErrorInput = terminalInput({
  staleHistoricalMutationError: true,
  historicalText: "apply_patch failed; later edit and write both succeeded.",
});
const staleErrorTerminal = resolveEmbeddedRunTerminal(
  staleErrorInput, { emptyResponseAttempts: 0 },
);
assert(
  staleErrorTerminal.action === "retry",
  "stale mutating-tool failure must not block a later settled successful batch",
);
const staleErrorDecision =
  staleErrorInput.attempt.__agentBenchContinuationDecision;
assert(
  staleErrorDecision?.candidate === "current-empty"
    && staleErrorDecision?.all_tools_settled === true
    && staleErrorDecision?.last_tool_error === true
    && staleErrorDecision?.decision === "continue"
    && staleErrorDecision?.deny_reason?.length === 0,
  "stale historical error diagnostics must preserve the allowed decision",
);
const staleErrorRetryState = { emptyResponseAttempts: 0 };
assert(
  retryTerminal({
    ...staleErrorInput,
    terminalState: { outcome: "ok" },
    finalAssistantVisibleText: "",
    attemptAssistant: staleErrorInput.attempt.currentAttemptAssistant,
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "stale-error", sessionId: "stale-error" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  }, staleErrorRetryState, null, staleErrorTerminal.instruction)?.action === "retry",
  "stale historical error must reach the outer empty retry",
);
assert(
  staleErrorRetryState.emptyResponseAttempts === 1,
  "stale historical error consumes the first outer empty retry",
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
  (() => {
    const input = terminalInput({
    content: [{ type: "image", data: "current-terminal-payload" }],
    });
    const result = resolveEmbeddedRunTerminal(
      input, { emptyResponseAttempts: 0 },
    );
    return result.action === "complete"
      && input.attempt.__agentBenchContinuationDecision?.decision === "deny"
      && input.attempt.__agentBenchContinuationDecision?.deny_reason
        ?.includes("current_payload");
  })(),
  "a current non-text terminal payload must suppress continuation",
);
const activeToolInput = terminalInput({ settled: false });
assert(
  resolveEmbeddedRunTerminal(
    activeToolInput, { emptyResponseAttempts: 0 },
  ).action === "complete"
    && activeToolInput.attempt.__agentBenchContinuationDecision
      ?.deny_reason?.includes("active_tool_work"),
  "active tool work must retain ownership and prevent a new model call",
);
const noSettledProofInput = terminalInput();
noSettledProofInput.attempt.messagesSnapshot = [];
noSettledProofInput.attempt.toolMetas = [];
noSettledProofInput.attempt.itemLifecycle = {
  startedCount: 0, completedCount: 0, activeCount: 0,
};
assert(
  resolveEmbeddedRunTerminal(
    noSettledProofInput, { emptyResponseAttempts: 0 },
  ).action === "retry",
  "current-empty without an active owner must not require settled-tool proof",
);
const noSettledProofLength = terminalInput({
  content: [{ type: "text", text: "More implementation remains." }],
  stopReason: "length",
});
noSettledProofLength.attempt.messagesSnapshot = [];
noSettledProofLength.attempt.toolMetas = [];
noSettledProofLength.attempt.itemLifecycle = {
  startedCount: 0, completedCount: 0, activeCount: 0,
};
assert(
  resolveEmbeddedRunTerminal(
    noSettledProofLength, { emptyResponseAttempts: 0 },
  ).action === "retry",
  "length without an active owner must not require settled-tool proof",
);
const activeLength = terminalInput({
  settled: false,
  content: [{ type: "text", text: "Waiting for the tool result." }],
  stopReason: "length",
});
assert(
  resolveEmbeddedRunTerminal(
    activeLength, { emptyResponseAttempts: 0 },
  ).action === "complete"
    && activeLength.attempt.__agentBenchContinuationDecision
      ?.deny_reason?.includes("active_tool_work"),
  "length must not start another model call while a tool owner is active",
);
activeLength.attempt.itemLifecycle.activeCount = 0;
assert(
  resolveEmbeddedRunTerminal(
    activeLength, { emptyResponseAttempts: 0 },
  ).action === "retry",
  "length continuation becomes eligible after native tool settlement",
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
'''
    )
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
    decision_lines = [
        line.split("] ", 1)[1]
        for line in completed.stderr.splitlines()
        if line.startswith("[agent-bench-continuation-decision] ")
    ]
    assert decision_lines
    decisions = [json.loads(line) for line in decision_lines]
    empty_decision = next(
        item for item in decisions if item["candidate"] == "current-empty"
    )
    assert empty_decision["decision"] == "continue"
    assert empty_decision["deny_reason"] == []
    assert empty_decision["current_stop_reason"] == "stop"
    assert empty_decision["current_visible_chars"] == 0
    assert empty_decision["current_has_tool_call"] is False
    assert empty_decision["current_has_payload"] is False
    assert empty_decision["cumulative_payload_count"] == 1
    assert empty_decision["all_tools_settled"] is True
    assert empty_decision["terminal_kind"] == "ok"
    assert empty_decision["last_tool_error"] is False
    assert empty_decision["empty_retry_attempt"] == 1
    assert empty_decision["length_retry_attempt"] == 0
    accumulated_empty_decision = next(
        item
        for item in decisions
        if item["candidate"] == "accumulated-empty"
    )
    assert accumulated_empty_decision["decision"] == "continue"
    assert accumulated_empty_decision["deny_reason"] == []
    assert accumulated_empty_decision["empty_retry_attempt"] == 1
    assert accumulated_empty_decision["length_retry_attempt"] == 0
    stale_error_decision = next(
        item
        for item in decisions
        if item["candidate"] == "current-empty"
        and item["last_tool_error"] is True
    )
    assert stale_error_decision["decision"] == "continue"
    assert stale_error_decision["deny_reason"] == []
    assert stale_error_decision["all_tools_settled"] is True
    assert stale_error_decision["cumulative_payload_count"] == 1
    assert stale_error_decision["empty_retry_attempt"] == 1
    length_decision = next(
        item for item in decisions if item["candidate"] == "length"
    )
    assert length_decision["decision"] == "continue"
    assert length_decision["empty_retry_attempt"] == 0
    assert length_decision["length_retry_attempt"] == 1


def test_v2026_9_2_context_pressure_terminal_uses_native_recovery(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    bundle = root / "dist/embedded-recovery-test.js"
    assert "input.normalizedAttempt" not in bundle.read_text()
    bundle.write_text(bundle.read_text() + r'''
function assert(value, message) {
  if (!value) throw new Error(message);
}
function recoveryInput({
  activeCount = 0, route = "compact_then_truncate",
  estimatedPromptTokens = 427191, promptBudgetBeforeReserve = 242144,
  stopReason = "length", content = [], truncationHelps = true,
  acceptedSessionSpawns = [], completedDelivery = false,
  runtimeContinuationStarted = false,
  yieldDetected = false,
  assistantPresent = true, assistantTexts = [],
  heartbeatToolResponse, successfulCronAdds = 0,
} = {}) {
  const contextRecoveryState = {
    autoCompactionCount: 0, overflowCompactionAttempts: 0,
    toolResultTruncationAttempted: false,
  };
  const terminalRetryState = {
    emptyResponseAttempts: 2, lengthNoProgressAttempts: 3,
  };
  const events = [];
  const assistant = assistantPresent ? { content, stopReason } : undefined;
  const attempt = {
    ...(assistant ? { currentAttemptCompletedAssistant: assistant } : {}),
    assistantTexts,
    terminal: { kind: "ok" },
    contextBudgetStatus: {
      route, source: "pre-prompt-estimate", estimatedPromptTokens,
      promptBudgetBeforeReserve,
    },
    itemLifecycle: { startedCount: 0, completedCount: 0, activeCount },
    messagesSnapshot: [], toolMetas: [],
    acceptedSessionSpawns, completedDelivery, runtimeContinuationStarted,
    yieldDetected,
    heartbeatToolResponse, successfulCronAdds,
    settledToolEvidence: {
      allToolsProvenSettled: false, intentionalTermination: false,
      parkedCodeModeRun: false, failedToolNames: new Set(),
    },
  };
  const workspace = { id: "shared-workspace" };
  const toolCapability = { exec: true, write: true };
  return {
    runInput: { runParams: { runId: "pressure", sessionId: "session" } },
    attemptEnvelope: {
      attempt,
      attemptAssistant: assistant,
      currentAttemptCompletedAssistant: assistant,
      terminalState: { outcome: "ok" },
    },
    contextRecoveryState,
    terminalRetryState,
    commonRecoveryInput: {
      state: contextRecoveryState,
      contextTokenBudget: 262144,
      genericCompactionRecoveryAllowed: true,
      attempt,
      runParams: {
        runId: "pressure", sessionId: "session",
        onAutoCompactionSucceeded(count) { events.push(["count", count]); },
      },
      events,
      truncateToolResults() {
        events.push(["truncate"]);
        return {
          truncated: truncationHelps,
          truncatedCount: truncationHelps ? 1 : 0,
          reason: truncationHelps ? undefined : "nothing reducible",
        };
      },
      armPostCompactionGuard() { events.push(["guard"]); },
      prepareCurrentTranscriptRetry() { events.push(["retry-transcript"]); },
      workspace,
      toolCapability,
    },
    events,
    workspace,
    toolCapability,
  };
}
(async () => {
  process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY = "continue";
  const pressure = recoveryInput();
  const result = await executeRecoveryContract(pressure);
  assert(result.action === "retry", "pressure length must retry");
  assert(
    pressure.contextRecoveryState.autoCompactionCount === 1,
    "pressure length must perform an actual compaction",
  );
  assert(
    pressure.events.some(([event, budget]) =>
      event === "compact" && budget === 242144),
    "native compaction must use the preflight prompt budget",
  );
  assert(
    pressure.events.some(([event]) => event === "truncate"),
    "compact_then_truncate must retain native post-compaction truncation",
  );
  assert(
    pressure.terminalRetryState.lengthNoProgressAttempts === 0,
    "successful compaction must reset the independent length streak",
  );
  assert(
    pressure.terminalRetryState.emptyResponseAttempts === 2,
    "length compaction must not consume or reset the empty budget",
  );
  assert(
    pressure.commonRecoveryInput.workspace === pressure.workspace
      && pressure.commonRecoveryInput.toolCapability === pressure.toolCapability,
    "compaction recovery must preserve session workspace and tool capability",
  );

  const active = recoveryInput({ activeCount: 1 });
  const whileActive = await executeRecoveryContract(active);
  assert(
    whileActive.action === "proceed"
      && active.contextRecoveryState.autoCompactionCount === 0,
    "active tool work must not start compaction or another model call",
  );
  active.attemptEnvelope.attempt.itemLifecycle.activeCount = 0;
  const afterCommit = await executeRecoveryContract(active);
  assert(
    afterCommit.action === "retry"
      && active.contextRecoveryState.autoCompactionCount === 1,
    "recovery may continue only after the tool result owner is quiescent",
  );

  const compactOnly = recoveryInput({ route: "compact_only" });
  const compactOnlyResult = await executeRecoveryContract(compactOnly);
  assert(compactOnlyResult.action === "retry", "compact_only must retry");
  assert(
    !compactOnly.events.some(([event]) => event === "truncate"),
    "compact_only must not run the truncate phase",
  );

  const settledSpawn = recoveryInput({
    acceptedSessionSpawns: [{
      childSessionKey: "agent:child:one", expectsCompletionMessage: false,
    }],
  });
  const settledSpawnResult = await executeRecoveryContract(settledSpawn);
  assert(
    settledSpawnResult.action === "retry"
      && settledSpawn.contextRecoveryState.autoCompactionCount === 1
      && settledSpawn.terminalRetryState.lengthNoProgressAttempts === 0,
    "historical settled session spawns must not veto pressure recovery",
  );

  const parentContinuedAfterSpawn = recoveryInput({
    acceptedSessionSpawns: [{
      childSessionKey: "agent:child:settled", expectsCompletionMessage: true,
    }],
  });
  const parentContinuedAfterSpawnResult = await executeRecoveryContract(
    parentContinuedAfterSpawn,
  );
  assert(
    parentContinuedAfterSpawnResult.action === "retry"
      && parentContinuedAfterSpawn.contextRecoveryState.autoCompactionCount === 1,
    "a settled spawn receipt must not retain terminal ownership",
  );

  const runtimeContinuation = recoveryInput({
    runtimeContinuationStarted: true,
  });
  const runtimeContinuationResult = await executeRecoveryContract(
    runtimeContinuation,
  );
  assert(
    runtimeContinuationResult.action === "proceed"
      && runtimeContinuation.contextRecoveryState.autoCompactionCount === 0,
    "an existing runtime continuation must veto competing recovery",
  );

  const nativeHandoff = recoveryInput({ yieldDetected: true });
  const nativeHandoffResult = await executeRecoveryContract(nativeHandoff);
  assert(
    nativeHandoffResult.action === "proceed"
      && nativeHandoff.contextRecoveryState.autoCompactionCount === 0,
    "an actual native handoff must veto competing recovery",
  );

  const completedDelivery = recoveryInput({ completedDelivery: true });
  const completedDeliveryResult = await executeRecoveryContract(
    completedDelivery,
  );
  assert(
    completedDeliveryResult.action === "proceed"
      && completedDelivery.contextRecoveryState.autoCompactionCount === 0,
    "completed external message delivery must retain terminal ownership",
  );

  const truncateOnly = recoveryInput({
    route: "truncate_tool_results_only",
  });
  const truncateOnlyResult = await executeRecoveryContract(truncateOnly);
  assert(truncateOnlyResult.action === "retry", "truncate-only must retry");
  assert(
    truncateOnly.contextRecoveryState.autoCompactionCount === 0
      && truncateOnly.events.some(([event]) => event === "truncate"),
    "truncate-only must use native truncation without compaction",
  );
  assert(
    truncateOnly.terminalRetryState.lengthNoProgressAttempts === 0
      && truncateOnly.terminalRetryState.emptyResponseAttempts === 2,
    "successful truncate-only must reset only the length streak",
  );

  const truncateOnlyEmpty = recoveryInput({
    route: "truncate_tool_results_only", stopReason: "stop",
    acceptedSessionSpawns: [{ childSessionKey: "agent:child:settled" }],
  });
  const truncateOnlyEmptyResult = await executeRecoveryContract(
    truncateOnlyEmpty,
  );
  assert(
    truncateOnlyEmptyResult.action === "retry"
      && truncateOnlyEmpty.contextRecoveryState.autoCompactionCount === 0
      && truncateOnlyEmpty.terminalRetryState.emptyResponseAttempts === 0
      && truncateOnlyEmpty.terminalRetryState.lengthNoProgressAttempts === 3,
    "pressure empty must share native truncate-only recovery and reset empty",
  );

  const ineffectiveTruncation = recoveryInput({
    route: "truncate_tool_results_only", truncationHelps: false,
  });
  const ineffectiveTruncationResult = await executeRecoveryContract(
    ineffectiveTruncation,
  );
  assert(
    ineffectiveTruncationResult.errorKind === "context_overflow"
      && ineffectiveTruncationResult.livenessState === "blocked"
      && ineffectiveTruncation.contextRecoveryState.autoCompactionCount === 0
      && ineffectiveTruncation.terminalRetryState.lengthNoProgressAttempts === 3
      && ineffectiveTruncation.terminalRetryState.emptyResponseAttempts === 2,
    "ineffective truncation must surface blocked overflow without resetting streak",
  );

  const emptyPressure = recoveryInput({ stopReason: "stop" });
  const emptyPressureResult = await executeRecoveryContract(emptyPressure);
  assert(emptyPressureResult.action === "retry", "pressure empty must retry");
  assert(
    emptyPressure.contextRecoveryState.autoCompactionCount === 1,
    "pressure empty must perform an actual compaction",
  );
  assert(
    emptyPressure.events.some(([event]) => event === "truncate"),
    "pressure empty must preserve compact_then_truncate behavior",
  );
  assert(
    emptyPressure.terminalRetryState.emptyResponseAttempts === 0,
    "successful empty compaction must reset the empty streak",
  );
  assert(
    emptyPressure.terminalRetryState.lengthNoProgressAttempts === 3,
    "empty compaction must not consume or reset the length budget",
  );

  const accumulatedEmptyPressure = recoveryInput({
    assistantPresent: false, assistantTexts: [],
  });
  const accumulatedEmptyResult = await executeRecoveryContract(
    accumulatedEmptyPressure,
  );
  assert(
    accumulatedEmptyResult.action === "retry"
      && accumulatedEmptyPressure.contextRecoveryState.autoCompactionCount === 1,
    "pressure accumulated-empty must use native compaction recovery",
  );
  assert(
    accumulatedEmptyPressure.terminalRetryState.emptyResponseAttempts === 0
      && accumulatedEmptyPressure.terminalRetryState.lengthNoProgressAttempts === 3,
    "successful accumulated-empty recovery must reset only the empty streak",
  );
  assert(
    accumulatedEmptyPressure.events.some(([event]) => event === "truncate"),
    "accumulated-empty compact_then_truncate must retain native truncation",
  );

  const historicalVisibleWithoutAssistant = recoveryInput({
    assistantPresent: false, assistantTexts: ["already visible"],
  });
  const historicalVisibleResult = await executeRecoveryContract(
    historicalVisibleWithoutAssistant,
  );
  assert(
    historicalVisibleResult.action === "proceed"
      && historicalVisibleWithoutAssistant.contextRecoveryState
        .autoCompactionCount === 0,
    "missing assistant with historical visible text is not accumulated-empty",
  );

  const heartbeatOwned = recoveryInput({
    assistantPresent: false, assistantTexts: [],
    heartbeatToolResponse: { text: "HEARTBEAT_OK" },
  });
  const heartbeatOwnedResult = await executeRecoveryContract(heartbeatOwned);
  assert(
    heartbeatOwnedResult.action === "proceed"
      && heartbeatOwned.contextRecoveryState.autoCompactionCount === 0,
    "heartbeat terminal ownership must veto accumulated-empty recovery",
  );

  const cronOwned = recoveryInput({
    assistantPresent: false, assistantTexts: [], successfulCronAdds: 1,
  });
  const cronOwnedResult = await executeRecoveryContract(cronOwned);
  assert(
    cronOwnedResult.action === "proceed"
      && cronOwned.contextRecoveryState.autoCompactionCount === 0,
    "successful cron terminal ownership must veto accumulated-empty recovery",
  );

  const explicitDoneLength = recoveryInput({
    stopReason: "length", assistantTexts: ["DONE"],
    content: [{ type: "text", text: "DONE" }],
  });
  const explicitDoneLengthResult = await executeRecoveryContract(
    explicitDoneLength,
  );
  assert(
    explicitDoneLengthResult.action === "proceed"
      && explicitDoneLength.contextRecoveryState.autoCompactionCount === 0,
    "explicit DONE length must not enter Agent_Bench pressure recovery",
  );

  const reasoningOnlyLength = recoveryInput({
    stopReason: "length",
    content: [{ type: "reasoning", text: "still reasoning" }],
  });
  const reasoningOnlyLengthResult = await executeRecoveryContract(
    reasoningOnlyLength,
  );
  assert(
    reasoningOnlyLengthResult.action === "proceed"
      && reasoningOnlyLength.contextRecoveryState.autoCompactionCount === 0,
    "reasoning-only length must not enter Agent_Bench pressure recovery",
  );

  const activeEmpty = recoveryInput({ activeCount: 1, stopReason: "stop" });
  const activeEmptyResult = await executeRecoveryContract(activeEmpty);
  assert(
    activeEmptyResult.action === "proceed"
      && activeEmpty.contextRecoveryState.autoCompactionCount === 0,
    "active tool work must retain ownership before empty compaction",
  );

  const healthyEmpty = recoveryInput({
    stopReason: "stop", estimatedPromptTokens: 200000,
  });
  const healthyEmptyResult = await executeRecoveryContract(healthyEmpty);
  assert(
    healthyEmptyResult.action === "proceed"
      && healthyEmpty.contextRecoveryState.autoCompactionCount === 0
      && healthyEmpty.terminalRetryState.emptyResponseAttempts === 2,
    "healthy-context empty must remain on the ordinary empty retry path",
  );

  const payloadEmpty = recoveryInput({
    stopReason: "stop",
    content: [{ type: "image", data: "terminal-payload" }],
  });
  const payloadEmptyResult = await executeRecoveryContract(payloadEmpty);
  assert(
    payloadEmptyResult.action === "proceed"
      && payloadEmpty.contextRecoveryState.autoCompactionCount === 0,
    "a current terminal payload must not be classified as pressure empty",
  );
  process.stdout.write("CONTEXT_PRESSURE_RECOVERY_CONTRACT_OK\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
''')
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY": "continue",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "CONTEXT_PRESSURE_RECOVERY_CONTRACT_OK" in completed.stdout
    assert '"recovery_reason":"context_pressure_compaction"' in (
        completed.stderr
    )
    assert '"candidate":"current-empty"' in completed.stderr
    assert '"candidate":"accumulated-empty"' in completed.stderr


def test_v2026_9_2_readable_recovery_captures_explicit_attempt_binding(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    bundle = root / "dist/embedded-recovery-test.js"
    source = bundle.read_text()
    declaration_anchor = (
        "const normalizedAttempt = await normalizeEmbeddedRunAttempt(input);"
    )
    assert source.count(declaration_anchor) == 1
    source = source.replace(
        declaration_anchor,
        "const capturedAttempt = await normalizeEmbeddedRunAttempt(input);",
    )
    call_anchor = "    normalizedAttempt,\n"
    assert source.count(call_anchor) == 1
    source = source.replace(
        call_anchor,
        "    normalizedAttempt: capturedAttempt,\n",
    )
    bundle.write_text(source)

    patch_runtime_tree(root)
    transformed = bundle.read_text()
    assert "input.normalizedAttempt" not in transformed
    assert "__abResolveRoadmapContextPressureTerminal(\n" in transformed
    assert "\t\tcapturedAttempt.attempt," in transformed
    bundle.write_text(transformed + r'''
(async () => {
  process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY = "continue";
  const result = await runPreparedEmbeddedLoop({
    attemptEnvelope: {
      attempt: {
        terminal: { kind: "ok" }, assistantTexts: ["complete"],
        itemLifecycle: { activeCount: 0 }, toolMetas: [],
        settledToolEvidence: {
          allToolsProvenSettled: false, intentionalTermination: false,
          parkedCodeModeRun: false, failedToolNames: new Set(),
        },
      },
      attemptAssistant: { content: [{ type: "text", text: "complete" }], stopReason: "stop" },
      currentAttemptCompletedAssistant: { content: [{ type: "text", text: "complete" }], stopReason: "stop" },
      terminalState: { outcome: "ok" },
    },
    runInput: {},
    contextRecoveryState: {
      autoCompactionCount: 0, toolResultTruncationAttempted: false,
    },
    terminalRetryState: {
      emptyResponseAttempts: 0, lengthNoProgressAttempts: 0,
    },
    commonRecoveryInput: {},
  });
  if (result.action !== "proceed") throw new Error("explicit binding failed");
  process.stdout.write("EXPLICIT_ATTEMPT_BINDING_OK\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
''')
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY": "continue",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "EXPLICIT_ATTEMPT_BINDING_OK" in completed.stdout


def test_v2026_9_2_minified_worker_executes_pressure_recovery(tmp_path):
    root = _runtime_tree(tmp_path)
    bundle = root / "dist/embedded-recovery-minified-test.js"
    bundle.write_text(r'''
function resolveSettledToolBatchEvidence(Ot){return Ot.settledToolEvidence}
function hasMessagingToolDeliveryEvidence(Ot){return Ot.completedDelivery===true}
function hasAsyncActivity(Ot){return(Ot??[]).some(Zt=>Zt.asyncStarted===true)}
function hasNonToolTerminalState(Ot){return Boolean(Ot.clientToolCalls||Ot.yieldDetected||Ot.didSendDeterministicApprovalPrompt||Ot.heartbeatToolResponse||Ot.hasToolMediaBlockReply||Ot.didDeliverSourceReplyViaMessageTool||(Ot.messagingToolSourceReplyPayloads?.length??0)>0||hasMessagingToolDeliveryEvidence(Ot)||(Ot.acceptedSessionSpawns?.length??0)>0||(Ot.successfulCronAdds??0)>0||hasAsyncActivity(Ot.toolMetas))}
function projectAgentRunAttemptTerminal(Ot){return{aborted:Ot.kind===`aborted`,promptErrorSource:Ot.promptErrorSource,timedOut:Ot.kind===`timeout`}}
function isEmbeddedRunTerminalInterrupted(Ot){return Ot===`aborted`||Ot===`timeout`}
function isProviderRefusalAssistantError(){return false}
function isLikelyContextOverflowError(Ot){return String(Ot).includes(`Context window exceeded`)}
async function compactEmbeddedRunForRecovery(Ot,Zt){Ot.events.push([`compact`,Zt.tokenBudget]);Ot.state.autoCompactionCount+=1;return{result:{compacted:true}}}
async function recoverEmbeddedRunOverflow(Ot){let Zt=!Ot.aborted&&!Ot.signalOwnedInterruption?(()=>{if(isProviderRefusalAssistantError(Ot.assistantOverflowCandidate))return null;return Ot.assistantErrorText&&isLikelyContextOverflowError(Ot.assistantErrorText)?{text:Ot.assistantErrorText,source:`assistantError`}:null})():null;if(!Zt||!Ot.genericCompactionRecoveryAllowed||Ot.contextTokenBudget===void 0)return{action:`none`};let Dn=projectAgentRunAttemptTerminal(Ot.attempt.terminal),cr=Ot.attempt.preflightRecovery,pr=Dn.promptErrorSource===`precheck`&&cr?.source===`mid-turn`&&typeof cr.promptBudgetBeforeReserve==`number`?cr.promptBudgetBeforeReserve:void 0,Br=false;if(!Br&&Ot.attemptCompactionCount>0&&Ot.state.overflowCompactionAttempts<3)return{action:`retry`};if(!Br&&Ot.attemptCompactionCount===0&&Ot.state.overflowCompactionAttempts<3){let Zt=await compactEmbeddedRunForRecovery(Ot,{tokenBudget:pr??Ot.contextTokenBudget});if(Zt.result.compacted){if(cr?.route===`compact_then_truncate`)Ot.truncateToolResults();return{action:`retry`}}}if(!Ot.state.toolResultTruncationAttempted){Ot.state.toolResultTruncationAttempted=true;let Zt=Ot.truncateToolResults();if(Zt.truncated)return{action:`retry`}}return{action:`surface`,kind:`context_overflow`}}
async function recoverEmbeddedRunAttempt(Ot){let{normalizedAttempt:Dn}=Ot,{attempt:pr,sessionIdUsed:xr,attemptAssistant:Mr,currentAttemptAssistant:Pr,currentAttemptCompletedAssistant:Br,terminalState:Hr}=Dn,{aborted:ta,externalAbort:na,promptError:ra,promptErrorSource:ia,timedOut:aa,timedOutDuringCompaction:oa}=projectAgentRunAttemptTerminal(pr.terminal),la=isEmbeddedRunTerminalInterrupted(Hr.outcome),ua=false,ma=false,ha=false,Mi=null,va=Br===void 0?Mr?.stopReason===`error`||Mr?.stopReason===`length`?Mr:void 0:Br.stopReason===`error`||Br.stopReason===`length`?Br:void 0,retry=()=>({action:`retry`});if(!ua&&!ma&&!ha)return{action:`proceed`};let Sa=await recoverEmbeddedRunOverflow({aborted:ta,signalOwnedInterruption:false,promptError:ra,assistantErrorText:Mi,assistantOverflowCandidate:va,attemptCompactionCount:0,attempt:pr,genericCompactionRecoveryAllowed:true,contextTokenBudget:262144,state:Ot.contextRecoveryState,events:Ot.events,truncateToolResults:Ot.truncateToolResults});if(Sa.action===`retry`)return retry();if(Sa.action===`surface`)return{action:`complete`,result:{errorKind:Sa.kind,livenessState:`blocked`}};return{action:`proceed`}}
function createEmbeddedRunTerminalRetryState(){return{emptyResponseAttempts:3,lengthNoProgressAttempts:3}}
async function executeWorkerContract(stopReason,{activeCount=0,content=[],route=`compact_then_truncate`,truncationHelps=true,acceptedSessionSpawns=[],runtimeContinuationStarted=false,yieldDetected=false,completedDelivery=false,assistantPresent=true,assistantTexts=[]}={}){let Da=createEmbeddedRunTerminalRetryState(),Sa={autoCompactionCount:0,overflowCompactionAttempts:0,toolResultTruncationAttempted:false},Br=assistantPresent?{content,stopReason}:void 0,pr={terminal:{kind:`ok`},assistantTexts,contextBudgetStatus:{route,estimatedPromptTokens:427191,promptBudgetBeforeReserve:242144},itemLifecycle:{activeCount},toolMetas:[],acceptedSessionSpawns,runtimeContinuationStarted,yieldDetected,completedDelivery,settledToolEvidence:{allToolsProvenSettled:false,intentionalTermination:false,parkedCodeModeRun:false}},za={attempt:pr,sessionIdUsed:`s`,attemptAssistant:Br,currentAttemptAssistant:Br,currentAttemptCompletedAssistant:Br,terminalState:{outcome:`ok`}},na={},ev=[],tr=()=>{ev.push([`truncate`]);return{truncated:truncationHelps,truncatedCount:truncationHelps?1:0}},ro=await recoverEmbeddedRunAttempt({runInput:na,normalizedAttempt:za,contextRecoveryState:Sa,events:ev,truncateToolResults:tr});if(ro.action===`complete`)return{...ro.result,retryState:Da,recoveryState:Sa,events:ev};if(ro.action===`retry`)return{action:`retry`,retryState:Da,recoveryState:Sa,events:ev};return{action:`proceed`,retryState:Da,recoveryState:Sa,events:ev}}
''')
    patch_runtime_tree(root)
    transformed = bundle.read_text()
    assert transformed.count("agentBenchContextPressure") >= 2
    assert "Context window exceeded under preflight pressure" in transformed
    bundle.write_text(bundle.read_text() + r'''
function assert(value,message){if(!value)throw new Error(message)}
(async()=>{
process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY=`continue`;
const empty=await executeWorkerContract(`stop`);
assert(empty.action===`retry`, `minified pressure empty must retry`);
assert(empty.recoveryState.autoCompactionCount===1, `minified empty must compact`);
assert(empty.retryState.emptyResponseAttempts===0, `minified empty streak must reset`);
assert(empty.retryState.lengthNoProgressAttempts===3, `empty must preserve length streak`);
assert(empty.events.some(([event,budget])=>event===`compact`&&budget===242144),`empty must use the pressure budget`);
assert(empty.events.some(([event])=>event===`truncate`), `empty must retain truncation`);
const length=await executeWorkerContract(`length`);
assert(length.action===`retry`, `minified pressure length must retry`);
assert(length.retryState.lengthNoProgressAttempts===0, `length streak must reset`);
assert(length.retryState.emptyResponseAttempts===3, `length must preserve empty streak`);
const settledSpawn=await executeWorkerContract(`length`,{acceptedSessionSpawns:[{childSessionKey:`settled`,expectsCompletionMessage:false}]});
assert(settledSpawn.action===`retry`&&settledSpawn.recoveryState.autoCompactionCount===1,`historical spawn must not veto minified recovery`);
const parentContinuedSpawn=await executeWorkerContract(`length`,{acceptedSessionSpawns:[{childSessionKey:`settled`,expectsCompletionMessage:true}]});
assert(parentContinuedSpawn.action===`retry`&&parentContinuedSpawn.recoveryState.autoCompactionCount===1,`settled completion-message spawn must not veto minified recovery`);
const runtimeContinuation=await executeWorkerContract(`length`,{runtimeContinuationStarted:true});
assert(runtimeContinuation.action===`proceed`&&runtimeContinuation.recoveryState.autoCompactionCount===0,`runtime continuation must veto minified recovery`);
const nativeHandoff=await executeWorkerContract(`length`,{yieldDetected:true});
assert(nativeHandoff.action===`proceed`&&nativeHandoff.recoveryState.autoCompactionCount===0,`native handoff must veto minified recovery`);
const delivered=await executeWorkerContract(`length`,{completedDelivery:true});
assert(delivered.action===`proceed`&&delivered.recoveryState.autoCompactionCount===0,`external delivery must veto minified recovery`);
const explicitDone=await executeWorkerContract(`length`,{content:[{type:`text`,text:`DONE`}],assistantTexts:[`DONE`]});
assert(explicitDone.action===`proceed`&&explicitDone.recoveryState.autoCompactionCount===0,`explicit DONE must veto minified pressure length`);
const reasoningLength=await executeWorkerContract(`length`,{content:[{type:`reasoning`,text:`working`}]});
assert(reasoningLength.action===`proceed`&&reasoningLength.recoveryState.autoCompactionCount===0,`reasoning-only length must veto minified pressure recovery`);
const truncateOnly=await executeWorkerContract(`length`,{route:`truncate_tool_results_only`});
assert(truncateOnly.action===`retry`, `minified truncate-only must retry`);
assert(truncateOnly.recoveryState.autoCompactionCount===0, `truncate-only must skip compaction`);
assert(truncateOnly.events.some(([event])=>event===`truncate`), `truncate-only must use native truncation`);
assert(truncateOnly.retryState.lengthNoProgressAttempts===0, `truncate-only must reset length streak`);
const ineffectiveTruncate=await executeWorkerContract(`length`,{route:`truncate_tool_results_only`,truncationHelps:false});
assert(ineffectiveTruncate.errorKind===`context_overflow`&&ineffectiveTruncate.livenessState===`blocked`&&ineffectiveTruncate.retryState.lengthNoProgressAttempts===3&&ineffectiveTruncate.retryState.emptyResponseAttempts===3,`ineffective truncation must surface blocked overflow without resetting streak`);
const active=await executeWorkerContract(`stop`,{activeCount:1});
assert(active.action===`proceed`&&active.recoveryState.autoCompactionCount===0,`active owner must block recovery`);
const payload=await executeWorkerContract(`stop`,{content:[{type:`image`,data:`x`}]});
assert(payload.action===`proceed`&&payload.recoveryState.autoCompactionCount===0,`payload must not be empty`);
const accumulated=await executeWorkerContract(void 0,{assistantPresent:false,assistantTexts:[]});
assert(accumulated.action===`retry`&&accumulated.recoveryState.autoCompactionCount===1,`minified accumulated empty must compact`);
assert(accumulated.retryState.emptyResponseAttempts===0&&accumulated.retryState.lengthNoProgressAttempts===3,`minified accumulated empty must reset only empty streak`);
const historical=await executeWorkerContract(void 0,{assistantPresent:false,assistantTexts:[`visible`]});
assert(historical.action===`proceed`&&historical.recoveryState.autoCompactionCount===0,`minified visible history is not accumulated empty`);
process.stdout.write(`MINIFIED_PRESSURE_RECOVERY_OK\n`);
})().catch(error=>{console.error(error);process.exitCode=1});
''')
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY": "continue",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "MINIFIED_PRESSURE_RECOVERY_OK" in completed.stdout


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
    for name in ("builtin-openclaw-test.js", "resolver-worker-test.js"):
        path = root / "dist" / name
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


def test_v2026_9_2_patcher_does_not_require_or_rewrite_split_call_site(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    path = root / "dist/embedded-terminal-call-test.js"
    original = path.read_text()

    patch_runtime_tree(root)

    assert path.read_text() == original
    assert "payloadCount, aborted, timedOut, attempt" in original
    assert "hasCurrentTerminalPayload" not in original


def test_v2026_9_2_verify_requires_every_resolver_definition_to_be_patched(
    tmp_path,
):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/resolver-worker-test.js"
    path.write_text(
        path.read_text().replace(
            CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
            "REMOVED_CURRENT_ASSISTANT_EMPTY_STOP_MARKER",
        )
    )

    with pytest.raises(RuntimeError, match="not every.*resolver definition"):
        verify_runtime_tree(root)


def test_v2026_9_2_patcher_requires_resolver_lexical_bindings(tmp_path):
    root = _runtime_tree(tmp_path)
    path = root / "dist/resolver-worker-test.js"
    path.write_text(
        path.read_text().replace(
            "function hasAsyncActivity(toolMetas) {\n"
            "  return (toolMetas ?? []).some((entry) => entry.asyncStarted);\n"
            "}\n",
            "",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="resolver lexical bindings missing.*hasAsyncActivity",
    ):
        patch_runtime_tree(root)


def test_v2026_9_2_verify_requires_resolver_lexical_bindings(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/resolver-worker-test.js"
    path.write_text(
        path.read_text().replace(
            "function hasAsyncActivity(toolMetas) {\n"
            "  return (toolMetas ?? []).some((entry) => entry.asyncStarted);\n"
            "}\n",
            "",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="resolver lexical bindings missing.*hasAsyncActivity",
    ):
        verify_runtime_tree(root)


@pytest.mark.parametrize(
    "legacy_marker",
    [
        LEGACY_CURRENT_ASSISTANT_EMPTY_STOP_MARKER,
        LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER,
        "hasCurrentTerminalPayload",
    ],
)
def test_v2026_9_2_patcher_rejects_stale_cross_bundle_current_empty_patch(
    tmp_path,
    legacy_marker,
):
    root = _runtime_tree(tmp_path)
    path = root / "dist/builtin-openclaw-test.js"
    path.write_text(path.read_text() + f"\n/* {legacy_marker} */\n")

    with pytest.raises(RuntimeError, match="obsolete cross-bundle"):
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


@pytest.mark.parametrize(
    ("terminal_marker", "reset_marker"),
    [
        (
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V2_BEGIN",
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "COMPACTION_RETRY_STREAK_RESET_V2_BEGIN",
        ),
        (
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V3_BEGIN",
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_RECOVERY_STREAK_RESET_V3_BEGIN",
        ),
        (
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V4_BEGIN",
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_RECOVERY_STREAK_RESET_V4_BEGIN",
        ),
        (
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V5_BEGIN",
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN",
        ),
        (
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_PRESSURE_TERMINAL_RECOVERY_V6_BEGIN",
            "AB_HARNESS_OPENCLAW_V2026_9_2_"
            "CONTEXT_RECOVERY_STREAK_RESET_V5_BEGIN",
        ),
    ],
)
def test_v2026_9_2_patcher_rejects_stale_pressure_patch(
    tmp_path,
    terminal_marker,
    reset_marker,
):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    path = root / "dist/embedded-recovery-test.js"
    text = path.read_text().replace(
        CONTEXT_PRESSURE_TERMINAL_RECOVERY_MARKER,
        terminal_marker,
    ).replace(
        CONTEXT_RECOVERY_STREAK_RESET_MARKER,
        reset_marker,
    )
    path.write_text(text)

    with pytest.raises(RuntimeError, match="predates current"):
        patch_runtime_tree(root)


def test_v2026_9_2_patcher_fails_closed_when_attempt_binding_moves(tmp_path):
    root = _runtime_tree(tmp_path)
    path = root / "dist/embedded-recovery-test.js"
    source = path.read_text()
    call_anchor = "    normalizedAttempt,\n"
    assert source.count(call_anchor) == 1
    source = source.replace(
        call_anchor,
        "    movedNormalizedAttempt,\n",
    )
    path.write_text(source)

    with pytest.raises(RuntimeError, match="context-pressure"):
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
    assert CONTINUATION_DECISION_DIAGNOSTIC_MARKER in audit
    assert LEGACY_CURRENT_TERMINAL_PAYLOAD_MARKER in audit
    assert (
        "all_resolver_definitions_have_current_empty_logic_and_bindings"
        in audit
    )
    for helper in (
        "classifyAssistantTurn",
        "resolveSettledToolBatchEvidence",
        "hasCompletedMessagingToolDeliveryEvidence",
        "hasAsyncActivity",
    ):
        assert helper in audit
    assert "no_obsolete_cross_bundle_current_empty_contract" in audit
    assert BOUNDED_LENGTH_CONTINUATION_MARKER in audit
    assert LENGTH_SEMANTIC_PROGRESS_MARKER in audit
    assert "stale_unbounded_length_continuation_present" in audit
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in audit
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in audit
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in audit
    assert LOOP_VETO_RECOVERY_PROBE_MARKER in audit
