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
    EMPTY_STOP_RETRY_LIMIT_MARKER,
    EXEC_MARKER,
    FS_CHECK_MARKER,
    FS_HELPER_MARKER,
    FS_READ_MARKER,
    FS_RUN_MARKER,
    FS_WRITE_MARKER,
    LITELLM_PRICING_URL,
    LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER,
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
    UNBOUNDED_LENGTH_CONTINUATION_MARKER,
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
    (dist / "embedded-agent-test.js").write_text(
        "function resolveSettledToolBatchEvidence(attempt) {\n"
        "  return attempt.settledToolEvidence ?? { allToolsProvenSettled: false };\n"
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
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in builtin_text
    assert "__abSettledToolEvidence.allToolsProvenSettled" in builtin_text
    assert "!__abSettledToolEvidence.hasUnsettledToolError" not in builtin_text
    assert "const __abAssistantState = classifyAssistantTurn(params)" in builtin_text
    assert '__abAssistantState.assistant?.stopReason === "length"' in builtin_text
    assert "const __abExplicitDone" in builtin_text
    assert "!__abExplicitDone" in builtin_text
    assert "!__abAssistantState.reasoningOnly" in builtin_text
    assert (
        "__abAssistantState.emptyResponse || __abIncompleteLength" in builtin_text
    )
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
    assert UNBOUNDED_LENGTH_CONTINUATION_MARKER in embedded_text
    assert UNBOUNDED_LENGTH_CONTINUATION_MARKER in worker_text
    assert 'Ot.attemptAssistant?.stopReason === "length"' in worker_text
    assert 'input.attemptAssistant?.stopReason === "length"' in embedded_text
    assert "if (__abLengthContinuation)" in embedded_text
    assert "continuing under the enclosing benchmark deadline" in embedded_text
    assert "if (!__abLengthContinuation)" not in embedded_text
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


def test_v2026_9_2_length_continues_without_consuming_empty_stop_budget(tmp_path):
    root = _runtime_tree(tmp_path)

    patch_runtime_tree(root)

    builtin = (root / "dist/builtin-openclaw-test.js").read_text()
    embedded = (root / "dist/embedded-agent-test.js").read_text()
    assert "__abAssistantState.emptyResponse || __abIncompleteLength" in builtin
    assert '__abAssistantState.assistant?.stopReason === "length"' in builtin
    assert "!__abExplicitDone" in builtin
    assert builtin.count("return EMPTY_RESPONSE_RETRY_INSTRUCTION;") == 1
    assert "maxEmptyResponseRetryAttempts" in embedded
    assert f"process.env.OPENCLAW_AGENT_BENCH_EMPTY_STOP_RETRIES" in embedded
    assert "Math.min(parsed, 16)" in embedded
    assert UNBOUNDED_LENGTH_CONTINUATION_MARKER in embedded
    assert "__abLengthContinuation" in embedded
    assert "retryState.emptyResponseAttempts += 1" in embedded
    assert embedded.index("if (__abLengthContinuation)") < embedded.index(
        "retryState.emptyResponseAttempts += 1"
    )

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


def test_v2026_9_2_transformed_bundle_executes_retry_state_contract(tmp_path):
    root = _runtime_tree(tmp_path)
    patch_runtime_tree(root)
    bundle = root / "dist/embedded-agent-test.js"
    bundle.write_text(bundle.read_text() + r'''
function abInput({ progress = "none", outcome = "ok", loopVeto = false } = {}) {
  const toolProgress = progress === "tool";
  return {
    attempt: {
      terminal: { kind: outcome === "timeout" ? "timeout" : "ok" },
      messagesSnapshot: loopVeto ? [{
        role: "toolResult",
        details: { status: "blocked", deniedReason: "tool-loop" },
      }] : [],
      toolMetas: toolProgress ? [{ toolCallId: "call-1" }] : [],
      settledToolEvidence: { allToolsProvenSettled: toolProgress },
    },
    terminalState: { outcome },
    finalAssistantVisibleText: progress === "assistant" ? "Implemented feature" : "",
    attemptAssistant: { stopReason: "stop" },
    maxEmptyResponseRetryAttempts: 4,
    runParams: { runId: "probe", sessionId: "probe" },
    activeErrorContext: { provider: "probe", model: "probe" },
    activateInternalPrompt() {},
  };
}
function empty(state) {
  return retryTerminal(abInput(), state, null, "continue");
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
process.stdout.write("TRANSFORMED_RETRY_CONTRACT_OK\n");
''')
    completed = subprocess.run(
        [*_node_command(), str(bundle)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
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


def test_v2026_9_2_patcher_upgrades_counter_patch_without_streak_reset(
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

    upgraded = patch_runtime_tree(root)

    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in path.read_text()
    assert str(path.relative_to(root)) in upgraded[
        "settled_tool_continuation"
    ]["files_changed"]
    assert patch_runtime_tree(root)["settled_tool_continuation"][
        "files_changed"
    ] == []


def test_v2026_9_2_patcher_migrates_legacy_consecutive_reset(tmp_path):
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

    upgraded = patch_runtime_tree(root)

    updated = path.read_text()
    assert LEGACY_CONSECUTIVE_EMPTY_STOP_RESET_MARKER not in updated
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in updated
    assert "classifyAssistantTurn" not in updated
    assert str(path.relative_to(root)) in upgraded[
        "settled_tool_continuation"
    ]["files_changed"]


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

    with pytest.raises(RuntimeError, match="predates empty-length or"):
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
    assert UNBOUNDED_LENGTH_CONTINUATION_MARKER in audit
    assert CONSECUTIVE_EMPTY_STOP_RESET_MARKER in audit
    assert SETTLED_CONTINUATION_TOOL_ERROR_MARKER in audit
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in audit
    assert LOOP_VETO_RECOVERY_PROBE_MARKER in audit
