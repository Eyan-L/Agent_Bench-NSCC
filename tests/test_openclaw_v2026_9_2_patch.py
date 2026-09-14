from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.harnesses.openclaw.v2026_9_2_patch import (
    EMPTY_STOP_RETRY_LIMIT_MARKER,
    EXEC_MARKER,
    FS_CHECK_MARKER,
    FS_HELPER_MARKER,
    FS_READ_MARKER,
    FS_RUN_MARKER,
    FS_WRITE_MARKER,
    LITELLM_PRICING_URL,
    OPENROUTER_PRICING_URL,
    ROADMAP_CONTINUATION_INSTRUCTION,
    SANDBOX_MARKER,
    SETTLED_CONTINUATION_INSTRUCTION_MARKER,
    SETTLED_CONTINUATION_POLICY_MARKER,
    SETTLED_CONTINUATION_RETRY_MARKER,
    SHARED_WORKSPACE_SPAWN_MARKER,
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
    assert "__abSettledToolEvidence.allToolsProvenSettled" in builtin_text
    assert "!__abSettledToolEvidence.hasUnsettledToolError" in builtin_text
    assert "classifyAssistantTurn(params).emptyResponse" in builtin_text
    assert 'params.attempt?.terminal?.kind !== "timeout"' in builtin_text
    assert "!hasCompletedMessagingToolDeliveryEvidence(params.attempt)" in (
        builtin_text
    )
    assert "!hasAsyncActivity(params.attempt.toolMetas)" in builtin_text
    assert builtin_text.index(SETTLED_CONTINUATION_RETRY_MARKER) < (
        builtin_text.index("shouldSkipNonVisibleTurnRetry(params)")
    )
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in embedded_text
    assert "if (raw === void 0 || raw === \"\") return 1" in embedded_text
    assert "Math.min(parsed, 16)" in embedded_text
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
        for path in (root / "dist").glob("*.js")
    }
    second = patch_runtime_tree(root)
    assert second["exec_bridge_files_changed"] == []
    assert second["sandbox_bypass_files_changed"] == []
    assert second["shared_workspace_spawn_files_changed"] == []
    assert second["settled_tool_continuation"]["files_changed"] == []
    assert second["fs_bridge_files_changed"] == []
    assert second["openrouter_pricing_url_replacements"] == 0
    assert second["litellm_pricing_url_replacements"] == 0
    assert snapshot == {
        path.name: path.read_text()
        for path in (root / "dist").glob("*.js")
    }


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
    assert EMPTY_STOP_RETRY_LIMIT_MARKER in audit
