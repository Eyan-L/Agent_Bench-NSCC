from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "agent_bench" / "harnesses" / "openclaw" / "apply_patch.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("openclaw_apply_patch", PATCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER_FIXTURE = r'''
async function runEmbeddedPiAgent(params) {
  const log = console;
  let autoCompactionCount = 0;
  let runLoopIterations = 0;
  while (true) {
    const attempt = await runEmbeddedAttemptWithBackend({
      sessionId: params.sessionId,
      abortSignal: params.abortSignal,
    });
    if (timedOut) {
      log.info(
        `[timeout-compaction] compaction succeeded for ${provider}/${modelId}; retrying prompt`,
      );
      continue;
    }
    if (overflowed) {
      autoCompactionCount += 1;
      log.info(`auto-compaction succeeded for ${provider}/${modelId}; retrying prompt`);
      continue;
    }
    return attempt;
  }
}
'''


OUTCOME_FIXTURE = r'''
const log = console;
async function recordLoopOutcome(args) {
  const sessionState = getDiagnosticSessionState({
    sessionKey: args.ctx.sessionKey,
    sessionId: args.ctx.agentId,
  });
  recordToolCallOutcome(sessionState, {
    toolName: args.toolName,
    toolParams: args.toolParams,
    toolCallId: args.toolCallId,
    result: args.result,
    error: args.error,
    config: args.ctx.loopDetection,
  });
}
'''


def test_runner_backport_arms_only_after_successful_compaction_retry():
    patcher = _load_patcher()
    transformed, changed = patcher._transform_post_compaction_runner(RUNNER_FIXTURE)

    assert changed is True
    assert patcher._POST_COMPACTION_GUARD_RUNNER_MARKER in transformed
    assert "windowSize: 3" in transformed
    assert "params.config?.tools?.loopDetection?.enabled !== false" in transformed
    assert "abortSignal: __abPostCompactionAbortController.signal" in transformed
    assert "__abPostCompactionGuard.abortController = __abPostCompactionAbortController" in transformed
    assert "const __abParentAbortSignal = params.abortSignal" in transformed
    assert "removeEventListener?.(\"abort\", __abRelayParentAbort)" in transformed
    assert "throw __abPostCompactionGuard.abortError ?? __abErr" in transformed
    assert "if (__abPostCompactionGuard.abortError)" in transformed
    assert "throw __abPostCompactionGuard.abortError" in transformed
    assert transformed.count("const __abPostCompactionAbortController = new AbortController();") == 1
    assert transformed.index("const __abPostCompactionAbortController = new AbortController();") > transformed.index(
        "while (true)"
    )
    assert transformed.count("__abArmPostCompactionGuard();") == 2
    assert "__abArmPostCompactionGuard();" not in transformed.split(
        "let autoCompactionCount = 0;", 1
    )[0]


def test_outcome_backport_uses_exact_tool_args_result_triple():
    patcher = _load_patcher()
    transformed, changed = patcher._transform_post_compaction_outcome(OUTCOME_FIXTURE)

    assert changed is True
    assert patcher._POST_COMPACTION_GUARD_OUTCOME_MARKER in transformed
    assert "entry.toolName === __abObservation.toolName" in transformed
    assert "entry.argsHash === __abObservation.argsHash" in transformed
    assert "entry.resultHash === __abObservation.resultHash" in transformed
    assert "PostCompactionLoopPersistedError" in transformed
    assert "detector=compaction_loop_persisted" in transformed
    assert ".abort(__abError)" in transformed


def test_runner_and_outcome_can_share_one_compiled_bundle():
    patcher = _load_patcher()
    transformed, runner_changed = patcher._transform_post_compaction_runner(
        RUNNER_FIXTURE + "\n" + OUTCOME_FIXTURE
    )
    transformed, outcome_changed = patcher._transform_post_compaction_outcome(transformed)

    assert runner_changed is True
    assert outcome_changed is True
    assert patcher._POST_COMPACTION_GUARD_RUNNER_MARKER in transformed
    assert patcher._POST_COMPACTION_GUARD_OUTCOME_MARKER in transformed


def test_backport_propagates_persisted_loop_error_after_attempt_unwinds(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the OpenClaw bundle transform integration fixture")

    patcher = _load_patcher()
    transformed, runner_changed = patcher._transform_post_compaction_runner(
        RUNNER_FIXTURE + "\n" + OUTCOME_FIXTURE
    )
    transformed, outcome_changed = patcher._transform_post_compaction_outcome(transformed)
    assert runner_changed is True
    assert outcome_changed is True

    harness = r'''
let timedOut = false;
let overflowed = true;
const provider = "test-provider";
const modelId = "test-model";
const __fixtureSessionState = { toolCallHistory: [] };

function getDiagnosticSessionState() {
  return __fixtureSessionState;
}

function recordToolCallOutcome(state, params) {
  const record = {
    toolName: params.toolName,
    argsHash: JSON.stringify(params.toolParams),
    toolCallId: params.toolCallId,
    resultHash: JSON.stringify(params.error ?? params.result),
  };
  state.toolCallHistory.push(record);
}

let __fixtureAttempt = 0;
async function runEmbeddedAttemptWithBackend(params) {
  __fixtureAttempt += 1;
  if (__fixtureAttempt === 1) {
    overflowed = true;
    return { attempt: 1 };
  }

  overflowed = false;
  for (let i = 0; i < 3; i += 1) {
    await recordLoopOutcome({
      ctx: {
        sessionKey: "fixture-session-key",
        sessionId: "fixture-session-id",
        runId: "fixture-run-id",
        loopDetection: {},
      },
      toolName: "read",
      toolParams: { path: "/same" },
      toolCallId: `call-${i}`,
      result: "same-result",
    });
  }
  if (!params.abortSignal.aborted) {
    throw new Error("post-compaction guard did not abort the active attempt signal");
  }
  // Deliberately resolve normally after observing the abort. The runner glue
  // must still surface the guard error instead of treating this as a generic
  // aborted/incomplete turn.
  return { attempt: 2, aborted: true };
}
'''
    assertion = r'''
runEmbeddedPiAgent({
  runId: "fixture-run-id",
  sessionId: "fixture-session-id",
  config: {},
  abortSignal: undefined,
}).then(() => {
  console.error("runner resolved unexpectedly");
  process.exitCode = 2;
}).catch((err) => {
  if (err?.name !== "PostCompactionLoopPersistedError") {
    console.error(`unexpected error name: ${err?.name}`);
    process.exitCode = 3;
    return;
  }
  if (err?.detector !== "compaction_loop_persisted") {
    console.error(`unexpected detector: ${err?.detector}`);
    process.exitCode = 4;
  }
});
'''

    script = tmp_path / "post_compaction_guard_fixture.js"
    script.write_text(harness + "\n" + transformed + "\n" + assertion)
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_backport_is_idempotent_per_role():
    patcher = _load_patcher()
    runner_once, _ = patcher._transform_post_compaction_runner(RUNNER_FIXTURE)
    runner_twice, changed = patcher._transform_post_compaction_runner(runner_once)
    assert changed is False
    assert runner_twice == runner_once

    outcome_once, _ = patcher._transform_post_compaction_outcome(OUTCOME_FIXTURE)
    outcome_twice, changed = patcher._transform_post_compaction_outcome(outcome_once)
    assert changed is False
    assert outcome_twice == outcome_once
