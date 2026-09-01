from __future__ import annotations

import importlib.util
from pathlib import Path


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
