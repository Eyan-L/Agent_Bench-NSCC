// Runs inside the SIF. Read the sealed package, not a host npm installation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';

const root = '/app';
assert.equal(JSON.parse(fs.readFileSync(`${root}/package.json`, 'utf8')).version, '2026.9.6');
const statusPath = `${root}/.agent-bench-v2026.9.2.json`;
const candidate = fs.existsSync(statusPath);
const status = candidate ? JSON.parse(fs.readFileSync(statusPath, 'utf8')) : null;
function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(file);
    else if (entry.isFile() && /\.(?:m?js|cjs)$/.test(entry.name)) yield file;
  }
}
const sources = new Map([...walk(`${root}/dist`)].map(file => [file, fs.readFileSync(file, 'utf8')]));
for (const token of ['compaction_loop_persisted', 'openclaw.model.usage', 'openclaw.tool.execution',
  'function settleRequesterRun(', 'requesterContinuationSettled', 'function buildDelegationGuidanceSection(']) {
  assert.ok([...sources.values()].some(source => source.includes(token)), `Missing native ${token}`);
}
if (candidate) {
  assert.equal(status.upstream_version, '2026.9.6');
  assert.equal(status.verification?.verified, true);
  assert.equal(status.native_child_settlement, true);
  assert.equal(status.implicit_child_handoff_patch_applied, false);
  const keys = ['exec_bridge', 'sandbox_bypass', 'shared_workspace_spawn', 'settled_continuation_policy',
    'settled_continuation_instruction', 'settled_continuation_retry', 'empty_stop_retry_limit',
    'loop_veto_recovery_probe', 'sessions_yield_no_pending_loop', 'context_pressure_terminal_recovery',
    'fs_helper', 'fs_run', 'fs_write', 'fs_read', 'fs_check'];
  const files = new Set(status.files_changed || []);
  for (const key of keys) {
    assert.ok(status.verification.markers[key]?.length, `Missing ${key}`);
    for (const file of status.verification.markers[key]) files.add(file);
  }
  const copies = [
    ['resolveSettledTurnFinalizationRequest', 'SETTLED_CONTINUATION_POLICY_BEGIN'],
    ['resolveEmptyResponseRetryInstruction', 'CURRENT_ASSISTANT_EMPTY_STOP_V2_BEGIN'],
    ['recoverEmbeddedRunOverflow', 'V2026_9_6_CONTEXT_PRESSURE_BEGIN'],
  ];
  for (const [file, source] of sources) {
    assert.ok(!source.includes('IMPLICIT_CHILD_HANDOFF_OWNERSHIP_V1_BEGIN'), `Obsolete handoff in ${file}`);
    assert.ok(!/AB_HARNESS_OPENCLAW_PATCH_(POST_COMPACTION_LOOP_GUARD|MODEL_USAGE_BEGIN|TOOL_CALL_BEGIN)/.test(source));
    assert.ok(!source.includes('https://openrouter.ai/api/v1/models'), `Online pricing in ${file}`);
    assert.ok(!source.includes('https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'));
    for (const [name, marker] of copies) {
      const definitions = [...source.matchAll(new RegExp(`function\\s+${name}\\(`, 'g'))].length;
      if (definitions) assert.equal(source.split(marker).length - 1, definitions, `Incomplete ${name} in ${file}`);
    }
    if (source.includes('function resolveEmptyResponseRetryInstruction(')) {
      assert.ok(source.includes('hasCompletionMessageSessionSpawn('));
      assert.ok(source.includes('hasCommittedMessagingToolDeliveryEvidence('));
      assert.ok(source.includes('SETTLED_CONTINUATION_INSTRUCTION_BEGIN'));
    }
  }
  for (const file of files) {
    const check = spawnSync('node', ['--check', path.join(root, file)], { encoding: 'utf8', timeout: 60000 });
    assert.equal(check.status, 0, `${file}: ${check.stderr}`);
  }
  console.log(`PASS sealed bridge markers; ${files.size} JavaScript bundles parsed`);
}

async function nativeBundle(prefix, definition = '') {
  const file = [...sources.keys()].find(file => path.basename(file).startsWith(prefix)
    && (!definition || sources.get(file).includes(`function ${definition}(`)));
  assert.ok(file, `Missing ${prefix}`);
  const source = sources.get(file);
  const exports = source.match(/export\s*\{([\s\S]*?)\};/);
  assert.ok(exports, `Missing exports in ${file}`);
  const module = await import(file);
  return {
    source,
    get(name) {
      const match = exports[1].match(new RegExp(`(?:^|,)\\s*${name}\\s+as\\s+([\\w$]+)`));
      assert.ok(match, `Missing native ${name} export`);
      return module[match[1]];
    },
  };
}

const delegation = await nativeBundle('watched-sessions-prompt-');
const resolveMode = delegation.get('resolveMainSessionDelegationMode');
const guidance = delegation.get('buildDelegationGuidanceSection');
const config = { agents: { defaults: { subagents: { delegationMode: 'prefer', maxConcurrent: 16, maxChildrenPerAgent: 16 } } } };
const mode = resolveMode({ config, agentId: 'roadmap_bench', sessionKey: 'agent:roadmap_bench:explicit:validation' });
assert.equal(mode, 'prefer');
assert.ok(guidance({ mode, isMinimal: false, hiddenDelegationTool: 'sessions_spawn', hasVisibleSessionSpawn: true }).join('\n').includes('delegate via sessions_spawn'));
assert.equal(guidance({ mode: 'suggest', isMinimal: false, hiddenDelegationTool: 'sessions_spawn', hasVisibleSessionSpawn: true }).length, 0);
assert.equal(guidance({ mode, isMinimal: true, hiddenDelegationTool: 'sessions_spawn', hasVisibleSessionSpawn: true }).length, 0);
console.log('PASS native prefer for benchmark sessions; minimal child and suggest unchanged');

for (const configPath of process.argv.slice(2)) {
  const validation = spawnSync('node', ['/app/openclaw.mjs', 'config', 'validate', '--json'], {
    encoding: 'utf8', timeout: 60000,
    env: { ...process.env, OPENCLAW_CONFIG_PATH: configPath, OPENCLAW_STATE_DIR: '/tmp/agent-bench-profile-audit' },
  });
  assert.equal(validation.status, 0, `Invalid ${configPath}: ${validation.stdout} ${validation.stderr}`);
  const result = JSON.parse(validation.stdout);
  assert.equal(result.valid, true);
  console.log(`PASS full native configuration: ${path.basename(configPath)}`);
}

if (candidate) {
  process.env.OPENCLAW_AGENT_BENCH_SETTLED_TOOL_POLICY = 'continue';
  const builtin = await nativeBundle('builtin-openclaw-', 'resolveEmptyResponseRetryInstruction');
  const retry = builtin.get('resolveEmptyResponseRetryInstruction');
  function attempt(text = '', stopReason = 'stop') {
    const assistant = { role: 'assistant', content: [{ type: 'text', text }], stopReason };
    return {
      terminal: { kind: 'ok' }, currentAttemptAssistant: assistant,
      currentAttemptCompletedAssistant: assistant, assistantTexts: [], toolMetas: [],
      itemLifecycle: { startedCount: 0, completedCount: 0, activeCount: 0 },
      acceptedSessionSpawns: [], messagesSnapshot: [{ role: 'user', content: 'task' }, assistant],
      replayMetadata: { hadPotentialSideEffects: false, replaySafe: true },
      messagingToolSentTexts: [], messagingToolSentTargets: [], messagingToolSentMediaUrls: [],
    };
  }
  function request(value, more = {}) {
    return { provider: 'agent_model', modelId: 'Qwen3.6-35B-A3B', modelApi: 'openai-completions',
      payloadCount: 0, aborted: false, timedOut: false, attempt: value, ...more };
  }
  assert.ok(retry(request(attempt())).includes('Use tools'));
  assert.ok(retry(request(attempt('implementation unfinished', 'length'))));
  const historical = attempt('implementation unfinished', 'length');
  historical.assistantTexts = ['DONE'];
  assert.ok(retry(request(historical)), 'Historical DONE must not end the current turn');
  assert.equal(retry(request(attempt('DONE'))), null);
  assert.equal(retry(request(attempt('DONE', 'length'))), null);
  const child = attempt();
  child.acceptedSessionSpawns = [{ runId: 'child', childSessionKey: 'child', expectsCompletionMessage: true }];
  assert.equal(retry(request(child)), null, 'Native child delivery owns completion');
  const pending = attempt(); pending.itemLifecycle.activeCount = 1;
  assert.equal(retry(request(pending)), null);
  const owner = attempt(); owner.runtimeContinuationStarted = true;
  assert.equal(retry(request(owner)), null);
  assert.equal(retry(request(attempt(), { aborted: true })), null);
  assert.equal(retry(request(attempt(), { timedOut: true })), null);

  const pressureFile = [...sources.keys()].find(file => path.basename(file).startsWith('embedded-agent-')
    && sources.get(file).includes('function __abV96ContextPressure('));
  const source = sources.get(pressureFile);
  const start = source.indexOf('function __abV96ContextPressure(');
  const end = source.indexOf('async function recoverEmbeddedRunOverflow(', start);
  assert.ok(start >= 0 && end > start);
  const helper = vm.runInNewContext(`${source.slice(start, end)}; __abV96ContextPressure`, {
    process, hasAttemptTerminalState: builtin.get('hasAttemptTerminalState'),
    hasAsyncActivity: builtin.get('hasAsyncActivity'),
    resolveSettledToolBatchEvidence: builtin.get('resolveSettledToolBatchEvidence'),
    resolveCurrentAttemptAssistant: builtin.get('resolveCurrentAttemptAssistant'),
  });
  const pressure = attempt('unfinished', 'length');
  pressure.contextBudgetStatus = { route: 'compact_only', estimatedPromptTokens: 900, promptBudgetBeforeReserve: 800 };
  assert.equal(helper(pressure)?.route, 'compact_only');
  pressure.currentAttemptAssistant.content[0].text = 'DONE';
  assert.equal(helper(pressure), null);
  pressure.currentAttemptAssistant.content[0].text = 'unfinished';
  pressure.itemLifecycle.activeCount = 1;
  assert.equal(helper(pressure), null);
  pressure.itemLifecycle.activeCount = 0;
  pressure.runtimeContinuationStarted = true;
  assert.equal(helper(pressure), null);
  console.log('PASS actual continuation protocol: current DONE, historical DONE, pending tools, native child ownership, abort and pressure guards');
}
console.log(`PASS OpenClaw 2026.9.6 ${candidate ? 'candidate' : 'base'} audit`);
