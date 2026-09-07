import { spawn } from "node:child_process";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import {
  createRemoteShellSandboxFsBridge,
  registerSandboxBackend,
} from "openclaw/plugin-sdk/sandbox";

const PLUGIN_ID = "agent-bench-singularity";

function requireBridgeEnv() {
  const bridge = process.env.OPENCLAW_EXEC_BRIDGE?.trim();
  const socket = process.env.OPENCLAW_EXEC_BRIDGE_SOCKET?.trim();
  const token = process.env.OPENCLAW_EXEC_BRIDGE_TOKEN?.trim();
  if (!bridge || !socket || !token) {
    throw new Error(
      "Agent_Bench Singularity backend requires OPENCLAW_EXEC_BRIDGE, " +
        "OPENCLAW_EXEC_BRIDGE_SOCKET, and OPENCLAW_EXEC_BRIDGE_TOKEN",
    );
  }
  return bridge;
}

function bridgeArgv({ workdir, env = {}, command, argv }) {
  const bridge = requireBridgeEnv();
  const out = [bridge, "--workdir", workdir, "--env-json", JSON.stringify(env)];
  if (command !== undefined) {
    out.push("--command", command);
  } else {
    out.push("--argv-json", JSON.stringify(argv ?? []));
  }
  return out;
}

async function runBridgeBuffered({
  workdir,
  env = {},
  argv,
  stdin,
  allowFailure = false,
  signal,
}) {
  const child = spawn(bridgeArgv({ workdir, env, argv }), {
    env: process.env,
    stdio: ["pipe", "pipe", "pipe"],
    signal,
  });

  const stdout = [];
  const stderr = [];
  child.stdout.on("data", (chunk) => stdout.push(Buffer.from(chunk)));
  child.stderr.on("data", (chunk) => stderr.push(Buffer.from(chunk)));

  if (stdin !== undefined) {
    child.stdin.end(stdin);
  } else {
    child.stdin.end();
  }

  const result = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code, closeSignal) => {
      resolve({
        stdout: Buffer.concat(stdout),
        stderr: Buffer.concat(stderr),
        code: typeof code === "number" ? code : closeSignal ? 128 : 125,
      });
    });
  });

  if (result.code !== 0 && !allowFailure) {
    const detail = result.stderr.toString("utf8").trim() || `exit ${result.code}`;
    throw new Error(`Agent_Bench sandbox bridge command failed: ${detail}`);
  }
  return result;
}

function normalizeRuntimeId(scopeKey) {
  return `agent-bench-${String(scopeKey).replace(/[^A-Za-z0-9._-]/g, "_")}`;
}

function createAgentBenchSandboxBackend(params) {
  requireBridgeEnv();

  const workdir = params.cfg.docker.workdir || "/app";
  let handle;
  handle = {
    // Registering under "docker" deliberately overrides the built-in Docker
    // factory. The existing Agent_Bench state renderer can therefore keep its
    // stable backend="docker" configuration while OpenClaw routes all sandbox
    // work through the already-running external bridge.
    id: "docker",
    runtimeId: normalizeRuntimeId(params.scopeKey),
    runtimeLabel: `Agent_Bench bridge (${params.scopeKey})`,
    workdir,
    env: params.cfg.docker.env,
    configLabel: "Agent_Bench external execution bridge",
    configLabelKind: "Bridge",
    workdirValidation: "backend",
    workdirRoots: [workdir],

    async validateWorkdir(candidate) {
      const result = await runBridgeBuffered({
        workdir: "/",
        argv: [
          "/bin/sh",
          "-c",
          'cd -- "$1" 2>/dev/null && pwd -P',
          "agent-bench-workdir",
          candidate,
        ],
        allowFailure: true,
      });
      if (result.code !== 0) {
        return null;
      }
      return result.stdout.toString("utf8").trim() || null;
    },

    async buildExecSpec({ command, workdir: requestedWorkdir, env, usePty }) {
      return {
        argv: bridgeArgv({
          workdir: requestedWorkdir || workdir,
          env,
          command,
        }),
        env: process.env,
        stdinMode: usePty ? "pipe-open" : "pipe-closed",
      };
    },

    async runShellCommand(command) {
      return await runBridgeBuffered({
        workdir,
        env: {},
        argv: [
          "/bin/sh",
          "-c",
          command.script,
          "openclaw-sandbox-fs",
          ...(command.args ?? []),
        ],
        stdin: command.stdin,
        allowFailure: command.allowFailure === true,
        signal: command.signal,
      });
    },

    // The runtime workspace exists in the task container rather than inside
    // the OpenClaw SIF. Use OpenClaw's supported remote-shell filesystem
    // bridge so read/write/edit/stat operations share the same backend path.
    remoteWorkspaceDir: workdir,
    remoteAgentWorkspaceDir: workdir,
    async runRemoteShellScript(command) {
      return await handle.runShellCommand(command);
    },
    createFsBridge({ sandbox }) {
      return createRemoteShellSandboxFsBridge({ sandbox, runtime: handle });
    },
  };
  return handle;
}

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "Agent Bench Singularity Sandbox",
  description:
    "Routes OpenClaw sandbox exec and filesystem operations through the Agent_Bench external execution bridge.",
  register(api) {
    if (api.registrationMode !== "full") {
      return;
    }
    // The legacy Agent_Bench config already selects backend="docker". Override
    // that registered backend rather than changing benchmark state rendering.
    const unregister = registerSandboxBackend("docker", createAgentBenchSandboxBackend);
    api.lifecycle.registerRuntimeLifecycle({
      id: "agent-bench-singularity-backend-cleanup",
      cleanup: ({ reason, sessionKey, runId }) => {
        if (sessionKey !== undefined || runId !== undefined) {
          return;
        }
        if (reason === "disable" || reason === "restart") {
          unregister();
        }
      },
    });
  },
});
