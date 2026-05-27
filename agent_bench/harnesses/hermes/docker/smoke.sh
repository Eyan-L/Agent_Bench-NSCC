#!/usr/bin/env bash
# Smoke test for the Hermes harness image. Pure image-level checks:
#   1. Hermes CLI starts.
#   2. Agent_Bench patch markers are baked in.
#   3. docker backend bridge mode validates missing bridge env clearly.
#   4. docker backend executes through a fake bridge without Docker/socket.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(dirname "$HERE")"
VERSION_FILE="$HARNESS_DIR/patch/VERSION"
HERMES_VERSION="$(awk -F': *' '/^upstream_version:/ {print $2; exit}' "$VERSION_FILE")"
HM_IMAGE_TAG="${HM_IMAGE_TAG:-agent-bench/hermes:${HERMES_VERSION}}"

pass()  { echo "  PASS  $*"; }
fail()  { echo "  FAIL  $*" >&2; exit 1; }
banner(){ echo; echo "[smoke] $* ($HM_IMAGE_TAG)"; }

banner "1/4 cli help"
docker run --rm "$HM_IMAGE_TAG" hermes --help >/tmp/hm_smoke_help.txt 2>&1 || {
    cat /tmp/hm_smoke_help.txt >&2
    fail "hermes --help failed"
}
grep -qi 'Hermes' /tmp/hm_smoke_help.txt || {
    cat /tmp/hm_smoke_help.txt >&2
    fail "hermes --help output unexpected"
}
pass "hermes --help OK"

banner "2/4 patch markers (mode=baked)"
docker run --rm \
    -e HERMES_PATCH_MODE=baked \
    "$HM_IMAGE_TAG" python3 /opt/hermes-tools/apply_patch.py status \
    >/tmp/hm_smoke_patch.txt 2>&1 || {
        cat /tmp/hm_smoke_patch.txt >&2
        fail "patch status command failed"
    }
grep -q 'applied' /tmp/hm_smoke_patch.txt || {
    cat /tmp/hm_smoke_patch.txt >&2
    fail "patch status reports no applied patch"
}
pass "patch status OK"

banner "3/4 docker backend bridge guard"
docker run --rm "$HM_IMAGE_TAG" python3 - <<'PY' >/tmp/hm_smoke_agent_bench.serving.txt 2>&1
import os
os.environ["TERMINAL_ENV"] = "docker"
os.environ["HERMES_EXEC_BRIDGE"] = "/usr/local/bin/hm-exec-bridge"
os.environ.pop("HERMES_EXEC_BRIDGE_SOCKET", None)
os.environ.pop("HERMES_EXEC_BRIDGE_TOKEN", None)
from tools.terminal_tool import check_terminal_requirements
from tools.environments.docker import DockerEnvironment
assert check_terminal_requirements() is False
try:
    DockerEnvironment(image="unused", cwd="/testbed")
except RuntimeError as e:
    assert "HERMES_EXEC_BRIDGE_SOCKET" in str(e)
    print("missing bridge env is explicit")
else:
    raise SystemExit("DockerEnvironment unexpectedly initialized without bridge socket/token")
PY
cat /tmp/hm_smoke_agent_bench.serving.txt
pass "docker bridge guard OK"

banner "4/4 docker backend bridge execution"
docker run --rm "$HM_IMAGE_TAG" python3 - <<'PY' >/tmp/hm_smoke_bridge.txt 2>&1
import os
from pathlib import Path

workdir = Path("/tmp/hm-task")
workdir.mkdir(parents=True, exist_ok=True)
bridge = Path("/tmp/fake-hm-bridge.py")
bridge.write_text("""#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", required=True)
ap.add_argument("--env-json", default="{}")
ap.add_argument("--command", default=None)
ap.add_argument("--argv-json", default=None)
ap.add_argument("--timeout-ms", default="0")
args = ap.parse_args()

assert os.environ.get("OPENCLAW_EXEC_BRIDGE_SOCKET") == "/tmp/hm.sock"
assert os.environ.get("OPENCLAW_EXEC_BRIDGE_TOKEN") == "hm-token"
if args.argv_json:
    argv = json.loads(args.argv_json)
else:
    argv = ["bash", "-c", args.command]
env = os.environ.copy()
env.update(json.loads(args.env_json))
stdin_data = None if sys.stdin.isatty() else sys.stdin.read()
proc = subprocess.run(
    argv,
    cwd=args.workdir,
    env=env,
    input=stdin_data,
    capture_output=True,
    text=True,
)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
raise SystemExit(proc.returncode)
""")
bridge.chmod(0o755)

os.environ.update({
    "TERMINAL_ENV": "docker",
    "HERMES_EXEC_BRIDGE": str(bridge),
    "HERMES_EXEC_BRIDGE_SOCKET": "/tmp/hm.sock",
    "HERMES_EXEC_BRIDGE_TOKEN": "hm-token",
})

from tools.environments.docker import DockerEnvironment

env = DockerEnvironment(image="unused", cwd=str(workdir), timeout=30)
result = env.execute("pwd && printf smoke-ok", timeout=10)
assert result["returncode"] == 0, result
assert str(workdir) in result["output"], result
assert "smoke-ok" in result["output"], result
print("bridge execution OK")
PY
cat /tmp/hm_smoke_bridge.txt
pass "docker bridge execution OK"

echo
echo "[smoke] all 4 checks passed for $HM_IMAGE_TAG"
