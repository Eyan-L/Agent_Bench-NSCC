#!/usr/bin/env bash
# Terminal-Bench in-loop verify helper. The `verify` skill points here.
# Runs /tests/test.sh inside this container, then reads /logs/verifier/reward.txt.
# Prints a one-line JSON status. Exits 0 on PASS, non-zero otherwise.
#
# Source of truth for the script the harness drops at /app/_oc_validate.sh.
# Edited in place — no codegen needed, _seed_workspace_helpers in
# applications/terminal_bench/app.py reads this file and copies it verbatim.
set -u

VERIFIER="${1:-/tests/test.sh}"
REWARD_FILE="/logs/verifier/reward.txt"

# Drop any stale reward from a previous attempt so we read this run's value.
rm -f "$REWARD_FILE" 2>/dev/null || true

bash "$VERIFIER" >/tmp/_oc_verify.stdout 2>/tmp/_oc_verify.stderr
rc=$?
reward=""
if [[ -f "$REWARD_FILE" ]]; then
    reward="$(tr -d '[:space:]' < "$REWARD_FILE")"
fi

if [[ "$reward" == "1" ]]; then
    status="PASS"
    out_code=0
elif [[ "$reward" == "0" ]]; then
    status="FAIL"
    out_code=1
elif [[ -z "$reward" ]]; then
    status="ERROR"
    out_code=2
else
    status="ERROR"
    out_code=3
fi

tail_chars=$(tail -c 1200 /tmp/_oc_verify.stdout 2>/dev/null \
    | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
    || echo '""')

printf '{"status": "%s", "verifier_rc": %d, "reward_raw": "%s", "verifier_stdout_tail": %s}\n' \
    "$status" "$rc" "$reward" "$tail_chars"
exit "$out_code"
