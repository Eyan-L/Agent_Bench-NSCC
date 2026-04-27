#!/usr/bin/env bash
# scripts/status.sh — show per-stage completion status.

set -uo pipefail

KIT_ROOT="${KIT_ROOT:-$(pwd)}"

check() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        printf "  \033[32m✓\033[0m %-26s %s\n" "$label" "${path#$KIT_ROOT/}"
    else
        printf "  \033[90m·\033[0m %-26s (not yet)\n" "$label"
    fi
}

echo
echo "== Generation =="
check "plan"        "$KIT_ROOT/artifacts/plan.md"
check "instrument"  "$KIT_ROOT/eval/trace/.generated"
check "runner"      "$KIT_ROOT/eval/runner/.generated"
check "analyzer"    "$KIT_ROOT/eval/analysis/.generated"
check "reporter"    "$KIT_ROOT/eval/report/.generated"

echo
echo "== Runs =="
if [[ -d "$KIT_ROOT/data/runs" ]]; then
    count=$(find "$KIT_ROOT/data/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${count:-0}" -gt 0 ]]; then
        printf "  \033[32m✓\033[0m %-26s %s run(s)\n" "data/runs/" "$count"
    else
        printf "  \033[90m·\033[0m %-26s (empty)\n" "data/runs/"
    fi
else
    printf "  \033[90m·\033[0m %-26s (no runs yet)\n" "data/runs/"
fi

echo
echo "== Analysis =="
check "long-format parquet" "$KIT_ROOT/data/analysis/long.parquet"
check "data_quality.md"     "$KIT_ROOT/data/analysis/data_quality.md"

echo
echo "== Report =="
check "report.md"    "$KIT_ROOT/report/report.md"
check "claims.json"  "$KIT_ROOT/report/claims.json"
check "pushback.md"  "$KIT_ROOT/report/pushback.md"
echo
