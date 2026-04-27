#!/usr/bin/env bash
# scripts/check_env.sh — verify environment before running experiments.
#
# Does not attempt to install anything. Reports ok/warn/fail, exits 0 unless
# something is structurally broken.

set -uo pipefail

OK=0
FAIL=0
WARN=0

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; OK=$((OK+1)); }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; WARN=$((WARN+1)); }

echo
echo "== Host =="
if command -v python3 >/dev/null 2>&1; then
    pass "python3: $(python3 --version 2>&1)"
else
    fail "python3 not found"
fi
if command -v docker >/dev/null 2>&1; then
    pass "docker: $(docker --version)"
else
    warn "docker not found (needed for SWE-bench task containers)"
fi
if command -v make >/dev/null 2>&1; then
    pass "make: $(make --version | head -n 1)"
else
    fail "make not found"
fi

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${gpus:-0}" -gt 0 ]]; then
        pass "nvidia-smi: ${gpus} GPU(s)"
    else
        warn "nvidia-smi present but reports 0 GPUs"
    fi
else
    warn "nvidia-smi not found — GPU telemetry will be unavailable"
fi
if python3 -c "import pynvml" 2>/dev/null; then
    pass "pynvml importable"
else
    warn "pynvml not importable (pip install nvidia-ml-py)"
fi
if command -v dcgmi >/dev/null 2>&1; then
    pass "dcgmi present (MFU/MBU/HBM BW counters available)"
else
    warn "dcgmi not found — MFU/MBU/HBM BW will be reported as unavailable (expected on most hosts)"
fi

echo
echo "== Python deps =="
for pkg in psutil pyarrow pandas numpy; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "$pkg importable"
    else
        warn "$pkg not importable"
    fi
done

echo
echo "== Claude =="
if command -v claude >/dev/null 2>&1; then
    pass "claude CLI on PATH — generation targets will invoke it"
else
    warn "claude CLI not on PATH — generation will fall back to manual-paste mode"
fi

echo
echo "== vLLM (serving) =="
if command -v vllm >/dev/null 2>&1; then
    vllm_ver="$(vllm --version 2>&1 | head -n1 | awk '{print $NF}' || echo unknown)"
    pass "vllm: $vllm_ver"
    # Qwen3.6 needs vllm >= 0.19.0.
    if python3 -c "
import sys
v = '$vllm_ver'.split('.')
try:
    major, minor = int(v[0]), int(v[1])
    sys.exit(0 if (major, minor) >= (0, 19) else 1)
except Exception:
    sys.exit(2)
" 2>/dev/null; then
        pass "vllm version satisfies Qwen3.6 requirement (≥0.19.0)"
    else
        warn "vllm version may be below 0.19.0 required for Qwen3.6 (pip install -U 'vllm>=0.19.0')"
    fi
else
    warn "vllm not on PATH (pip install 'vllm>=0.19.0')"
fi
if python3 -c "import huggingface_hub" 2>/dev/null; then
    pass "huggingface_hub importable"
else
    warn "huggingface_hub not importable (pip install 'huggingface_hub[cli]')"
fi

echo
echo "== TP sanity =="
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    tp="${TP:-2}"
    if [[ "${gpu_count:-0}" -ge "$tp" ]]; then
        pass "GPU count ($gpu_count) ≥ TP ($tp)"
    else
        warn "GPU count ($gpu_count) < TP ($tp) — reduce TP in .env or add GPUs"
    fi
fi

echo
echo "== SWE-bench =="
if python3 -c "import swebench" 2>/dev/null; then
    pass "swebench importable"
else
    warn "swebench not importable (pip install swebench)"
fi

echo
echo "== OpenClaw =="
if command -v openclaw >/dev/null 2>&1; then
    pass "openclaw CLI on PATH"
elif python3 -c "import openclaw" 2>/dev/null; then
    pass "openclaw package importable"
else
    warn "OpenClaw not found — install before running experiments"
fi

echo
echo "== Serving endpoint =="
if [[ -n "${SERVING_ENDPOINT:-}" ]]; then
    if curl -sf -o /dev/null -m 3 "${SERVING_ENDPOINT%/}/models" 2>/dev/null; then
        pass "$SERVING_ENDPOINT reachable (/models)"
    elif curl -sf -o /dev/null -m 3 "${SERVING_ENDPOINT%/}/" 2>/dev/null; then
        pass "$SERVING_ENDPOINT reachable (root)"
    else
        warn "$SERVING_ENDPOINT not reachable yet (start it before run-phaseN)"
    fi
else
    warn "SERVING_ENDPOINT not set (configure in .env or export)"
fi

echo
echo "Summary: $OK ok, $WARN warn, $FAIL fail"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
