#!/usr/bin/env python3
"""Static checks for the profile layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_bench.core.config import load_composed_config
from agent_bench.core.profile_io import load_profile


def _profile_files(root: Path) -> list[Path]:
    suffixes = {".json", ".toml", ".yaml", ".yml"}
    return sorted(path for path in root.glob("**/*") if path.suffix in suffixes)


def _load(path: Path) -> dict:
    return load_profile(path)


def _check_all_profiles() -> list[str]:
    errors: list[str] = []
    for path in _profile_files(ROOT / "profiles"):
        try:
            _load(path)
        except Exception as exc:
            errors.append(f"{path}: invalid profile: {exc}")
    return errors


def _check_axis_boundaries() -> list[str]:
    errors: list[str] = []
    for path in _profile_files(ROOT / "profiles" / "benchmarks"):
        data = _load(path)
        output_dir = str(data.get("output_dir") or "")
        if not output_dir:
            errors.append(f"{path}: missing top-level `output_dir`")
        elif Path(output_dir).name != output_dir:
            errors.append(
                f"{path}: output_dir should be a run-set leaf, not a path "
                f"(got {output_dir!r})"
            )
    for path in _profile_files(ROOT / "profiles" / "harnesses"):
        data = _load(path)
        if "model" in data:
            errors.append(f"{path}: harness configs must not contain top-level `model`")
        if "harness" not in data:
            errors.append(f"{path}: missing top-level `harness`")
    for path in _profile_files(ROOT / "profiles" / "models"):
        model = _load(path).get("model")
        if not isinstance(model, dict):
            errors.append(f"{path}: missing `model` object")
            continue
        for key in ("config_id", "name", "max_tokens", "context_window"):
            if key not in model:
                errors.append(f"{path}: model.{key} missing")
        if int(model.get("context_window", 0)) <= 0:
            errors.append(f"{path}: model.context_window must be positive")
        if int(model.get("max_tokens", 0)) > int(model.get("context_window", 0)):
            errors.append(f"{path}: model.max_tokens exceeds model.context_window")
    for path in _profile_files(ROOT / "profiles" / "serving"):
        serving = _load(path).get("serving")
        if not isinstance(serving, dict):
            errors.append(f"{path}: missing `serving` object")
            continue
        for key in ("config_id", "backend", "endpoint"):
            if key not in serving:
                errors.append(f"{path}: serving.{key} missing")
        if "served_context_length" in serving:
            errors.append(
                f"{path}: serving configs must not contain served_context_length; "
                "model.context_window is the source of truth"
            )
        runtime = serving.get("runtime") or {}
        if isinstance(runtime, dict) and "tp" in runtime:
            errors.append(
                f"{path}: serving.runtime.tp is runtime launch state; "
                "set VLLM_TP/SGLANG_TP/TP when starting the job"
            )
    return errors


def _check_combo(args: argparse.Namespace) -> list[str]:
    if not (args.app and args.harness and args.model and args.serving):
        return []
    try:
        load_composed_config(
            Path(args.app),
            Path(args.harness),
            Path(args.model),
            Path(args.serving),
        )
    except Exception as exc:
        return [f"selected combo invalid: {exc}"]
    return []


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app")
    parser.add_argument("--harness")
    parser.add_argument("--model")
    parser.add_argument("--serving")
    args = parser.parse_args(argv)

    errors = []
    errors.extend(_check_all_profiles())
    errors.extend(_check_axis_boundaries())
    errors.extend(_check_combo(args))

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("config validation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
