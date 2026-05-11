"""Validation checks before Phase 1."""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_ar_issue_only_run(run_dir: Path) -> bool:
    """Validate AR-issue-only run produces one LLM_INFER step."""
    
    step_trace_path = run_dir / "steps.jsonl"
    verdict_path = run_dir / "verdict.json"
    
    if not step_trace_path.exists():
        logger.error(f"Missing step trace: {step_trace_path}")
        return False
    
    if not verdict_path.exists():
        logger.error(f"Missing verdict: {verdict_path}")
        return False
    
    # Check step trace
    llm_steps = 0
    with open(step_trace_path) as f:
        for line in f:
            step = json.loads(line)
            if step["step_type"] == "LLM_INFER":
                llm_steps += 1
    
    if llm_steps != 1:
        logger.error(f"Expected 1 LLM_INFER step, found {llm_steps}")
        return False
    
    logger.info("✓ AR-issue-only validation passed")
    return True


def validate_openclaw_run(run_dir: Path) -> bool:
    """Validate OpenClaw run produces mixed step types."""
    
    step_trace_path = run_dir / "steps.jsonl"
    verdict_path = run_dir / "verdict.json"
    
    if not step_trace_path.exists():
        logger.error(f"Missing step trace: {step_trace_path}")
        return False
    
    if not verdict_path.exists():
        logger.error(f"Missing verdict: {verdict_path}")
        return False
    
    # Check step trace has mixed types
    step_types = set()
    with open(step_trace_path) as f:
        for line in f:
            step = json.loads(line)
            step_types.add(step["step_type"])
    
    if "LLM_INFER" not in step_types:
        logger.error("No LLM_INFER steps found")
        return False
    
    if len(step_types) < 2:
        logger.error(f"Expected mixed step types, found only: {step_types}")
        return False
    
    logger.info(f"✓ OpenClaw validation passed (step types: {step_types})")
    return True


def validate_resume(runs_dir: Path, expected_run_count: int) -> bool:
    """Validate resume does not re-execute completed runs."""
    
    manifests = list(runs_dir.glob("*/manifest.json"))
    
    if len(manifests) != expected_run_count:
        logger.error(
            f"Expected {expected_run_count} manifests, found {len(manifests)}"
        )
        return False
    
    # Check all manifests have terminal status
    for manifest_path in manifests:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        if manifest["status"] not in ("PASS", "FAIL", "TIMEOUT", "ERROR"):
            logger.error(f"Non-terminal status: {manifest['status']}")
            return False
    
    logger.info("✓ Resume validation passed")
    return True


def validate_serving_sweep(runs_dir: Path) -> bool:
    """Validate serving sweep produces distinct config_ids on same instance."""
    
    manifests = list(runs_dir.glob("*/manifest.json"))
    
    # Group by instance_id
    by_instance = {}
    for manifest_path in manifests:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        instance_id = manifest["instance_id"]
        by_instance.setdefault(instance_id, []).append(manifest)
    
    # Check each instance has runs with different serving_config_id
    for instance_id, instance_manifests in by_instance.items():
        config_ids = {m["serving_config_id"] for m in instance_manifests}
        
        if len(config_ids) != 3:
            logger.error(
                f"Instance {instance_id}: expected 3 distinct config_ids, "
                f"found {len(config_ids)}: {config_ids}"
            )
            return False
    
    logger.info("✓ Serving sweep validation passed")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True,
                       choices=["ar_issue_only", "openclaw", "resume", "serving_sweep"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    
    args = parser.parse_args()
    
    if args.check == "ar_issue_only":
        success = validate_ar_issue_only_run(args.run_dir)
    elif args.check == "openclaw":
        success = validate_openclaw_run(args.run_dir)
    elif args.check == "resume":
        success = validate_resume(args.runs_dir, args.expected_count)
    elif args.check == "serving_sweep":
        success = validate_serving_sweep(args.runs_dir)
    
    exit(0 if success else 1)
