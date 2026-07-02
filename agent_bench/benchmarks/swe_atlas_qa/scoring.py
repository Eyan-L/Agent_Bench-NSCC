from __future__ import annotations

import logging
import requests
from typing import Any, Mapping

from agent_bench.core.config import load_active_model_and_serving

logger = logging.getLogger(__name__)


def grade_verifier(
    endpoint: str,
    api_key: str,
    model_name: str,
    question: str,
    reference: str,
    answer: str,
    criterion: str,
) -> bool:
    """Grade a single rubric criterion using the active model serving endpoint."""
    prompt = (
        "You are an expert grading assistant evaluating an agent's answer to a codebase question.\n\n"
        f"Question: {question}\n\n"
        f"Reference Answer: {reference}\n\n"
        f"Agent Answer: {answer}\n\n"
        f"Criterion to verify: {criterion}\n\n"
        "Does the Agent Answer demonstrate or satisfy the Criterion? Respond with exactly 'YES' or 'NO' and nothing else."
    )
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return "YES" in content
        else:
            logger.error(
                f"Judge call failed with status code {resp.status_code}: {resp.text}"
            )
    except Exception as e:
        logger.error(f"Error calling judge LLM: {e}")
    return False


def score_answer(
    instance_data: Mapping[str, Any],
    answer_text: str,
) -> dict[str, Any]:
    """Grade the answer against the rubric positive and negative verifiers."""
    import os
    model_cfg, _ = load_active_model_and_serving()
    if os.environ.get("DEEPSEEK_API_KEY"):
        endpoint = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        model_name = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    else:
        endpoint = os.environ.get("OPENAI_BASE_URL") or (model_cfg.endpoint if model_cfg else "")
        api_key = os.environ.get("OPENAI_API_KEY") or (model_cfg.api_key if model_cfg else "")
        model_name = model_cfg.name if model_cfg else ""

    question = instance_data.get("question", "")
    reference = instance_data.get("reference_answer", "")
    rubric = instance_data.get("rubric") or {}

    positives = (
        rubric.get("positive_verifiers")
        or rubric.get("must_have")
        or rubric.get("should_have")
        or []
    )
    negatives = (
        rubric.get("negative_verifiers")
        or rubric.get("must_not_have")
        or rubric.get("should_not_have")
        or []
    )

    if isinstance(positives, str):
        positives = [positives]
    if isinstance(negatives, str):
        negatives = [negatives]

    pos_results = []
    for criterion in positives:
        met = grade_verifier(
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name,
            question=question,
            reference=reference,
            answer=answer_text,
            criterion=criterion,
        )
        pos_results.append(met)

    neg_results = []
    for criterion in negatives:
        triggered = grade_verifier(
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name,
            question=question,
            reference=reference,
            answer=answer_text,
            criterion=criterion,
        )
        neg_results.append(triggered)

    all_pos_met = all(pos_results) if pos_results else True
    any_neg_triggered = any(neg_results) if neg_results else False

    passed = all_pos_met and not any_neg_triggered
    score = 1.0 if passed else 0.0

    return {
        "status": "PASS" if passed else "FAIL",
        "test_pass_rate": score,
        "passed": 1 if passed else 0,
        "failed": 0 if passed else 1,
        "positives_met": pos_results,
        "negatives_triggered": neg_results,
    }
