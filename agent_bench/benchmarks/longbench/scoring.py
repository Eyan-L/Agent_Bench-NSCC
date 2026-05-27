"""LongBench v1 metrics — vendored from THUDM/LongBench (metrics.py + eval.py scorer).

Heavy deps (``jieba``, ``rouge``, ``fuzzywuzzy``) are imported lazily inside the
functions that need them, so English / QA scoring (qa_f1, count, retrieval,
classification) works with the stdlib alone. Install the extras for the zh / rouge /
code subtasks::

    pip install jieba rouge fuzzywuzzy python-Levenshtein
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Dict, List, Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove (EN+ZH) punctuation and extra whitespace."""
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』"
        "【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    no_punc = "".join(ch for ch in s.lower() if ch not in all_punctuation)
    return "".join(no_punc.split())


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(gt_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"段落(\d+)", ground_truth)
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(gt_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    from fuzzywuzzy import fuzz  # lazy
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"] or []
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    return (1.0 / len(em_match_list)) if ground_truth in em_match_list else 0.0


def rouge_score(prediction, ground_truth, **kwargs):
    from rouge import Rouge  # lazy
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    import jieba  # lazy
    prediction = " ".join(jieba.cut(prediction, cut_all=False))
    ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_score(prediction, ground_truth)


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    pred = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    return f1_score(pred, gt)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    import jieba  # lazy
    pred = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gt = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred = [t for t in pred if t]
    gt = [t for t in gt if t]
    return f1_score(pred, gt)


METRICS: Dict[str, Callable] = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}

# LongBench truncates the prediction to its first line for these few-shot/classification tasks.
_FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def score_one(
    dataset: str,
    metric_name: str,
    prediction: str,
    answers: List[str],
    all_classes: Optional[List[str]] = None,
) -> float:
    """Score one prediction against its ground-truth answers (LongBench eval.py scorer).

    Returns a 0..1 score: the max over all reference answers. Aggregate by mean and
    ``round(100 * mean, 2)`` for the conventional LongBench number.
    """
    fn = METRICS[metric_name]
    pred = prediction
    if dataset in _FIRST_LINE_DATASETS:
        pred = pred.lstrip("\n").split("\n")[0]
    best = 0.0
    for gt in (answers or [""]):
        best = max(best, float(fn(pred, gt, all_classes=all_classes)))
    return best
