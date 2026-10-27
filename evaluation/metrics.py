"""Answer and retrieval metrics (SQuAD-style normalisation)."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_answer(pred: str, gold: str) -> float:
    """Lenient match: gold answer appears in the prediction (useful for chatty models)."""
    g = normalize(gold)
    return float(bool(g) and g in normalize(pred))


def recall_at_k(ranked_relevance: list[bool], k: int) -> float:
    return float(any(ranked_relevance[:k]))


def reciprocal_rank(ranked_relevance: list[bool]) -> float:
    for i, rel in enumerate(ranked_relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0
