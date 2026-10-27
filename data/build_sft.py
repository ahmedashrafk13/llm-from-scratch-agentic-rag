"""Build the instruction-tuning and domain-QA datasets.

Outputs (data/processed/):
  sft_train.jsonl / sft_val.jsonl  {"prompt", "response", "source"}
  domain_eval.jsonl                held-out domain questions with gold answers (never trained on)
  rag_corpus.jsonl                 passages for the RAG index

Sources
  * General instructions: databricks-dolly-15k, filtered to short closed-QA /
    open-QA / brainstorm items without context (a 37M model with a 512-token
    window cannot use long contexts).
  * Domain QA, created for this project from rag-datasets/rag-mini-wikipedia:
    each question-answer pair is turned into two training examples -
      closed-book : "Q -> A"
      open-book   : "Context: <passage containing the answer> Q -> A"
    so the same SFT set teaches both answering and reading a retrieved passage.
    Questions are split 50/50 by a stable hash into train vs held-out eval.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from rank_bm25 import BM25Okapi

OUT = Path(__file__).resolve().parent / "processed"


def tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def is_eval(q: str) -> bool:
    return int(hashlib.md5(q.encode()).hexdigest(), 16) % 2 == 0


def best_passage(question: str, answer: str, bm25, passages: list[str]) -> str | None:
    scores = bm25.get_scores(tok(question + " " + answer))
    for i in sorted(range(len(passages)), key=lambda i: -scores[i])[:20]:
        if answer.lower() in passages[i].lower():
            return passages[i]
    return None


def trim_passage(p: str, answer: str, max_words: int = 120) -> str:
    words = p.split()
    if len(words) <= max_words:
        return p
    pos = p.lower().find(answer.lower())
    centre = len(p[:pos].split()) if pos >= 0 else 0
    lo = max(0, centre - max_words // 2)
    return " ".join(words[lo:lo + max_words])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    passages = [r["passage"].strip() for r in corpus if len(r["passage"].split()) >= 8]
    with open(OUT / "rag_corpus.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(passages):
            f.write(json.dumps({"id": i, "text": p}) + "\n")
    bm25 = BM25Okapi([tok(p) for p in passages])

    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    domain_train, domain_eval = [], []
    seen = set()
    for r in qa:
        q, a = r["question"].strip(), str(r["answer"]).strip()
        if not q or not a or q in seen or len(a.split()) > 25:
            continue
        seen.add(q)
        (domain_eval if is_eval(q) else domain_train).append({"question": q, "answer": a})

    sft = []
    for ex in domain_train:
        sft.append({"prompt": ex["question"], "response": ex["answer"], "source": "domain_closed"})
        p = best_passage(ex["question"], ex["answer"], bm25, passages)
        if p:
            ctx = trim_passage(p, ex["answer"])
            sft.append({"prompt": f"Context: {ctx}\n\nQuestion: {ex['question']}",
                        "response": ex["answer"], "source": "domain_open"})

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    general = [{"prompt": r["instruction"].strip(), "response": r["response"].strip(), "source": "dolly"}
               for r in dolly
               if not r["context"] and r["category"] in ("open_qa", "general_qa", "brainstorming", "classification")
               and len(r["response"].split()) <= 80 and len(r["instruction"].split()) <= 40]
    random.shuffle(general)
    sft += general[:4000]
    random.shuffle(sft)
    n_val = 300
    for name, rows in {"sft_val": sft[:n_val], "sft_train": sft[n_val:]}.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(OUT / "domain_eval.jsonl", "w", encoding="utf-8") as f:
        for r in domain_eval:
            f.write(json.dumps(r) + "\n")
    stats = {"passages": len(passages), "domain_train_q": len(domain_train), "domain_eval_q": len(domain_eval),
             "sft_train": len(sft) - n_val, "sft_val": n_val,
             "by_source": {s: sum(r["source"] == s for r in sft) for s in ("domain_closed", "domain_open", "dolly")}}
    print(stats)
    (OUT / "sft_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
