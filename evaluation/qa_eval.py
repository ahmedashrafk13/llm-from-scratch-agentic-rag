"""End-to-end QA experiments on the held-out domain questions.

Experiments (all on the same seeded sample of held-out questions):
  1. alignment stages    scratch base vs SFT vs SFT+DPO        (closed-book and with RAG)
  2. scale               Qwen2.5-1.5B QLoRA vs scratch SFT+DPO on identical tasks
  3. RAG vs fine-tune    RAG-only (Qwen base + RAG) / FT-only (QLoRA, no RAG) / RAG+FT
  4. agent ablations     top-k in {1,3,5}, self-correction on/off   (RAG+FT system)
  5. hallucination       hand-built set: answerable + unanswerable questions

Answer metrics: EM, token-F1, contains-gold. RAG runs add RAGAS-style metrics.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import datasets  # noqa: F401
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import contains_answer, exact_match, normalize, token_f1  # noqa: E402
from evaluation.rag_metrics import RagJudge  # noqa: E402
from rag.agent import ABSTAIN, RAGAgent  # noqa: E402
from rag.generators import QwenGenerator, ScratchGenerator  # noqa: E402
from rag.retriever import Retriever, chunk_corpus, load_passages  # noqa: E402
from training.chat_format import read_jsonl  # noqa: E402

OUT = ROOT / "results" / "qa_experiments.json"
SAMPLES = ROOT / "results" / "qa_samples.jsonl"
ABSTAIN_MARKERS = ("i don't know", "i do not know", "not in the documents", "cannot be determined",
                   "not mentioned", "no information")


def is_abstain(ans: str) -> bool:
    a = ans.lower()
    return any(m in a for m in ABSTAIN_MARKERS)


def closed_book(gen, qs) -> tuple[dict, list]:
    rows = []
    for q in qs:
        a = gen.generate(q["question"])
        rows.append({"q": q["question"], "gold": q["answer"], "pred": a})
    return answer_metrics(rows), rows


def answer_metrics(rows) -> dict:
    return {"exact_match": round(np.mean([exact_match(r["pred"], r["gold"]) for r in rows]), 4),
            "token_f1": round(np.mean([token_f1(r["pred"], r["gold"]) for r in rows]), 4),
            "contains_gold": round(np.mean([contains_answer(r["pred"], r["gold"]) for r in rows]), 4),
            "n": len(rows)}


def with_rag(agent: RAGAgent, qs, judge: RagJudge) -> tuple[dict, list]:
    rows, faith, rel, cprec = [], [], [], []
    t0 = time.time()
    for q in qs:
        res = agent.run(q["question"])
        rows.append({"q": q["question"], "gold": q["answer"], "pred": res.answer, "trace": res.trace,
                     "abstained": res.abstained})
        cprec.append(judge.context_precision(res.contexts, q["answer"], normalize))
        if not res.abstained and res.answer:
            faith.append(judge.faithfulness(q["question"], res.answer, res.contexts))
            rel.append(judge.answer_relevancy(q["question"], res.answer))
    m = answer_metrics(rows)
    m.update({"faithfulness": round(float(np.mean(faith)) if faith else 0.0, 4),
              "answer_relevancy": round(float(np.mean(rel)) if rel else 0.0, 4),
              "context_precision": round(float(np.mean(cprec)), 4),
              "abstain_rate": round(np.mean([r["abstained"] for r in rows]), 4),
              "self_correct_rate": round(np.mean([any("self-correct" in t for t in r["trace"]) for r in rows]), 4),
              "s_per_q": round((time.time() - t0) / len(qs), 2)})
    return m, rows


def hallucination(agent_or_gen, testset, judge: RagJudge, use_rag: bool) -> dict:
    """Hallucination = an answer that is neither correct, nor supported, nor an abstention.
       unanswerable: any non-abstaining answer counts as a hallucination.
       answerable:   a non-abstaining answer with token-F1 < 0.5 vs gold AND faithfulness < 0.5."""
    un, ans, rows = [], [], []
    for item in testset:
        if use_rag:
            r = agent_or_gen.run(item["question"])
            pred, ctx, abst = r.answer, r.contexts, r.abstained or is_abstain(r.answer)
        else:
            pred, ctx = agent_or_gen.generate(item["question"]), []
            abst = is_abstain(pred)
        if item["answerable"]:
            correct = token_f1(pred, item["answer"]) >= 0.5 or contains_answer(pred, item["answer"])
            supported = bool(ctx) and judge.faithfulness(item["question"], pred, ctx) >= 0.5
            ans.append(not abst and not correct and not supported)
        else:
            un.append(not abst)
        rows.append({"q": item["question"], "answerable": item["answerable"], "pred": pred})
    return {"hallucination_rate": round(float(np.mean(un + ans)), 4),
            "unanswerable_answered_rate": round(float(np.mean(un)), 4),
            "answerable_unsupported_wrong_rate": round(float(np.mean(ans)), 4),
            "n_unanswerable": len(un), "n_answerable": len(ans)}, rows


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiment groups to (re)run")
    args = ap.parse_args()
    random.seed(0)
    qs = read_jsonl(ROOT / "data" / "processed" / "domain_eval.jsonl")
    random.Random(0).shuffle(qs)
    qs = qs[:args.n]
    hset = read_jsonl(ROOT / "evaluation" / "hallucination_testset.jsonl")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    samples = open(SAMPLES, "a", encoding="utf-8")

    def record(group, key, metrics, rows):
        results.setdefault(group, {})[key] = metrics
        print(group, key, metrics, flush=True)
        OUT.write_text(json.dumps(results, indent=2))
        for r in rows[:15]:
            samples.write(json.dumps({"group": group, "system": key, **r}) + "\n")
        samples.flush()

    want = lambda g: args.only is None or g in args.only  # noqa: E731
    judge = RagJudge()
    retr = Retriever(chunk_corpus(load_passages(), 128, 32))

    if want("1_alignment_stages"):
        for ck in ("base", "sft", "dpo"):
            g = ScratchGenerator(ck)
            record("1_alignment_stages", f"scratch-{ck} closed-book", *closed_book(g, qs))
            record("1_alignment_stages", f"scratch-{ck} + RAG", *with_rag(RAGAgent(retr, g), qs, judge))
            free(g)

    need_qwen = any(want(g) for g in ("2_scale", "3_rag_vs_finetune", "4_agent_ablations", "5_hallucination"))
    if need_qwen:
        base = QwenGenerator(adapter=None)
        if want("3_rag_vs_finetune"):
            record("3_rag_vs_finetune", "base (no RAG, no FT)", *closed_book(base, qs))
            record("3_rag_vs_finetune", "RAG only (base + RAG)", *with_rag(RAGAgent(retr, base), qs, judge))
        if want("5_hallucination"):
            m, rows = hallucination(base, hset, judge, use_rag=False)
            record("5_hallucination", "qwen base, no RAG", m, rows)
            m, rows = hallucination(RAGAgent(retr, base), hset, judge, use_rag=True)
            record("5_hallucination", "qwen base + agentic RAG", m, rows)
        free(base)
        ft = QwenGenerator(adapter="qlora")
        if want("3_rag_vs_finetune"):
            record("3_rag_vs_finetune", "FT only (QLoRA, no RAG)", *closed_book(ft, qs))
            record("3_rag_vs_finetune", "RAG + FT (QLoRA + RAG)", *with_rag(RAGAgent(retr, ft), qs, judge))
        if want("2_scale"):
            sc = ScratchGenerator("dpo")
            for name, g in (("scratch-dpo 37M", sc), ("qwen-qlora 1.5B", ft)):
                record("2_scale", f"{name} closed-book", *closed_book(g, qs))
                record("2_scale", f"{name} + RAG", *with_rag(RAGAgent(retr, g), qs, judge))
            free(sc)
        if want("4_agent_ablations"):
            for k in (1, 3, 5):
                record("4_agent_ablations", f"top-k={k}", *with_rag(RAGAgent(retr, ft, k=k), qs, judge))
            record("4_agent_ablations", "top-k=3, no reranker",
                   *with_rag(RAGAgent(retr, ft, k=3, rerank=False), qs, judge))
            record("4_agent_ablations", "top-k=3, no self-correction",
                   *with_rag(RAGAgent(retr, ft, k=3, self_correct=False), qs, judge))
        if want("5_hallucination"):
            m, rows = hallucination(ft, hset, judge, use_rag=False)
            record("5_hallucination", "qwen QLoRA, no RAG", m, rows)
            m, rows = hallucination(RAGAgent(retr, ft), hset, judge, use_rag=True)
            record("5_hallucination", "qwen QLoRA + agentic RAG", m, rows)
            m, rows = hallucination(RAGAgent(retr, ft, self_correct=False), hset, judge, use_rag=True)
            record("5_hallucination", "qwen QLoRA + RAG, no self-correction", m, rows)
    samples.close()


if __name__ == "__main__":
    main()
