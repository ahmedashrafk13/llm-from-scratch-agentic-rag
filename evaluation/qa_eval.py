"""End-to-end QA experiments on the held-out domain questions.

Experiments (all on the same seeded sample of held-out questions):
  1. alignment stages    scratch base vs SFT vs SFT+DPO        (closed-book and with RAG)
  2. scale               Qwen2.5-1.5B QLoRA vs scratch SFT+DPO on identical tasks
  3. RAG vs fine-tune    RAG-only (Qwen base + RAG) / FT-only (QLoRA, no RAG) / RAG+FT
  4. agent ablations     top-k in {1,3,5}, self-correction on/off   (RAG+FT system)
  5. hallucination       hand-built set: answerable + unanswerable questions

Answer metrics: EM, token-F1, contains-gold. RAG runs add RAGAS-style metrics.

Resumable: results are saved after every experiment and finished ones are skipped on the
next run (use --fresh to recompute). Set RESULTS_MIRROR=<dir> to also copy them there
after each experiment (the Colab notebook points it at Google Drive).
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
from evaluation.store import mirror  # noqa: E402
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


class Lazy:
    """Build an expensive object (model, index) only when an experiment actually needs it."""

    def __init__(self, factory):
        self.factory, self.obj = factory, None

    def get(self):
        if self.obj is None:
            self.obj = self.factory()
        return self.obj

    def free(self):
        self.obj = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiment groups to (re)run")
    ap.add_argument("--judge-device", default="cuda", help="'cpu' frees ~1 GB of VRAM on 4 GB cards")
    ap.add_argument("--fresh", action="store_true", help="ignore saved results and recompute everything")
    args = ap.parse_args()
    random.seed(0)
    qs = read_jsonl(ROOT / "data" / "processed" / "domain_eval.jsonl")
    random.Random(0).shuffle(qs)
    qs = qs[:args.n]
    hset = read_jsonl(ROOT / "evaluation" / "hallucination_testset.jsonl")
    results = {} if args.fresh or not OUT.exists() else json.loads(OUT.read_text())
    if results.get("_n", args.n) != args.n:
        raise SystemExit(f"{OUT.name} holds n={results['_n']} results; pass --n {results['_n']} or --fresh")
    results["_n"] = args.n
    # config -> [group, key] of the experiment that measured it. Decoding is greedy, so identical
    # configurations (e.g. "RAG + FT" and "top-k=3") give identical answers: measure once, reuse.
    index = results.setdefault("_index", {})
    samples = open(SAMPLES, "a", encoding="utf-8")

    def save():
        OUT.write_text(json.dumps(results, indent=2))
        mirror(OUT, SAMPLES)

    def run(group, key, cfg, fn):
        if args.only is not None and group not in args.only:
            return
        if key in results.get(group, {}):
            print(f"[skip] {group} / {key} (done)", flush=True)
            return
        if cfg in index:
            g0, k0 = index[cfg]
            metrics, rows = results[g0][k0], []
            print(f"[reuse] {group} / {key} == {g0} / {k0}", flush=True)
        else:
            metrics, rows = fn()
            index[cfg] = [group, key]
        results.setdefault(group, {})[key] = metrics
        print(group, key, metrics, flush=True)
        for r in rows[:15]:
            samples.write(json.dumps({"group": group, "system": key, **r}) + "\n")
        samples.flush()
        save()

    judge = Lazy(lambda: RagJudge(device=args.judge_device))
    retr = Lazy(lambda: Retriever(chunk_corpus(load_passages(), 128, 32)))

    def rag_cfg(name, k=3, rerank=True, self_correct=True):
        ctx = 150 if name.startswith("scratch") else 400  # RAGAgent's default context budget
        return f"rag|{name}|k={k}|rerank={rerank}|selfcorrect={self_correct}|hybrid|ctx={ctx}"

    def rag(gen, **kw):
        return lambda: with_rag(RAGAgent(retr.get(), gen.get(), **kw), qs, judge.get())

    def closed(gen):
        return lambda: closed_book(gen.get(), qs)

    def hall(gen, use_rag, **kw):
        def f():
            target = RAGAgent(retr.get(), gen.get(), **kw) if use_rag else gen.get()
            return hallucination(target, hset, judge.get(), use_rag=use_rag)
        return f

    G1 = "1_alignment_stages"
    for ck in ("base", "sft", "dpo"):
        g = Lazy(lambda ck=ck: ScratchGenerator(ck))
        run(G1, f"scratch-{ck} closed-book", f"closed|scratch-{ck}", closed(g))
        run(G1, f"scratch-{ck} + RAG", rag_cfg(f"scratch-{ck}"), rag(g))
        g.free()

    base = Lazy(lambda: QwenGenerator(adapter=None))
    QB = "qwen1.5b-base"
    run("3_rag_vs_finetune", "base (no RAG, no FT)", f"closed|{QB}", closed(base))
    run("3_rag_vs_finetune", "RAG only (base + RAG)", rag_cfg(QB), rag(base))
    run("5_hallucination", "qwen base, no RAG", f"hall|{QB}|norag", hall(base, False))
    run("5_hallucination", "qwen base + agentic RAG", f"hall|{rag_cfg(QB)}", hall(base, True))
    base.free()

    ft = Lazy(lambda: QwenGenerator(adapter="qlora"))
    QF = "qwen1.5b-qlora"
    run("3_rag_vs_finetune", "FT only (QLoRA, no RAG)", f"closed|{QF}", closed(ft))
    run("3_rag_vs_finetune", "RAG + FT (QLoRA + RAG)", rag_cfg(QF), rag(ft))
    sc = Lazy(lambda: ScratchGenerator("dpo"))
    for name, g, gname in (("scratch-dpo 37M", sc, "scratch-dpo"), ("qwen-qlora 1.5B", ft, QF)):
        run("2_scale", f"{name} closed-book", f"closed|{gname}", closed(g))
        run("2_scale", f"{name} + RAG", rag_cfg(gname), rag(g))
    sc.free()
    for k in (1, 3, 5):
        run("4_agent_ablations", f"top-k={k}", rag_cfg(QF, k=k), rag(ft, k=k))
    run("4_agent_ablations", "top-k=3, no reranker", rag_cfg(QF, rerank=False), rag(ft, k=3, rerank=False))
    run("4_agent_ablations", "top-k=3, no self-correction", rag_cfg(QF, self_correct=False),
        rag(ft, k=3, self_correct=False))
    run("5_hallucination", "qwen QLoRA, no RAG", f"hall|{QF}|norag", hall(ft, False))
    run("5_hallucination", "qwen QLoRA + agentic RAG", f"hall|{rag_cfg(QF)}", hall(ft, True))
    run("5_hallucination", "qwen QLoRA + RAG, no self-correction", f"hall|{rag_cfg(QF, self_correct=False)}",
        hall(ft, True, self_correct=False))
    ft.free()
    samples.close()
    print("all QA experiments done", flush=True)


if __name__ == "__main__":
    main()
