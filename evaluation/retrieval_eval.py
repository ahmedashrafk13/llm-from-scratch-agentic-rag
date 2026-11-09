"""Retrieval ablations: Recall@k and MRR@10.

Queries: every domain question whose answer is not yes/no (yes/no answers occur
in almost every chunk, so string-match relevance would be meaningless for them).
Relevance: a chunk is relevant if it contains the normalised gold answer.

Caveat (stated, not hidden): string-match relevance favours larger chunks, since
a bigger chunk is more likely to contain the answer string by chance. The chunk-size
table therefore also reports mean chunk length; compare sizes with that in mind.

Ablations
  A. retriever mode   dense / BM25 / hybrid(RRF)      x reranker off / on
  B. chunk size       64 / 128 / 256 words            x overlap 0 / 32
  C. embedding model  BGE-small vs E5-small (dense only)

Resumable: each finished configuration is saved and skipped on the next run (--fresh recomputes).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import datasets  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import normalize, recall_at_k, reciprocal_rank  # noqa: E402
from evaluation.store import mirror  # noqa: E402
from rag.retriever import Retriever, chunk_corpus, load_passages  # noqa: E402
from training.chat_format import read_jsonl  # noqa: E402

KS = (1, 3, 5, 10)


def queries() -> list[dict]:
    proc = ROOT / "data" / "processed"
    qs = read_jsonl(proc / "domain_eval.jsonl")
    return [q for q in qs if normalize(q["answer"]) not in ("yes", "no", "")]


def score(retr: Retriever, qs, mode: str, rerank: bool) -> dict:
    chunk_norm = [normalize(c.text) for c in retr.chunks]
    rec = {k: 0.0 for k in KS}
    mrr = 0.0
    t0 = time.time()
    for q in qs:
        hits = retr.search(q["question"], k=10, mode=mode, rerank=rerank, pool=30)
        a = normalize(q["answer"])
        rel = [a in chunk_norm[i] for i, _ in hits]
        for k in KS:
            rec[k] += recall_at_k(rel, k)
        mrr += reciprocal_rank(rel)
    n = len(qs)
    return {**{f"recall@{k}": round(v / n, 4) for k, v in rec.items()}, "mrr@10": round(mrr / n, 4),
            "ms_per_query": round(1000 * (time.time() - t0) / n, 1), "n_queries": n}


def main():
    qs = queries()
    passages = load_passages()
    path = ROOT / "results" / "retrieval_ablation.json"
    out = json.loads(path.read_text()) if path.exists() and "--fresh" not in sys.argv else {}
    for sec in ("A_mode_x_rerank", "B_chunking", "C_embedder"):
        out.setdefault(sec, {})
    cache: dict = {}

    def retriever(size=128, ov=32, embedder="bge-small"):
        if (size, ov, embedder) not in cache:
            cache[(size, ov, embedder)] = Retriever(chunk_corpus(passages, size, ov), embedder=embedder)
        return cache[(size, ov, embedder)]

    def run(sec, key, fn):  # resumable: each finished config is saved and skipped next time
        if key in out[sec]:
            print(f"[skip] {sec} / {key} (done)", flush=True)
            return
        out[sec][key] = fn()
        print(key, out[sec][key], flush=True)
        path.write_text(json.dumps(out, indent=2))
        mirror(path)

    for mode in ("dense", "sparse", "hybrid"):
        for rr in (False, True):
            run("A_mode_x_rerank", f"{mode}{'+rerank' if rr else ''}",
                lambda mode=mode, rr=rr: score(retriever(), qs, mode, rr))

    for size in (64, 128, 256):
        for ov in (0, 32):
            def chunking(size=size, ov=ov):
                r = retriever(size, ov)
                return {**score(r, qs, "hybrid", True), "n_chunks": len(r.chunks),
                        "mean_words": round(sum(len(c.text.split()) for c in r.chunks) / len(r.chunks), 1)}
            run("B_chunking", f"size{size}_overlap{ov}", chunking)
            if (size, ov) != (128, 32):  # keep the default index; drop the others to save memory
                cache.pop((size, ov, "bge-small"), None)

    run("C_embedder", "bge-small", lambda: score(retriever(), qs, "dense", False))
    run("C_embedder", "e5-small", lambda: score(retriever(embedder="e5-small"), qs, "dense", False))
    path.write_text(json.dumps(out, indent=2))
    mirror(path)


if __name__ == "__main__":
    main()
