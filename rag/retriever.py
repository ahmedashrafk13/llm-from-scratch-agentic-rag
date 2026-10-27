"""Chunking, dense / sparse / hybrid retrieval and cross-encoder reranking."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "rag_corpus.jsonl"

EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
}
RERANKER = "BAAI/bge-reranker-base"


@dataclass
class Chunk:
    id: int
    text: str


def load_passages() -> list[str]:
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def chunk_corpus(passages: list[str], size: int = 128, overlap: int = 32) -> list[Chunk]:
    """Sliding word windows (size, overlap) over the passage stream. The corpus is
    ordered by article, so a window can join adjacent paragraphs of one article; at
    article boundaries a window may straddle two articles."""
    assert 0 <= overlap < size
    words: list[str] = []
    for p in passages:
        words.extend(p.split())
    step = size - overlap
    return [Chunk(i, " ".join(words[s:s + size]))
            for i, s in enumerate(range(0, max(1, len(words) - overlap), step))]


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: str = "bge-small", device: str | None = None):
        import faiss
        import torch
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name, self.q_prefix, self.d_prefix = EMBEDDERS[embedder]
        self.encoder = SentenceTransformer(name, device=self.device)
        emb = self.encoder.encode([self.d_prefix + c.text for c in chunks], batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=False)
        self.index = faiss.IndexFlatIP(emb.shape[1])  # exact cosine search; corpus is small
        self.index.add(emb.astype(np.float32))
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER, device=self.device, max_length=512)
        return self._reranker

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.encoder.encode([self.q_prefix + query], normalize_embeddings=True)
        s, i = self.index.search(q.astype(np.float32), k)
        return list(zip(i[0].tolist(), s[0].tolist()))

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def hybrid(self, query: str, k: int, pool: int = 50, rrf_k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion of BM25 and dense rankings."""
        fused: dict[int, float] = {}
        for ranking in (self.dense(query, pool), self.sparse(query, pool)):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]

    def rerank(self, query: str, cands: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        scores = self.reranker.predict([(query, self.chunks[i].text) for i, _ in cands],
                                       batch_size=32, show_progress_bar=False)
        order = np.argsort(-scores)[:k]
        return [(cands[j][0], float(scores[j])) for j in order]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool = True,
               pool: int = 30) -> list[tuple[int, float]]:
        first = {"dense": self.dense, "sparse": self.sparse, "hybrid": self.hybrid}[mode](query, pool if rerank else k)
        return self.rerank(query, first, k) if rerank else first[:k]
