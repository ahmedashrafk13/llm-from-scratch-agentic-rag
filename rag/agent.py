"""Agentic RAG controller with tool use and a self-correction loop.

    question
      |-- arithmetic? ----------------------------> calculator tool -> answer
      |
      v
    search_docs(question)  hybrid BM25+dense -> cross-encoder rerank
      |
      |-- top rerank score >= tau ----------------> generate answer from context
      |
      v  (weak retrieval: self-correction)
    rewrite query (keywords + entity lookup tool) -> search_docs again, merge
      |
      |-- still < tau ----------------------------> abstain ("not in the documents")
      v
    generate answer from context

The controller is deterministic Python, not LLM-planned: a 37M model cannot emit
reliable tool-call syntax, and a fixed policy keeps every generator comparable on
identical retrieval. The LLM is used only to read the context and write the answer.
Every step is recorded in `trace` so the demo can show its reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import Retriever, load_passages
from .tools import calculator, extract_arithmetic, lookup

ABSTAIN = "I don't know - the answer is not in the documents."
STOP = set("what who whom whose which when where why how is are was were do does did the a an of in on to for "
           "and or by with from at as that this it its be been name list".split())


@dataclass
class AgentResult:
    answer: str
    contexts: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    retries: int = 0
    abstained: bool = False


def rewrite_query(q: str, passages: list[str]) -> str:
    """Keyword query + definitions of any capitalised entities found via the lookup tool."""
    kws = [w for w in re.findall(r"[A-Za-z0-9']+", q) if w.lower() not in STOP]
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", q)
    extra = []
    for e in entities[:2]:
        hits = lookup(e, passages, max_hits=1)
        if hits:
            extra.append(" ".join(hits[0].split()[:15]))
    return " ".join(kws + extra)


def build_prompt(question: str, contexts: list[str], max_words: int) -> str:
    ctx, used = [], 0
    for c in contexts:
        w = c.split()
        take = w[:max(0, max_words - used)]
        if not take:
            break
        ctx.append(" ".join(take))
        used += len(take)
    return f"Context: {' '.join(ctx)}\n\nQuestion: {question}"


class RAGAgent:
    def __init__(self, retriever: Retriever, generator, k: int = 3, tau: float = 0.1,
                 self_correct: bool = True, rerank: bool = True, mode: str = "hybrid", max_ctx_words: int | None = None):
        self.r, self.gen = retriever, generator
        self.k, self.tau, self.self_correct, self.rerank, self.mode = k, tau, self_correct, rerank, mode
        # the 512-token from-scratch model gets a tighter context budget than Qwen
        self.max_ctx_words = max_ctx_words or (150 if getattr(generator, "name", "").startswith("scratch") else 400)
        self.passages = load_passages()

    def _search(self, q: str):
        return self.r.search(q, k=self.k, mode=self.mode, rerank=self.rerank)

    def run(self, question: str) -> AgentResult:
        res = AgentResult(answer="")
        expr = extract_arithmetic(question)
        if expr:
            res.trace.append(f"tool:calculator({expr})")
            res.answer = calculator(expr)
            return res

        hits = self._search(question)
        top = hits[0][1] if hits else float("-inf")
        res.trace.append(f"tool:search_docs -> top score {top:.3f}")
        # tau is defined on reranker probabilities; without reranking there is no calibrated score
        weak = self.rerank and top < self.tau
        if weak and self.self_correct:
            q2 = rewrite_query(question, self.passages)
            res.trace.append(f"self-correct: weak retrieval, rewrite -> {q2!r}")
            hits2 = self._search(q2)
            res.retries = 1
            merged = {i: s for i, s in hits + hits2}
            hits = sorted(merged.items(), key=lambda x: -x[1])[:self.k]
            top = hits[0][1]
            res.trace.append(f"tool:search_docs (retry) -> top score {top:.3f}")
            weak = top < self.tau
        res.contexts = [self.r.chunks[i].text for i, _ in hits]
        if weak:
            res.trace.append("abstain: retrieval below threshold")
            res.answer, res.abstained = ABSTAIN, True
            return res
        res.answer = self.gen.generate(build_prompt(question, res.contexts, self.max_ctx_words))
        res.trace.append(f"generate:{getattr(self.gen, 'name', 'llm')}")
        return res
