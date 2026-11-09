"""RAGAS-style metrics computed locally (no API judge).

The `ragas` package needs an LLM judge (by default OpenAI). To stay offline and
reproducible, the same three quantities are computed with local models; the
definitions follow RAGAS, the judges are swapped:

  faithfulness       share of answer claims entailed by the retrieved context.
                     Judge: NLI cross-encoder (DeBERTa-v3, MNLI/SNLI). Short answers
                     are turned into a claim "Q -> A" before entailment.
  answer_relevancy   RAGAS generates questions from the answer and compares them with
                     the original; here: cosine(question, answer) in BGE space
                     (an approximation - documented as such in the README).
  context_precision  RAGAS definition: mean over relevant ranks k of precision@k, with
                     relevance = context contains the gold answer (string judge).
"""
from __future__ import annotations

import re

import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
# Bump when a judge definition changes: qa_eval then recomputes every judged result.
# v2: yes/no answers are judged as declarative statements (v1 scored them all as neutral).
JUDGE_VERSION = 2
YES_NO_QUESTION = re.compile(r"^(?:is|was|are|were|do|does|did|can|could|has|have|had|will|would|should|shall|may|"
                             r"might|must)\s+(.+?)\s*\?*$", re.I)


class RagJudge:
    def __init__(self, device: str = "cuda"):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.nli = CrossEncoder(NLI_MODEL, device=device, max_length=512)
        self.label_contra, self.label_entail = 0, 1  # model card label order: contradiction, entailment, neutral
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    def claims(self, question: str, answer: str) -> list[tuple[str, int]]:
        """Answer -> [(hypothesis, NLI label that counts as support)]."""
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) >= 3]
        if sents:
            return [(s, self.label_entail) for s in sents]
        short = answer.strip().rstrip(".!").lower()
        m = YES_NO_QUESTION.match(question.strip())
        if m and short in ("yes", "no"):
            # "Is X the fifth president?" + yes/no -> hypothesis "X the fifth president." An NLI model
            # reads a question followed by "yes" as neutral, so the claim must be a statement; a "no"
            # answer is supported when the context contradicts the statement.
            return [(m.group(1).rstrip("? ") + ".", self.label_entail if short == "yes" else self.label_contra)]
        return [(f"{question.rstrip('?')}? {answer}.", self.label_entail)]  # short factoid answer

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        ctx = " ".join(contexts)[:3000]
        cl = self.claims(question, answer)
        probs = self.nli.predict([(ctx, h) for h, _ in cl], apply_softmax=True, show_progress_bar=False)
        return float(np.mean(np.argmax(probs, -1) == np.array([lab for _, lab in cl])))

    def answer_relevancy(self, question: str, answer: str) -> float:
        q, a = self.emb.encode([question, answer], normalize_embeddings=True)
        return float(q @ a)

    @staticmethod
    def context_precision(contexts: list[str], gold: str, norm) -> float:
        rel = [norm(gold) in norm(c) for c in contexts]
        if not any(rel):
            return 0.0
        hits, total = 0, 0.0
        for k, r in enumerate(rel, 1):
            if r:
                hits += 1
                total += hits / k
        return total / sum(rel)
