"""Build evaluation/hallucination_testset.jsonl.

Unanswerable questions were written by hand for this project: plausible factoid
questions about topics that look like the corpus (people, animals, countries,
science) but are absent from it. Each carries a key term; a question is kept only
if that term never appears in the corpus, so "the answer is not in the documents"
is verifiably the correct response.

Answerable questions: a seeded sample of held-out domain_eval items (non yes/no).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import normalize  # noqa: E402
from training.chat_format import read_jsonl  # noqa: E402

UNANSWERABLE = [
    ("Who designed the Sydney Opera House?", "Opera House"),
    ("In what year did the Chernobyl disaster happen?", "Chernobyl"),
    ("What is the top speed of a cheetah?", "cheetah"),
    ("Who wrote the novel Moby-Dick?", "Moby"),
    ("What is the capital city of Kazakhstan?", "Kazakhstan"),
    ("Who was the first woman to win a Nobel Prize?", "Marie Curie"),
    ("How many moons does Neptune have?", "Neptune"),
    ("What instrument did Jimi Hendrix play?", "Hendrix"),
    ("Who founded the company Tesla Motors?", "Tesla Motors"),
    ("What is the national animal of Scotland?", "Scotland"),
    ("Which river flows through Budapest?", "Budapest"),
    ("Who invented the World Wide Web?", "World Wide Web"),
    ("What is the lifespan of a giant tortoise?", "tortoise"),
    ("Who painted The Starry Night?", "Starry Night"),
    ("In which year was the Berlin Wall torn down?", "Berlin Wall"),
    ("What language is spoken in Andorra?", "Andorra"),
    ("Who was the 46th president of the United States?", "Biden"),
    ("What is the main ingredient in guacamole?", "guacamole"),
    ("How tall is Mount Kilimanjaro?", "Kilimanjaro"),
    ("Who composed The Four Seasons?", "Vivaldi"),
    ("What does a platypus eat?", "platypus"),
    ("Where is the Taj Mahal located?", "Taj Mahal"),
    ("Who discovered penicillin?", "penicillin"),
    ("What is the currency of Thailand?", "Thailand"),
    ("Who directed the film Jurassic Park?", "Jurassic"),
    ("What is the largest desert in Asia?", "Gobi"),
    ("Which planet has the Great Red Spot?", "Great Red Spot"),
    ("Who was the first person to climb Mount Everest?", "Everest"),
    ("What year did the Titanic sink?", "Titanic"),
    ("What is the chemical symbol for tungsten?", "tungsten"),
]


def main(n_answerable: int = 40):
    corpus = " ".join(json.loads(l)["text"] for l in open(ROOT / "data" / "processed" / "rag_corpus.jsonl",
                                                            encoding="utf-8"))
    rows, dropped = [], []
    for q, term in UNANSWERABLE:
        if re.search(r"\b" + re.escape(term) + r"\b", corpus, re.I):
            dropped.append(term)
        else:
            rows.append({"question": q, "answer": None, "answerable": False, "key_term": term})
    ev = [r for r in read_jsonl(ROOT / "data" / "processed" / "domain_eval.jsonl")
          if normalize(r["answer"]) not in ("yes", "no")]
    random.Random(1).shuffle(ev)
    rows += [{"question": r["question"], "answer": r["answer"], "answerable": True} for r in ev[:n_answerable]]
    with open(ROOT / "evaluation" / "hallucination_testset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"unanswerable kept {sum(not r['answerable'] for r in rows)}, dropped (term in corpus): {dropped}")


if __name__ == "__main__":
    main()
