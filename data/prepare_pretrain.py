"""Download, clean, filter and deduplicate the pre-training corpus.

Sources
  * TinyStories (roneneldan/TinyStories) - simple, high-signal English for a small model.
  * FineWeb sample-10BT (HuggingFaceFW/fineweb), streamed - real web text so the
    model sees more than children's stories.

Pipeline
  1. Normalise whitespace / unicode.
  2. Quality filters (Gopher/C4-style heuristics): length, alphabetic ratio,
     mean word length, duplicate-line ratio, symbol ratio, boilerplate phrases.
  3. Exact dedup on a hash of normalised text.
  4. Near-dedup with MinHash + LSH (5-word shingles, 64 perms, 16 bands x 4 rows).
Writes data/processed/{tinystories,fineweb}.jsonl and a filter report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import datasets  # noqa: F401  (import before numpy/torch on Windows; see README)
import numpy as np
from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "processed"
BOILERPLATE = ("lorem ipsum", "javascript is disabled", "cookie policy", "all rights reserved",
               "terms of use", "click here", "privacy policy")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_reason(text: str, min_words: int, max_words: int) -> str | None:
    """Return the name of the first failed filter, or None if the doc passes."""
    words = text.split()
    n = len(words)
    if n < min_words:
        return "too_short"
    if n > max_words:
        return "too_long"
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < 0.70:
        return "low_alpha_ratio"
    mwl = sum(len(w) for w in words) / n
    if not 3 <= mwl <= 10:
        return "mean_word_len"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and 1 - len(set(lines)) / len(lines) > 0.3:
        return "dup_lines"
    if sum(text.count(s) for s in "#{}<>|") / n > 0.1:
        return "symbols"
    if sum(not c.isascii() for c in text) / len(text) > 0.1:
        return "non_english"
    low = text.lower()
    if any(b in low for b in BOILERPLATE):
        return "boilerplate"
    return None


class MinHashLSH:
    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = np.uint64((1 << 61) - 1)
        self.a = rng.integers(1, 1 << 61, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 61, num_perm, dtype=np.uint64)
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list[dict] = [defaultdict(list) for _ in range(bands)]

    def signature(self, text: str) -> np.ndarray:
        w = text.lower().split()
        shingles = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
        h = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") >> 3
                      for s in shingles], dtype=np.uint64)
        return ((np.outer(h, self.a) + self.b) % self.p).min(axis=0)

    def is_duplicate_then_add(self, doc_id: int, text: str) -> bool:
        sig = self.signature(text)
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
        dup = any(self.buckets[i].get(k) for i, k in enumerate(keys))
        if not dup:
            for i, k in enumerate(keys):
                self.buckets[i][k].append(doc_id)
        return dup


def process(name: str, docs, limit: int, min_words: int, max_words: int, near_dedup: bool) -> dict:
    stats: Counter[str] = Counter()
    seen: set[bytes] = set()
    lsh = MinHashLSH() if near_dedup else None
    kept = 0
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for doc in docs:
            if kept >= limit:
                break
            stats["seen"] += 1
            text = normalise(doc)
            reason = quality_reason(text, min_words, max_words)
            if reason:
                stats[f"drop_{reason}"] += 1
                continue
            h = hashlib.md5(re.sub(r"\W+", "", text.lower()).encode()).digest()
            if h in seen:
                stats["drop_exact_dup"] += 1
                continue
            seen.add(h)
            if lsh is not None and lsh.is_duplicate_then_add(kept, text):
                stats["drop_near_dup"] += 1
                continue
            f.write(json.dumps({"text": text}) + "\n")
            kept += 1
    stats["kept"] = kept
    print(name, dict(stats))
    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinystories", type=int, default=600_000)
    ap.add_argument("--fineweb", type=int, default=25_000)
    args = ap.parse_args()

    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ts_stats = process("tinystories", (r["text"] for r in ts), args.tinystories, 20, 2000, near_dedup=False)
    ts_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    process("tinystories_val", (r["text"] for r in ts_val), 5_000, 20, 2000, near_dedup=False)

    fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    fw_stats = process("fineweb", (r["text"] for r in fw), args.fineweb, 50, 1500, near_dedup=True)

    Path(__file__).resolve().parents[1].joinpath("results").mkdir(exist_ok=True)
    (Path(__file__).resolve().parents[1] / "results" / "data_filter_report.json").write_text(
        json.dumps({"tinystories": ts_stats, "fineweb": fw_stats}, indent=2))


if __name__ == "__main__":
    main()
