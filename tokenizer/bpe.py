"""Byte-level BPE tokenizer, written from scratch.

Design (same family as GPT-2 / Llama-3 tokenizers):
  1. Pre-tokenize text with a regex so merges never cross word / number /
     punctuation boundaries.
  2. Each pre-token is a sequence of raw UTF-8 bytes (base vocab = 256), so
     there is no <unk>: any string round-trips exactly.
  3. Training greedily merges the most frequent adjacent pair. Pair counts are
     kept incrementally over a {word: frequency} table, so each merge only
     touches words that contain that pair instead of rescanning the corpus.
  4. Special tokens are appended after the merges and are never split.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

# GPT-4-style split pattern (contractions, letters, 1-3 digit groups, punctuation, whitespace).
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None, special_tokens: list[str] | None = None):
        self.pattern = re.compile(SPLIT_PATTERN)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair in merges or []:
            new_id = 256 + len(self.merges)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.special_tokens: dict[str, int] = {}
        for tok in special_tokens or SPECIAL_TOKENS:
            idx = len(self.vocab)
            self.special_tokens[tok] = idx
            self.vocab[idx] = tok.encode("utf-8")
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, texts, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges > 0
        pat = re.compile(SPLIT_PATTERN)
        word_freq: Counter[str] = Counter()
        for t in texts:
            word_freq.update(pat.findall(t))

        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> word indices containing it
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(wi)

        merges: list[tuple[int, int]] = []
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))  # deterministic tie-break
            if pair_counts[best] < 2:
                break
            new_id = 256 + m
            merges.append(best)
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):  # remove this word's old pairs
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                w = _merge(w, best, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(best, None)
            if verbose and (m + 1) % 500 == 0:
                print(f"merge {m + 1}/{n_merges}")
        return cls(merges)

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge present (= lowest new id)
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        parts = self._special_re.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue
            for chunk in self.pattern.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- misc
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_id(self, special: str) -> int:
        return self.special_tokens[special]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "merges": [list(p) for p in self.merges],
            "special_tokens": list(self.special_tokens),
        }))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls([tuple(p) for p in d["merges"]], d["special_tokens"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
