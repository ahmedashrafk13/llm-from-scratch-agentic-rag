"""Encode the cleaned corpus into flat uint16 token streams (train.bin / val.bin).

Documents are separated by <|endoftext|>; the trainer later packs the stream
into fixed-length sequences, so no compute is spent on padding.
FineWeb docs are mixed in proportionally across the stream (not appended at the end)
so every training phase sees both domains.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizer.bpe import BPETokenizer  # noqa: E402

_tok = None


def _init():
    global _tok
    _tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")


def _encode(text: str) -> list[int]:
    return _tok.encode(text, allow_special=False) + [_tok.token_id("<|endoftext|>")]


def read(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def encode_all(texts: list[str]) -> np.ndarray:
    with Pool(6, initializer=_init) as pool:
        chunks = pool.map(_encode, texts, chunksize=512)
    return np.fromiter((i for c in chunks for i in c), dtype=np.uint16)


def main():
    proc = ROOT / "data" / "processed"
    ts, fw = read(proc / "tinystories.jsonl"), read(proc / "fineweb.jsonl")
    fw_val, fw = fw[:500], fw[500:]
    # interleave: one fineweb doc every k tinystories docs
    k = max(1, len(ts) // max(len(fw), 1))
    mixed, j = [], 0
    for i, t in enumerate(ts):
        mixed.append(t)
        if i % k == k - 1 and j < len(fw):
            mixed.append(fw[j]); j += 1
    mixed.extend(fw[j:])

    train = encode_all(mixed)
    train.tofile(proc / "train.bin")
    for name, texts in {"val_tinystories": read(proc / "tinystories_val.jsonl"), "val_fineweb": fw_val}.items():
        encode_all(texts).tofile(proc / f"{name}.bin")
    print(f"train tokens: {len(train):,}")
    (proc / "meta.json").write_text(json.dumps({"train_tokens": int(len(train)), "docs": len(mixed)}))


if __name__ == "__main__":
    main()
