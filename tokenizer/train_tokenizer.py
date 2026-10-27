"""Train the BPE tokenizer and compare it with GPT-2's tokenizer.

Comparison metrics on held-out text from both domains:
  * bytes per token (compression; higher = fewer tokens per document)
  * exact round-trip (decode(encode(x)) == x)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizer.bpe import BPETokenizer  # noqa: E402


def read_jsonl(path: Path, n: int | None = None) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if n and len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--n-tinystories", type=int, default=40_000)
    ap.add_argument("--n-fineweb", type=int, default=8_000)
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    ts, fw = read_jsonl(proc / "tinystories.jsonl", 60_000), read_jsonl(proc / "fineweb.jsonl", 12_000)
    random.seed(0)
    train_texts = ts[:args.n_tinystories] + fw[:args.n_fineweb]
    heldout = {"tinystories": ts[-2000:], "fineweb": fw[-1000:]}

    t0 = time.time()
    tok = BPETokenizer.train(train_texts, args.vocab_size, verbose=True)
    train_s = time.time() - t0
    tok.save(ROOT / "tokenizer" / "tokenizer.json")
    print(f"trained {tok.vocab_size} tokens in {train_s:.0f}s")

    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    report = {"vocab_size": tok.vocab_size, "train_seconds": round(train_s, 1),
              "train_docs": len(train_texts), "heldout": {}}
    for name, texts in heldout.items():
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        ours = [tok.encode(t, allow_special=False) for t in texts]
        theirs = [gpt2.encode(t) for t in texts]
        report["heldout"][name] = {
            "ours_bytes_per_token": round(nbytes / sum(map(len, ours)), 3),
            "gpt2_bytes_per_token": round(nbytes / sum(map(len, theirs)), 3),
            "ours_roundtrip_exact": all(tok.decode(i) == t for i, t in zip(ours, texts)),
        }
    sample = "Once upon a time, Lily found a shiny red ball. The GDP grew 3.4% in 2023!"
    report["example"] = {"text": sample, "ours": [tok.decode([i]) for i in tok.encode(sample)],
                         "gpt2": gpt2.tokenize(sample)}
    report["longest_tokens"] = sorted((tok.vocab[i].decode("utf-8", "replace") for i in range(256, tok.vocab_size - 4)),
                                      key=len, reverse=True)[:15]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tokenizer_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["heldout"], indent=2))


if __name__ == "__main__":
    main()
