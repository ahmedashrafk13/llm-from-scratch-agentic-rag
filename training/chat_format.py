"""Chat template shared by SFT, DPO, evaluation and serving for the from-scratch model.

    <|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>

Loss is computed on response tokens only (prompt tokens are masked with -100).
"""
from __future__ import annotations

import json
from pathlib import Path


def prompt_text(prompt: str) -> str:
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def encode_example(tok, prompt: str, response: str, max_len: int):
    p = tok.encode(prompt_text(prompt))
    r = tok.encode(response, allow_special=False) + [tok.token_id("<|endoftext|>")]
    if len(p) + len(r) > max_len:  # keep the end of the prompt (question) and as much response as fits
        r = r[:max_len // 2]
        p = p[-(max_len - len(r)):]
    ids = p + r
    labels = [-100] * len(p) + r
    return ids, labels


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
