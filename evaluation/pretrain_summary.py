"""Rebuild results/pretrain_summary.json from the base checkpoint + training log.

Also a tokenizer/data integrity check: the trainer evaluated on fixed validation windows
(seed 1234), so re-running that exact 20-iteration evaluation must reproduce the val loss
logged at the final step. A different tokenizer or validation stream would not.

    python evaluation/pretrain_summary.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Transformer  # noqa: E402
from model.attention import attention_backend  # noqa: E402
from training.pretrain import DATA, evaluate  # noqa: E402


def main(batch: int = 32, seq: int = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer.load(ROOT / "checkpoints" / "base", device)
    vals = {n: np.memmap(DATA / f"val_{n}.bin", dtype=np.uint16, mode="r") for n in ("tinystories", "fineweb")}
    log = pd.read_csv(ROOT / "results" / "pretrain_log.csv")
    last = log.dropna(subset=["val_tinystories"]).iloc[-1]

    check = evaluate(model, vals, batch, seq, 20, device)
    integrity = {k: {"logged": round(float(last[f"val_{k}"]), 4), "recomputed": round(v, 4),
                     "abs_diff": round(abs(v - float(last[f"val_{k}"])), 4)} for k, v in check.items()}
    print("integrity check (step", int(last.step), "):", json.dumps(integrity))

    final = evaluate(model, vals, batch, seq, 100, device)
    steps = int(log.step.max())
    summary = {"params_M": round(model.num_params() / 1e6, 2), "steps": steps,
               "tokens": int(log.tokens.max()),
               **{f"val_loss_{k}": round(v, 4) for k, v in final.items()},
               **{f"val_ppl_{k}": round(math.exp(v), 2) for k, v in final.items()},
               "train_tok_per_s_T4": int(log.tok_per_s.median()),
               "attention_kernel_eval_gpu": attention_backend()}
    (ROOT / "results" / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "results" / "tokenizer_integrity.json").write_text(json.dumps(integrity, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
