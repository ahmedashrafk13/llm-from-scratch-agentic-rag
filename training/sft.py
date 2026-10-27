"""Supervised fine-tuning of the pre-trained from-scratch model.

Prompt tokens are masked so the model is only trained to produce responses.
Sequences are length-sorted into buckets to cut padding waste.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Transformer  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from training.chat_format import encode_example, read_jsonl  # noqa: E402
from training.pretrain import lr_at, make_optimizer  # noqa: E402


def collate(batch, pad_id: int, device: str):
    T = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(ids) - 1] = torch.tensor(labels[1:])  # shift: predict next token
    return x.to(device), y.to(device)


def batches(examples, bs: int, shuffle: bool):
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        yield [examples[i] for i in g]


@torch.no_grad()
def eval_loss(model, examples, pad_id, device, bs=32) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in batches(examples, bs, False):
        x, y = collate(b, pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        k = (y != -100).sum().item()
        tot, n = tot + loss.item() * k, n + k
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "base"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "sft"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    model = Transformer.load(args.init, device)
    L = model.cfg.max_seq_len
    proc = ROOT / "data" / "processed"
    train = [encode_example(tok, r["prompt"], r["response"], L) for r in read_jsonl(proc / "sft_train.jsonl")]
    val = [encode_example(tok, r["prompt"], r["response"], L) for r in read_jsonl(proc / "sft_val.jsonl")]
    pad = tok.token_id("<|pad|>")

    opt = make_optimizer(model, args.lr, 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    steps_per_epoch = math.ceil(len(train) / args.bs)
    total = steps_per_epoch * args.epochs
    history = [{"epoch": 0, "val_loss": eval_loss(model, val, pad, device)}]
    print(history[-1])
    best, step = float("inf"), 0
    model.train()
    for ep in range(1, args.epochs + 1):
        for b in batches(train, args.bs, True):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, 50, args.lr, args.lr * 0.1)
            x, y = collate(b, pad, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
        vl = eval_loss(model, val, pad, device)
        history.append({"epoch": ep, "val_loss": vl})
        print(history[-1], flush=True)
        if vl < best:
            best = vl
            model.save(args.out, extra={"sft_epoch": ep})
    (ROOT / "results" / "sft_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
