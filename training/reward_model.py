"""Bradley-Terry reward model - the first half of the RLHF/PPO alternative to DPO.

A scalar head on the SFT model's final hidden state (last response token) is trained
so r(chosen) > r(rejected) on the same preference pairs DPO uses.

PPO itself is documented in the README rather than run: it needs the policy, a
frozen reference, this reward model and a value head resident at once plus an
online generation loop - roughly 4x DPO's memory and an order of magnitude more
wall-clock on a 4 GB GPU, which is exactly the trade-off DPO was designed to remove.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Transformer  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from training.chat_format import encode_example, read_jsonl  # noqa: E402
from training.dpo import PAIRS  # noqa: E402


class RewardModel(nn.Module):
    def __init__(self, backbone: Transformer):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.cfg.d_model, 1)

    def forward(self, x, last_idx):
        h = self.backbone.tok_emb(x)
        for layer in self.backbone.layers:
            h = layer(h, self.backbone.rope_cos, self.backbone.rope_sin)
        h = self.backbone.norm(h)
        return self.head(h[torch.arange(x.size(0)), last_idx].float()).squeeze(-1)


def main():
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    rm = RewardModel(Transformer.load(ROOT / "checkpoints" / "sft", device)).to(device)
    pairs = read_jsonl(PAIRS)
    random.shuffle(pairs)
    val, train = pairs[:150], pairs[150:]
    pad, L = tok.token_id("<|pad|>"), rm.backbone.cfg.max_seq_len

    def scores(batch):
        enc = [encode_example(tok, p["prompt"], p["chosen"], L)[0] for p in batch] + \
              [encode_example(tok, p["prompt"], p["rejected"], L)[0] for p in batch]
        T = max(map(len, enc))
        x = torch.full((len(enc), T), pad, dtype=torch.long)
        for i, e in enumerate(enc):
            x[i, :len(e)] = torch.tensor(e)
        last = torch.tensor([len(e) - 1 for e in enc])
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            r = rm(x.to(device), last.to(device))
        return r[:len(batch)], r[len(batch):]

    @torch.no_grad()
    def acc():
        rm.eval()
        c = 0
        for i in range(0, len(val), 16):
            rc, rr = scores(val[i:i + 16])
            c += (rc > rr).sum().item()
        rm.train()
        return c / len(val)

    opt = torch.optim.AdamW(rm.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    hist = [{"epoch": 0, "val_pairwise_acc": acc()}]
    for ep in range(1, 3):
        random.shuffle(train)
        for i in range(0, len(train), 8):
            rc, rr = scores(train[i:i + 8])
            loss = -F.logsigmoid(rc - rr).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        hist.append({"epoch": ep, "val_pairwise_acc": acc()})
        print(hist[-1], flush=True)
    (ROOT / "results" / "reward_model.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
