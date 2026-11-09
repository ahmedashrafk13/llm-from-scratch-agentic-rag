"""Post-training metrics measured directly from the saved checkpoints.

  * SFT validation loss (response tokens only) for base / SFT / DPO on sft_val.jsonl
  * DPO preference metrics on the held-out preference pairs, policy vs the frozen SFT
    reference: loss, reward accuracy (implicit reward of chosen > rejected) and mean margin.
    The SFT row is the epoch-0 baseline (policy == reference, so margin 0 and loss log 2).

Preference pairs are rebuilt with the same seeded procedure as training/dpo.py
(SFT greedy answers as on-policy rejections) and written to data/processed/dpo_pairs.jsonl,
which training/reward_model.py also reads.

    python evaluation/post_training_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Transformer  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from training.chat_format import encode_example, read_jsonl  # noqa: E402
from training.dpo import PAIRS, build_pairs, pair_batch, seq_logps  # noqa: E402
from training.sft import eval_loss  # noqa: E402

CK = ROOT / "checkpoints"


@torch.no_grad()
def dpo_metrics(policy, ref, tok, pairs, beta: float, device: str, bs: int = 8) -> dict:
    L = policy.cfg.max_seq_len
    losses, accs, margins = [], [], []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        x, y = pair_batch(tok, batch, L, device)
        n = len(batch)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            r = seq_logps(ref, x, y)
            p = seq_logps(policy, x, y)
        m = beta * ((p[:n] - r[:n]) - (p[n:] - r[n:]))
        losses.append(-F.logsigmoid(m).mean().item())
        accs.append(((m > 0).float() + 0.5 * (m == 0).float()).mean().item())  # ties count half
        margins.append(m.mean().item())
    k = len(losses)
    return {"dpo_val_loss": round(sum(losses) / k, 4), "reward_acc": round(sum(accs) / k, 4),
            "margin": round(sum(margins) / k, 4), "n_pairs": len(pairs)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    pad = tok.token_id("<|pad|>")
    val_rows = read_jsonl(ROOT / "data" / "processed" / "sft_val.jsonl")
    out: dict = {"sft_val_loss": {}, "dpo_heldout_pairs": {}}

    for ck in ("base", "sft", "dpo"):
        m = Transformer.load(CK / ck, device).eval()
        val = [encode_example(tok, r["prompt"], r["response"], m.cfg.max_seq_len) for r in val_rows]
        out["sft_val_loss"][ck] = round(eval_loss(m, val, pad, device), 4)
        print(ck, "sft val loss", out["sft_val_loss"][ck], flush=True)
        del m

    # same seeding order as training/dpo.py main()
    random.seed(0)
    torch.manual_seed(0)
    sft = Transformer.load(CK / "sft", device).eval()
    pairs = build_pairs(sft, tok, device) if not PAIRS.exists() else read_jsonl(PAIRS)
    random.shuffle(pairs)
    val = pairs[:150]
    out["n_pairs_total"] = len(pairs)
    if all("on_policy" in p for p in pairs):  # flag written by training/dpo.py build_pairs
        out["on_policy_rejected"] = sum(p["on_policy"] for p in pairs)
    beta = json.loads((CK / "dpo" / "config.json").read_text()).get("dpo_beta", 0.1)
    out["dpo_heldout_pairs"]["sft (reference, epoch 0)"] = dpo_metrics(sft, sft, tok, val, beta, device)
    dpo = Transformer.load(CK / "dpo", device).eval()
    out["dpo_heldout_pairs"]["dpo (final)"] = dpo_metrics(dpo, sft, tok, val, beta, device)
    print(json.dumps(out["dpo_heldout_pairs"], indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "post_training_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
