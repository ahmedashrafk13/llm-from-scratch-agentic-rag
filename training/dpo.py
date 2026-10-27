"""Direct Preference Optimization (Rafailov et al., 2023) on the SFT model.

Preference pairs (built here, on-policy where possible):
  chosen   = the reference answer from the SFT data
  rejected = the SFT model's own greedy answer when it is wrong (token-F1 < 0.5),
             otherwise the answer to a different, randomly chosen question
             (a fluent but incorrect answer - the classic hallucination shape).

Loss:  -log sigmoid( beta * [(log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))] )
The reference model is the frozen SFT checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import token_f1  # noqa: E402
from model import Transformer  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from training.chat_format import encode_example, prompt_text, read_jsonl  # noqa: E402
from training.pretrain import make_optimizer  # noqa: E402
from training.sft import collate  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "dpo_pairs.jsonl"


@torch.no_grad()
def greedy(model, tok, prompt: str, device, max_new=48) -> str:
    ids = torch.tensor([tok.encode(prompt_text(prompt))], device=device)
    out = model.generate(ids, max_new, temperature=0, eos_id=tok.token_id("<|endoftext|>"))
    return tok.decode(out[0, ids.size(1):].tolist()).strip()


def build_pairs(model, tok, device, n_general: int = 600) -> list[dict]:
    rows = read_jsonl(ROOT / "data" / "processed" / "sft_train.jsonl")
    domain = [r for r in rows if r["source"].startswith("domain")]
    general = [r for r in rows if r["source"] == "dolly"][:n_general]
    pool = [r["response"] for r in domain]
    pairs, on_policy = [], 0
    for r in domain + general:
        gen = greedy(model, tok, r["prompt"], device)
        if gen and token_f1(gen, r["response"]) < 0.5:
            rejected, on_policy = gen, on_policy + 1
        else:
            rejected = random.choice([a for a in pool if a != r["response"]])
        pairs.append({"prompt": r["prompt"], "chosen": r["response"], "rejected": rejected, "source": r["source"]})
    print(f"{len(pairs)} pairs, {on_policy} with on-policy rejected answers")
    with open(PAIRS, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return pairs


def seq_logps(model, x, y) -> torch.Tensor:
    logits, _ = model(x)
    lp = F.log_softmax(logits.float(), -1)
    mask = y != -100
    tok_lp = lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(-1)


def pair_batch(tok, pairs, L, device):
    pad = tok.token_id("<|pad|>")
    enc = [encode_example(tok, p["prompt"], p["chosen"], L) for p in pairs] + \
          [encode_example(tok, p["prompt"], p["rejected"], L) for p in pairs]
    return collate(enc, pad, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "checkpoints" / "sft"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "dpo"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rebuild-pairs", action="store_true")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
    policy = Transformer.load(args.init, device)
    ref = Transformer.load(args.init, device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pairs = build_pairs(policy, tok, device) if args.rebuild_pairs or not PAIRS.exists() else read_jsonl(PAIRS)
    random.shuffle(pairs)
    val, train = pairs[:150], pairs[150:]
    L = policy.cfg.max_seq_len

    def run(batch, train_mode: bool):
        x, y = pair_batch(tok, batch, L, device)
        n = len(batch)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            with torch.no_grad():
                r = seq_logps(ref, x, y)
            p = seq_logps(policy, x, y) if train_mode else seq_logps(policy, x, y).detach()
        margin = args.beta * ((p[:n] - r[:n]) - (p[n:] - r[n:]))
        return -F.logsigmoid(margin).mean(), (margin > 0).float().mean().item(), margin.mean().item()

    @torch.no_grad()
    def evaluate():
        policy.eval()
        out = [run(val[i:i + args.bs], False) for i in range(0, len(val), args.bs)]
        policy.train()
        return {"loss": sum(o[0].item() for o in out) / len(out),
                "reward_acc": sum(o[1] for o in out) / len(out),
                "margin": sum(o[2] for o in out) / len(out)}

    opt = make_optimizer(policy, args.lr, 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history = [{"epoch": 0, **evaluate()}]
    print(history[-1])
    policy.train()
    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        for i in range(0, len(train), args.bs):
            loss, _, _ = run(train[i:i + args.bs], True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        history.append({"epoch": ep, **evaluate()})
        print(history[-1], flush=True)
    policy.save(args.out, extra={"dpo_beta": args.beta})
    (ROOT / "results" / "dpo_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
