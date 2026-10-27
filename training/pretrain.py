"""Pre-train the from-scratch Transformer on packed token streams.

Features: AdamW (decoupled weight decay on matrices only), linear warmup + cosine
decay, gradient clipping, gradient accumulation, fp16 mixed precision with a
dynamic loss scaler (Turing GPUs have no bf16), optional gradient checkpointing,
and atomic checkpoint save / resume - locally and optionally to the Hugging Face Hub.

    python training/pretrain.py --max-steps 12000              # fresh run
    python training/pretrain.py --resume                       # continue from checkpoints/pretrain/last
    python training/pretrain.py --resume --hub-repo user/llm-scratch   # also sync checkpoints to the Hub
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import ModelConfig, Transformer  # noqa: E402
from model.attention import attention_backend  # noqa: E402

DATA = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "pretrain"


def get_batch(data: np.memmap, batch: int, seq: int, device: str, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - seq - 1, batch)
    x = torch.from_numpy(np.stack([data[i:i + seq].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + seq].astype(np.int64) for i in ix]))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step: int, max_steps: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, vals: dict[str, np.memmap], batch: int, seq: int, iters: int, device: str) -> dict:
    model.eval()
    rng = np.random.default_rng(1234)  # fixed windows -> comparable across steps
    out = {}
    for name, data in vals.items():
        losses = []
        for _ in range(iters):
            x, y = get_batch(data, batch, seq, device, rng)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def make_optimizer(model, lr: float, wd: float):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available())


def save_checkpoint(model, opt, scaler, step, args, hub_repo=None):
    tmp, final = CKPT / "last.tmp", CKPT / "last"
    model.save(tmp, extra={"step": step})
    torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "args": vars(args), "rng": np.random.get_state()}, tmp / "trainer_state.pt")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    if hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hub_repo, exist_ok=True, private=True)
        api.upload_folder(folder_path=str(final), repo_id=hub_repo, path_in_repo="last",
                          commit_message=f"step {step}")


def maybe_download_from_hub(hub_repo: str):
    from huggingface_hub import snapshot_download
    try:
        p = snapshot_download(hub_repo, allow_patterns=["last/*"])
    except Exception as e:  # first run: repo does not exist yet
        print(f"no Hub checkpoint to resume ({type(e).__name__}); starting fresh")
        return
    if (Path(p) / "last").exists():
        shutil.copytree(Path(p) / "last", CKPT / "last", dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=12_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    vals = {n: np.memmap(DATA / f"val_{n}.bin", dtype=np.uint16, mode="r") for n in ("tinystories", "fineweb")}

    if args.resume and args.hub_repo and not (CKPT / "last").exists():
        maybe_download_from_hub(args.hub_repo)

    if args.resume and (CKPT / "last").exists():
        model = Transformer.load(CKPT / "last", device)
        state = torch.load(CKPT / "last" / "trainer_state.pt", map_location=device, weights_only=False)
        start = state["step"]
        print(f"resumed from step {start}")
    else:
        from tokenizer import BPETokenizer
        vocab = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json").vocab_size
        cfg = ModelConfig(vocab_size=vocab,max_seq_len=args.seq, gradient_checkpointing=args.grad_ckpt)
        model = Transformer(cfg).to(device)
        state, start = None, 0
    model.cfg.gradient_checkpointing = args.grad_ckpt
    print(f"params: {model.num_params() / 1e6:.1f}M | attention kernel: {attention_backend()}")

    opt = make_optimizer(model, args.lr, args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    if state:
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
    fwd = torch.compile(model) if args.compile else model

    tokens_per_step = args.batch * args.accum * args.seq
    log_path = ROOT / "results" / "pretrain_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    new_log = not (args.resume and log_path.exists())
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["step", "tokens", "lr", "train_loss", "val_tinystories", "val_fineweb", "tok_per_s"])

    rng = np.random.default_rng(start)
    model.train()
    t0, running = time.time(), []
    for step in range(start, args.max_steps):
        lr = lr_at(step, args.max_steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, args.seq, device, rng)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = fwd(x, y)
            scaler.scale(loss / args.accum).backward()
            running.append(loss.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % 50 == 0:
            dt = time.time() - t0
            tps = 50 * tokens_per_step / dt
            print(f"step {step + 1} | loss {np.mean(running):.3f} | lr {lr:.2e} | {tps:,.0f} tok/s", flush=True)
            row_loss, t0 = float(np.mean(running)), time.time()
            running = []
            ev = evaluate(model, vals, args.batch, args.seq, 20, device) if (step + 1) % args.eval_every == 0 else {}
            if ev:
                print(f"  val {ev}", flush=True)
            log.writerow([step + 1, (step + 1) * tokens_per_step, lr, row_loss,
                          ev.get("tinystories", ""), ev.get("fineweb", ""), round(tps)])
            logf.flush()
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.max_steps:
            save_checkpoint(model, opt, scaler, step + 1, args, args.hub_repo)

    final = evaluate(model, vals, args.batch, args.seq, 100, device)
    model.save(ROOT / "checkpoints" / "base", extra={"step": args.max_steps})
    summary = {"params_M": round(model.num_params() / 1e6, 2), "steps": args.max_steps,
               "tokens": args.max_steps * tokens_per_step,
               **{f"val_loss_{k}": round(v, 4) for k, v in final.items()},
               **{f"val_ppl_{k}": round(math.exp(v), 2) for k, v in final.items()},
               "attention_kernel": attention_backend()}
    (ROOT / "results" / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
