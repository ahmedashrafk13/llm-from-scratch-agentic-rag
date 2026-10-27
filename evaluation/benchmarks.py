"""Zero-shot multiple-choice benchmarks scored by log-likelihood (lm-eval-harness style).

For each question every option is appended to the context and scored by the sum
of its token log-probabilities. We report
  acc       - argmax of summed log-prob
  acc_norm  - argmax of log-prob / option length in bytes (removes the length bias)
on a fixed, seeded subset of each benchmark (n per task, default 500), so all
models see exactly the same items.

Models: from-scratch base / SFT / DPO, GPT-2 (124M) baseline, and Qwen2.5-1.5B
(4-bit, base and QLoRA) as an upper reference.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import datasets  # noqa: F401  (import before torch on Windows)
import torch
import torch.nn.functional as F
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHANCE = {"mmlu": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25}


def load_task(task: str, n: int, seed: int = 0) -> list[dict]:
    """Return [{"ctx": str, "options": [str], "label": int}]"""
    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        items = [{"ctx": f"Question: {r['question']}\nAnswer:", "options": [" " + c for c in r["choices"]],
                  "label": r["answer"]} for r in ds]
    elif task.startswith("arc"):
        name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", name, split="test")
        items = []
        for r in ds:
            if r["answerKey"] not in r["choices"]["label"]:
                continue
            items.append({"ctx": f"Question: {r['question']}\nAnswer:",
                          "options": [" " + t for t in r["choices"]["text"]],
                          "label": r["choices"]["label"].index(r["answerKey"])})
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        items = [{"ctx": r["ctx"], "options": [" " + e for e in r["endings"]], "label": int(r["label"])} for r in ds]
    else:
        raise ValueError(task)
    random.Random(seed).shuffle(items)
    return items[:n]


class Scorer:
    """Wraps a model + tokenizer into `logprob(context, continuation)`."""

    def __init__(self, kind: str, path: str | None = None):
        self.kind = kind
        if kind == "scratch":
            from model import Transformer
            from tokenizer import BPETokenizer
            self.tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
            self.model = Transformer.load(path, "cuda").eval()
            self.enc = lambda s: self.tok.encode(s, allow_special=False)
            self.max_len = self.model.cfg.max_seq_len
        elif kind == "gpt2":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
            self.enc = self.tok.encode
            self.max_len = 1024
        elif kind.startswith("qwen"):
            from training.qlora import load_4bit
            self.model, self.tok = load_4bit()
            if kind == "qwen-qlora":
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, str(ROOT / "checkpoints" / "qlora"))
            self.model.eval()
            self.enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
            self.max_len = 1024

    @torch.no_grad()
    def option_logprobs(self, ctx: str, options: list[str]) -> list[float]:
        c = self.enc(ctx)
        seqs, spans = [], []
        for o in options:
            oi = self.enc(o)
            ids = (c + oi)[-self.max_len:]
            seqs.append(ids)
            spans.append(len(oi))
        T = max(map(len, seqs))
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.model(x)[0] if self.kind == "scratch" else self.model(input_ids=x).logits
        lp = F.log_softmax(logits.float(), -1)
        out = []
        for i, (s, n) in enumerate(zip(seqs, spans)):
            L = len(s)
            tgt = torch.tensor(s[L - n:], device="cuda")
            out.append(lp[i, L - n - 1:L - 1].gather(-1, tgt[:, None]).sum().item())
        return out


def evaluate(scorer: Scorer, items: list[dict]) -> dict:
    acc = acc_norm = 0
    for it in items:
        lps = scorer.option_logprobs(it["ctx"], it["options"])
        acc += int(max(range(len(lps)), key=lambda i: lps[i]) == it["label"])
        norm = [lp / len(o.encode()) for lp, o in zip(lps, it["options"])]
        acc_norm += int(max(range(len(norm)), key=lambda i: norm[i]) == it["label"])
    return {"acc": round(acc / len(items), 4), "acc_norm": round(acc_norm / len(items), 4), "n": len(items)}


MODELS = {
    "scratch-base": ("scratch", ROOT / "checkpoints" / "base"),
    "scratch-sft": ("scratch", ROOT / "checkpoints" / "sft"),
    "scratch-dpo": ("scratch", ROOT / "checkpoints" / "dpo"),
    "gpt2-124M": ("gpt2", None),
    "qwen2.5-1.5B-4bit": ("qwen-base", None),
    "qwen2.5-1.5B-qlora": ("qwen-qlora", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tasks", nargs="*", default=list(CHANCE))
    args = ap.parse_args()
    out_path = ROOT / "results" / "benchmarks.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    tasks = {t: load_task(t, args.n) for t in args.tasks}
    for name in args.models:
        kind, path = MODELS[name]
        scorer = Scorer(kind, str(path) if path else None)
        results.setdefault(name, {})
        for t, items in tasks.items():
            results[name][t] = evaluate(scorer, items)
            print(name, t, results[name][t], flush=True)
            out_path.write_text(json.dumps(results, indent=2))
        del scorer
        torch.cuda.empty_cache()
    results["_chance"] = CHANCE
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
