"""QLoRA fine-tune of Qwen2.5-1.5B on the same SFT data as the from-scratch model.

* Base weights frozen and quantised to 4-bit NF4 with double quantisation
  (bitsandbytes); compute in fp16.
* LoRA adapters (r=16, alpha=32) on all attention and MLP projections (PEFT).
* Gradient checkpointing + paged 8-bit AdamW keep the run inside 4 GB of VRAM.
* Same prompt-masked loss as training/sft.py; the chat template is Qwen's own.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.chat_format import read_jsonl  # noqa: E402

BASE = "Qwen/Qwen2.5-1.5B"


def load_4bit(base: str = BASE):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16)
    return model, tok


def qwen_prompt(tok, prompt: str) -> str:
    # Qwen base ships the same ChatML template as the instruct model.
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def encode(tok, prompt, response, max_len):
    p = tok(qwen_prompt(tok, prompt), add_special_tokens=False)["input_ids"]
    r = tok(response + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + r)[-max_len:]
    labels = ([-100] * len(p) + r)[-max_len:]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "qlora"))
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--n-general", type=int, default=1700)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb

    model, tok = load_4bit()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable, total = model.get_nb_trainable_parameters()

    rows = read_jsonl(ROOT / "data" / "processed" / "sft_train.jsonl")
    domain = [r for r in rows if r["source"].startswith("domain")]
    general = [r for r in rows if r["source"] == "dolly"][:args.n_general]
    data = [encode(tok, r["prompt"], r["response"], args.max_len) for r in domain + general]
    random.shuffle(data)
    val = [encode(tok, r["prompt"], r["response"], args.max_len)
           for r in read_jsonl(ROOT / "data" / "processed" / "sft_val.jsonl")[:150]]

    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(data) / args.accum)

    def loss_on(ids, labels):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        return model(input_ids=x, labels=y).loss

    @torch.no_grad()
    def val_loss():
        model.eval()
        l = sum(loss_on(*v).item() for v in val) / len(val)
        model.train()
        return l

    hist = [{"step": 0, "val_loss": val_loss()}]
    print(f"trainable {trainable / 1e6:.1f}M / {total / 1e6:.0f}M | {hist[-1]}", flush=True)
    model.train()
    t0, step = time.time(), 0
    for i, (ids, labels) in enumerate(data):
        loss = loss_on(ids, labels) / args.accum
        loss.backward()
        if (i + 1) % args.accum == 0 or i + 1 == len(data):
            lr = args.lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{total_steps} loss {loss.item() * args.accum:.3f} "
                      f"{(time.time() - t0) / step:.1f}s/step", flush=True)
            if step % 100 == 0 or step == total_steps:
                hist.append({"step": step, "val_loss": val_loss()})
                print(hist[-1], flush=True)
    model.save_pretrained(args.out)
    summary = {"base": BASE, "trainable_params_M": round(trainable / 1e6, 2), "total_params_M": round(total / 1e6),
               "examples": len(data), "steps": total_steps, "minutes": round((time.time() - t0) / 60, 1),
               "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "history": hist}
    (ROOT / "results" / "qlora_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
