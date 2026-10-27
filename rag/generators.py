"""Uniform `generate(prompt) -> str` / `stream(prompt)` wrappers for every model in the project."""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ScratchGenerator:
    """The from-scratch model (base / sft / dpo checkpoint)."""

    def __init__(self, ckpt: str = "dpo", device: str | None = None):
        from model import Transformer
        from tokenizer import BPETokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
        self.model = Transformer.load(ROOT / "checkpoints" / ckpt, self.device).eval()
        self.name = f"scratch-{ckpt}"
        self.max_prompt_tokens = self.model.cfg.max_seq_len - 64

    def _ids(self, prompt: str):
        from training.chat_format import prompt_text
        ids = self.tok.encode(prompt_text(prompt))[-self.max_prompt_tokens:]
        return torch.tensor([ids], device=self.device)

    def stream(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.0):
        eos = self.tok.token_id("<|endoftext|>")
        buf = []
        for t in self.model.generate(self._ids(prompt), max_new_tokens, temperature=temperature,
                                     eos_id=eos, stream=True):
            buf.append(t)
            text = self.tok.decode(buf)
            if not text.endswith("�"):  # wait for complete UTF-8 sequences
                yield text
                buf = []

    def generate(self, prompt: str, max_new_tokens: int = 48, temperature: float = 0.0) -> str:
        return "".join(self.stream(prompt, max_new_tokens, temperature)).strip()


class QwenGenerator:
    """Qwen2.5-1.5B in 4-bit, optionally with the QLoRA adapter attached."""

    def __init__(self, adapter: str | None = "qlora"):
        from training.qlora import load_4bit
        self.model, self.tok = load_4bit()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(ROOT / "checkpoints" / adapter))
        self.model.eval()
        self.name = "qwen1.5b-qlora" if adapter else "qwen1.5b-base"

    def _inputs(self, prompt: str):
        from training.qlora import qwen_prompt
        return self.tok(qwen_prompt(self.tok, prompt), return_tensors="pt", add_special_tokens=False).to("cuda")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inp = self._inputs(prompt)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                  temperature=temperature or None, top_p=None, top_k=None,
                                  pad_token_id=self.tok.eos_token_id,
                                  eos_token_id=[self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id])
        return self.tok.decode(out[0, inp["input_ids"].size(1):], skip_special_tokens=True).strip()

    def stream(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0):
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**self._inputs(prompt), max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  streamer=streamer, pad_token_id=self.tok.eos_token_id,
                  eos_token_id=[self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id])
        Thread(target=self.model.generate, kwargs=kw, daemon=True).start()
        yield from streamer
