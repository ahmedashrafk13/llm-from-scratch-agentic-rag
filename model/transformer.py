"""Decoder-only Transformer (pre-norm, Llama-style)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import GroupedQueryAttention
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU
from .rope import precompute_rope


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # one matrix for input and output
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style: scale residual-output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0, loss_mask=None):
        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = layer(x, self.rope_cos, self.rope_sin, cache, start_pos)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100, reduction="none")
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            return logits, (loss * m).sum() / m.sum().clamp(min=1)
        valid = (targets.reshape(-1) != -100).float()
        return logits, (loss * valid).sum() / valid.sum().clamp(min=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = 0.95, eos_id: int | None = None, stream: bool = False):
        """KV-cached sampling. If stream=True yields token ids one at a time."""
        gen = self._generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id)
        if stream:
            return gen
        out = list(gen)
        return torch.cat([idx, torch.tensor([out], device=idx.device, dtype=idx.dtype)], 1) if out else idx

    def _generate(self, idx, max_new_tokens, temperature, top_k, top_p, eos_id):
        self.eval()
        idx = idx[:, -(self.cfg.max_seq_len - max_new_tokens):]
        caches = [{} for _ in self.layers]
        logits, _ = self(idx, kv_caches=caches, start_pos=0)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            if pos >= self.cfg.max_seq_len:
                break
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, -1)
                if top_p:
                    sp, si = probs.sort(descending=True)
                    cut = sp.cumsum(-1) - sp > top_p
                    sp[cut] = 0
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            tok = nxt.item()
            if eos_id is not None and tok == eos_id:
                break
            yield tok
            logits, _ = self(nxt, kv_caches=caches, start_pos=pos)
            pos += 1

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "config.json").write_text(json.dumps({**self.cfg.to_dict(), **(extra or {})}, indent=2))

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "Transformer":
        path = Path(path)
        d = json.loads((path / "config.json").read_text())
        cfg = ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})
        m = cls(cfg)
        m.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
        return m.to(device)
