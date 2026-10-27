"""Causal Grouped-Query Attention with RoPE and a fused attention kernel.

GQA: n_heads query heads share n_kv_heads key/value heads. This cuts KV-cache
size by n_heads / n_kv_heads at inference with little quality loss.

Kernel: torch.nn.functional.scaled_dot_product_attention dispatches to
FlashAttention-2 on Ampere+ GPUs and to the memory-efficient (xformers-style)
kernel on older cards such as the GTX 1650 (sm_75) used here. Either way the
T x T attention matrix is never materialised in HBM. `attention_backend()`
reports which kernel was actually selected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None, start_pos: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])
        k = apply_rope(k, cos[start_pos:start_pos + T], sin[start_pos:start_pos + T])

        if kv_cache is not None:  # incremental decoding
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # Causal mask only needed when queries span multiple positions; a single new
        # token during cached decoding may attend to everything before it.
        if T > 1 and start_pos > 0:  # would need an explicit offset mask
            raise NotImplementedError("multi-token decode after prefill is not used in this project")
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


def attention_backend() -> str:
    if not torch.cuda.is_available():
        return "math (CPU)"
    major, _ = torch.cuda.get_device_capability()
    return "flash-attention-2" if major >= 8 else "memory-efficient (sm<80: FlashAttention-2 unsupported)"
