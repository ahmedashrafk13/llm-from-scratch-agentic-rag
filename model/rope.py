"""Rotary Position Embeddings (Su et al., 2021).

Each pair of channels (x[2i], x[2i+1]) in a query/key head is rotated by angle
pos * theta^(-2i/d). The dot product q_m . k_n then depends only on (m - n), so
attention sees relative position with no learned position parameters.
"""
import torch


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10_000.0, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)           # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). cos/sin: (T, D/2). Rotates interleaved channel pairs."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)
