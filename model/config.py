from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8          # query heads
    n_kv_heads: int = 2       # key/value heads (GQA: 4 query heads share each KV head)
    ffn_hidden: int = 1376    # SwiGLU hidden; ~(8/3)*d_model rounded to a multiple of 32
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)
