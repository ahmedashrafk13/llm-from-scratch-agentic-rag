import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import exact_match, reciprocal_rank, token_f1  # noqa: E402
from model import ModelConfig, Transformer  # noqa: E402
from model.rope import apply_rope, precompute_rope  # noqa: E402
from tokenizer.bpe import BPETokenizer  # noqa: E402

TEXTS = ["the cat sat on the mat. the cat ran.", "Once upon a time there was a cat 123,456!", "naïve café 東京"]


def test_bpe_roundtrip_and_specials():
    tok = BPETokenizer.train(TEXTS * 20, vocab_size=300)
    for t in TEXTS + ["unseen wörds ✓ \n\n tabs\t"]:
        assert tok.decode(tok.encode(t)) == t
    ids = tok.encode("<|user|>\nhi<|endoftext|>")
    assert ids[0] == tok.token_id("<|user|>") and ids[-1] == tok.token_id("<|endoftext|>")
    assert len(tok.encode("the cat")) < len("the cat".encode())  # merges compress


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer.train(TEXTS * 20, vocab_size=300)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXTS[1]) == tok.encode(TEXTS[1])


def test_rope_relative():
    cos, sin = precompute_rope(16, 64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rope(q.expand(1, 1, 64, 16).clone(), cos, sin)[0, 0, m]
        kn = apply_rope(k.expand(1, 1, 64, 16).clone(), cos, sin)[0, 0, n]
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(40, 37), atol=1e-4)


def tiny():
    return Transformer(ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                                   ffn_hidden=96, max_seq_len=32))


def test_causality():
    m = tiny().eval()
    x = torch.randint(0, 100, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(0, 100, (6,))
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = tiny().eval()
    x = torch.randint(0, 100, (1, 12))
    full, _ = m(x)
    caches = [{} for _ in m.layers]
    out, _ = m(x[:, :8], kv_caches=caches, start_pos=0)
    steps = [out[:, -1]]
    for p in range(8, 12):
        o, _ = m(x[:, p:p + 1], kv_caches=caches, start_pos=p)
        steps.append(o[:, -1])
    assert torch.allclose(full[0, 7:12], torch.cat(steps), atol=1e-4)  # positions 7..11


def test_tied_embeddings_and_overfit():
    torch.manual_seed(0)
    m = tiny()
    assert m.lm_head.weight is m.tok_emb.weight
    x = torch.randint(0, 100, (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(100):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.5


def test_metrics():
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1
    assert 0 < token_f1("Abraham Lincoln was", "Abraham Lincoln") < 1
    assert reciprocal_rank([False, False, True]) == 1 / 3
