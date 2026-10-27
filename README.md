# LLM From Scratch + Post-Training + Agentic RAG

A decoder-only language model implemented, pre-trained, aligned and served from scratch, then used inside a retrieval-augmented agent. A QLoRA fine-tune of Qwen2.5-1.5B on the same data is included for comparison.

> **Status: training in progress.** The code is complete. The result tables, loss curves and findings will be added from real runs. No numbers appear here until they have been measured.

## Why
Most portfolios call an LLM API or fine-tune a downloaded model. This repo covers the layers underneath: how a Transformer is built, trained and aligned, and how retrieval changes its behaviour.

## What is built
| Area | Location | What it covers |
|---|---|---|
| Tokenizer | `tokenizer/` | Byte-level BPE from scratch: regex pre-tokenization, incremental pair counting, special tokens, exact round-trip, and a comparison with GPT-2 |
| Model | `model/` | Token embeddings, **RoPE**, **Grouped-Query Attention** (8 query heads / 2 KV heads), a fused SDPA kernel (FlashAttention-2 on Ampere+, memory-efficient kernel on Turing), **SwiGLU**, **RMSNorm**, tied embeddings and a KV-cached sampler. About 37M parameters. |
| Data | `data/` | TinyStories plus deduplicated FineWeb. Quality filters (Gopher/C4-style), exact dedup and MinHash-LSH near-dedup. Packed uint16 token streams. |
| Pre-training | `training/pretrain.py` | AdamW, warmup + cosine schedule, gradient clipping and accumulation, fp16 AMP, gradient checkpointing, and checkpoint save/resume locally and to the **Hugging Face Hub** |
| SFT | `training/sft.py` | Prompt-masked instruction tuning on Dolly plus domain QA pairs built for this project (closed-book and open-book formats) |
| DPO | `training/dpo.py` | Direct Preference Optimization against a frozen SFT reference, with on-policy rejected answers |
| RM / PPO | `training/reward_model.py` | A Bradley-Terry reward model. PPO is documented as the comparison. |
| QLoRA | `training/qlora.py` | Qwen2.5-1.5B in 4-bit NF4 with double quantization, LoRA r=16 on all projections, gradient checkpointing and paged 8-bit AdamW |
| Agentic RAG | `rag/` | Chunking, BGE/E5 embeddings, FAISS, BM25, hybrid RRF, a BGE cross-encoder reranker, tools (calculator, document search, lookup) and **self-correcting re-retrieval** with abstention |
| Evaluation | `evaluation/` | Perplexity; MMLU / ARC / HellaSwag versus GPT-2; Recall@k and MRR; RAGAS-style faithfulness, answer relevancy and context precision; hallucination rate on a hand-built set |
| Serving | `serve/` | FastAPI with SSE token streaming, a Gradio demo and a Dockerfile |

## Experiments
- Base vs SFT vs SFT+DPO.
- QLoRA 1.5B vs the from-scratch 37M model on identical tasks.
- RAG only vs fine-tune only vs RAG + fine-tune.
- Ablations: chunk size and overlap, top-k, reranker on/off, retriever mode, embedder, and self-correction on/off.

## Run it
**On Colab (recommended):** open `colab_train.ipynb` on a T4 runtime and add `HF_TOKEN` as a Colab secret.

**Locally:**
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest tests -q
python data/prepare_pretrain.py && python tokenizer/train_tokenizer.py && python data/tokenize_corpus.py
python data/build_sft.py && python evaluation/build_hallucination_set.py
python training/pretrain.py --hub-repo <user>/llm-scratch-checkpoints   # add --resume after a disconnect
python training/sft.py && python training/dpo.py && python training/reward_model.py
python training/qlora.py
python evaluation/benchmarks.py && python evaluation/retrieval_eval.py && python evaluation/qa_eval.py
python evaluation/make_report.py
```
On Windows, import `datasets` before `torch`; the scripts already do this.

## Design notes
- **Evaluation is offline.** The `ragas` package needs an API judge, so its three metrics are recomputed with local judges (an NLI cross-encoder, BGE similarity and string-match relevance). `evaluation/rag_metrics.py` documents each substitution.
- **Agent control flow is deterministic.** A 37M model cannot emit reliable tool calls. A fixed policy also keeps every generator comparable on identical retrieval.
- **vLLM is not used locally.** vLLM does not run on native Windows, so local serving uses HF generate with a streamer. The API contract is identical.
