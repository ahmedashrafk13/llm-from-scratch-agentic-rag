"""Gradio demo: (1) the raw from-scratch LM, (2) the agentic RAG system with its trace.

    python serve/app_gradio.py            # GENERATOR=qwen-qlora to use the 1.5B model for answers
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.api import agent, generator  # noqa: E402


def complete(prompt, model_name, temperature, max_tokens):
    g = generator(model_name)
    if model_name == "scratch-base":
        ids = torch.tensor([g.tok.encode(prompt, allow_special=False)], device=g.device)
        toks, text = [], prompt
        for t in g.model.generate(ids, int(max_tokens), temperature=temperature,
                                  eos_id=g.tok.token_id("<|endoftext|>"), stream=True):
            toks.append(t)
            yield prompt + g.tok.decode(toks)
        return
    out = ""
    for piece in g.stream(prompt, int(max_tokens), temperature):
        out += piece
        yield out


def ask(question):
    r = agent().run(question)
    ctx = "\n\n---\n\n".join(r.contexts) or "(none)"
    return r.answer, "\n".join(f"{i + 1}. {t}" for i, t in enumerate(r.trace)), ctx


with gr.Blocks(title="LLM from scratch + Agentic RAG") as demo:
    gr.Markdown("# LLM from scratch + Agentic RAG\nA 37M-parameter Transformer trained from scratch, "
                "aligned with SFT + DPO, and used inside a retrieval agent.")
    with gr.Tab("Language model"):
        p = gr.Textbox(label="Prompt", value="Once upon a time, a little fox")
        m = gr.Radio(["scratch-base", "scratch-sft", "scratch-dpo"], value="scratch-base", label="Checkpoint")
        t = gr.Slider(0, 1.5, 0.8, label="Temperature")
        n = gr.Slider(16, 300, 150, step=1, label="Max new tokens")
        o = gr.Textbox(label="Output", lines=10)
        gr.Button("Generate").click(complete, [p, m, t, n], o)
    with gr.Tab("Agentic RAG"):
        gr.Markdown(f"Answer model: `{os.getenv('GENERATOR', 'scratch-dpo')}` - corpus: rag-mini-wikipedia")
        q = gr.Textbox(label="Question", value="Who was Abraham Lincoln's wife?")
        a = gr.Textbox(label="Answer")
        tr = gr.Textbox(label="Agent trace", lines=5)
        c = gr.Textbox(label="Retrieved context", lines=10)
        gr.Button("Ask").click(ask, q, [a, tr, c])

if __name__ == "__main__":
    demo.launch()
