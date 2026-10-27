"""FastAPI server with token streaming (Server-Sent Events).

    uvicorn serve.api:app --port 8000
    curl -N -X POST localhost:8000/generate -H "content-type: application/json" \
         -d '{"prompt": "Once upon a time", "model": "scratch-base"}'
    curl -X POST localhost:8000/ask -H "content-type: application/json" \
         -d '{"question": "Who was the 16th president of the United States?"}'

Environment: GENERATOR=qwen-qlora | scratch-dpo (default scratch-dpo, which runs on CPU).
vLLM is the production choice for the Qwen model on Linux (PagedAttention, continuous
batching); it does not run on native Windows, so this server uses HF generate + a
streamer thread. The endpoint contract is the same either way.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

app = FastAPI(title="LLM-from-scratch + Agentic RAG")


@lru_cache(maxsize=4)
def generator(name: str):
    from rag.generators import QwenGenerator, ScratchGenerator
    if name.startswith("scratch-"):
        return ScratchGenerator(name.removeprefix("scratch-"))
    return QwenGenerator(adapter="qlora" if name == "qwen-qlora" else None)


@lru_cache(maxsize=1)
def agent():
    from rag.agent import RAGAgent
    from rag.retriever import Retriever, chunk_corpus, load_passages
    return RAGAgent(Retriever(chunk_corpus(load_passages(), 128, 32)), generator(os.getenv("GENERATOR", "scratch-dpo")))


class GenRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    model: str = "scratch-dpo"
    max_new_tokens: int = Field(128, le=400)
    temperature: float = Field(0.7, ge=0, le=2)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(req: GenRequest):
    g = generator(req.model)
    if req.model == "scratch-base":  # raw LM: continue the text, no chat template
        import torch
        ids = torch.tensor([g.tok.encode(req.prompt, allow_special=False)], device=g.device)
        toks = g.model.generate(ids, req.max_new_tokens, temperature=req.temperature,
                                eos_id=g.tok.token_id("<|endoftext|>"), stream=True)
        pieces = (g.tok.decode([t]) for t in toks)
    else:
        pieces = g.stream(req.prompt, req.max_new_tokens, req.temperature)

    def sse():
        for p in pieces:
            yield f"data: {json.dumps({'token': p})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/ask")
def ask(req: AskRequest):
    r = agent().run(req.question)
    return {"answer": r.answer, "trace": r.trace, "contexts": r.contexts, "abstained": r.abstained}
