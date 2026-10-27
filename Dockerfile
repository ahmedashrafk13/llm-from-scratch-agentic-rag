# CPU image serving the from-scratch model + agentic RAG (FastAPI, SSE streaming).
# For the Qwen/QLoRA generator use a CUDA base image and set GENERATOR=qwen-qlora.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 GENERATOR=scratch-dpo
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY model/ model/
COPY tokenizer/ tokenizer/
COPY training/__init__.py training/chat_format.py training/
COPY rag/ rag/
COPY serve/ serve/
COPY data/processed/rag_corpus.jsonl data/processed/
# checkpoints are not in git; mount or COPY them at build time
COPY checkpoints/dpo/ checkpoints/dpo/
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
