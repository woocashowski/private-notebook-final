"""
Mock Ollama — drop-in replacement for localhost:11434 during testing.
Returns deterministic fake embeddings and streamed fake chat tokens.
Just enough to prove the pipeline is wired correctly.
"""
import hashlib
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


def fake_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic pseudo-embedding — just hashes the text into a vector."""
    h = hashlib.sha256(text.encode()).digest()
    # Repeat hash to fill dim floats in [-1, 1]
    vec = []
    i = 0
    while len(vec) < dim:
        vec.append((h[i % len(h)] - 128) / 128.0)
        i += 1
    # Also mix in word-level signal so similar texts end up closer.
    for word in text.lower().split()[:50]:
        wh = hashlib.md5(word.encode()).digest()
        idx = wh[0] % dim
        vec[idx] += 0.3
    return vec


@app.get("/api/tags")
def tags():
    return {
        "models": [
            {"name": "llama3.2:3b"},
            {"name": "nomic-embed-text"},
        ]
    }


@app.post("/api/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    return {"embedding": fake_embed(body.get("prompt", ""))}


@app.post("/api/embed")
async def embed_batch(req: Request):
    body = await req.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {"embeddings": [fake_embed(t) for t in inputs]}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    # Tiny canned response that pretends to have read the context.
    reply = (
        "Based on the provided excerpts, here is a brief answer. "
        "The documents describe the topic of your question. See [Source 1] "
        "for the most relevant passage, and [Source 2] for additional context. "
        "(this is a mock response — swap in real Ollama to get a real answer)"
    )
    words = reply.split(" ")

    async def stream():
        for i, w in enumerate(words):
            tok = w + (" " if i < len(words) - 1 else "")
            yield json.dumps({"message": {"content": tok}, "done": False}) + "\n"
            await asyncio.sleep(0.02)
        yield json.dumps({"message": {"content": ""}, "done": True}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
