"""
End-to-end integration test.
Uses httpx ASGI transport — no network ports, no background processes.
Monkey-patches rag.embed + rag.chat_stream so no real Ollama is needed.
Exercises the FULL server stack: upload → chunk → embed → retrieve → stream.
"""
import asyncio
import hashlib
import io
import json
from pathlib import Path

import httpx

# --- Nuke any old index before importing rag -------------------------------
import shutil
shutil.rmtree("chroma_db", ignore_errors=True)

import rag
import server


# --- Monkey-patch Ollama calls ---------------------------------------------
def fake_vec(text: str, dim: int = 384) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec = [(h[i % len(h)] - 128) / 128.0 for i in range(dim)]
    for word in text.lower().split()[:50]:
        wh = hashlib.md5(word.encode()).digest()
        vec[wh[0] % dim] += 0.3
    return vec


async def fake_embed(texts):
    return [fake_vec(t) for t in texts]


async def fake_chat_stream(messages, model=None):
    reply = (
        "Based on the excerpts, here is the answer. The documents cover the "
        "topic asked about. See [Source 1] for the most relevant passage. "
        "(this is a mock — swap real Ollama for real answers)"
    )
    for word in reply.split():
        yield word + " "
        await asyncio.sleep(0.005)


async def fake_check():
    return {
        "ok": True,
        "llm_present": True,
        "embed_present": True,
        "llm_model": "mock-llm",
        "embed_model": "mock-embed",
        "available": ["mock-llm", "mock-embed"],
    }


rag.embed = fake_embed
rag.chat_stream = fake_chat_stream
rag.check_ollama = fake_check


# --- Sample doc -------------------------------------------------------------
SAMPLE = """Offline RAG Architecture Guide

This document describes a retrieval-augmented generation system that runs
entirely on local hardware. No data is ever sent to external services.

Chapter 1: Ingestion Pipeline

When a user uploads a document, the system extracts raw text, splits it into
overlapping chunks of approximately 900 characters, and computes a dense
vector embedding for each chunk. The embeddings are stored in ChromaDB.

Chapter 2: Retrieval

At query time, the user's question is embedded with the same model. The top
five most similar chunks are retrieved by cosine distance and injected into
the prompt as context for the language model.

Chapter 3: Privacy Guarantees

Because both the embedding model and the generation model run locally via
Ollama, no plaintext from user documents ever crosses a network boundary.
The vector database also lives on local disk under chroma_db/.
"""


# --- Test run ---------------------------------------------------------------
async def main():
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        print("━" * 60)
        print("1. GET /api/health")
        print("━" * 60)
        r = await client.get("/api/health")
        print(json.dumps(r.json(), indent=2))

        print("\n" + "━" * 60)
        print("2. POST /api/notebooks  (create a notebook)")
        print("━" * 60)
        nb = (await client.post("/api/notebooks", json={"name": "Test Notebook"})).json()
        nb_id = nb["id"]
        print(json.dumps(nb, indent=2))

        print("\n" + "━" * 60)
        print("3. GET /api/documents  (empty state)")
        print("━" * 60)
        r = await client.get(f"/api/documents?notebook_id={nb_id}")
        print(r.json())

        print("\n" + "━" * 60)
        print("4. POST /api/upload  (ingest sample.txt)")
        print("━" * 60)
        files = {"file": ("sample.txt", io.BytesIO(SAMPLE.encode()), "text/plain")}
        r = await client.post("/api/upload", files=files, data={"notebook_id": nb_id})
        print(json.dumps(r.json(), indent=2))

        print("\n" + "━" * 60)
        print("5. GET /api/documents  (after upload)")
        print("━" * 60)
        r = await client.get(f"/api/documents?notebook_id={nb_id}")
        print(json.dumps(r.json(), indent=2))

        print("\n" + "━" * 60)
        print("6. POST /api/conversations + /api/chat  (streaming SSE)")
        print("━" * 60)
        conv = (await client.post(
            "/api/conversations", json={"notebook_id": nb_id}
        )).json()
        print("Query: 'How does the retrieval step work?'\n")

        async with client.stream(
            "POST", "/api/chat",
            json={
                "query": "How does the retrieval step work?",
                "k": 3,
                "conversation_id": conv["id"],
            },
        ) as r:
            buffer = ""
            answer = ""
            sources_printed = False
            async for chunk in r.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    lines = event.split("\n")
                    ev = "message"; data = ""
                    for ln in lines:
                        if ln.startswith("event:"): ev = ln[6:].strip()
                        elif ln.startswith("data:"): data += ln[5:].strip()
                    if not data: continue
                    if ev == "sources":
                        srcs = json.loads(data)
                        print(f"↪ retrieved {len(srcs)} source chunk(s):")
                        for s in srcs:
                            print(f"   [{s['n']}] {s['filename']} · chunk {s['chunk_index']} · d={s['distance']:.3f}")
                            print(f"       {s['preview'][:90]}...")
                        print("\n↪ streaming answer:")
                        sources_printed = True
                    elif ev == "token":
                        tok = json.loads(data)
                        answer += tok
                        print(tok, end="", flush=True)
                    elif ev == "done":
                        print("\n\n[stream complete]")
                    elif ev == "error":
                        print(f"\n[error] {data}")

        print("\n" + "━" * 60)
        print("7. DELETE /api/documents/{id}")
        print("━" * 60)
        docs = (await client.get(f"/api/documents?notebook_id={nb_id}")).json()
        if docs:
            r = await client.delete(f"/api/documents/{docs[0]['doc_id']}")
            print(r.json())
        r = await client.get(f"/api/documents?notebook_id={nb_id}")
        print(f"docs after delete: {r.json()}")

        print("\n✓ all endpoints exercised successfully")


asyncio.run(main())
