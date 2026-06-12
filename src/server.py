"""
FastAPI server. Endpoints:
  GET    /                           -> frontend (static/index.html)
  GET    /api/health                 -> Ollama + models status
  GET    /api/models                 -> available LLMs with `installed` flag
  POST   /api/pull                   -> stream pull progress for a model (SSE)

  GET    /api/notebooks              -> list notebooks
  POST   /api/notebooks              -> create a notebook
  PATCH  /api/notebooks/{id}         -> rename a notebook
  DELETE /api/notebooks/{id}         -> delete notebook + its docs + its chats

  GET    /api/documents?notebook_id= -> list docs (optionally in one notebook)
  POST   /api/upload                 -> upload & ingest a file (form: notebook_id)
  POST   /api/reset                  -> clear docs (body: {notebook_id?})
  DELETE /api/documents/{doc_id}

  GET    /api/conversations?notebook_id=
  POST   /api/conversations          -> create (body: {model?, notebook_id})
  GET    /api/conversations/{id}
  DELETE /api/conversations/{id}

  POST   /api/chat                   -> stream answer, persist turn (SSE)

Run:
  uvicorn src.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import rag
from .parsers import parse_file, PARSERS

app = FastAPI(title="Offline RAG")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Ensure at least one notebook exists; migrate orphan data on first start.
rag.ensure_default_notebook()


class ChatRequest(BaseModel):
    query: str
    k: int = 5
    model: Optional[str] = None
    conversation_id: Optional[str] = None


class PullRequest(BaseModel):
    model: str


class ConversationCreate(BaseModel):
    model: Optional[str] = None
    notebook_id: Optional[str] = None


class NotebookCreate(BaseModel):
    name: str


class NotebookUpdate(BaseModel):
    name: str


class ResetRequest(BaseModel):
    notebook_id: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return await rag.check_ollama()


@app.get("/api/models")
async def models():
    return await rag.list_models_with_status()


@app.post("/api/pull")
async def pull(req: PullRequest):
    """Stream pull progress as SSE. Each event is one Ollama status line."""
    async def event_stream():
        try:
            async for status in rag.pull_model(req.model):
                yield f"event: progress\ndata: {json.dumps(status)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Notebooks ──────────────────────────────────────────────────────────────
@app.get("/api/notebooks")
def notebooks_list():
    return rag.list_notebooks()


@app.post("/api/notebooks")
def notebooks_create(body: NotebookCreate):
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=400, detail="nazwa nie może być pusta")
    return rag.create_notebook(body.name)


@app.patch("/api/notebooks/{nb_id}")
def notebooks_rename(nb_id: str, body: NotebookUpdate):
    nb = rag.rename_notebook(nb_id, body.name)
    if nb is None:
        raise HTTPException(status_code=404, detail="notebook not found")
    return nb


@app.delete("/api/notebooks/{nb_id}")
def notebooks_delete(nb_id: str):
    ok = rag.delete_notebook(nb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="notebook not found")
    return {"ok": True}


# ── Documents ──────────────────────────────────────────────────────────────
@app.get("/api/documents")
def documents(notebook_id: Optional[str] = None):
    return rag.list_documents(notebook_id=notebook_id)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), notebook_id: str = Form(...)):
    if not rag.get_notebook(notebook_id):
        raise HTTPException(status_code=400, detail="nieznany notebook_id")

    ext = Path(file.filename).suffix.lower()
    if ext not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{ext}'. Allowed: {sorted(PARSERS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        text = parse_file(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted (scanned PDF? try OCR).")
        result = await rag.ingest_document(file.filename, text, notebook_id=notebook_id)
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    n = rag.delete_document(doc_id)
    return {"deleted_chunks": n}


@app.post("/api/reset")
def reset(body: ResetRequest):
    if body.notebook_id:
        n = rag.reset_notebook(body.notebook_id)
        return {"ok": True, "deleted": n, "scope": "notebook"}
    rag.reset_all()
    return {"ok": True, "scope": "all"}


# ── Conversations ──────────────────────────────────────────────────────────
@app.get("/api/conversations")
def conversations_list(notebook_id: Optional[str] = None):
    return rag.list_conversations(notebook_id=notebook_id)


@app.post("/api/conversations")
def conversations_create(body: ConversationCreate):
    if not body.notebook_id:
        raise HTTPException(status_code=400, detail="notebook_id wymagany")
    if not rag.get_notebook(body.notebook_id):
        raise HTTPException(status_code=400, detail="nieznany notebook_id")
    return rag.create_conversation(model=body.model, notebook_id=body.notebook_id)


@app.get("/api/conversations/{conv_id}")
def conversations_get(conv_id: str):
    c = rag.get_conversation(conv_id)
    if c is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return c


@app.delete("/api/conversations/{conv_id}")
def conversations_delete(conv_id: str):
    ok = rag.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Server-Sent Events stream.
    First event carries retrieved sources, then token chunks, then 'done'.
    Retrieval is scoped to the conversation's notebook.
    """
    conv = None
    history: list[dict] = []
    notebook_id: Optional[str] = None
    if req.conversation_id:
        conv = rag.get_conversation(req.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        history = conv.get("messages", [])
        notebook_id = conv.get("notebook_id")

    model = req.model or (conv["model"] if conv else None)

    # Short follow-ups ("a ile wynosi kara?") embed poorly on their own;
    # prepend the previous user question so retrieval keeps the topic.
    retrieval_query = req.query
    if history and len(req.query.strip()) < 60:
        prev_user = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "user"), ""
        )
        if prev_user:
            retrieval_query = f"{prev_user}\n{req.query}"

    hits = await rag.retrieve(retrieval_query, k=req.k, notebook_id=notebook_id)
    messages = rag.build_prompt_with_history(req.query, hits, history, model=model)

    async def event_stream():
        sources = [
            {
                "n": i + 1,
                "filename": h["metadata"]["filename"],
                "chunk_index": h["metadata"]["chunk_index"],
                "page": h["metadata"].get("page"),
                "preview": h["text"][:240],
                "distance": h["distance"],
            }
            for i, h in enumerate(hits)
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        answer_parts: list[str] = []
        try:
            async for token in rag.chat_stream(messages, model=model):
                answer_parts.append(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return

        if conv is not None:
            answer = "".join(answer_parts)
            if not answer.strip():
                fallback = "Model nie zwrócił odpowiedzi. Spróbuj ponownie lub przeformułuj pytanie."
                yield f"event: token\ndata: {json.dumps(fallback)}\n\n"
                answer = fallback
            new_messages = [
                {"role": "user", "content": req.query},
                {"role": "assistant", "content": answer, "sources": sources},
            ]
            title = conv.get("title")
            if not title and req.query.strip():
                title = req.query.strip()[:60]
            rag.update_conversation(
                conv["id"],
                append_messages=new_messages,
                title=title,
                model=model,
            )

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
