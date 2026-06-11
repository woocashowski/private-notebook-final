"""
RAG core.

- Chunking: simple recursive splitter (paragraphs -> sentences -> words).
- Embeddings: Ollama `/api/embeddings`.
- Vector store: ChromaDB persistent client.
- Generation: Ollama `/api/chat` with streaming.

Everything is local. No network calls outside localhost.
"""
from __future__ import annotations

import os
import uuid
import json
import re
import time
import logging
import threading
from pathlib import Path
from typing import AsyncIterator, Iterable

# Silence chromadb telemetry warnings (known noisy-but-harmless logs).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import httpx
import chromadb
from chromadb.config import Settings

# --- Config (override via env) -----------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHATS_FILE = Path(os.getenv("CHATS_FILE", "./chats.json"))
NOTEBOOKS_FILE = Path(os.getenv("NOTEBOOKS_FILE", "./notebooks.json"))
COLLECTION = "documents"

CHUNK_SIZE = 900        # chars (~200-250 tokens — safe for small embed models)
CHUNK_OVERLAP = 150

# --- Available LLMs (for UI dropdown) ---------------------------------------
AVAILABLE_MODELS = [
    {
        "id": "llama3.2:3b",
        "name": "Llama 3.2 3B",
        "size_gb": 2.0,
        "pros": "Szybki, niewielki, działa na CPU, dobry ogólny model.",
        "cons": "Słabszy w złożonym rozumowaniu i długim kontekście.",
        "recommended_for": "Domyślny wybór — pytania faktograficzne, krótkie odpowiedzi.",
    },
    {
        "id": "qwen2.5:1.5b",
        "name": "Qwen 2.5 1.5B",
        "size_gb": 1.0,
        "pros": "Najszybszy, minimalny narzut pamięci, dobry na słabym sprzęcie.",
        "cons": "Jakość odpowiedzi wyraźnie niższa niż modeli 3B+.",
        "recommended_for": "Stary laptop bez GPU / szybkie testy.",
    },
    {
        "id": "qwen2.5:7b",
        "name": "Qwen 2.5 7B",
        "size_gb": 4.7,
        "pros": "Silny w matematyce i kodzie, wielojęzyczny (także PL).",
        "cons": "Wolny na CPU, ~8 GB RAM w trakcie generacji.",
        "recommended_for": "GPU lub Apple Silicon — najlepsza jakość PL.",
    },
    {
        "id": "llama3.1:8b",
        "name": "Llama 3.1 8B",
        "size_gb": 4.9,
        "pros": "Bardzo dobry w rozumowaniu i długim kontekście (128k).",
        "cons": "Wymaga GPU dla sensownej prędkości.",
        "recommended_for": "Złożone pytania wymagające kontekstu z wielu źródeł.",
    },
    {
        "id": "phi3.5:3.8b",
        "name": "Phi 3.5 3.8B",
        "size_gb": 2.2,
        "pros": "Dobry kompromis wielkość/jakość, mocny w zadaniach logicznych.",
        "cons": "Głównie angielski — PL słabszy.",
        "recommended_for": "Dokumenty w języku angielskim, analiza techniczna.",
    },
    {
        "id": "gemma2:2b",
        "name": "Gemma 2 2B",
        "size_gb": 1.6,
        "pros": "Lekki, bezpieczne odpowiedzi, dobra jakość przy tym rozmiarze.",
        "cons": "Krótszy kontekst, słabszy w kodowaniu.",
        "recommended_for": "Szybkie odpowiedzi faktograficzne na słabszym sprzęcie.",
    },
    {
        "id": "hf.co/mradermacher/Bielik-7B-polish-law-GGUF:Q2_K",
        "name": "Bielik 7B Polish Law low",
        "size_gb": 2.8,
        "pros": "Dobra jakość ogólna, wytrenowany na polskim systemie prawnym.",
        "cons": "Najslabszy model Bielika - DEMO",
        "recommended_for": "Uniwersalny model do różnorodnych zadań, jeśli masz GPU.",
    },
    {
        "id": "hf.co/mradermacher/Bielik-7B-polish-law-GGUF:Q4_K_S",
        "name": "Bielik 7B Polish Law mid",
        "size_gb": 4.2,
        "pros": "Lepsza jakość ogólna, wytrenowany na polskim systemie prawnym.",
        "cons": "Średni model Bielika",
        "recommended_for": "Uniwersalny model do różnorodnych zadań, jeśli masz GPU.",
    },
    {
        "id": "hf.co/mradermacher/Bielik-7B-polish-law-GGUF:Q8_0",
        "name": "Bielik 7B Polish Law up",
        "size_gb": 7.8,
        "pros": "Prawie najlepsza jakość ogólna, wytrenowany na polskim systemie prawnym.",
        "cons": "wolny model Bielika",
        "recommended_for": "Uniwersalny model do różnorodnych zadań, jeśli masz GPU.",
    },
    {
        "id": "hf.co/mradermacher/Bielik-7B-polish-law-GGUF:f16",
        "name": "Bielik 7B Polish Law top",
        "size_gb": 14.6,
        "pros": "Najlepsza jakość ogólna, wytrenowany na polskim systemie prawnym.",
        "cons": "bardzo wolny model Bielika",
        "recommended_for": "Najmocniejszy model do różnorodnych zadań, jeśli masz GPU.",
    }
]

# --- Chroma ------------------------------------------------------------------
_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)
# We compute embeddings ourselves (via Ollama) so disable Chroma's default.
_collection = _client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)


# --- Chunking ----------------------------------------------------------------
def _split_long(text: str, size: int) -> list[str]:
    """Hard-split when a single paragraph exceeds size."""
    out = []
    for i in range(0, len(text), size):
        out.append(text[i:i + size])
    return out


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Greedy paragraph-aware chunker.
    Groups whole paragraphs up to `size`, adds tail of previous chunk as overlap.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs: list[str] = []
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        if len(p) > size:
            paragraphs.extend(_split_long(p, size))
        else:
            paragraphs.append(p)

    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            # Overlap: carry tail of previous chunk into new one.
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n\n{p}" if tail else p
    if buf:
        chunks.append(buf)
    return chunks


# --- Ollama client -----------------------------------------------------------
async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama. Returns list of vectors."""
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for t in texts:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
    return out


async def chat_stream(messages: list[dict], model: str | None = None) -> AsyncIterator[str]:
    """Stream chat tokens from Ollama using the given model (defaults to LLM_MODEL)."""
    target_model = model or LLM_MODEL
    
    if "hf.co/" in target_model:
        target_model = target_model.lower()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",  
            json={
                "model": target_model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip(): 
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("error"):
                    raise RuntimeError(data["error"])

                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

                if data.get("done"):
                    break

async def installed_models() -> list[str]:
    """Return names of models currently installed in Ollama."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


def _model_installed(installed: list[str], model_id: str) -> bool:
    """Strict 1:1 match or base-name match fallback for tags."""
    if model_id in installed:
        return True        
    base = model_id.split(":")[0]
    for m in installed:
        if m == model_id:
            return True
        if m.split(":")[0] == base and (":" not in model_id or m == model_id):
            return True
    return False


async def list_models_with_status() -> list[dict]:
    """AVAILABLE_MODELS enriched with an `installed` flag."""
    try:
        installed = await installed_models()
    except Exception:
        installed = []
    return [
        {**m, "installed": _model_installed(installed, m["id"])}
        for m in AVAILABLE_MODELS
    ]


async def pull_model(model_id: str) -> AsyncIterator[dict]:
    """
    Stream Ollama's pull progress. Each yielded dict is one Ollama status line.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/pull",
            json={"model": model_id, "stream": True},
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


async def check_ollama() -> dict:
    """Smoke-check Ollama and required models."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            
            def has(name: str) -> bool:
                return any(m == name or m.startswith(name + ":") or m.split(":")[0] == name.split(":")[0] for m in models)
            return {
                "ok": True,
                "llm_present": has(LLM_MODEL),
                "embed_present": has(EMBED_MODEL),
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "available": models,
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "llm_model": LLM_MODEL, "embed_model": EMBED_MODEL}


# --- Public API --------------------------------------------------------------
async def ingest_document(filename: str, text: str, notebook_id: str) -> dict:
    """Chunk, embed, and store a document's text scoped to a notebook."""
    chunks = chunk_text(text)
    if not chunks:
        return {"filename": filename, "chunks": 0, "skipped": True}

    doc_id = uuid.uuid4().hex[:12]
    vectors = await embed(chunks)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "notebook_id": notebook_id,
        }
        for i in range(len(chunks))
    ]
    _collection.add(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
    )
    return {"filename": filename, "doc_id": doc_id, "chunks": len(chunks), "notebook_id": notebook_id}


async def retrieve(query: str, k: int = 5, notebook_id: str | None = None) -> list[dict]:
    """Return top-k chunks relevant to query, optionally scoped to one notebook."""
    if _collection.count() == 0:
        return []
    q_vec = (await embed([query]))[0]
    kwargs = {
        "query_embeddings": [q_vec],
        "n_results": min(k, _collection.count()),
    }
    if notebook_id:
        kwargs["where"] = {"notebook_id": notebook_id}
    res = _collection.query(**kwargs)
    hits = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]
        if distance > 1.2:
            continue
        hits.append({
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": distance,
        })
    return hits


def build_prompt(query: str, hits: list[dict]) -> list[dict]:
    """Construct chat messages with a smart hybrid approach for Bielik."""
    
    # 1. Sprawdzamy, czy RAG faktycznie coś znalazł w bazie danych
    if not hits:
        # Informujemy model wprost: opieraj się TYLKO na swojej wewnętrznej wiedzy prawniczej
        system = (
            "Jesteś profesjonalnym i elitarnym asystentem ds. polskiego prawa. "
            "Twoim zadaniem jest udzielenie precyzyjnej, wyczerpującej i poprawnej merytorycznie "
            "odpowiedzi na pytanie użytkownika, korzystając z CAŁEJ SWOJEJ WIEDZY na temat "
            "polskich ustaw, kodeksów i przepisów (np. Kodeksu cywilnego, karnego, Prawa o ruchu drogowym).\n"
            "ZASADY:\n"
            "- Odpowiadaj naturalnie, pełnymi zdaniami, czystą polszczyzną.\n"
            "- Podaj podstawę prawną (np. artykuł, ustawę), jeśli ją pamiętasz.\n"
            "- Twoja odpowiedź musi być merytorycznym wyjaśnieniem — NIGDY nie odpowiadaj pytaniem "
            "ani nie powtarzaj pytania użytkownika.\n"
            "- Ponieważ użytkownik nie załączył dodatkowych dokumentów do tego pytania, NIE używaj "
            "żadnych przypisów typu [Dokument N] ani [Source N] — po prostu odpowiedz na pytanie.\n"
            "- Pod żadnym pozorem nie pisz po angielsku."
        )
        user = f"Pytanie prawne: {query}"
        
    else:
        # 2. Jeśli dokumenty są w bazie, uruchamiamy tryb RAG z przypisami
        parts = []
        for i, h in enumerate(hits, 1):
            src = h["metadata"]["filename"]
            parts.append(f"[Dokument {i}: {src}]\n{h['text']}")
        context = "\n\n---\n\n".join(parts)

        system = (
            "Jesteś profesjonalnym asystentem ds. polskiego prawa. Twoim zadaniem jest odpowiedź "
            "na pytanie użytkownika w języku polskim.\n"
            "ZASADY:\n"
            "- Odpowiadaj WYŁĄCZNIE własnymi słowami — nigdy nie przepisuj ani nie cytuj dosłownie treści materiałów.\n"
            "- Materiały mogą zawierać pytania innych osób lub fragmenty dyskusji — ignoruj je całkowicie; "
            "odpowiadasz TYLKO na pytanie użytkownika podane na końcu.\n"
            "- Twoja odpowiedź musi być merytorycznym wyjaśnieniem — NIGDY nie odpowiadaj pytaniem "
            "ani nie powtarzaj pytania z materiałów.\n"
            "- Nie wklejaj adresów URL ani żadnych placeholderów (<url>, [link] itp.) z materiałów.\n"
            "- Jeśli materiały zawierają odpowiedź, użyj ich w pierwszej kolejności i dopisz na końcu "
            "zdania przypis wskazujący nazwę pliku, np. [Dokument 1: nazwa.pdf].\n"
            "- Jeśli materiały są niewystarczające, uzupełnij odpowiedź o swoją własną wiedzę prawniczą.\n"
            "- Nigdy nie używaj języka angielskiego.\n\n"
            f"Materiały pomocnicze:\n{context}"
        )
        user = f"Pytanie: {query}"
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def list_documents(notebook_id: str | None = None) -> list[dict]:
    """Unique documents in the store with chunk counts, optionally filtered by notebook."""
    kwargs = {"include": ["metadatas"]}
    if notebook_id:
        kwargs["where"] = {"notebook_id": notebook_id}
    all_meta = _collection.get(**kwargs)
    counts: dict[str, dict] = {}
    for m in all_meta["metadatas"]:
        key = m["doc_id"]
        if key not in counts:
            counts[key] = {
                "doc_id": key,
                "filename": m["filename"],
                "chunks": 0,
                "notebook_id": m.get("notebook_id"),
            }
        counts[key]["chunks"] += 1
    return sorted(counts.values(), key=lambda d: d["filename"].lower())


def delete_document(doc_id: str) -> int:
    """Remove all chunks of a doc. Returns number deleted."""
    got = _collection.get(where={"doc_id": doc_id}, include=[])
    ids = got["ids"]
    if ids:
        _collection.delete(ids=ids)
    return len(ids)


def reset_notebook(notebook_id: str) -> int:
    """Delete all chunks belonging to a notebook. Returns number deleted."""
    got = _collection.get(where={"notebook_id": notebook_id}, include=[])
    ids = got["ids"]
    if ids:
        _collection.delete(ids=ids)
    return len(ids)


def reset_all() -> None:
    """Nuke the collection (all notebooks)."""
    global _collection
    _client.delete_collection(COLLECTION)
    _collection = _client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


# --- Notebooks (collections) ------------------------------------------------
_notebooks_lock = threading.Lock()


def _load_notebooks() -> dict:
    if not NOTEBOOKS_FILE.exists():
        return {"notebooks": []}
    try:
        return json.loads(NOTEBOOKS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"notebooks": []}


def _save_notebooks(data: dict) -> None:
    NOTEBOOKS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_notebooks() -> list[dict]:
    """Return all notebooks, newest first, with doc and chat counts."""
    with _notebooks_lock:
        data = _load_notebooks()
    notebooks = list(data["notebooks"])

    # Count docs per notebook from Chroma.
    all_meta = _collection.get(include=["metadatas"])
    doc_counts: dict[str, set] = {}
    for m in all_meta["metadatas"]:
        nb = m.get("notebook_id")
        if nb:
            doc_counts.setdefault(nb, set()).add(m["doc_id"])

    # Count chats per notebook.
    chats_data = _load_chats()
    chat_counts: dict[str, int] = {}
    for c in chats_data["conversations"]:
        nb = c.get("notebook_id")
        if nb:
            chat_counts[nb] = chat_counts.get(nb, 0) + 1

    out = []
    for nb in notebooks:
        out.append({
            **nb,
            "doc_count": len(doc_counts.get(nb["id"], set())),
            "chat_count": chat_counts.get(nb["id"], 0),
        })
    return sorted(out, key=lambda n: n.get("updated_at") or 0, reverse=True)


def get_notebook(nb_id: str) -> dict | None:
    with _notebooks_lock:
        data = _load_notebooks()
    for nb in data["notebooks"]:
        if nb["id"] == nb_id:
            return nb
    return None


def create_notebook(name: str) -> dict:
    now = int(time.time())
    nb = {
        "id": uuid.uuid4().hex[:12],
        "name": (name or "Bez nazwy").strip() or "Bez nazwy",
        "created_at": now,
        "updated_at": now,
    }
    with _notebooks_lock:
        data = _load_notebooks()
        data["notebooks"].append(nb)
        _save_notebooks(data)
    return nb


def rename_notebook(nb_id: str, name: str) -> dict | None:
    with _notebooks_lock:
        data = _load_notebooks()
        for nb in data["notebooks"]:
            if nb["id"] == nb_id:
                nb["name"] = (name or "").strip() or nb["name"]
                nb["updated_at"] = int(time.time())
                _save_notebooks(data)
                return nb
        return None


def delete_notebook(nb_id: str) -> bool:
    """Delete a notebook together with its chunks and conversations."""
    with _notebooks_lock:
        data = _load_notebooks()
        before = len(data["notebooks"])
        data["notebooks"] = [n for n in data["notebooks"] if n["id"] != nb_id]
        if len(data["notebooks"]) == before:
            return False
        _save_notebooks(data)

    # Cascade: delete chunks in this notebook.
    try:
        reset_notebook(nb_id)
    except Exception:
        pass

    # Cascade: delete conversations tied to this notebook.
    with _chats_lock:
        cdata = _load_chats()
        cdata["conversations"] = [c for c in cdata["conversations"] if c.get("notebook_id") != nb_id]
        _save_chats(cdata)
    return True


def ensure_default_notebook() -> str:
    """Create a default notebook if none exists, plus migrate legacy chunks/chats to it."""
    with _notebooks_lock:
        data = _load_notebooks()
        if data["notebooks"]:
            return data["notebooks"][0]["id"]

    default = create_notebook("Domyślny")
    default_id = default["id"]

    # Migrate orphan chunks (no notebook_id) to default notebook.
    try:
        all_items = _collection.get(include=["metadatas"])
        orphan_ids = []
        orphan_metas = []
        for mid, meta in zip(all_items["ids"], all_items["metadatas"]):
            if not meta.get("notebook_id"):
                meta = {**meta, "notebook_id": default_id}
                orphan_ids.append(mid)
                orphan_metas.append(meta)
        if orphan_ids:
            _collection.update(ids=orphan_ids, metadatas=orphan_metas)
    except Exception:
        pass

    # Migrate orphan chats to default notebook.
    with _chats_lock:
        cdata = _load_chats()
        changed = False
        for c in cdata["conversations"]:
            if not c.get("notebook_id"):
                c["notebook_id"] = default_id
                changed = True
        if changed:
            _save_chats(cdata)

    return default_id


# --- Conversations ----------------------------------------------------------
_chats_lock = threading.Lock()


def _load_chats() -> dict:
    if not CHATS_FILE.exists():
        return {"conversations": []}
    try:
        return json.loads(CHATS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"conversations": []}


def _save_chats(data: dict) -> None:
    CHATS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_conversations(notebook_id: str | None = None) -> list[dict]:
    """Return conversations, newest-updated first. Optionally filter by notebook."""
    with _chats_lock:
        data = _load_chats()
    items = data["conversations"]
    if notebook_id:
        items = [c for c in items if c.get("notebook_id") == notebook_id]
    summaries = [
        {
            "id": c["id"],
            "title": c.get("title") or "(bez tytułu)",
            "model": c.get("model"),
            "notebook_id": c.get("notebook_id"),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
            "message_count": len(c.get("messages", [])),
        }
        for c in items
    ]
    return sorted(summaries, key=lambda c: c["updated_at"] or 0, reverse=True)


def get_conversation(conv_id: str) -> dict | None:
    with _chats_lock:
        data = _load_chats()
    for c in data["conversations"]:
        if c["id"] == conv_id:
            return c
    return None


def create_conversation(model: str | None = None, notebook_id: str | None = None) -> dict:
    now = int(time.time())
    conv = {
        "id": uuid.uuid4().hex[:12],
        "title": None,
        "model": model or LLM_MODEL,
        "notebook_id": notebook_id,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    with _chats_lock:
        data = _load_chats()
        data["conversations"].append(conv)
        _save_chats(data)
    return conv


def update_conversation(conv_id: str, *, append_messages: list[dict] | None = None,
                         title: str | None = None, model: str | None = None) -> dict | None:
    with _chats_lock:
        data = _load_chats()
        for c in data["conversations"]:
            if c["id"] == conv_id:
                if append_messages:
                    c.setdefault("messages", []).extend(append_messages)
                if title is not None:
                    c["title"] = title
                if model is not None:
                    c["model"] = model
                c["updated_at"] = int(time.time())
                _save_chats(data)
                return c
        return None


def delete_conversation(conv_id: str) -> bool:
    with _chats_lock:
        data = _load_chats()
        before = len(data["conversations"])
        data["conversations"] = [c for c in data["conversations"] if c["id"] != conv_id]
        if len(data["conversations"]) == before:
            return False
        _save_chats(data)
        return True


def build_prompt_with_history(query: str, hits: list[dict], history: list[dict]) -> list[dict]:
    """Like build_prompt but prepends prior conversation turns as chat history."""
    base = build_prompt(query, hits)
    system, user = base[0], base[1]
    # History excludes system; include only user/assistant turns, capped at last 6 (3 exchanges).
    prior = [m for m in history if m.get("role") in ("user", "assistant")]
    prior = prior[-6:]
    return [system, *prior, user]
