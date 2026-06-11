# Offline RAG — MVP

A local-only "NotebookLM-lite": drop files in, ask questions, get answers with
citations. Nothing leaves your machine.

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  PDF / DOCX  │ →  │  chunk+embed  │ →  │   ChromaDB   │
│  MD / TXT    │    │   (Ollama)    │    │  (local SQL) │
└──────────────┘    └───────────────┘    └──────┬───────┘
                                                │
                     ┌──────────────┐    ┌──────┴───────┐
     question  ───→  │   retrieve   │ ←──│  top-k chunks│
                     │   + prompt   │    └──────────────┘
                     └──────┬───────┘
                            │
                     ┌──────┴───────┐
                     │ Ollama chat  │  →  streaming answer + citations
                     │  (llama3.2)  │
                     └──────────────┘
```

## Stack

- **Runtime**: Ollama (local LLM + embeddings server)
- **LLM**: `llama3.2:3b` (default — change via `LLM_MODEL` env)
- **Embeddings**: `nomic-embed-text` (change via `EMBED_MODEL`)
- **Vector DB**: ChromaDB (persistent, SQLite under the hood)
- **Backend**: FastAPI + httpx
- **Frontend**: single HTML file, vanilla JS, SSE streaming

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — install from [ollama.com](https://ollama.com/download), then:
   ```bash
   ollama serve &                     # starts the local server
   ollama pull llama3.2:3b            # ~2 GB
   ollama pull nomic-embed-text       # ~274 MB
   ```

## Run

```bash
bash run.sh
```

Open <http://localhost:8000>.

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --port 8000 --reload
```

## Polish / multilingual docs

The default embeddings (`nomic-embed-text`) are English-first. For Polish or
mixed-language files, swap to a multilingual model:

```bash
ollama pull bge-m3
EMBED_MODEL=bge-m3 bash run.sh
```

⚠  If you change the embedding model, **wipe the index first** — vectors from
different models are not comparable. Click "clear all" in the sidebar, or:

```bash
rm -rf chroma_db/
```

## What works in this MVP

- Upload PDF / DOCX / MD / TXT via drag-and-drop
- Automatic chunking (~900 chars, paragraph-aware, 150-char overlap)
- Streaming answers with inline `[Source N]` citations
- Expandable source panels showing the actual chunk used
- Per-document delete + full reset
- Live health check for Ollama + required models

## What's NOT here (deliberately — it's an MVP)

- No OCR (scanned PDFs will be empty — use the full project's Phase 2)
- No conversation history (each question is standalone)
- No notebooks / collections (all docs share one index)
- No re-ranking (raw cosine distance only)
- No semantic chunking (recursive paragraph-based only)
- No auth — runs on localhost

## Config (env vars)

| Variable      | Default              | What it does                         |
|---------------|----------------------|--------------------------------------|
| `OLLAMA_URL`  | `http://localhost:11434` | Ollama server address            |
| `LLM_MODEL`   | `llama3.2:3b`        | Generation model                     |
| `EMBED_MODEL` | `nomic-embed-text`   | Embedding model                      |
| `CHROMA_DIR`  | `./chroma_db`        | Where the vector DB lives on disk    |

## File layout

```
offline-rag/
├── server.py          # FastAPI app, endpoints, SSE streaming
├── rag.py             # chunking, embeddings, retrieval, generation
├── parsers.py         # PDF / DOCX / MD / TXT text extraction
├── static/
│   └── index.html     # UI (single file, no build step)
├── requirements.txt
├── run.sh
└── chroma_db/         # generated — vector store
```

## Troubleshooting

**"ollama offline" in the sidebar** — run `ollama serve` in another terminal.

**"missing: llama3.2:3b"** — run `ollama pull llama3.2:3b`.

**Slow answers on CPU** — try a smaller model: `ollama pull qwen2.5:1.5b` then
`LLM_MODEL=qwen2.5:1.5b bash run.sh`. On Apple Silicon / NVIDIA, you can go
larger: `llama3.1:8b` or `qwen2.5:7b`.

**"No text could be extracted"** — the PDF is probably scanned. OCR is out of
scope for the MVP.

**Answers are wrong or generic** — try `k=8` retrieval (edit the `k` param in
the frontend fetch call) or a larger LLM. Also check the sources panel: if the
right chunk isn't in the retrieved set, the problem is retrieval quality, not
the model.
