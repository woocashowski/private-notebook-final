#!/usr/bin/env bash
set -e

# venv
if [ ! -d ".venv" ]; then
  echo "→ creating venv"
  python3 -m venv .venv
fi
source .venv/bin/activate

# deps
if [ ! -f ".venv/.installed" ]; then
  echo "→ installing requirements"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch .venv/.installed
fi

# check ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "⚠  Ollama is not running on localhost:11434"
  echo "   Install: https://ollama.com/download"
  echo "   Then:    ollama serve &"
  echo "            ollama pull llama3.2:3b"
  echo "            ollama pull nomic-embed-text"
  exit 1
fi

echo "→ starting server on http://localhost:8000"
uvicorn src.server:app --port 8000 --reload
