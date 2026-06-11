@echo off
setlocal

REM venv
if not exist ".venv" (
    echo -^> creating venv
    python -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

REM deps
if not exist ".venv\.installed" (
    echo -^> installing requirements
    python -m pip install -q --upgrade pip
    if errorlevel 1 goto :error
    pip install -q -r requirements.txt
    if errorlevel 1 goto :error
    type nul > ".venv\.installed"
)

REM check ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama is not running on localhost:11434
    echo     Install: https://ollama.com/download
    echo     Then:    ollama serve
    echo              ollama pull llama3.2:3b
    echo              ollama pull nomic-embed-text
    exit /b 1
)

echo -^> starting server on http://localhost:8000
uvicorn server:app --port 8000 --reload
goto :eof

:error
echo [!] command failed
exit /b 1
