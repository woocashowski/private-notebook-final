"""
Document parsers. Extracts raw text from supported file types.
Each parser returns a single string. Keep it simple — for MVP.
"""
from pathlib import Path
from pypdf import PdfReader
import docx


def parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = ""
        if text.strip():
            parts.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(parts)


def parse_docx(path: Path) -> str:
    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


def parse_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
    ".markdown": parse_text,
}


def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(PARSERS)}")
    return PARSERS[ext](path)
