"""
Functies voor het inlezen en voorbereiden (chunken, schonen) van teksten.

Gebruik deze module éénmalig om ruwe documenten om
te zetten in nette tekst-chunks + bijbehorende FAISS-index.
"""

from __future__ import annotations

import re
import string
import json
from pathlib import Path
from typing import List

import nltk
from PyPDF2 import PdfReader

from .config import CHUNK_SIZE, OVERLAP, CHUNKS_FILE

try:
    nltk.data.find("tokenizers/punkt")
except LookupError: 
    nltk.download("punkt")


#   Extraction & cleaning

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Leest een PDF en geeft alle pagina-tekst als één string terug.
    """
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    """
    Basisschoonmaak:
    - Lower-case
    - Alleen ASCII-letters & standaard leestekens
    - Witruimte normaliseren
    """
    text = text.lower()
    allowed = string.ascii_lowercase + string.digits + string.punctuation + " \n"
    text = "".join(ch for ch in text if ch in allowed)
    text = re.sub(r"\s+", " ", text).strip()
    return text


#   Chunking

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Verdeelt `text` in overlappende stukken van ~`size` tokens.
    Overlap helpt de RAG-retriever om context aan elkaar te plakken.
    """
    tokens = nltk.word_tokenize(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


#   Pipeline-helper

def pdfs_to_chunks(pdf_paths: List[str | Path]) -> List[str]:
    """
    Extract + clean + chunk in één keer voor meerdere PDF’s.
    """
    all_chunks: List[str] = []
    for path in pdf_paths:
        raw = extract_text_from_pdf(path)
        clean = clean_text(raw)
        all_chunks.extend(chunk_text(clean))
    return all_chunks


def save_chunks(chunks: List[str], file_path: str | Path = CHUNKS_FILE) -> None:
    """
    Slaat de tekst-chunks op als JSON-lijst.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


#   Command-line entry-point

if __name__ == "__main__": 
    import argparse

    ap = argparse.ArgumentParser(description="Build text chunks from PDFs.")
    ap.add_argument("pdfs", nargs="+", help="Één of meer PDF-bestanden")
    args = ap.parse_args()

    chunks = pdfs_to_chunks(args.pdfs)
    save_chunks(chunks)
    print(f"{len(chunks)} chunks opgeslagen in {CHUNKS_FILE}")