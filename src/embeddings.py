"""
Laadt de sentence-transformer, bouwt (of herlaadt) de FAISS-index
en stelt één helperfunctie beschikbaar: `retrieve(question)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    DEVICE,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    CHUNKS_FILE,
    TOP_K,
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

# Load tekst-chunks
with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks: List[str] = json.load(f)


# FAISS-index

INDEX_DIR.mkdir(parents=True, exist_ok=True)
index_file: Path = INDEX_DIR / "index.faiss"

if index_file.exists():
    index = faiss.read_index(str(index_file))
else:
    print("Geen bestaande FAISS-index gevonden – bouwen …")
    vectors = embedding_model.encode(chunks, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_file))
    print(f"Index opgeslagen in {index_file}")

def retrieve(question: str, top_k: int = TOP_K) -> Tuple[List[str], List[int]]:
    """
    Geeft de `top_k` relevante tekst-chunks én hun indices terug.
    """
    q_emb = embedding_model.encode([question]).astype("float32")
    _dists, idx = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in idx[0]], idx[0].tolist()