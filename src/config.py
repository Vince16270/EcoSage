"""
Globale configuratie-instellingen voor de chatbot-applicatie.

Hiermee hou je paden, model-namen en andere ‘magic numbers’
op één centrale plek, zodat je ze niet door de code hoeft te zoeken.
"""

from pathlib import Path
import os
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = MODELS_DIR / "faiss"
CHUNKS_FILE = MODELS_DIR / "chunks.json"

# ── Models / RAG params ───────────────────────────────────────────
EMBEDDING_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME  = "perplexity-ai/r1-1776-distill-llama-70b"
TOP_K        = 5
CHUNK_SIZE   = 300
OVERLAP      = 50

# ── Device selection (CPU · CUDA · Apple MPS) ─────────────────────
device_env = os.getenv("DEVICE", "auto").lower()          

if device_env == "auto":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"            
    else:
        DEVICE = "cpu"
else:
    DEVICE = device_env          

# ── Prompt template ───────────────────────────────────────────────
PROMPT_TEMPLATE = """You're an expert in European policy.
You have access to the following context:
{context}

Your question is: {question}
Answer concisely."""