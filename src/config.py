"""
Configuration settings for the EcoSage Chat application.
"""

from pathlib import Path
import os
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Paths 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = MODELS_DIR / "faiss"
CHUNKS_FILE = MODELS_DIR / "chunks.json"

# Models / RAG params
EMBEDDING_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME  = "NousResearch/Hermes-3-Llama-3.2-3B"
TRANSLATION_MODEL      = "Helsinki-NLP/opus-mt-en-nl"
TOP_K        = 5
CHUNK_SIZE   = 512
OVERLAP      = 50

# Device selection 
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

# Prompt
PROMPT_TEMPLATE = """You are an expert in European energy and climate policy.
You have access to the following context:
{context}

Your question is: {question}
Answer concisely."""