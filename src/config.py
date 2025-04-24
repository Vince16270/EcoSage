"""
Globale configuratie-instellingen voor de chatbot-applicatie.

Hiermee hou je paden, model-namen en andere ‘magic numbers’
op één centrale plek, zodat je ze niet door de code hoeft te zoeken.
"""

from pathlib import Path
import torch

# --------------------------------------------------
# Bestands- en directory-paden
# --------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
INDEX_DIR: Path = MODELS_DIR / "faiss"
CHUNKS_FILE: Path = MODELS_DIR / "chunks.json"   # opgeslagen context-snippets

# --------------------------------------------------
# Retrieval- & generatie-instellingen
# --------------------------------------------------
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME: str = "openai-community/gpt2-medium"      # voorbeeld
TOP_K: int = 5                           # aantal passages uit FAISS
CHUNK_SIZE: int = 300                    # tokens/woorden per chunk
OVERLAP: int = 50                        # overlap tussen chunks

# --------------------------------------------------
# Hardware
# --------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Prompt-template
# --------------------------------------------------
PROMPT_TEMPLATE: str = """You're an expert in European policy.
You have access to the following context:
{context}

Your question is: {question}
Answer concisely."""