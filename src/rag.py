"""
Retrieve-and-Generate chatmodule.

Gebruik:
    >>> from src.rag import RAGChat
    >>> bot = RAGChat()
    >>> print(bot.answer("What is the EU Green Deal?"))
"""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, GENERATION_MODEL_NAME, PROMPT_TEMPLATE, TOP_K
from .embeddings import retrieve


# --------------------------------------------------------------------------- #
#   Model & tokenizer
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME).to(DEVICE)
model.eval()

MAX_MODEL_LEN = model.config.max_position_embeddings  # bv. 1024 tokens voor GPT-2-medium


# --------------------------------------------------------------------------- #
#   Prompt‐hulp
# --------------------------------------------------------------------------- #
def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Plakt context samen en vult de prompt‐template in."""
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# --------------------------------------------------------------------------- #
#   RAGChat-klasse
# --------------------------------------------------------------------------- #
class RAGChat:
    """
    Combineert retrieval (FAISS) en generatie (GPT-2, Llama-2, …).

    Parameters
    ----------
    top_k : int
        Aantal tekst-chunks om op te halen.
    max_new_tokens : int
        Lengte van het antwoord dat het LM mag toevoegen.
    """

    def __init__(self, *, top_k: int = TOP_K, max_new_tokens: int = 128) -> None:
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    # --------------------------------------------------------------------- #
    #   Publieke API
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def answer(self, question: str) -> str:
        """Geef een beknopt antwoord op `question`."""
        context_chunks, _ = retrieve(question, self.top_k)
        prompt = build_prompt(question, context_chunks)

        # Tokenize met AFKAPPEN zodat we nooit > model.max_position_embeddings gaan
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_LEN,
        ).to(DEVICE)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,               # puur greedy / beam-0
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)