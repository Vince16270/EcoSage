"""
Retrieve-and-Generate (RAG) module.

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

tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME).to(DEVICE)
model.eval()

MAX_MODEL_LEN: int = model.config.max_position_embeddings  # bv. 1024 tokens


def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Zet context en vraag in de template."""
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)

class RAGChat:
    """
    Combineert FAISS-retrieval met LM-generatie.

    Parameters
    ----------
    top_k : int
        Hoeveel tekst-chunks ophalen.
    max_new_tokens : int
        Maximale lengte van het antwoord.
    """

    def __init__(self, *, top_k: int = TOP_K, max_new_tokens: int = 128) -> None:
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def answer(self, question: str) -> str:
        """Geef een beknopt antwoord op `question`."""
        context_chunks, _ = retrieve(question, self.top_k)
        prompt = build_prompt(question, context_chunks)

        max_prompt_tokens = MAX_MODEL_LEN - self.max_new_tokens
        if max_prompt_tokens < 1:
            raise ValueError("`max_new_tokens` overschrijdt de modelcontext!")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(DEVICE)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            max_length=MAX_MODEL_LEN,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)