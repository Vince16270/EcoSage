"""
Bevat de RAGChat-klasse: retrieve + generate.
Los te gebruiken in cli.py, api.py of tests.
"""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, GENERATION_MODEL_NAME, PROMPT_TEMPLATE, TOP_K
from .embeddings import retrieve

# --------------------------------------------------
# 1. Laad generatief model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME).to(DEVICE)
model.eval()


# --------------------------------------------------
# 2. Prompt builder
# --------------------------------------------------
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# --------------------------------------------------
# 3. RAG-klasse
# --------------------------------------------------
class RAGChat:
    """
    >>> chat = RAGChat()
    >>> chat.answer("What is the EU Green Deal?")
    'The European Green Deal is ...'
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k

    # ---------------- Retrieval + generatie ---------------- #
    def answer(self, question: str, max_new_tokens: int = 100) -> str:
        context_chunks, _ = retrieve(question, self.top_k)
        prompt = build_prompt(question, context_chunks)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(ids[0], skip_special_tokens=True)