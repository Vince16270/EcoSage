"""
Retrieve-and-Generate (RAG) met DeepSeek-R1-Distill-Qwen-7B.

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

tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_NAME,
    trust_remote_code=True,   
)

model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_NAME,
    torch_dtype=torch.float16, 
    device_map={"": DEVICE},                         
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval()

MAX_MODEL_LEN: int = model.config.max_position_embeddings  

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)

class RAGChat:
    """
    Koppelt FAISS-retrieval aan DeepSeek-Qwen-generatie.

    Parameters
    ----------
    top_k : int
        Aantal context-chunks om op te halen.
    max_new_tokens : int
        Maximale lengte van het antwoord.
    """

    def __init__(self, *, top_k: int = TOP_K, max_new_tokens: int = 256) -> None:
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def answer(self, question: str) -> str:
        """Geef een antwoord op `question` met context‐retrieval."""
        context_chunks, _ = retrieve(question, self.top_k)

        if not context_chunks:
            return "Het spijt me, maar ik beschik niet over een direct antwoord op uw vraag en kon ook geen relevante informatie vinden."

        prompt = build_prompt(question, context_chunks)

        max_prompt_tokens = MAX_MODEL_LEN - self.max_new_tokens
        if max_prompt_tokens < 1:
            raise ValueError("`max_new_tokens` overschrijdt de model-context!")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(DEVICE)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,       
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        answer_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not answer_text:
            context_info = "\n\n".join(context_chunks)
            return (
                "Het spijt me, maar ik beschik niet over een direct antwoord op uw vraag. "
                "Hier is wel relevante informatie omtrent het onderwerp:\n\n"
                f"{context_info}"
            )

        return answer_text
