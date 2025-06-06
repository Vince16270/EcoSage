"""
Retrieve-and-Generate (RAG) with NousResearch/Hermes-3-Llama-3.2-3B.
Defaults to English; met lang='nl' wordt het antwoord vertaald naar het Nederlands.
"""

from __future__ import annotations
from typing import List
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from .config import (
    DEVICE,
    GENERATION_MODEL_NAME,
    PROMPT_TEMPLATE,
    TOP_K,
    TRANSLATION_MODEL,
)
from .embeddings import retrieve

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generator – NousResearch/Hermes-3-Llama-3.2-3B
gen_tok = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, trust_remote_code=True)
gen_mod = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
    trust_remote_code=True,
    attn_implementation="eager",
).eval()
MAX_MODEL_LEN: int = gen_mod.config.max_position_embeddings

# 2. Translator – Helsinki-NLP/opus-mt-en-nl
trans_tok = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
trans_mod = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL,
    torch_dtype=torch.float16,
)
trans_mod.to(DEVICE)
trans_mod.eval()
# ─────────────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def en2nl(text: str, max_new_tokens: int = 256) -> str:
    """Vertaal van Engels naar Nederlands met Opus-MT."""
    ins = trans_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    # ── Belangrijk: verplaats elk tensor‐veld naar trans_mod.device ──
    ins = {k: v.to(trans_mod.device) for k, v in ins.items()}

    ids = trans_mod.generate(
        **ins,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        temperature=0.0,
        pad_token_id=trans_tok.eos_token_id,
    )
    return trans_tok.decode(ids[0], skip_special_tokens=True).strip()


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class RAGChat:
    """Combineert FAISS-retrieval met Qwen-generatie."""

    def __init__(self, *, top_k: int = TOP_K, max_new_tokens: int = 256) -> None:
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def answer(self, question: str, *, lang: str = "en") -> str:
        # 1) Retrieve
        context_chunks, _ = retrieve(question, self.top_k)
        if not context_chunks:
            return (
                "Sorry, ik kon geen relevante informatie vinden."
                if lang.lower().startswith("nl")
                else "Sorry, I couldn't find relevant information."
            )

        # 2) Genereer
        prompt = build_prompt(question, context_chunks)
        max_prompt_tokens = MAX_MODEL_LEN - self.max_new_tokens
        if max_prompt_tokens < 1:
            raise ValueError("`max_new_tokens` overschrijdt de model-context!")

        # ────────── Hier de ingreep: batch = gen_tok(...); batch naar DEVICE ──────────
        batch = gen_tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        out = gen_mod.generate(
            **batch,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_beams=1,
            pad_token_id=gen_tok.eos_token_id,
        )
        # ────────────────────────────────────────────────────────────────────────────

        gen_ids = out[0][batch["input_ids"].shape[-1]:]
        answer = gen_tok.decode(gen_ids, skip_special_tokens=True).strip()

        # schoon kleine "(100 words)"-restjes uit context
        answer = re.sub(r"\([^)]*words?[^)]*\)", "", answer, flags=re.I).strip()

        if not answer:
            # Fallback‐sectie: vangen we een lijst met bronnen/snippets
            lines: List[str] = []
            if lang.lower().startswith("nl"):
                lines.append("Sorry, ik kon geen antwoord genereren. Hier zijn wel enkele relevante stukken:\n")
            else:
                lines.append("Sorry, I couldn't generate an answer. Here are some relevant sources:\n")

            for i, chunk in enumerate(context_chunks, start=1):
                if isinstance(chunk, dict):
                    source = chunk.get("file_name", "Unknown source")
                    cid = chunk.get("chunk_id", "?")
                    text = chunk.get("text", "")
                else:
                    source = "Unknown source"
                    cid = "?"
                    text = str(chunk)

                snippet = text.replace("\n", " ")[:150].rstrip() + "…"
                lines.append(f"{i}. {source} (chunk {cid}): {snippet}")

            return "\n".join(lines)

        # 3) Vertaal indien NL gevraagd
        if lang.lower().startswith("nl"):
            answer = en2nl(answer)

        return answer