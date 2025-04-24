"""
Eenvoudige REPL voor snelle tests.

Voorbeeld:
$ python -m src.cli
User: What is REPowerEU?
Assistant: REPowerEU is ...
"""

from __future__ import annotations

import argparse
import readline  # pijltjes/up-history

from .rag import RAGChat


def main() -> None:
    ap = argparse.ArgumentParser(description="Open een RAG-chat in de terminal.")
    ap.add_argument("--top-k", type=int, default=5, help="Aantal context-chunks (default: 5)")
    args = ap.parse_args()

    chat = RAGChat(top_k=args.top_k)
    print("💬  Type 'exit' of Ctrl-C om te stoppen.\n")

    try:
        while True:
            question = input("User: ").strip()
            if question.lower() in {"exit", "quit", "stop"}:
                break
            answer = chat.answer(question)
            print(f"Assistant: {answer}\n")
    except (EOFError, KeyboardInterrupt):
        pass

    print("Chat afgesloten.")


if __name__ == "__main__":  # pragma: no cover
    main()