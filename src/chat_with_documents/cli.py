"""Command-line entry point, mainly for scripted checks without the UI.

Example:
    python -m chat_with_documents.cli paper.pdf -q "What is the main result?"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from chat_with_documents.chunking import chunk_pages
from chat_with_documents.config import Settings
from chat_with_documents.embeddings import SentenceTransformerEmbedder
from chat_with_documents.index import VectorIndex
from chat_with_documents.llm import OllamaClient, OllamaError
from chat_with_documents.loader import load_pdfs
from chat_with_documents.rag import answer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about PDF files.")
    parser.add_argument("pdfs", nargs="*", help="PDF files to index")
    parser.add_argument("-q", "--question", action="append", help="question (repeatable)")
    parser.add_argument("--model", help="Ollama model name (default: OLLAMA_MODEL or llama3.1)")
    parser.add_argument("--top-k", type=int, help="number of passages to retrieve")
    parser.add_argument("--list-models", action="store_true", help="list Ollama models and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    llm = OllamaClient(settings.ollama_host)

    if args.list_models:
        try:
            print("\n".join(llm.list_models()))
        except OllamaError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    if not args.pdfs or not args.question:
        _parser().print_usage(sys.stderr)
        print("error: at least one PDF and one --question are required", file=sys.stderr)
        return 2

    pages = load_pdfs(args.pdfs)
    if not pages:
        print("error: no extractable text found in the given PDFs", file=sys.stderr)
        return 1
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    index = VectorIndex.build(chunks, SentenceTransformerEmbedder(settings.embedding_model))
    print(f"Indexed {len(pages)} pages into {len(chunks)} chunks.\n")

    model = args.model or settings.ollama_model
    top_k = args.top_k or settings.top_k
    history: List[dict] = []
    for question in args.question:
        print(f"Q: {question}")
        try:
            result = answer(question, index, llm, model, history, top_k, settings.history_turns)
        except OllamaError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"A: {result.text.strip()}\n")
        for number, hit in enumerate(result.sources, start=1):
            print(f"  [{number}] {hit.chunk.label} (score {hit.score:.2f})")
        print()
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.text},
        ]
    return 0


if __name__ == "__main__":
    sys.exit(main())
