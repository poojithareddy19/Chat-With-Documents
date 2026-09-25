"""Measure page-level retrieval quality against a hand-written question set.

Example:
    python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json
    python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json \
        --chunk-size 400 --chunk-overlap 80 --verbose

No Ollama needed: this exercises the embedder and the index only.
"""

from __future__ import annotations

import argparse
import sys
import time

from chat_with_documents.chunking import chunk_pages
from chat_with_documents.config import Settings
from chat_with_documents.embeddings import SentenceTransformerEmbedder
from chat_with_documents.evaluation import evaluate, load_questions
from chat_with_documents.index import VectorIndex
from chat_with_documents.loader import load_pdf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf")
    parser.add_argument("questions")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--verbose", action="store_true", help="list the misses at the largest k")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap
    model_name = args.embedding_model or settings.embedding_model

    pages = load_pdf(args.pdf)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    started = time.perf_counter()
    index = VectorIndex.build(chunks, SentenceTransformerEmbedder(model_name))
    index_seconds = time.perf_counter() - started

    questions = load_questions(args.questions)
    report = evaluate(index, questions, ks=sorted(args.k))

    print(f"Document: {args.pdf} ({len(pages)} pages with text)")
    print(f"Embedding model: {model_name}")
    print(f"Chunking: size {chunk_size}, overlap {chunk_overlap} -> {len(chunks)} chunks")
    print(f"Index build time: {index_seconds:.1f} s")
    print(f"Questions: {len(questions)}\n")
    print(report.to_markdown())

    if args.verbose:
        k = max(args.k)
        misses = report.misses(k)
        print(f"Misses at k={k}: {len(misses)}")
        for r in misses:
            got = ", ".join(f"p.{h.chunk.page_number}" for h in r.hits[:k])
            print(f"- {r.question.question}")
            print(f"    wanted pages {r.question.relevant_pages}, got {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
