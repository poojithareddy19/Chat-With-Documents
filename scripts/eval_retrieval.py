"""Measure page-level retrieval quality against hand-written question sets.

Example:
    python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json
    python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json \
        --mode all --k 1 3 4 5 --frozen --verbose
    python scripts/eval_retrieval.py a.pdf eval/a.json b.pdf eval/b.json --mode all

Arguments come in PDF / question-set pairs. Each PDF gets its own index (the
relevant pages in a question set refer to that one document); the combined
row pools every question from every document.

No Ollama needed: this exercises the embedder and the retrievers only.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

from chat_with_documents.chunking import chunk_pages
from chat_with_documents.config import RETRIEVAL_MODES, Settings
from chat_with_documents.embeddings import SentenceTransformerEmbedder
from chat_with_documents.evaluation import EvalReport, evaluate, load_questions
from chat_with_documents.hybrid import Retriever
from chat_with_documents.index import VectorIndex
from chat_with_documents.loader import load_pdf


def _table(rows: List[Tuple[str, str, EvalReport]], ks: List[int]) -> str:
    head = "| Document | Mode | Questions | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR |"
    lines = [head, "| " + " | ".join(["---"] * (len(ks) + 4)) + " |"]
    for document, mode, report in rows:
        recalls = " | ".join(f"{report.recall_at_k[k]:.2f}" for k in ks)
        lines.append(
            f"| {document} | {mode} | {len(report.results)} | {recalls} | {report.mrr:.2f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pairs", nargs="+", metavar="PDF QUESTIONS")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=[*RETRIEVAL_MODES, "all"],
        default=["dense"],
        help="retrieval mode(s) to evaluate (default: dense)",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="ignore pages added after reviewing misses (the pre-adjudication answer key)",
    )
    parser.add_argument("--verbose", action="store_true", help="list the misses at the largest k")
    args = parser.parse_args(argv)
    if len(args.pairs) % 2:
        parser.error("arguments must come in PDF QUESTIONS pairs")

    settings = Settings.from_env()
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap
    model_name = args.embedding_model or settings.embedding_model
    modes = list(RETRIEVAL_MODES) if "all" in args.mode else list(dict.fromkeys(args.mode))
    ks = sorted(set(args.k))

    embedder = SentenceTransformerEmbedder(model_name)
    print(f"Embedding model: {model_name}")
    print(f"Chunking: size {chunk_size}, overlap {chunk_overlap}")
    print(f"Answer key: {'frozen (before adjudication)' if args.frozen else 'adjudicated'}\n")

    reports: Dict[Tuple[str, str], EvalReport] = {}
    documents = []
    for pdf, questions_path in zip(args.pairs[::2], args.pairs[1::2]):
        pages = load_pdf(pdf)
        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
        started = time.perf_counter()
        index = VectorIndex.build(chunks, embedder)
        seconds = time.perf_counter() - started
        questions = load_questions(questions_path, frozen=args.frozen)
        name = pdf.replace("\\", "/").rsplit("/", 1)[-1]
        documents.append(name)
        print(
            f"{name}: {len(pages)} pages with text, {len(chunks)} chunks, "
            f"indexed in {seconds:.1f} s, {len(questions)} questions"
        )
        for mode in modes:
            reports[name, mode] = evaluate(Retriever(index, mode), questions, ks=ks)

    rows = [(d, m, reports[d, m]) for d in documents for m in modes]
    if len(documents) > 1:
        for mode in modes:
            pooled = [r for d in documents for r in reports[d, mode].results]
            rows.append(("**combined**", mode, EvalReport(pooled, ks)))
    print()
    print(_table(rows, ks))

    if args.verbose:
        k = max(ks)
        for document, mode, report in rows:
            if document == "**combined**":
                continue
            misses = report.misses(k)
            print(f"\n{document}, {mode}: misses at k={k}: {len(misses)}")
            for r in misses:
                got = ", ".join(f"p.{h.chunk.page_number}" for h in r.hits[:k])
                print(f"- {r.question.question}")
                print(f"    wanted pages {r.question.relevant_pages}, got {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
