"""Collect answers for a manual faithfulness audit, then summarise the labels.

Step 1, run questions through the full pipeline (needs Ollama) and write a CSV
with the answer, the pages it cites and the pages retrieved:

    python scripts/answer_eval.py run data/argo_user_manual.pdf eval/argo_user_manual.json \
        --sample 10 --out eval/answer_eval.csv
    python scripts/answer_eval.py run data/other.pdf eval/other.json --sample 10 \
        --out eval/answer_eval.csv --append
    python scripts/answer_eval.py run data/argo_user_manual.pdf eval/unanswerable.json \
        --out eval/answer_eval.csv --append

A question set may mark questions with "answerable": false (the answer is not
in the PDF). Those rows test whether the model refuses instead of inventing.

Step 2, label by hand in a spreadsheet: supported (yes/partly/no),
citation_correct (yes/no) and, for unanswerable rows, refused (yes/no).

Step 3, count the labels:

    python scripts/answer_eval.py summarize eval/answer_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from chat_with_documents.chunking import chunk_pages
from chat_with_documents.config import RETRIEVAL_MODES, Settings
from chat_with_documents.embeddings import SentenceTransformerEmbedder
from chat_with_documents.evaluation import ANSWER_AUDIT_COLUMNS, cited_numbers, summarize_audit
from chat_with_documents.hybrid import Retriever
from chat_with_documents.index import VectorIndex
from chat_with_documents.llm import OllamaClient
from chat_with_documents.loader import load_pdf
from chat_with_documents.rag import answer


def _pages(hits) -> str:
    return "; ".join(str(h.chunk.page_number) for h in hits)


def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data["questions"]
    if args.sample and args.sample < len(questions):
        questions = random.Random(args.seed).sample(questions, args.sample)

    pages = load_pdf(args.pdf)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    index = VectorIndex.build(chunks, SentenceTransformerEmbedder(settings.embedding_model))
    retriever = Retriever(index, args.mode or settings.retrieval_mode)
    llm = OllamaClient(settings.ollama_host)
    model = args.model or settings.ollama_model
    document = Path(args.pdf).name

    out = Path(args.out)
    new_file = not (args.append and out.exists())
    with out.open("w" if new_file else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANSWER_AUDIT_COLUMNS)
        if new_file:
            writer.writeheader()
        for number, q in enumerate(questions, start=1):
            print(f"[{number}/{len(questions)}] {q['question']}", flush=True)
            # Each question stands alone: no chat history.
            result = answer(q["question"], retriever, llm, model, top_k=settings.top_k)
            numbers = [n for n in cited_numbers(result.text) if 0 < n <= len(result.sources)]
            cited = [result.sources[n - 1] for n in numbers]
            writer.writerow(
                {
                    "document": document,
                    "question": q["question"],
                    "answerable": "no" if q.get("answerable") is False else "yes",
                    "answer": result.text.strip(),
                    "cited_pages": _pages(cited),
                    "retrieved_pages": _pages(result.sources),
                }
            )
            handle.flush()
    print(f"Wrote {len(questions)} rows to {out}. Now label them by hand.")
    return 0


def summarize(args: argparse.Namespace) -> int:
    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {
        "faithful": "Faithfulness (supported = yes)",
        "citation_correct": "Citation accuracy",
        "refused_unanswerable": "Refused unanswerable questions",
    }
    for key, (hits, total) in summarize_audit(rows).items():
        print(f"{labels[key]}: {hits}/{total}" if total else f"{labels[key]}: no labels yet")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="answer questions and write an unlabelled CSV")
    p_run.add_argument("pdf")
    p_run.add_argument("questions", help="question set JSON (same format as eval/*.json)")
    p_run.add_argument("--out", default="eval/answer_eval.csv")
    p_run.add_argument("--append", action="store_true", help="add rows to an existing CSV")
    p_run.add_argument("--sample", type=int, help="pick this many questions at random")
    p_run.add_argument("--seed", type=int, default=0, help="random seed for --sample")
    p_run.add_argument("--mode", choices=RETRIEVAL_MODES)
    p_run.add_argument("--model", help="Ollama model (default: OLLAMA_MODEL or llama3.1)")
    p_run.set_defaults(func=run)

    p_sum = sub.add_parser("summarize", help="count the hand-written labels")
    p_sum.add_argument("csv")
    p_sum.set_defaults(func=summarize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
