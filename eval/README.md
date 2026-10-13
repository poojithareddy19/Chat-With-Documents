# Evaluation data

| File | What it is |
| --- | --- |
| `argo_user_manual.json` | 30 retrieval questions over the Argo User's Manual, frozen before any retrieval run |
| `answer_eval.csv` | Manual answer audit: one row per answer, labelled by hand (see `scripts/answer_eval.py`) |

## Question set format

```json
{
  "document": "file_name.pdf",
  "source_url": "https://...",
  "version": "title, version and date of the exact edition used",
  "method": "How the questions were written: which pages were read, in whose words, and that it was done before any retrieval run.",
  "questions": [
    {"question": "In your own words?", "relevant_pages": [12]},
    {"question": "A question added to after review?", "relevant_pages": [12, 40], "added_after_review": [40]},
    {"question": "Something the PDF does not answer?", "relevant_pages": [], "answerable": false}
  ]
}
```

- `relevant_pages` are 1-based, as shown in a PDF viewer.
- `added_after_review` lists pages added when misses were adjudicated after a run. `scripts/eval_retrieval.py --frozen` ignores them, which reproduces the answer key as it was first committed.
- `answerable: false` is only for the answer audit (`scripts/answer_eval.py`). Keep unanswerable questions in their own file, since the retrieval evaluation has no page to look for.

## Adding a new document

1. Pick a public PDF and record its URL, version and date.
2. Read part of it and write the questions and their pages **before** running any retrieval. Mix exact identifiers (section numbers, defined terms, codes), paraphrases and questions answered by a table.
3. Commit the JSON on its own, before the first evaluation run. That commit is the proof that the set was frozen first.
4. Then run all three modes, with both documents in one command so the combined row is computed:

   ```bash
   python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json data/<new>.pdf eval/<new>.json --mode all --k 1 3 4 5 --frozen --verbose
   ```
