# Retrieval evaluation

The unit tests prove the retriever returns the right shape. This page measures whether it returns the right pages, on a real document, with numbers that were produced the honest way.

## Setup

| Item | Value |
| --- | --- |
| Date | 25 September 2026 (dense baseline); 13 October 2026 (recall@4, BM25 and hybrid) |
| Document | Argo User's Manual v3.44.0 (July 2025), 99 pages, [DOI 10.13155/29825](https://doi.org/10.13155/29825), [PDF](https://archimer.ifremer.fr/doc/00187/29825/120885.pdf) |
| Question set | [eval/argo_user_manual.json](../eval/argo_user_manual.json), 30 questions |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval modes | dense (FAISS cosine), BM25 (`rank-bm25`, Okapi), hybrid (reciprocal rank fusion of the two) |
| Metric | Page-level: a question is a hit at rank r if the r-th retrieved chunk comes from one of its relevant pages. Recall@k is the share of questions with a hit in the top k. MRR is the mean of 1/rank of the first hit. |
| Hardware | Laptop CPU, no GPU |

Reproduce with:

```bash
python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json --mode all --k 1 3 4 5 --frozen --verbose
python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json --mode all --k 1 3 4 5 --verbose
```

`--frozen` ignores the pages marked `added_after_review` in the question set, which gives the answer key exactly as it was before any retrieval run.

## How the question set was made

1. The manual's text was dumped page by page and read directly (pages 14 to 31, 58 to 63 and 80 to 99, roughly half the document).
2. Thirty questions were written in the author's own words about facts on those pages, each with the page numbers where the fact appears. This was done before any retrieval was run, so the retriever's output could not influence the questions.
3. The evaluation was run once with this frozen set.
4. Each miss was then checked against the pages the retriever returned. Two misses turned out to be hits on pages the author had not read: the CF conventions version also appears on page 48, and the Julian reference date follows from the JULD units on pages 21 and 35. Those pages were added. The other four misses were confirmed as genuine and left as they were.

Step 4 only ever adds pages that verifiably answer the question, so it corrects the ground truth without letting the retriever grade itself. Both scores are reported so the effect of the correction is visible. Pages that were never read may still hold valid answers, which makes even the adjudicated numbers conservative.

## Results

The app shows the model `TOP_K = 4` passages, so **recall@4 is the number that describes the app**. Recall@5 is kept because it was the originally reported metric.

Dense retrieval, default settings (chunk size 800, overlap 150, 488 chunks):

| Metric | Frozen ground truth | After adjudication |
| --- | --- | --- |
| Recall@1 | 0.57 | 0.63 |
| Recall@3 | 0.73 | 0.80 |
| Recall@4 | 0.77 | 0.83 |
| Recall@5 | 0.80 | 0.87 |
| MRR | 0.66 | 0.73 |

Smaller chunks (size 400, overlap 80, 1010 chunks), as a sensitivity check:

| Metric | Frozen ground truth | After adjudication |
| --- | --- | --- |
| Recall@1 | 0.53 | 0.53 |
| Recall@3 | 0.77 | 0.77 |
| Recall@5 | 0.83 | 0.83 |
| MRR | 0.65 | 0.65 |

Smaller chunks trade a little precision at rank 1 for a little more recall at rank 5, and double the index size and build time. The adjudication did not change their scores because the two added pages were not among their top five either. The default stays at 800/150.

## The four genuine misses (dense)

| Question | Wanted page | What came back | Why | Fixed by hybrid? |
| --- | --- | --- | --- | --- |
| What information does cycle 0 contain? | 16 | 38, 36, 33 | Those pages define cycle 0 as "launch cycle" but do not say what it contains. Vocabulary overlap beat meaning. | Yes |
| Why was NetCDF chosen as the Argo file format? | 15 | 17, 14 | Page 17 describes NetCDF at length; page 15 gives the three reasons in a short bullet list that the embedding does not single out. | No |
| Which POSITION_QC flag value tells a user that a position was changed? | 96 | 27, 23, 21 | Exact-identifier lookup. The dense embedding matches other pages full of `*_QC` variable definitions. | Yes |
| What is the glossary definition of a float? | 99 | 53, 52, 31 | The glossary says "autonomous platform", never "float definition". Pure paraphrase gap. | Yes |

Two of the four are the classic dense-retrieval weakness with exact tokens and short factual lines. That made BM25 fused with the dense ranking the obvious next experiment, measured below against the same frozen question set.

## Dense vs BM25 vs hybrid

Hybrid mode takes the top 20 chunks from dense search and the top 20 from BM25 and fuses them with reciprocal rank fusion: each chunk scores the sum of `1 / (60 + rank)` over the two lists, and the top k are kept. RRF uses ranks only, so the cosine scores (in [-1, 1]) and BM25 scores (unbounded) never need to be put on a common scale. BM25 tokenises on letters, digits and underscores, so `POSITION_QC` stays one token. Both 20 and 60 are standard defaults, not values tuned on these questions.

Frozen ground truth (the headline numbers):

| Mode | Recall@1 | Recall@3 | Recall@4 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Dense (baseline) | 0.57 | 0.73 | 0.77 | 0.80 | 0.66 |
| BM25 only | 0.53 | 0.77 | 0.83 | 0.87 | 0.67 |
| Hybrid (RRF) | **0.70** | **0.90** | **0.93** | **0.93** | **0.79** |

After adjudication:

| Mode | Recall@1 | Recall@3 | Recall@4 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Dense (baseline) | 0.63 | 0.80 | 0.83 | 0.87 | 0.73 |
| BM25 only | 0.57 | 0.77 | 0.83 | 0.87 | 0.68 |
| Hybrid (RRF) | **0.77** | **0.93** | **0.97** | **0.97** | **0.84** |

What happened:

- **The two retrievers fail on different questions.** Dense misses exact identifiers and short lines. BM25 alone misses paraphrases: "How should I cite Argo data in a publication?", "Where is the reason a float stopped working recorded?" and "What is the file name pattern for a delayed-mode trajectory file?" all have answers that use different words from the question. Because the failures barely overlap, fusing the rankings fixes most of both.
- **Hybrid fixes three of the four genuine dense misses** (cycle 0, POSITION_QC, the glossary definition of a float). The NetCDF question is still missed by every mode: page 15 lists the reasons in a short bullet list, while page 17 matches the question's words better for both retrievers.
- With the frozen key, hybrid also misses the CF conventions question, because it returns page 48. That is the page the adjudication later added, so it is a correct answer the frozen key does not credit.

How much to trust this:

- 30 questions means one question is 0.033. Hybrid's recall@4 gain on the frozen key, 0.77 to 0.93, is five questions.
- The idea to try hybrid came from reading misses on this same question set, even though no parameter was tuned on it. A second document with its own frozen question set is the real test, and the app's default stays `dense` until that result is in. Set `RETRIEVAL_MODE=hybrid` to use hybrid now.
- Cost: BM25 indexing of 488 chunks is instant, and hybrid search scores every chunk densely once per query, which a flat FAISS index does anyway.

## Limits of this evaluation

- Thirty questions from one document, written by one person who also built the system. Good enough to catch regressions and compare settings, not a benchmark. See [eval/README.md](../eval/README.md) for adding a second document.
- Page-level matching is generous: a chunk from the right page might still miss the exact sentence.
- The metric ignores answer generation entirely. See [verification.md](verification.md) for one end-to-end run through the language model.
