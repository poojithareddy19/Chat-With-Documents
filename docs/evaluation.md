# Retrieval evaluation

The unit tests prove the retriever returns the right shape. This page measures whether it returns the right pages, on a real document, with numbers that were produced the honest way.

## Setup

| Item | Value |
| --- | --- |
| Date | 25 September 2026 |
| Document | Argo User's Manual v3.44.0 (July 2025), 99 pages, [DOI 10.13155/29825](https://doi.org/10.13155/29825), [PDF](https://archimer.ifremer.fr/doc/00187/29825/120885.pdf) |
| Question set | [eval/argo_user_manual.json](../eval/argo_user_manual.json), 30 questions |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Metric | Page-level: a question is a hit at rank r if the r-th retrieved chunk comes from one of its relevant pages. Recall@k is the share of questions with a hit in the top k. MRR is the mean of 1/rank of the first hit. |
| Hardware | Laptop CPU, no GPU |

Reproduce with:

```bash
python scripts/eval_retrieval.py data/argo_user_manual.pdf eval/argo_user_manual.json --verbose
```

## How the question set was made

1. The manual's text was dumped page by page and read directly (pages 14 to 31, 58 to 63 and 80 to 99, roughly half the document).
2. Thirty questions were written in the author's own words about facts on those pages, each with the page numbers where the fact appears. This was done before any retrieval was run, so the retriever's output could not influence the questions.
3. The evaluation was run once with this frozen set.
4. Each miss was then checked against the pages the retriever returned. Two misses turned out to be hits on pages the author had not read: the CF conventions version also appears on page 48, and the Julian reference date follows from the JULD units on pages 21 and 35. Those pages were added. The other four misses were confirmed as genuine and left as they were.

Step 4 only ever adds pages that verifiably answer the question, so it corrects the ground truth without letting the retriever grade itself. Both scores are reported so the effect of the correction is visible. Pages that were never read may still hold valid answers, which makes even the adjudicated numbers conservative.

## Results

Default settings (chunk size 800, overlap 150, 488 chunks):

| Metric | Frozen ground truth | After adjudication |
| --- | --- | --- |
| Recall@1 | 0.57 | 0.63 |
| Recall@3 | 0.73 | 0.80 |
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

## The four genuine misses

| Question | Wanted page | What came back | Why |
| --- | --- | --- | --- |
| What information does cycle 0 contain? | 16 | 38, 36, 33 | Those pages define cycle 0 as "launch cycle" but do not say what it contains. Vocabulary overlap beat meaning. |
| Why was NetCDF chosen as the Argo file format? | 15 | 17, 14 | Page 17 describes NetCDF at length; page 15 gives the three reasons in a short bullet list that the embedding does not single out. |
| Which POSITION_QC flag value tells a user that a position was changed? | 96 | 27, 23, 21 | Exact-identifier lookup. The dense embedding matches other pages full of `*_QC` variable definitions. |
| What is the glossary definition of a float? | 99 | 53, 52, 31 | The glossary says "autonomous platform", never "float definition". Pure paraphrase gap. |

Two of the four are the classic dense-retrieval weakness with exact tokens and short factual lines. A BM25 component fused with the dense scores is the obvious next experiment, and this question set is what it should be measured against.

## Limits of this evaluation

- Thirty questions from one document, written by one person who also built the system. Good enough to catch regressions and compare settings, not a benchmark.
- Page-level matching is generous: a chunk from the right page might still miss the exact sentence.
- The metric ignores answer generation entirely. See [verification.md](verification.md) for one end-to-end run through the language model.
