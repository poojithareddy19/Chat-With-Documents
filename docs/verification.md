# End-to-end verification

Unit tests prove the plumbing. This page records one real run so the README's claims can be checked against actual output rather than taken on trust.

## Setup

| Item | Value |
| --- | --- |
| Date | 25 September 2026 |
| Machine | Windows 11 laptop, 15 GB RAM, no GPU, about 5 GB RAM free at the time |
| Python | 3.9.13 |
| Ollama model | `llama3.1:latest` (8 B, Q4, 100% CPU) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Document | Argo user manual, 99 pages, 1.5 MB (a public oceanography instrument manual) |
| Settings | defaults: chunk 800/150, top-k 4 |

## Command

```bash
python -m chat_with_documents.cli argo_user_manual.pdf --model llama3.1 \
  -q "What is an Argo float and what does it measure?" \
  -q "How deep do they typically go?"
```

## Output (verbatim)

```
Indexed 99 pages into 488 chunks.

Q: What is an Argo float and what does it measure?
A: An Argo float is an autonomous platform deployed in the sea that performs environmental monitoring [1].
According to passage [1], an Argo float measures a physical parameter using a sensor, and the output is provided in parameter counts that need to be converted into physical units using a calibration equation [1].

  [1] argo_user_manual.pdf, p.99 (score 0.61)
  [2] argo_user_manual.pdf, p.15 (score 0.57)
  [3] argo_user_manual.pdf, p.59 (score 0.56)
  [4] argo_user_manual.pdf, p.28 (score 0.53)

Q: How deep do they typically go?
A: The documents don't contain specific information about the typical depth range of Argo floats. However, passage [1] lists a "Deepest sampling" category with a 1000 decibars slice thickness, but it doesn't provide information about typical or maximum depths.

  [1] argo_user_manual.pdf, p.84 (score 0.33)
  [2] argo_user_manual.pdf, p.41 (score 0.22)
  [3] argo_user_manual.pdf, p.40 (score 0.21)
  [4] argo_user_manual.pdf, p.19 (score 0.21)
```

Wall-clock time for the whole command, including loading both models, embedding 488 chunks and generating two answers on CPU: 4 min 15 s.

## What this shows

- Citations point at real pages of the source file; the UI shows the same passages under each answer.
- The second question is a follow-up ("they") and was understood through the chat history.
- When retrieval is weak (top score 0.33 instead of 0.61), the model followed the system prompt and said the documents do not contain the answer instead of guessing. That is the intended behaviour, not a failure.
- CPU-only generation with an 8 B model is slow. A smaller model, or a GPU, is the lever if latency matters.
