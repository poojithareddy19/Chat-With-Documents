# Chat with Documents

Ask questions about your own PDFs and get answers that cite the file and page they came from. Everything runs on your machine: PDF parsing, embeddings, vector search and the language model (through [Ollama](https://ollama.com)). No API keys, no data leaves the laptop.

[![CI](https://github.com/poojithareddy19/Chat-With-Documents/actions/workflows/ci.yml/badge.svg)](https://github.com/poojithareddy19/Chat-With-Documents/actions/workflows/ci.yml)

## What it does

1. Extracts text from each uploaded PDF page by page (`pypdf`).
2. Splits every page into overlapping chunks that never cross a page boundary, so each chunk has an exact `file, p.N` label.
3. Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a FAISS cosine-similarity index (in memory).
4. For each question, retrieves the top-k chunks, builds a prompt with numbered passages, and streams the answer from an Ollama model. The model is instructed to answer only from the passages and to cite them as `[1]`, `[2]`.
5. Shows the answer with an expandable list of the passages used, including similarity scores.

Follow-up questions work: the last few turns are sent back to the model as chat history.

## Quick start

Prerequisites: Python 3.9+, and Ollama running with at least one chat model.

```bash
ollama pull llama3.1
```

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install --no-deps -e .
streamlit run app.py
```

Upload one or more PDFs in the sidebar, pick a model, click **Process**, then ask away. The first run downloads the embedding model (about 90 MB).

### Command line

Useful for scripted checks or when you do not want the UI:

```bash
python -m chat_with_documents.cli manual.pdf -q "What is this document about?" -q "Summarise section 2"
python -m chat_with_documents.cli --list-models
```

### Configuration

All settings are optional environment variables (or a `.env` file, see [.env.example](.env.example)):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.1` | Model preselected in the UI and used by the CLI |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model |
| `CHUNK_SIZE` | `800` | Chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Passages retrieved per question |
| `HISTORY_TURNS` | `4` | Previous question/answer pairs sent to the model |

## Project layout

```
app.py                         Streamlit UI (thin; all logic lives in the package)
src/chat_with_documents/
  loader.py                    PDF -> Page(source, page_number, text)
  chunking.py                  Page -> Chunk, page-bounded overlapping windows
  embeddings.py                Embedder protocol + sentence-transformers backend
  index.py                     VectorIndex over FAISS IndexFlatIP, returns Hit(chunk, score)
  llm.py                       OllamaClient: list models, streaming chat, clear errors
  rag.py                       Prompt assembly, history trimming, answer / answer_stream
  config.py                    Settings.from_env()
  cli.py                       python -m chat_with_documents.cli
  evaluation.py                page-level recall@k and MRR over a question set
eval/argo_user_manual.json     30 hand-written questions with their answer pages
scripts/eval_retrieval.py      runs the evaluation against a PDF
tests/                         29 unit tests, no model download or Ollama needed
.github/workflows/ci.yml       ruff + pytest on Python 3.9 and 3.12
```

## Testing

```bash
pip install -r requirements-ci.txt
pip install --no-deps -e .
ruff check . && ruff format --check . && pytest
```

The unit tests replace the two heavy pieces with fakes: a deterministic bag-of-words embedder (so retrieval ranking is still meaningful) and a fake HTTP session for the Ollama client. They also build small PDFs by hand instead of shipping binary fixtures. That keeps CI under a minute and torch-free.

What the tests do not cover is answer quality, which depends on the model. See the next section for a real run.

## Measured

**Retrieval quality.** Thirty hand-written questions over a public 99-page manual, each tagged with the pages that answer it, written before any retrieval was run. With default settings the right page is the top hit 63% of the time and within the top five 87% of the time (MRR 0.73). Method, both pre- and post-adjudication scores, and an analysis of every miss are in [docs/evaluation.md](docs/evaluation.md). Reproduce with:

```bash
python scripts/eval_retrieval.py path/to/argo_user_manual.pdf eval/argo_user_manual.json --verbose
```

**End to end.** [docs/verification.md](docs/verification.md) has the verbatim output of a real run on this machine (an 8 B model on CPU, no GPU), including the citations returned and a case where the model correctly declined to answer.

## Design decisions

- **No LangChain.** The original version of this project used LangChain's `ConversationalRetrievalChain`. Every import it relied on is now deprecated, and the chain hid the prompt, the retrieval scores and the citation logic. The replacement is about 250 lines of plain Python with three well-known libraries, and every step is unit-testable.
- **Chunks never cross pages.** Splitting the concatenated text of all files (the common tutorial approach) loses the page number. Page-bounded chunks make citations exact at the cost of slightly more chunks.
- **Cosine similarity via inner product.** Embeddings are L2-normalised, so `IndexFlatIP` gives cosine scores in `[-1, 1]` that are meaningful to show in the UI.
- **Retrieve first, then stream.** Sources are known before the first token arrives, so the UI can show them even if generation is interrupted.
- **In-memory index.** Indexing a few hundred pages takes seconds on CPU, and persisting the index would raise questions about invalidation when files change. Persistence is an easy addition (`faiss.write_index`) if a use case needs it.

## Limitations

- Scanned PDFs have no extractable text. The app tells you which files were skipped, but does not run OCR.
- One index per browser session; there is no multi-user state or authentication.
- Answer quality is bounded by the local model. Small quantised models occasionally ignore the citation instruction.
- Retrieval is pure dense search. A hybrid with BM25 would help for exact identifiers and part numbers.

## License

MIT
