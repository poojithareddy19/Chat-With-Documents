"""Chat with Documents: local retrieval-augmented question answering over PDFs.

Pipeline: PDF pages -> page-bounded chunks -> sentence-transformer embeddings
-> FAISS cosine index, optionally fused with BM25 -> Ollama chat completion
with numbered citations.
"""

from chat_with_documents.chunking import Chunk, chunk_pages, split_text
from chat_with_documents.config import Settings
from chat_with_documents.embeddings import Embedder, SentenceTransformerEmbedder
from chat_with_documents.hybrid import BM25Index, Retriever, reciprocal_rank_fusion
from chat_with_documents.index import Hit, VectorIndex
from chat_with_documents.llm import OllamaClient, OllamaError
from chat_with_documents.loader import Page, load_pdf, load_pdfs
from chat_with_documents.rag import (
    Answer,
    answer,
    answer_stream,
    build_messages,
    flag_injections,
)

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "Embedder",
    "Hit",
    "OllamaClient",
    "OllamaError",
    "Retriever",
    "Page",
    "SentenceTransformerEmbedder",
    "Settings",
    "VectorIndex",
    "answer",
    "answer_stream",
    "build_messages",
    "chunk_pages",
    "flag_injections",
    "load_pdf",
    "load_pdfs",
    "reciprocal_rank_fusion",
    "split_text",
]
