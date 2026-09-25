"""Streamlit UI for Chat with Documents.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from typing import List, Sequence

import streamlit as st
from dotenv import load_dotenv

from chat_with_documents import (
    Hit,
    OllamaClient,
    OllamaError,
    SentenceTransformerEmbedder,
    Settings,
    VectorIndex,
    answer_stream,
    chunk_pages,
    load_pdfs,
)

load_dotenv()
SETTINGS = Settings.from_env()
SNIPPET_CHARS = 300


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)


@st.cache_resource
def get_llm(host: str) -> OllamaClient:
    return OllamaClient(host)


def render_sources(hits: Sequence[Hit]) -> None:
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})"):
        for number, hit in enumerate(hits, start=1):
            snippet = hit.chunk.text[:SNIPPET_CHARS].replace("\n", " ")
            if len(hit.chunk.text) > SNIPPET_CHARS:
                snippet += "..."
            st.markdown(f"**[{number}] {hit.chunk.label}** (similarity {hit.score:.2f})")
            st.caption(snippet)


def process_uploads(uploads: List) -> None:
    pages = load_pdfs(uploads)
    chunks = chunk_pages(pages, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    st.session_state.index = VectorIndex.build(chunks, get_embedder(SETTINGS.embedding_model))
    st.session_state.messages = []
    st.session_state.summary = (
        f"{len(uploads)} file(s), {len(pages)} page(s) with text, {len(chunks)} chunk(s)"
    )
    unreadable = sorted({u.name for u in uploads} - {p.source for p in pages})
    if unreadable:
        st.warning(
            "No text could be extracted from: "
            + ", ".join(unreadable)
            + ". Scanned PDFs need OCR before they can be searched."
        )


def sidebar() -> str:
    """Render the sidebar and return the selected model name (or empty)."""
    st.subheader("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    llm = get_llm(SETTINGS.ollama_host)
    model = ""
    try:
        models = llm.list_models()
    except OllamaError as exc:
        models = []
        st.error(str(exc))
    if models:
        # Ollama reports "llama3.1:latest" for a model pulled as "llama3.1".
        preferred = {SETTINGS.ollama_model, f"{SETTINGS.ollama_model}:latest"}
        default = next((i for i, name in enumerate(models) if name in preferred), 0)
        model = st.selectbox("Ollama model", models, index=default)

    if st.button("Process", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing..."):
            process_uploads(uploads)
    if "summary" in st.session_state:
        st.caption("Indexed: " + st.session_state.summary)
    return model


def main() -> None:
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("index", None)

    with st.sidebar:
        model = sidebar()

    st.title("Chat with Documents")
    st.caption(
        "Answers come only from your PDFs, with file and page citations. "
        "Everything runs locally through Ollama."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    if st.session_state.index is None:
        st.warning("Upload PDFs and click Process first.")
        return
    if not model:
        st.error("No Ollama model is available. Start Ollama and pull a model.")
        return

    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("assistant"):
        try:
            hits, stream = answer_stream(
                question,
                st.session_state.index,
                get_llm(SETTINGS.ollama_host),
                model,
                history=history,
                top_k=SETTINGS.top_k,
                history_turns=SETTINGS.history_turns,
            )
            text = st.write_stream(stream)
        except OllamaError as exc:
            st.error(str(exc))
            return
        render_sources(hits)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": hits})


if __name__ == "__main__":
    main()
