"""Retrieval-augmented answering with numbered citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

from chat_with_documents.index import Hit, VectorIndex
from chat_with_documents.llm import OllamaClient

SYSTEM_PROMPT = (
    "You answer questions about the user's documents.\n"
    "Use only the numbered context passages below. If the answer is not in the "
    "passages, say that the documents do not contain it. Do not invent facts.\n"
    "Cite the passages you rely on with their numbers in square brackets, "
    "for example [1] or [2][3]. Answer concisely."
)

NO_CONTEXT_REPLY = "No documents are indexed yet, so I cannot answer from your files."

Message = Dict[str, str]


@dataclass(frozen=True)
class Answer:
    text: str
    sources: List[Hit]


def format_context(hits: Sequence[Hit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit.chunk.label}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str, hits: Sequence[Hit], history: Sequence[Message] = ()
) -> List[Message]:
    """Assemble the chat messages sent to the model.

    `history` holds earlier user/assistant turns (without context blocks) so
    follow-up questions can refer back to them. Context for the current
    question is attached only to the current user message.
    """
    user_content = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": user_content},
    ]


def _trim_history(history: Sequence[Message], turns: int) -> List[Message]:
    if turns <= 0:
        return []
    return list(history)[-2 * turns :]


def answer_stream(
    question: str,
    index: VectorIndex,
    llm: OllamaClient,
    model: str,
    history: Sequence[Message] = (),
    top_k: int = 4,
    history_turns: int = 4,
) -> Tuple[List[Hit], Iterator[str]]:
    """Retrieve first, then return the hits together with a token stream."""
    hits = index.search(question, top_k)
    if not hits:
        return [], iter([NO_CONTEXT_REPLY])
    messages = build_messages(question, hits, _trim_history(history, history_turns))
    return hits, llm.chat_stream(model, messages)


def answer(
    question: str,
    index: VectorIndex,
    llm: OllamaClient,
    model: str,
    history: Sequence[Message] = (),
    top_k: int = 4,
    history_turns: int = 4,
) -> Answer:
    hits, stream = answer_stream(question, index, llm, model, history, top_k, history_turns)
    return Answer(text="".join(stream), sources=hits)
