from chat_with_documents.chunking import Chunk
from chat_with_documents.index import Hit, VectorIndex
from chat_with_documents.rag import NO_CONTEXT_REPLY, SYSTEM_PROMPT, answer, build_messages


class FakeLLM:
    def __init__(self, reply="The answer is 42 [1]."):
        self.reply = reply
        self.calls = []

    def chat_stream(self, model, messages):
        self.calls.append((model, list(messages)))
        for start in range(0, len(self.reply), 4):  # emulate token-sized pieces
            yield self.reply[start : start + 4]


def _hits():
    return [
        Hit(Chunk("a.pdf", 2, "Passage about apples."), 0.9),
        Hit(Chunk("b.pdf", 5, "Passage about bananas."), 0.4),
    ]


def test_messages_have_system_history_and_numbered_context():
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    messages = build_messages("Which fruit?", _hits(), history)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:3] == history
    last = messages[-1]
    assert last["role"] == "user"
    assert "[1] a.pdf, p.2\nPassage about apples." in last["content"]
    assert "[2] b.pdf, p.5\nPassage about bananas." in last["content"]
    assert last["content"].endswith("Question: Which fruit?")


def test_answer_returns_text_and_sources(fake_embedder):
    index = VectorIndex.build(
        [Chunk("a.pdf", 1, "apples are red"), Chunk("b.pdf", 1, "bananas are yellow")],
        fake_embedder,
    )
    llm = FakeLLM("Apples are red [1].")

    result = answer("what colour are apples", index, llm, model="m", top_k=1)

    assert result.text == "Apples are red [1]."
    assert [h.chunk.source for h in result.sources] == ["a.pdf"]
    model, messages = llm.calls[0]
    assert model == "m"
    assert "apples are red" in messages[-1]["content"]


def test_history_is_trimmed_to_the_last_turns(fake_embedder):
    index = VectorIndex.build([Chunk("a.pdf", 1, "text")], fake_embedder)
    llm = FakeLLM()
    history = []
    for i in range(6):
        history += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]

    answer("text", index, llm, model="m", history=history, history_turns=2)

    _, messages = llm.calls[0]
    kept = [m["content"] for m in messages[1:-1]]
    assert kept == ["q4", "a4", "q5", "a5"]


def test_empty_index_short_circuits_without_calling_the_model(fake_embedder):
    llm = FakeLLM()

    result = answer("anything", VectorIndex(fake_embedder), llm, model="m")

    assert result.text == NO_CONTEXT_REPLY
    assert result.sources == []
    assert llm.calls == []
