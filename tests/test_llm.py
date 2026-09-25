import json

import pytest
import requests

from chat_with_documents.llm import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=()):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = list(lines)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self):
        return iter(self._lines)


class FakeSession:
    def __init__(self, get=None, post=None, error=None):
        self._get, self._post, self._error = get, post, error
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        if self._error:
            raise self._error
        return self._get

    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        if self._error:
            raise self._error
        return self._post


def _stream(*chunks, done=True):
    lines = [json.dumps({"message": {"role": "assistant", "content": c}}).encode() for c in chunks]
    lines.append(b"")  # keep-alive blank line must be ignored
    if done:
        lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())
    return lines


def test_list_models_parses_names():
    session = FakeSession(get=FakeResponse(payload={"models": [{"name": "llama3.1:latest"}]}))
    client = OllamaClient("http://host:11434/", session=session)

    assert client.list_models() == ["llama3.1:latest"]
    assert client.is_available()
    assert session.requests[0][1] == "http://host:11434/api/tags"


def test_unreachable_server_raises_ollama_error():
    session = FakeSession(error=requests.ConnectionError("refused"))
    client = OllamaClient("http://host:11434", session=session)

    with pytest.raises(OllamaError, match="not reachable"):
        client.list_models()
    assert not client.is_available()


def test_chat_concatenates_streamed_tokens_and_stops_at_done():
    session = FakeSession(post=FakeResponse(lines=_stream("Hel", "lo", " world")))
    client = OllamaClient("http://host:11434", session=session)

    assert client.chat("m", [{"role": "user", "content": "hi"}]) == "Hello world"
    method, url, kwargs = session.requests[0]
    assert (method, url) == ("POST", "http://host:11434/api/chat")
    assert kwargs["json"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def test_missing_model_gives_actionable_message():
    session = FakeSession(post=FakeResponse(status_code=404, payload={"error": "not found"}))
    client = OllamaClient("http://host:11434", session=session)

    with pytest.raises(OllamaError, match="ollama pull mistral"):
        client.chat("mistral", [])


def test_error_event_inside_stream_is_raised():
    lines = [json.dumps({"error": "out of memory"}).encode()]
    session = FakeSession(post=FakeResponse(lines=lines))
    client = OllamaClient("http://host:11434", session=session)

    with pytest.raises(OllamaError, match="out of memory"):
        client.chat("m", [])
