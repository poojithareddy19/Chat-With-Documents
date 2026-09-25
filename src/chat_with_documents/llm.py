"""Thin client for the Ollama HTTP API (no SDK, just requests)."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

import requests


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or rejects a request."""


class OllamaClient:
    def __init__(self, host: str, timeout: float = 300.0, session: Optional[Any] = None):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    def is_available(self) -> bool:
        try:
            self.list_models()
        except OllamaError:
            return False
        return True

    def list_models(self) -> List[str]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama is not reachable at {self.host}: {exc}") from exc
        return [m["name"] for m in response.json().get("models", [])]

    def chat_stream(self, model: str, messages: Sequence[Dict[str, str]]) -> Iterator[str]:
        """Yield the assistant reply token by token."""
        payload = {"model": model, "messages": list(messages), "stream": True}
        try:
            response = self._session.post(
                f"{self.host}/api/chat", json=payload, stream=True, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama is not reachable at {self.host}: {exc}") from exc
        if response.status_code != 200:
            raise OllamaError(_error_message(response, model))
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise OllamaError(event["error"])
            content = event.get("message", {}).get("content", "")
            if content:
                yield content
            if event.get("done"):
                break

    def chat(self, model: str, messages: Sequence[Dict[str, str]]) -> str:
        return "".join(self.chat_stream(model, messages))


def _error_message(response: Any, model: str) -> str:
    try:
        detail = response.json().get("error", "")
    except ValueError:
        detail = getattr(response, "text", "")
    if response.status_code == 404:
        return f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}"
    return f"Ollama returned HTTP {response.status_code}: {detail}"
