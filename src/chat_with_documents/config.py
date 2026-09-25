"""Runtime settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


def _normalise_host(host: str) -> str:
    """Accept the forms the Ollama CLI itself accepts, such as 0.0.0.0:11434."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    history_turns: int = 4

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        default = cls()
        return cls(
            ollama_host=_normalise_host(env.get("OLLAMA_HOST", default.ollama_host)),
            ollama_model=env.get("OLLAMA_MODEL", default.ollama_model),
            embedding_model=env.get("EMBEDDING_MODEL", default.embedding_model),
            chunk_size=int(env.get("CHUNK_SIZE", default.chunk_size)),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", default.chunk_overlap)),
            top_k=int(env.get("TOP_K", default.top_k)),
            history_turns=int(env.get("HISTORY_TURNS", default.history_turns)),
        )
