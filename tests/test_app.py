"""Render check for the Streamlit UI. Skipped where Streamlit is not installed
(the torch-free CI job); the full-requirements job runs it."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("dotenv")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_renders_without_an_exception(monkeypatch):
    # Point at a closed port so the model list fails fast and shows an error
    # message instead of needing a running Ollama.
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")

    app = AppTest.from_file("../app.py", default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "Chat with Documents"
    assert app.sidebar.button[0].label == "Process"
