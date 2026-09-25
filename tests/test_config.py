from chat_with_documents.config import Settings


def test_defaults_when_env_is_empty():
    settings = Settings.from_env({})

    assert settings == Settings()
    assert settings.ollama_host == "http://localhost:11434"


def test_env_overrides_and_host_normalisation():
    settings = Settings.from_env(
        {"OLLAMA_HOST": "0.0.0.0:11434/", "OLLAMA_MODEL": "qwen2.5", "TOP_K": "7"}
    )

    assert settings.ollama_host == "http://0.0.0.0:11434"
    assert settings.ollama_model == "qwen2.5"
    assert settings.top_k == 7
