"""Tests for OpenAICompatibleClient configuration precedence."""

import pytest

from persona_agent.llm import OpenAICompatibleClient


def test_model_config_api_base_overrides_top_level(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "provider-b",
            "api_key": "top-key",
            "api_base": "https://top.example/v1",
            "model_configs": [
                {
                    "model": "provider-b",
                    "api_base": "https://provider-b.example/v1",
                    "api_key": "model-key",
                }
            ],
        }
    )

    assert str(client._client.base_url) == "https://provider-b.example/v1/"
    assert client._client.api_key == "model-key"


def test_api_base_falls_back_to_top_level(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "provider-a",
            "api_key": "top-key",
            "api_base": "https://top.example/v1",
            "model_configs": [{"model": "provider-a"}],
        }
    )

    assert str(client._client.base_url) == "https://top.example/v1/"


def test_env_api_base_overrides_config(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")

    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "provider-a",
            "api_key": "top-key",
            "api_base": "https://top.example/v1",
            "model_configs": [
                {"model": "provider-a", "api_base": "https://model.example/v1"}
            ],
        }
    )

    assert str(client._client.base_url) == "https://env.example/v1/"


def test_model_api_key_overrides_global_env_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "global-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "provider-a",
            "api_key": "top-key",
            "api_base": "https://top.example/v1",
            "model_configs": [
                {
                    "model": "provider-a",
                    "api_key": "provider-a-key",
                    "api_base": "https://provider-a.example/v1",
                }
            ],
        }
    )

    assert client._client.api_key == "provider-a-key"
    assert str(client._client.base_url) == "https://provider-a.example/v1/"


def test_from_config_honors_default_model_override_and_timeout() -> None:
    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "file-model",
            "api_key": "top-key",
            "api_settings": {"timeout": 45},
            "model_configs": [{"model": "file-model", "temperature": 0.2}],
        },
        default_model="override-model",
    )

    assert client.model == "override-model"
    assert client.timeout == 45.0


async def test_unconfigured_api_key_defers_error_until_chat(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleClient.from_config({"default_model": "gpt-4o-mini"})
    assert client._client is None

    with pytest.raises(RuntimeError, match="LLM API key is not configured"):
        await client.chat(messages=[{"role": "user", "content": "hi"}])
