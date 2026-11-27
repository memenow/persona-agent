"""Tests for application settings and environment parsing."""

import json

from persona_agent.config import Settings, load_settings


def test_load_settings_filters_blank_allowed_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("API_ALLOWED_KEYS", "key1, ,,key2,")
    monkeypatch.delenv("LLM_CONFIG_PATH", raising=False)
    settings = load_settings()
    assert settings.allowed_api_keys == ["key1", "key2"]


def test_load_settings_drops_purely_blank_allowed_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("API_ALLOWED_KEYS", " , , ")
    monkeypatch.delenv("LLM_CONFIG_PATH", raising=False)
    settings = load_settings()
    assert settings.allowed_api_keys == []


def test_load_settings_strips_allowed_origins(monkeypatch) -> None:
    monkeypatch.setenv(
        "API_ALLOWED_ORIGINS", " https://a.example.com , , https://b.example.com "
    )
    settings = load_settings()
    assert settings.allowed_origins == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_load_settings_reads_public_base_url(monkeypatch) -> None:
    monkeypatch.setenv("API_PUBLIC_BASE_URL", "https://agents.example.com")
    assert load_settings().public_base_url == "https://agents.example.com"


def test_load_settings_default_model_env_overrides_file(tmp_path, monkeypatch) -> None:
    llm_config = tmp_path / "llm_config.json"
    llm_config.write_text(json.dumps({"default_model": "file-model"}), encoding="utf-8")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(llm_config))
    monkeypatch.setenv("DEFAULT_MODEL", "env-model")
    assert load_settings().default_model == "env-model"


def test_resolve_public_base_url_uses_explicit_setting() -> None:
    settings = Settings(
        host="0.0.0.0",
        port=8000,
        public_base_url="https://agents.example.com/root/",
    )
    assert settings.resolve_public_base_url() == "https://agents.example.com/root"


def test_resolve_public_base_url_never_publishes_wildcard_bind_host() -> None:
    assert Settings(host="0.0.0.0", port=8000).resolve_public_base_url() == (
        "http://localhost:8000"
    )
    assert Settings(host="::", port=9000).resolve_public_base_url() == (
        "http://localhost:9000"
    )


def test_resolve_public_base_url_brackets_ipv6_host() -> None:
    assert Settings(host="::1", port=8000).resolve_public_base_url() == (
        "http://[::1]:8000"
    )
