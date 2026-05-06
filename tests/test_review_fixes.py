"""Regression tests for configuration and persistence fixes."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from persona_agent.api.agent_factory import AgentFactory
from persona_agent.api.config import ApiConfig, load_config
from persona_agent.api.dependencies import get_persona_manager
from persona_agent.api.persona_manager import PersonaManager
from persona_agent.api.routes import persona as persona_routes
from persona_agent.api.server import _resolve_public_base_url, create_app
from persona_agent.llm.client import OpenAICompatibleClient
from persona_agent.mcp.direct_mcp import DirectMCPManager


class CapturingMCPManager(DirectMCPManager):
    def __init__(self) -> None:
        super().__init__()
        self.captured_env: dict[str, str] = {}

    async def _connect_service(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> bool:
        self.captured_env = env
        return True


async def test_mcp_config_substitutes_env_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "real-key")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brave": {
                        "command": "node",
                        "args": ["server.js"],
                        "env": {
                            "BRAVE_API_KEY": "${BRAVE_API_KEY}",
                            "PATH": "${PATH}:/opt/mcp/bin",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = CapturingMCPManager()
    assert await manager.load_config(str(config_path)) is True
    assert manager.captured_env == {
        "BRAVE_API_KEY": "real-key",
        "PATH": "/usr/local/bin:/usr/bin:/opt/mcp/bin",
    }


def test_default_mcp_config_does_not_override_path() -> None:
    config = json.loads(Path("config/mcp_config.json").read_text(encoding="utf-8"))
    service = config["mcpServers"]["mcp-server-fetch"]
    assert "PATH" not in service.get("env", {})


async def test_agent_factory_reuses_initialized_injected_mcp_manager(tmp_path) -> None:
    llm_config_path = tmp_path / "llm_config.json"
    llm_config_path.write_text("{}", encoding="utf-8")
    mcp_manager = MagicMock()
    mcp_manager.is_initialized = True
    mcp_manager.load_config = AsyncMock()

    factory = AgentFactory(
        llm_config_path=str(llm_config_path),
        llm_client=MagicMock(),
        mcp_manager=mcp_manager,
    )

    assert await factory._ensure_mcp() is mcp_manager
    mcp_manager.load_config.assert_not_awaited()


def test_delete_persona_removes_loaded_file(tmp_path) -> None:
    persona_file = tmp_path / "alice.yaml"
    persona_file.write_text(
        "id: alice\nname: Alice\ndescription: Example persona\n",
        encoding="utf-8",
    )
    manager = PersonaManager(str(tmp_path))

    assert manager.delete_persona("alice") is True
    assert not persona_file.exists()
    assert PersonaManager(str(tmp_path)).get_persona("alice") is None


def test_delete_persona_does_not_remove_file_for_different_declared_id(tmp_path) -> None:
    aliased_file = tmp_path / "alice.yaml"
    aliased_file.write_text(
        "id: bob\nname: Bob\ndescription: Stored in alice.yaml\n",
        encoding="utf-8",
    )
    filename_match_file = tmp_path / "bob.yaml"
    filename_match_file.write_text(
        "id: carol\nname: Carol\ndescription: Stored in bob.yaml\n",
        encoding="utf-8",
    )
    manager = PersonaManager(str(tmp_path))

    assert manager.delete_persona("bob") is True

    assert not aliased_file.exists()
    assert filename_match_file.exists()
    reloaded = PersonaManager(str(tmp_path))
    assert reloaded.get_persona("bob") is None
    assert reloaded.get_persona("carol") is not None


def test_delete_persona_failure_returns_500_and_keeps_state(tmp_path, monkeypatch) -> None:
    persona_file = tmp_path / "alice.yaml"
    persona_file.write_text(
        "id: alice\nname: Alice\ndescription: Example persona\n",
        encoding="utf-8",
    )
    manager = PersonaManager(str(tmp_path))

    def fail_remove(path: str) -> None:
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr("persona_agent.api.persona_manager.os.remove", fail_remove)

    app = FastAPI()
    app.dependency_overrides[get_persona_manager] = lambda: manager
    app.include_router(persona_routes.router, prefix="/api/v1")

    response = TestClient(app).delete("/api/v1/personas/alice")

    assert response.status_code == 500
    assert manager.get_persona("alice") is not None
    assert str(persona_file) in manager._persona_files["alice"]  # noqa: SLF001


def test_llm_model_config_api_base_overrides_top_level() -> None:
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


def test_llm_api_base_falls_back_to_top_level() -> None:
    client = OpenAICompatibleClient.from_config(
        {
            "default_model": "provider-a",
            "api_key": "top-key",
            "api_base": "https://top.example/v1",
            "model_configs": [{"model": "provider-a"}],
        }
    )

    assert str(client._client.base_url) == "https://top.example/v1/"


def test_llm_env_api_base_overrides_config(monkeypatch) -> None:
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


def test_llm_model_api_key_overrides_global_openai_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "global-openai-key")

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


def test_public_base_url_uses_explicit_config() -> None:
    config = ApiConfig(
        host="0.0.0.0",
        port=8000,
        public_base_url="https://agents.example.com/root/",
    )

    assert _resolve_public_base_url(config) == "https://agents.example.com/root"


def test_public_base_url_never_publishes_wildcard_bind_host() -> None:
    config = ApiConfig(host="0.0.0.0", port=8000)

    assert _resolve_public_base_url(config) == "http://localhost:8000"


def test_load_config_reads_api_public_base_url(monkeypatch) -> None:
    monkeypatch.setenv("API_PUBLIC_BASE_URL", "https://agents.example.com")

    assert load_config().public_base_url == "https://agents.example.com"


def test_a2a_routes_use_public_base_url(tmp_path) -> None:
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "alice.yaml").write_text(
        "id: alice\nname: Alice\ndescription: Example persona\n",
        encoding="utf-8",
    )
    llm_config_path = tmp_path / "llm_config.json"
    llm_config_path.write_text(
        json.dumps({"default_model": "gpt-4o-mini", "api_key": ""}),
        encoding="utf-8",
    )

    app = asyncio.run(
        create_app(
            ApiConfig(
                host="0.0.0.0",
                port=8000,
                public_base_url="https://agents.example.com",
                personas_dir=str(personas_dir),
                llm_config_path=str(llm_config_path),
                mcp_config_path=str(tmp_path / "missing_mcp_config.json"),
            )
        )
    )

    with TestClient(app) as client:
        aggregate = client.get("/.well-known/agent.json")
        listing = client.get("/a2a/personas")
        card = client.get("/a2a/alice/.well-known/agent-card.json")

    assert aggregate.status_code == 200
    assert listing.status_code == 200
    assert card.status_code == 200
    assert aggregate.json()["url"] == "https://agents.example.com/a2a/"
    assert listing.json()["agents"][0]["url"] == "https://agents.example.com/a2a/alice/"
    assert card.json()["url"] == "https://agents.example.com/a2a/alice/"
    assert "0.0.0.0" not in aggregate.text + listing.text + card.text
