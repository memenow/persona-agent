"""Tests for A2A v1.0 discovery, sub-app mounting, and v1.0 / v0.3 JSON-RPC."""

import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from persona_agent.api import create_app
from persona_agent.config import Settings
from persona_agent.llm import ChatResponse


def _make_app(tmp_path, *, persona_id: str = "alice", persona_name: str = "Alice"):
    personas_dir = tmp_path / f"personas_{persona_id}"
    personas_dir.mkdir()
    (personas_dir / f"{persona_id}.yaml").write_text(
        f"id: {persona_id}\nname: {persona_name}\ndescription: Example persona\n",
        encoding="utf-8",
    )
    llm_config_path = tmp_path / f"llm_{persona_id}.json"
    llm_config_path.write_text(
        json.dumps({"default_model": "gpt-4o-mini", "api_key": ""}),
        encoding="utf-8",
    )
    return create_app(
        Settings(
            host="0.0.0.0",
            port=8000,
            public_base_url="https://agents.example.com",
            personas_dir=str(personas_dir),
            llm_config_path=str(llm_config_path),
            mcp_config_path=str(tmp_path / "missing_mcp_config.json"),
        )
    )


def test_a2a_discovery_endpoints_use_v1_schema_and_public_base_url(tmp_path) -> None:
    app = _make_app(tmp_path)

    with TestClient(app) as client:
        canonical = client.get("/.well-known/agent-card.json")
        legacy = client.get("/.well-known/agent.json")
        listing = client.get("/a2a/personas")
        card = client.get("/a2a/alice/.well-known/agent-card.json")

    assert canonical.status_code == 200
    assert legacy.status_code == 200
    assert canonical.json() == legacy.json()
    assert listing.status_code == 200
    assert card.status_code == 200

    assert canonical.json()["supportedInterfaces"][0]["url"] == (
        "https://agents.example.com/a2a/"
    )
    assert listing.json()["agents"][0]["url"] == (
        "https://agents.example.com/a2a/alice/"
    )
    assert card.json()["supportedInterfaces"][0]["url"] == (
        "https://agents.example.com/a2a/alice/"
    )
    assert "0.0.0.0" not in canonical.text + listing.text + card.text


def test_a2a_jsonrpc_supports_v1_and_v03_clients(tmp_path) -> None:
    app = _make_app(tmp_path)
    app.state.llm_client.chat = AsyncMock(
        return_value=ChatResponse(content="hello from alice")
    )

    with TestClient(app) as client:
        resp_v1 = client.post(
            "/a2a/alice/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "m-v1",
                        "role": "ROLE_USER",
                        "parts": [{"text": "hi"}],
                    }
                },
            },
        )
        resp_v03 = client.post(
            "/a2a/alice/",
            headers={"A2A-Version": "0.3"},
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "m-v03",
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                    }
                },
            },
        )

    assert resp_v1.status_code == 200
    v1_body = resp_v1.json()
    assert v1_body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert (
        v1_body["result"]["task"]["status"]["message"]["parts"][0]["text"]
        == "hello from alice"
    )

    assert resp_v03.status_code == 200
    v03_body = resp_v03.json()
    assert v03_body["result"]["status"]["state"] == "completed"
    assert (
        v03_body["result"]["status"]["message"]["parts"][0]["text"]
        == "hello from alice"
    )


def test_multiple_apps_have_isolated_a2a_registries(tmp_path) -> None:
    app_a = _make_app(tmp_path, persona_id="alice", persona_name="Alice")
    app_b = _make_app(tmp_path, persona_id="bob", persona_name="Bob")

    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        agents_a = [
            a["persona_id"] for a in client_a.get("/a2a/personas").json()["agents"]
        ]
        agents_b = [
            a["persona_id"] for a in client_b.get("/a2a/personas").json()["agents"]
        ]

    assert agents_a == ["alice"]
    assert agents_b == ["bob"]
