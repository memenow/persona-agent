"""Integration tests for FastAPI routes, error handling, and health endpoint."""

import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from persona_agent import __version__
from persona_agent.api import create_app
from persona_agent.config import Settings
from persona_agent.llm import ChatResponse


def _make_test_app(tmp_path, *, seed_alice: bool = False):
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    if seed_alice:
        (personas_dir / "alice.yaml").write_text(
            "id: alice\nname: Alice\ndescription: Example persona\n",
            encoding="utf-8",
        )
    llm_config_path = tmp_path / "llm_config.json"
    llm_config_path.write_text(
        json.dumps({"default_model": "gpt-4o-mini"}), encoding="utf-8"
    )

    return create_app(
        Settings(
            host="127.0.0.1",
            port=8000,
            personas_dir=str(personas_dir),
            llm_config_path=str(llm_config_path),
            mcp_config_path=str(tmp_path / "missing_mcp_config.json"),
        )
    )


def test_health_endpoint_reports_version_and_posture(tmp_path) -> None:
    app = _make_test_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"] == __version__
    assert body["protocol"] == "a2a"
    assert body["auth"] == {
        "enabled": False,
        "configured": False,
        "header": "X-API-Key",
    }
    assert body["cors"] == {
        "enabled": True,
        "wildcard_origin": True,
        "credentials": False,
    }


def test_global_exception_handler_does_not_leak_exception_text(tmp_path) -> None:
    app = _make_test_app(tmp_path)
    secret_marker = "SUPER-SECRET-INTERNAL-DETAIL"

    @app.get("/__boom")
    async def boom() -> None:
        raise RuntimeError(secret_marker)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/__boom")

    assert response.status_code == 500
    assert secret_marker not in response.text
    assert response.json() == {"detail": "Internal server error"}


def test_delete_persona_os_error_returns_500_and_preserves_state(
    tmp_path, monkeypatch
) -> None:
    app = _make_test_app(tmp_path, seed_alice=True)

    def fail_unlink(self, *args, **kwargs) -> None:
        raise OSError(f"cannot remove {self}")

    monkeypatch.setattr("pathlib.Path.unlink", fail_unlink)

    with TestClient(app) as client:
        response = client.delete("/api/v1/personas/alice")

    assert response.status_code == 500
    assert app.state.persona_repo.get_persona("alice") is not None


def test_upload_persona_validation_error_returns_422(tmp_path) -> None:
    app = _make_test_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/personas/upload",
            files={
                "file": (
                    "bad.json",
                    json.dumps({"id": "../bad", "name": "Bad"}),
                    "application/json",
                )
            },
        )

    assert response.status_code == 422


def test_update_persona_save_failure_rolls_back_memory(tmp_path, monkeypatch) -> None:
    app = _make_test_app(tmp_path, seed_alice=True)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(app.state.persona_repo, "save_persona", fail_save)

    with TestClient(app) as client:
        response = client.put("/api/v1/personas/alice", json={"name": "Mutated"})

    assert response.status_code == 500
    assert app.state.persona_repo.get_persona("alice").name == "Alice"


def test_rest_end_to_end_persona_agent_session_chat(tmp_path) -> None:
    app = _make_test_app(tmp_path)
    app.state.llm_client.chat = AsyncMock(
        return_value=ChatResponse(content="Greetings from Ada")
    )

    with TestClient(app) as client:
        p_resp = client.post(
            "/api/v1/personas",
            json={"name": "Ada", "description": "Mathematician"},
        )
        assert p_resp.status_code == 201
        persona_id = p_resp.json()["id"]

        a_resp = client.post("/api/v1/agents", json={"persona_id": persona_id})
        assert a_resp.status_code == 201
        agent_id = a_resp.json()["id"]

        s_resp = client.post("/api/v1/sessions", json={"agent_id": agent_id})
        assert s_resp.status_code == 201
        session_id = s_resp.json()["id"]

        m_resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "Hello"},
        )
        assert m_resp.status_code == 200
        assert m_resp.json()["success"] is True
        assert m_resp.json()["response"] == "Greetings from Ada"

        hist_resp = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert hist_resp.status_code == 200
        assert len(hist_resp.json()["messages"]) == 2
