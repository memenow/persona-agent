"""Regression tests for the P0 review findings.

Coverage:

* ``AgentSession`` exposes Unix wall-clock timestamps so REST clients can
  compare them against message ``timestamp`` values written by the executor.
* The global exception handler does not echo ``str(exc)`` to clients.
* ``load_config`` filters blank entries from ``API_ALLOWED_KEYS`` so a
  trailing comma cannot inject an empty allow-listed key.
* ``make_api_key_dependency`` uses constant-time comparison and refuses an
  empty header even if a blank entry slips into the allow list.
"""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from persona_agent.api.agent_factory import AgentSession
from persona_agent.api.auth import make_api_key_dependency
from persona_agent.api.config import ApiConfig, load_config
from persona_agent.api.dependencies import clear_dependency_caches
from persona_agent.api.server import create_app


def test_agent_session_timestamps_are_wall_clock_seconds() -> None:
    executor = MagicMock()
    before = time.time()
    session = AgentSession(agent_id="a", persona_id="p", executor=executor)
    after = time.time()
    assert before <= session.created_at <= after
    assert before <= session.last_active <= after


async def test_global_exception_handler_does_not_leak_str_exc(tmp_path) -> None:
    from fastapi.testclient import TestClient

    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    llm_config_path = tmp_path / "llm_config.json"
    llm_config_path.write_text("{}", encoding="utf-8")
    cfg = ApiConfig(
        host="127.0.0.1",
        port=8000,
        personas_dir=str(personas_dir),
        llm_config_path=str(llm_config_path),
        mcp_config_path=str(tmp_path / "missing_mcp_config.json"),
    )
    app = await create_app(cfg)

    secret_marker = "SUPER-SECRET-INTERNAL-DETAIL"

    @app.get("/__boom")
    async def boom() -> None:
        raise RuntimeError(secret_marker)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/__boom")

    assert response.status_code == 500
    assert secret_marker not in response.text
    assert response.json() == {"detail": "Internal server error"}


def test_load_config_filters_blank_allowed_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("API_ALLOWED_KEYS", "key1, ,,key2,")
    monkeypatch.delenv("LLM_CONFIG_PATH", raising=False)
    clear_dependency_caches()
    try:
        cfg = load_config()
    finally:
        clear_dependency_caches()
    assert cfg.allowed_api_keys == ["key1", "key2"]


def test_load_config_drops_purely_blank_allowed_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("API_ALLOWED_KEYS", " , , ")
    monkeypatch.delenv("LLM_CONFIG_PATH", raising=False)
    clear_dependency_caches()
    try:
        cfg = load_config()
    finally:
        clear_dependency_caches()
    assert cfg.allowed_api_keys == []


async def test_verify_api_key_rejects_empty_header() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="")
    assert exc.value.status_code == 401


async def test_verify_api_key_skips_blank_allowlist_entries() -> None:
    """Defense-in-depth: even if a blank entry slips past ``load_config``'s
    filter, the constant-time loop must not authenticate an empty header."""
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["", "real-key"])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="")
    assert exc.value.status_code == 401


async def test_verify_api_key_constant_time_accepts_correct_key() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["alpha", "beta"])
    verify = make_api_key_dependency(cfg)
    await verify(api_key_header="beta")


async def test_verify_api_key_constant_time_refuses_wrong_key() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["alpha", "beta"])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="gamma")
    assert exc.value.status_code == 401
