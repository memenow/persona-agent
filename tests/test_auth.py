"""Tests for API key authentication middleware dependency."""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from persona_agent.api.auth import make_api_key_dependency
from persona_agent.config import Settings


async def test_disabled_auth_passes_without_header() -> None:
    verify = make_api_key_dependency(Settings(enable_auth=False))
    await verify(api_key_header=None)


async def test_disabled_auth_passes_with_arbitrary_header() -> None:
    verify = make_api_key_dependency(Settings(enable_auth=False))
    await verify(api_key_header="anything")


async def test_misconfigured_auth_returns_503() -> None:
    verify = make_api_key_dependency(Settings(enable_auth=True, allowed_api_keys=[]))
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="any")
    assert exc.value.status_code == 503


async def test_missing_header_returns_401() -> None:
    verify = make_api_key_dependency(
        Settings(enable_auth=True, allowed_api_keys=["secret"])
    )
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header=None)
    assert exc.value.status_code == 401


async def test_empty_header_returns_401() -> None:
    verify = make_api_key_dependency(
        Settings(enable_auth=True, allowed_api_keys=["secret"])
    )
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="")
    assert exc.value.status_code == 401


async def test_skips_blank_allowlist_entries() -> None:
    verify = make_api_key_dependency(
        Settings(enable_auth=True, allowed_api_keys=["", "real-key"])
    )
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="")
    assert exc.value.status_code == 401


async def test_wrong_key_returns_401() -> None:
    verify = make_api_key_dependency(
        Settings(enable_auth=True, allowed_api_keys=["alpha", "beta"])
    )
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="gamma")
    assert exc.value.status_code == 401


async def test_correct_key_passes() -> None:
    verify = make_api_key_dependency(
        Settings(enable_auth=True, allowed_api_keys=["alpha", "beta"])
    )
    await verify(api_key_header="beta")


def test_binds_configured_header_name() -> None:
    settings = Settings(
        enable_auth=True,
        allowed_api_keys=["secret"],
        api_key_header="X-Custom-Auth",
    )
    verify = make_api_key_dependency(settings)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(verify)])
    def probe() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/probe", headers={"X-API-Key": "secret"}).status_code == 401
    assert client.get("/probe", headers={"X-Custom-Auth": "secret"}).status_code == 200
