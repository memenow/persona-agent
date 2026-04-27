"""Regression tests for the X-API-Key dependency.

These cover the three documented states (auth disabled / misconfigured /
enforced) so future refactors cannot silently disable enforcement.
"""

import pytest
from fastapi import HTTPException

from persona_agent.api.auth import make_api_key_dependency
from persona_agent.api.config import ApiConfig


async def test_disabled_auth_passes_without_header() -> None:
    cfg = ApiConfig(enable_auth=False)
    verify = make_api_key_dependency(cfg)
    await verify(api_key_header=None)


async def test_disabled_auth_passes_with_arbitrary_header() -> None:
    cfg = ApiConfig(enable_auth=False)
    verify = make_api_key_dependency(cfg)
    await verify(api_key_header="anything")


async def test_misconfigured_auth_returns_503() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=[])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="any")
    assert exc.value.status_code == 503


async def test_missing_header_returns_401() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header=None)
    assert exc.value.status_code == 401


async def test_wrong_key_returns_401() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    verify = make_api_key_dependency(cfg)
    with pytest.raises(HTTPException) as exc:
        await verify(api_key_header="wrong")
    assert exc.value.status_code == 401


async def test_correct_key_passes() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    verify = make_api_key_dependency(cfg)
    await verify(api_key_header="secret")


def test_factory_binds_configured_header_name() -> None:
    """The configured ``api_key_header`` must reach the wire layer.

    Round-trip via FastAPI ``TestClient`` so we exercise the same extraction
    path real requests use, instead of relying on closure reflection.
    """
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    cfg = ApiConfig(
        enable_auth=True,
        allowed_api_keys=["secret"],
        api_key_header="X-Custom-Auth",
    )
    verify = make_api_key_dependency(cfg)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(verify)])
    def probe():
        return {"ok": True}

    client = TestClient(app)
    # The default header is no longer accepted.
    assert client.get("/probe", headers={"X-API-Key": "secret"}).status_code == 401
    # The configured header is accepted.
    assert client.get("/probe", headers={"X-Custom-Auth": "secret"}).status_code == 200
