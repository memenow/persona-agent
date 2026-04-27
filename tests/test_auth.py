"""Regression tests for the X-API-Key dependency.

These cover the three documented states (auth disabled / misconfigured /
enforced) so future refactors cannot silently disable enforcement.
"""

import pytest
from fastapi import HTTPException

from persona_agent.api.auth import verify_api_key
from persona_agent.api.config import ApiConfig


async def test_disabled_auth_passes_without_header() -> None:
    cfg = ApiConfig(enable_auth=False)
    await verify_api_key(config=cfg, api_key_header=None)


async def test_disabled_auth_passes_with_arbitrary_header() -> None:
    cfg = ApiConfig(enable_auth=False)
    await verify_api_key(config=cfg, api_key_header="anything")


async def test_misconfigured_auth_returns_503() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=[])
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(config=cfg, api_key_header="any")
    assert exc.value.status_code == 503


async def test_missing_header_returns_401() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(config=cfg, api_key_header=None)
    assert exc.value.status_code == 401


async def test_wrong_key_returns_401() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(config=cfg, api_key_header="wrong")
    assert exc.value.status_code == 401


async def test_correct_key_passes() -> None:
    cfg = ApiConfig(enable_auth=True, allowed_api_keys=["secret"])
    await verify_api_key(config=cfg, api_key_header="secret")
