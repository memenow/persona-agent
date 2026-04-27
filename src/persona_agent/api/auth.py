"""API key authentication dependency."""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from persona_agent.api.config import ApiConfig
from persona_agent.api.dependencies import get_config

logger = logging.getLogger(__name__)

# Module-level singletons keep the security scheme out of argument defaults,
# which keeps ruff's B008 happy and mirrors FastAPI's recommended Annotated
# pattern.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

ConfigDep = Annotated[ApiConfig, Depends(get_config)]
ApiKeyDep = Annotated[str | None, Security(_API_KEY_HEADER)]


def make_api_key_dependency() -> Callable:
    """Build a FastAPI dependency that enforces X-API-Key when auth is enabled.

    The header name and allowed keys come from the active ApiConfig.
    When ``enable_auth`` is False, the dependency is a no-op so unauthenticated
    callers continue to work in dev.

    TODO: The header name is currently fixed to ``X-API-Key`` because
    ``APIKeyHeader`` resolves the header name at instance construction time.
    Supporting an arbitrary ``config.api_key_header`` per request would
    require building the security scheme inside a closure that reads the
    active configuration, which is deferred until a real need arises.
    """

    async def verify_api_key(
        config: ConfigDep,
        api_key_header: ApiKeyDep,
    ) -> None:
        if not config.enable_auth:
            return
        if not config.allowed_api_keys:
            logger.error(
                "enable_auth=True but allowed_api_keys is empty; refusing all requests"
            )
            raise HTTPException(status_code=503, detail="Authentication misconfigured")
        if api_key_header is None or api_key_header not in config.allowed_api_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key


verify_api_key = make_api_key_dependency()
