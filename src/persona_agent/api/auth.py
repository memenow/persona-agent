"""API key authentication dependency."""

import logging
import secrets
from collections.abc import Callable
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from persona_agent.api.config import ApiConfig

logger = logging.getLogger(__name__)


def make_api_key_dependency(config: ApiConfig) -> Callable:
    """Build a FastAPI dependency that enforces ``config.api_key_header``.

    The returned coroutine reads the configured header (default
    ``X-API-Key``) and validates it against ``config.allowed_api_keys``.
    When ``config.enable_auth`` is False the dependency is a no-op so
    unauthenticated callers continue to work in dev. Construction time
    binds the security scheme to the active ``config.api_key_header``,
    so renaming the header takes effect on app restart.

    Comparison uses :func:`secrets.compare_digest` over every configured key
    so a timing side-channel cannot reveal which prefix matched. Empty
    allowlist entries are skipped defensively in case ``load_config``'s
    blank-key filter is ever bypassed.
    """

    api_key_scheme = APIKeyHeader(name=config.api_key_header, auto_error=False)
    ApiKeyDep = Annotated[str | None, Security(api_key_scheme)]

    async def verify_api_key(api_key_header: ApiKeyDep = None) -> None:
        if not config.enable_auth:
            return
        if not config.allowed_api_keys:
            logger.error(
                "enable_auth=True but allowed_api_keys is empty; refusing all requests"
            )
            raise HTTPException(status_code=503, detail="Authentication misconfigured")
        if not api_key_header:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        matched = False
        for allowed in config.allowed_api_keys:
            if not allowed:
                continue
            if secrets.compare_digest(api_key_header, allowed):
                matched = True
        if not matched:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key
