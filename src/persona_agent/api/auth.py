"""API key authentication dependency for REST routes."""

import logging
import secrets
from collections.abc import Callable
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from persona_agent.config import Settings

logger = logging.getLogger(__name__)


def make_api_key_dependency(settings: Settings) -> Callable:
    """Create a FastAPI dependency enforcing ``settings.api_key_header``.

    Header name binding happens at app construction time, so changing
    ``api_key_header`` requires restarting the application.
    """
    api_key_scheme = APIKeyHeader(name=settings.api_key_header, auto_error=False)
    ApiKeyDep = Annotated[str | None, Security(api_key_scheme)]

    async def verify_api_key(api_key_header: ApiKeyDep = None) -> None:
        if not settings.enable_auth:
            return
        if not settings.allowed_api_keys:
            logger.error(
                "enable_auth=True with empty allowed_api_keys; refusing protected request"
            )
            raise HTTPException(status_code=503, detail="Authentication misconfigured")
        if not api_key_header:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        # Compare against every entry without short-circuiting to avoid timing leaks.
        matched = False
        for allowed in settings.allowed_api_keys:
            if not allowed:
                continue
            if secrets.compare_digest(api_key_header, allowed):
                matched = True

        if not matched:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key
