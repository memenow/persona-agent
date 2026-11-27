"""FastAPI application factory, lifespan wiring, and server runner."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from persona_agent import __version__
from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.a2a.registry import A2ARegistry
from persona_agent.agents.service import AgentService
from persona_agent.api.auth import make_api_key_dependency
from persona_agent.api.routes import (
    a2a_router,
    agents_router,
    personas_router,
    sessions_router,
)
from persona_agent.config import (
    ENV_DEFAULT_MODEL,
    Settings,
    load_llm_config,
    load_settings,
)
from persona_agent.llm.client import OpenAICompatibleClient
from persona_agent.mcp.manager import MCPManager
from persona_agent.personas.repository import PersonaRepository

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize MCP connections and mount per-persona A2A sub-apps at startup."""
    settings: Settings = app.state.settings
    mcp_manager: MCPManager = app.state.mcp_manager
    persona_repo: PersonaRepository = app.state.persona_repo
    llm_client = app.state.llm_client

    mcp_config_path = (
        Path(settings.mcp_config_path)
        if settings.mcp_config_path
        else Path(settings.llm_config_path).parent / "mcp_config.json"
    )
    if mcp_config_path.is_file():
        await mcp_manager.load_config(mcp_config_path)
        logger.info(
            "MCP services initialized: %d tools", len(mcp_manager.get_all_tools())
        )

    registry = A2ARegistry(base_url=settings.resolve_public_base_url())
    for persona in persona_repo.list_personas():
        executor = PersonaAgentExecutor(
            persona_id=persona.id,
            persona_name=persona.name,
            system_prompt=persona.build_system_prompt(),
            llm_client=llm_client,
            mcp_manager=mcp_manager,
        )
        registry.register_persona(persona, executor)

    app.state.a2a_registry = registry
    registry.mount_all(app)
    logger.info(
        "A2A registry initialized with %d personas", len(registry.list_personas())
    )

    yield

    await mcp_manager.close()
    logger.info("MCP services closed")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and wire the FastAPI application."""
    if settings is None:
        settings = load_settings()

    app = FastAPI(
        title="Persona Agent API",
        description="Persona agents exposed over A2A v1.0 and REST with MCP tool support",
        version=__version__,
        lifespan=lifespan,
    )

    llm_config = load_llm_config(settings.llm_config_path)
    model_override = settings.default_model if ENV_DEFAULT_MODEL in os.environ else None
    llm_client = OpenAICompatibleClient.from_config(
        llm_config, default_model=model_override
    )
    mcp_manager = MCPManager()
    persona_repo = PersonaRepository(settings.personas_dir)
    agent_service = AgentService(
        llm_config_path=settings.llm_config_path,
        llm_client=llm_client,
        mcp_manager=mcp_manager,
        mcp_config_path=settings.mcp_config_path,
        default_model=model_override,
    )

    app.state.settings = settings
    app.state.llm_client = llm_client
    app.state.mcp_manager = mcp_manager
    app.state.persona_repo = persona_repo
    app.state.agent_service = agent_service
    app.state.a2a_registry = None

    cors_wildcard_origin = any(origin == "*" for origin in settings.allowed_origins)
    cors_credentials = settings.enable_cors and not cors_wildcard_origin
    if settings.enable_cors:
        if cors_wildcard_origin:
            logger.warning(
                "CORS allow_credentials disabled because allowed_origins contains '*'."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=cors_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    verify_api_key = make_api_key_dependency(settings)
    protected = [Depends(verify_api_key)]

    app.include_router(
        personas_router, prefix=settings.api_prefix, dependencies=protected
    )
    app.include_router(
        agents_router, prefix=settings.api_prefix, dependencies=protected
    )
    app.include_router(
        sessions_router, prefix=settings.api_prefix, dependencies=protected
    )

    # A2A discovery and JSON-RPC endpoints remain public per A2A protocol design.
    app.include_router(a2a_router)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, object]:
        return {
            "status": "ok",
            "api_version": __version__,
            "protocol": "a2a",
            "default_model": settings.default_model,
            "mcp_tools": len(mcp_manager.get_all_tools())
            if mcp_manager.is_initialized
            else 0,
            "auth": {
                "enabled": settings.enable_auth,
                "configured": bool(settings.allowed_api_keys),
                "header": settings.api_key_header,
            },
            "cors": {
                "enabled": settings.enable_cors,
                "wildcard_origin": cors_wildcard_origin,
                "credentials": cors_credentials,
            },
        }

    logger.info(
        "Effective config: host=%s port=%d auth=%s api_keys=%d "
        "cors=%s wildcard_origin=%s",
        settings.host,
        settings.port,
        settings.enable_auth,
        len(settings.allowed_api_keys),
        settings.enable_cors,
        cors_wildcard_origin,
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never echo raw exception text to callers; it may include paths or secrets.
        logger.exception(
            "Unhandled exception while processing %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


def run_server(settings: Settings | None = None) -> None:
    """Configure logging, build the app, and start uvicorn."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if settings is None:
        settings = load_settings()

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run_server()
