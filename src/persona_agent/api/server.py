"""Main API server module.

Provides the FastAPI server with A2A protocol support, REST API,
and MCP tool integration managed via lifespan events.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.api.agent_factory import AgentFactory
from persona_agent.api.auth import make_api_key_dependency
from persona_agent.api.config import ApiConfig, load_config
from persona_agent.api.dependencies import (
    get_agent_factory,
    get_config,
    get_persona_manager,
)
from persona_agent.api.persona_manager import PersonaManager
from persona_agent.api.routes import agent, persona, session
from persona_agent.api.routes.a2a import A2ARegistry, set_registry
from persona_agent.api.routes.a2a import router as a2a_router
from persona_agent.llm.client import OpenAICompatibleClient
from persona_agent.mcp.direct_mcp import DirectMCPManager

API_VERSION = "0.2.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP service lifecycle and A2A registry setup."""
    config: ApiConfig = app.state.config
    mcp_manager: DirectMCPManager = app.state.mcp_manager

    # Initialize MCP services
    import os

    mcp_config_path = config.mcp_config_path or os.path.join(
        os.path.dirname(config.llm_config_path), "mcp_config.json"
    )
    if mcp_config_path and os.path.exists(mcp_config_path):
        await mcp_manager.load_config(mcp_config_path)
        logger.info(
            "MCP services initialized: %d tools", len(mcp_manager.get_all_tools())
        )

    # Set up A2A registry with all loaded personas
    persona_manager: PersonaManager = app.state.persona_manager
    llm_client = app.state.llm_client
    registry = A2ARegistry(base_url=f"http://{config.host}:{config.port}")

    for p_info in persona_manager.list_personas():
        p = persona_manager.get_persona(p_info["id"])
        if p:
            executor = PersonaAgentExecutor(
                persona_id=p.id,
                persona_name=p.name,
                system_prompt=p.generate_system_prompt(),
                llm_client=llm_client,
                mcp_manager=mcp_manager,
            )
            registry.register_persona(p, executor)

    set_registry(registry)
    # Mount each persona's A2A sub-app using the SDK's public build() API
    registry.mount_all(app)
    logger.info(
        "A2A registry initialized with %d personas", len(registry.list_personas())
    )

    yield

    # Cleanup
    await mcp_manager.close()
    logger.info("MCP services closed")


async def create_app(config: ApiConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="Persona Agent API",
        description="AI persona agents powered by A2A protocol with MCP tool integration",
        version=API_VERSION,
        lifespan=lifespan,
    )

    # Store shared objects in app state
    app.state.config = config

    llm_client = OpenAICompatibleClient.from_config(
        _load_llm_config(config.llm_config_path)
    )
    app.state.llm_client = llm_client

    mcp_manager = DirectMCPManager()
    app.state.mcp_manager = mcp_manager

    persona_manager = PersonaManager(config.personas_dir)
    app.state.persona_manager = persona_manager

    agent_factory = AgentFactory(
        llm_config_path=config.llm_config_path,
        llm_client=llm_client,
        mcp_manager=mcp_manager,
    )

    # CORS
    cors_wildcard_origin = any(origin == "*" for origin in config.allowed_origins)
    cors_credentials = config.enable_cors and not cors_wildcard_origin
    if config.enable_cors:
        if cors_wildcard_origin:
            logger.warning(
                "CORS allow_credentials disabled because allowed_origins contains '*'."
            )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_credentials=cors_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Dependency overrides
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_persona_manager] = lambda: persona_manager
    app.dependency_overrides[get_agent_factory] = lambda: agent_factory

    # Build the auth dependency now that we know the active config; this is
    # the only point where ``config.api_key_header`` is read, so renaming the
    # header requires an app restart.
    verify_api_key = make_api_key_dependency(config)

    # REST API routes (protected by API key when enable_auth is True)
    app.include_router(
        persona.router,
        prefix=config.api_prefix,
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        agent.router,
        prefix=config.api_prefix,
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        session.router,
        prefix=config.api_prefix,
        dependencies=[Depends(verify_api_key)],
    )

    # A2A discovery routes (aggregate agent card, persona listing) are public
    # by protocol design: external agents must be able to discover this hub
    # without holding an API key. The mounted SDK sub-apps are similarly
    # public. Authentication only gates the REST CRUD surface above.
    app.include_router(a2a_router)

    # Health check (always public; surfaces effective auth/CORS posture so
    # operators can verify configuration at a glance).
    @app.get("/health", tags=["health"])
    async def health_check():
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "protocol": "a2a",
            "default_model": config.default_model,
            "mcp_tools": len(mcp_manager.get_all_tools())
            if mcp_manager.is_initialized
            else 0,
            "auth": {
                "enabled": config.enable_auth,
                "configured": bool(config.allowed_api_keys),
                "header": config.api_key_header,
            },
            "cors": {
                "enabled": config.enable_cors,
                "wildcard_origin": cors_wildcard_origin,
                "credentials": cors_credentials,
            },
        }

    # Surface effective config in startup logs so operators don't have to
    # hit /health to confirm.
    logger.info(
        "Effective config: host=%s port=%d auth=%s api_keys=%d "
        "cors=%s wildcard_origin=%s",
        config.host,
        config.port,
        config.enable_auth,
        len(config.allowed_api_keys),
        config.enable_cors,
        cors_wildcard_origin,
    )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import traceback

        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "message": str(exc)},
        )

    return app


def _load_llm_config(path: str) -> dict:
    """Load LLM config from JSON file, returning empty dict on failure."""
    import json
    import os

    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load LLM config from %s: %s", path, e)
        return {}


def run_server(config: ApiConfig | None = None) -> None:
    """Run the FastAPI server."""
    if config is None:
        config = load_config()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(create_app(config))

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if config.debug else "info",
    )


if __name__ == "__main__":
    run_server()
