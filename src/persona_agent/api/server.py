"""Main API server module.

This module provides the FastAPI server implementation for the Persona Agent API.
"""

import asyncio
import logging
import os
from typing import Annotated, List, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.persona_agent.api.config import ApiConfig, load_config
from src.persona_agent.api.dependencies import get_config, get_persona_manager, get_agent_factory
from src.persona_agent.api.persona_manager import PersonaManager
from src.persona_agent.api.agent_factory import AgentFactory
from src.persona_agent.api.routes import persona, agent, session

# API version constant
API_VERSION = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_server")

# Load configuration
config = load_config()


async def create_app(config: ApiConfig = None):
    """Create and configure the FastAPI application.
    
    This function initializes and configures a FastAPI application with all necessary
    routes, middleware, and dependencies for the Persona Agent API.
    
    Args:
        config: Optional API configuration settings. If not provided, 
            default settings will be loaded.
               
    Returns:
        FastAPI: A fully configured FastAPI application instance ready to be run.
    """
    if config is None:
        config = load_config()
    
    app = FastAPI(
        title="Persona Agent API",
        description="API server for interacting with AI personas",
        version=API_VERSION,
    )
    
    # Log important configuration information
    logger.info(f"LLM config path: {config.llm_config_path}")
    logger.info(f"Default model: {config.default_model}")
    
    # Configure CORS
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Create persona manager
    persona_manager = PersonaManager(config.personas_dir)
    
    # Create agent factory
    agent_factory = AgentFactory(
        llm_config_path=config.llm_config_path
    )
    
    # Dependency overrides
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_persona_manager] = lambda: persona_manager
    app.dependency_overrides[get_agent_factory] = lambda: agent_factory
    
    # Include routers
    app.include_router(persona.router, prefix=config.api_prefix)
    app.include_router(agent.router, prefix=config.api_prefix)
    app.include_router(session.router, prefix=config.api_prefix)
    
    # Add health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint.
        
        Returns:
            dict: A dictionary with status information.
        """
        try:
            return {
                "status": "ok", 
                "api_version": API_VERSION,
                "default_model": config.default_model
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # Error handler for authentication errors
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with proper response.
        
        Args:
            request: The incoming request.
            exc: The HTTP exception that was raised.
            
        Returns:
            JSONResponse: A formatted JSON response with error details.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions with error logging and friendly response.
        
        Args:
            request: The incoming request.
            exc: The exception that was raised.
            
        Returns:
            JSONResponse: A formatted JSON response with error details.
        """
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(exc)}\n{error_detail}")
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "message": str(exc),
                "type": exc.__class__.__name__
            }
        )
    
    return app


def run_server(config: ApiConfig = None):
    """Run the FastAPI server with the given configuration.
    
    Args:
        config: Optional API configuration settings. If not provided, 
            default settings will be loaded.
    """
    if config is None:
        config = load_config()
    
    # Create an event loop for running the async app creation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create the app
    app = loop.run_until_complete(create_app(config))

    # Run the server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if config.debug else "info"
    )


if __name__ == "__main__":
    run_server()
