"""API routes for the Persona Agent system.

This package contains route handlers for API endpoints, organized by resource type:
- persona: Endpoints for managing persona definitions
- agent: Endpoints for creating and managing agents
- session: Endpoints for managing conversation sessions
"""

from .agent import router as agent_router
from .persona import router as persona_router
from .session import router as session_router

__all__ = ["agent_router", "persona_router", "session_router"] 