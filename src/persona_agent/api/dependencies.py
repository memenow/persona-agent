"""FastAPI dependency providers backed by ``app.state``."""

from fastapi import Request

from persona_agent.a2a.registry import A2ARegistry
from persona_agent.agents.service import AgentService
from persona_agent.config import Settings
from persona_agent.personas.repository import PersonaRepository


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_persona_repository(request: Request) -> PersonaRepository:
    return request.app.state.persona_repo


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_a2a_registry(request: Request) -> A2ARegistry:
    registry: A2ARegistry | None = getattr(request.app.state, "a2a_registry", None)
    if registry is None:
        raise RuntimeError("A2A registry not initialized")
    return registry
