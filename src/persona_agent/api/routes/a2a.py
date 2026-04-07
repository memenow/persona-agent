"""A2A protocol routes for persona agent discovery and communication.

Provides JSON-RPC endpoints for A2A protocol and agent card discovery.
Each persona is exposed as an independent A2A agent with its own card.
"""

import logging
from typing import Any

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from persona_agent.a2a.agent_card import build_agent_card
from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.api.persona_manager import Persona

logger = logging.getLogger(__name__)

router = APIRouter(tags=["a2a"])


class A2ARegistry:
    """Registry managing A2A applications for each persona.

    Holds the mapping from persona_id to their A2A app, executor,
    and agent card.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self._apps: dict[str, A2AFastAPIApplication] = {}
        self._executors: dict[str, PersonaAgentExecutor] = {}
        self._cards: dict[str, AgentCard] = {}
        self._task_stores: dict[str, InMemoryTaskStore] = {}

    def register_persona(
        self,
        persona: Persona,
        executor: PersonaAgentExecutor,
    ) -> None:
        """Register a persona as an A2A agent.

        Args:
            persona: The persona definition.
            executor: The PersonaAgentExecutor for this persona.
        """
        persona_id = persona.id
        card = build_agent_card(
            persona_id=persona_id,
            name=persona.name,
            description=persona.description,
            knowledge_domains=persona.knowledge_domains or None,
            base_url=self.base_url,
        )

        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
            queue_manager=InMemoryQueueManager(),
        )

        app = A2AFastAPIApplication(
            agent_card=card,
            http_handler=request_handler,
        )

        self._apps[persona_id] = app
        self._executors[persona_id] = executor
        self._cards[persona_id] = card
        self._task_stores[persona_id] = task_store

        logger.info(
            "Registered A2A agent for persona: %s (%s)", persona.name, persona_id
        )

    def get_app(self, persona_id: str) -> A2AFastAPIApplication | None:
        return self._apps.get(persona_id)

    def get_card(self, persona_id: str) -> AgentCard | None:
        return self._cards.get(persona_id)

    def get_executor(self, persona_id: str) -> PersonaAgentExecutor | None:
        return self._executors.get(persona_id)

    def list_personas(self) -> list[dict[str, Any]]:
        """List all registered A2A persona agents."""
        return [
            {
                "persona_id": pid,
                "name": card.name,
                "description": card.description,
                "url": card.url,
                "skills": len(card.skills),
            }
            for pid, card in self._cards.items()
        ]

    def build_aggregate_card(self) -> dict[str, Any]:
        """Build an aggregate agent card listing all persona agents."""
        all_skills = []
        for card in self._cards.values():
            all_skills.extend(card.skills)

        aggregate = AgentCard(
            name="Persona Agent Hub",
            description="Multi-persona AI agent hub powered by A2A protocol",
            url=f"{self.base_url}/a2a/",
            version="1.0.0",
            capabilities=AgentCard.model_fields["capabilities"].default
            if "capabilities" in AgentCard.model_fields
            else next(iter(self._cards.values())).capabilities,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=all_skills,
        )
        return aggregate.model_dump(exclude_none=True)


# Global registry instance — initialized in server.py lifespan
_registry: A2ARegistry | None = None


def get_registry() -> A2ARegistry:
    """Get the global A2A registry."""
    if _registry is None:
        raise RuntimeError("A2A registry not initialized")
    return _registry


def set_registry(registry: A2ARegistry) -> None:
    """Set the global A2A registry."""
    global _registry
    _registry = registry


@router.get("/.well-known/agent.json")
async def get_aggregate_agent_card() -> JSONResponse:
    """Return aggregate agent card for all personas."""
    registry = get_registry()
    return JSONResponse(content=registry.build_aggregate_card())


@router.get("/a2a/personas")
async def list_a2a_personas() -> JSONResponse:
    """List all A2A persona agents."""
    registry = get_registry()
    return JSONResponse(content={"agents": registry.list_personas()})


@router.get("/a2a/{persona_id}/.well-known/agent.json")
async def get_persona_agent_card(persona_id: str) -> JSONResponse:
    """Return agent card for a specific persona."""
    registry = get_registry()
    card = registry.get_card(persona_id)
    if not card:
        return JSONResponse(
            status_code=404,
            content={"error": f"Persona '{persona_id}' not found"},
        )
    return JSONResponse(content=card.model_dump(exclude_none=True))


@router.api_route("/a2a/{persona_id}/", methods=["POST"])
async def handle_a2a_request(persona_id: str, request: Request) -> JSONResponse:
    """Handle A2A JSON-RPC requests for a specific persona.

    Delegates to the persona's A2AFastAPIApplication instance.
    """
    registry = get_registry()
    app = registry.get_app(persona_id)
    if not app:
        return JSONResponse(
            status_code=404,
            content={"error": f"Persona '{persona_id}' not registered as A2A agent"},
        )

    # Delegate to the A2A app's handler
    body = await request.json()
    # Use the app's internal handler to process the JSON-RPC request
    try:
        response = await app._handle_requests(request)
        return response
    except Exception:
        logger.exception("Error handling A2A request for persona %s", persona_id)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error"},
                "id": body.get("id"),
            },
        )
