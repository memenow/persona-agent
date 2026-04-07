"""A2A protocol routes for persona agent discovery and communication.

Provides agent card discovery endpoints and manages per-persona A2A
sub-applications. Each persona is mounted as an independent ASGI app
using the a2a-sdk's public ``A2AFastAPIApplication.build()`` API.
"""

import logging
from typing import Any

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from persona_agent.a2a.agent_card import build_agent_card
from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.api.persona_manager import Persona

logger = logging.getLogger(__name__)

router = APIRouter(tags=["a2a"])


class A2ARegistry:
    """Registry managing A2A sub-applications for each persona.

    Each persona gets its own ``A2AFastAPIApplication`` built via the
    SDK's public ``build()`` method and mounted as an ASGI sub-app.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self._apps: dict[str, A2AFastAPIApplication] = {}
        self._sub_apps: dict[str, Any] = {}
        self._executors: dict[str, PersonaAgentExecutor] = {}
        self._cards: dict[str, AgentCard] = {}

    def register_persona(
        self,
        persona: Persona,
        executor: PersonaAgentExecutor,
    ) -> None:
        """Register a persona as an A2A agent.

        Creates a ``DefaultRequestHandler`` and builds the A2A sub-app
        via the SDK's public ``build()`` API.

        Args:
            persona: The persona definition.
            executor: The PersonaAgentExecutor bound to this persona.
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

        a2a_app = A2AFastAPIApplication(
            agent_card=card,
            http_handler=request_handler,
        )
        # build() returns a FastAPI/Starlette app with routes for
        # agent-card and JSON-RPC already wired up.
        sub_app = a2a_app.build()

        self._apps[persona_id] = a2a_app
        self._sub_apps[persona_id] = sub_app
        self._executors[persona_id] = executor
        self._cards[persona_id] = card

        logger.info("Registered A2A agent: %s (%s)", persona.name, persona_id)

    def get_sub_app(self, persona_id: str) -> Any | None:
        """Get the built ASGI sub-application for a persona."""
        return self._sub_apps.get(persona_id)

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
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
                state_transition_history=True,
            ),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=all_skills,
        )
        return aggregate.model_dump(exclude_none=True)

    def mount_all(self, parent_app: Any) -> None:
        """Mount every persona's A2A sub-app onto the parent FastAPI app.

        Each sub-app is mounted at ``/a2a/{persona_id}``, so the SDK's
        default ``/`` RPC route becomes ``/a2a/{persona_id}/`` and the
        agent card at ``/.well-known/agent-card.json`` becomes
        ``/a2a/{persona_id}/.well-known/agent-card.json``.
        """
        for persona_id, sub_app in self._sub_apps.items():
            mount_path = f"/a2a/{persona_id}"
            parent_app.mount(mount_path, sub_app)
            logger.info("Mounted A2A sub-app at %s", mount_path)


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


# --- Discovery & listing routes (not handled by SDK sub-apps) ---


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
