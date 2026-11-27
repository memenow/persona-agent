"""Registry for per-persona A2A v1.0 FastAPI sub-applications."""

import logging
from typing import Any

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from fastapi import FastAPI

from persona_agent.a2a.cards import (
    build_agent_card,
    build_aggregate_card,
    serialize_agent_card,
)
from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.personas.model import Persona

logger = logging.getLogger(__name__)


class A2ARegistry:
    """Owns per-persona A2A sub-apps and mounts them onto the parent FastAPI app."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._sub_apps: dict[str, FastAPI] = {}
        self._executors: dict[str, PersonaAgentExecutor] = {}
        self._cards: dict[str, AgentCard] = {}

    def register_persona(
        self,
        persona: Persona,
        executor: PersonaAgentExecutor,
    ) -> None:
        """Build and register an isolated A2A FastAPI sub-app for a persona."""
        persona_id = persona.id
        card = build_agent_card(
            persona_id=persona_id,
            name=persona.name,
            description=persona.description,
            knowledge_domains=persona.knowledge_domains or None,
            base_url=self.base_url,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            agent_card=card,
        )

        sub_app = FastAPI()
        add_a2a_routes_to_fastapi(
            sub_app,
            agent_card_routes=create_agent_card_routes(card),
            jsonrpc_routes=create_jsonrpc_routes(
                request_handler,
                rpc_url="/",
                enable_v0_3_compat=True,
            ),
        )

        self._sub_apps[persona_id] = sub_app
        self._executors[persona_id] = executor
        self._cards[persona_id] = card

        logger.info("Registered A2A agent: %s (%s)", persona.name, persona_id)

    def get_sub_app(self, persona_id: str) -> FastAPI | None:
        return self._sub_apps.get(persona_id)

    def get_card(self, persona_id: str) -> AgentCard | None:
        return self._cards.get(persona_id)

    def get_executor(self, persona_id: str) -> PersonaAgentExecutor | None:
        return self._executors.get(persona_id)

    def list_personas(self) -> list[dict[str, Any]]:
        """Return summary metadata for all registered A2A persona agents."""
        items: list[dict[str, Any]] = []
        for persona_id, card in self._cards.items():
            url = (
                card.supported_interfaces[0].url
                if card.supported_interfaces
                else f"{self.base_url}/a2a/{persona_id}/"
            )
            items.append(
                {
                    "persona_id": persona_id,
                    "name": card.name,
                    "description": card.description,
                    "url": url,
                    "skills": len(card.skills),
                }
            )
        return items

    def build_aggregate_card(self) -> dict[str, Any]:
        """Serialize an aggregate AgentCard spanning all registered personas."""
        aggregate = build_aggregate_card(self._cards.values(), base_url=self.base_url)
        return serialize_agent_card(aggregate)

    def mount_all(self, parent_app: FastAPI) -> None:
        """Mount each persona sub-app under /a2a/{persona_id}."""
        for persona_id, sub_app in self._sub_apps.items():
            mount_path = f"/a2a/{persona_id}"
            parent_app.mount(mount_path, sub_app)
            logger.info("Mounted A2A sub-app at %s", mount_path)
