"""A2A v1.0 AgentCard builders and JSON serialization helpers."""

from collections.abc import Iterable
from typing import Any

from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0


def build_agent_card(
    persona_id: str,
    name: str,
    description: str,
    knowledge_domains: dict[str, Any] | None = None,
    base_url: str = "http://localhost:8000",
    version: str = "1.0.0",
) -> AgentCard:
    """Build an A2A v1.0 AgentCard for a single persona."""
    normalized_base = base_url.rstrip("/")
    skills = _build_skills(persona_id, name, description, knowledge_domains)

    return AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=[
            AgentInterface(
                url=f"{normalized_base}/a2a/{persona_id}/",
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
        provider=AgentProvider(
            organization="Persona Agent",
            url=normalized_base,
        ),
    )


def build_aggregate_card(
    cards: Iterable[AgentCard],
    base_url: str = "http://localhost:8000",
    version: str = "1.0.0",
) -> AgentCard:
    """Build an A2A v1.0 AgentCard aggregating skills across all registered personas."""
    normalized_base = base_url.rstrip("/")
    all_skills: list[AgentSkill] = []
    for card in cards:
        all_skills.extend(card.skills)

    return AgentCard(
        name="Persona Agent Hub",
        description="Multi-persona A2A agent hub",
        version=version,
        supported_interfaces=[
            AgentInterface(
                url=f"{normalized_base}/a2a/",
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=all_skills,
        provider=AgentProvider(
            organization="Persona Agent",
            url=normalized_base,
        ),
    )


def serialize_agent_card(card: AgentCard) -> dict[str, Any]:
    """Serialize a Protobuf AgentCard to its canonical A2A JSON dictionary."""
    return agent_card_to_dict(card)


def _build_skills(
    persona_id: str,
    name: str,
    description: str,
    knowledge_domains: dict[str, Any] | None,
) -> list[AgentSkill]:
    skills: list[AgentSkill] = [
        AgentSkill(
            id=f"{persona_id}-conversation",
            name=f"Conversation with {name}",
            description=f"Have a conversation with {name}. {description}".strip(),
            tags=["conversation", "persona", persona_id],
            examples=[
                f"Tell me about yourself, {name}",
                "What do you think about current events?",
            ],
        )
    ]

    if knowledge_domains:
        for domain, topics in knowledge_domains.items():
            topic_list = topics if isinstance(topics, list) else [str(topics)]
            domain_label = domain.replace("_", " ").title()
            skills.append(
                AgentSkill(
                    id=f"{persona_id}-{domain}",
                    name=f"{name} on {domain_label}",
                    description=(
                        f"Ask {name} about {domain_label}: {', '.join(topic_list[:5])}"
                    ),
                    tags=[domain, persona_id, "expertise"],
                    examples=[
                        f"What's your view on {topic_list[0]}?"
                        if topic_list
                        else f"Tell me about {domain_label}",
                    ],
                )
            )

    return skills
