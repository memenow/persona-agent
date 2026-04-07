"""AgentCard builder for persona agents.

Converts persona definitions into A2A Agent Cards for agent discovery.
"""

from typing import Any

from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill


def build_agent_card(
    persona_id: str,
    name: str,
    description: str,
    knowledge_domains: dict[str, Any] | None = None,
    base_url: str = "http://localhost:8000",
    version: str = "1.0.0",
) -> AgentCard:
    """Build an A2A AgentCard from persona attributes.

    Args:
        persona_id: Unique identifier for the persona.
        name: Persona display name.
        description: Persona description.
        knowledge_domains: Dict of domain -> topics for skill generation.
        base_url: Base URL where the agent is hosted.
        version: Agent version string.

    Returns:
        A fully populated AgentCard.
    """
    skills = _build_skills(persona_id, name, description, knowledge_domains)

    return AgentCard(
        name=name,
        description=description,
        url=f"{base_url}/a2a/{persona_id}/",
        version=version,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
        provider=AgentProvider(
            organization="Persona Agent",
            url=base_url,
        ),
    )


def _build_skills(
    persona_id: str,
    name: str,
    description: str,
    knowledge_domains: dict[str, Any] | None,
) -> list[AgentSkill]:
    """Generate AgentSkill entries from persona knowledge domains.

    Each knowledge domain becomes a separate skill, plus a general
    conversation skill for the persona.
    """
    skills: list[AgentSkill] = []

    # General conversation skill
    skills.append(
        AgentSkill(
            id=f"{persona_id}-conversation",
            name=f"Conversation with {name}",
            description=f"Have a conversation with {name}. {description}",
            tags=["conversation", "persona", persona_id],
            examples=[
                f"Tell me about yourself, {name}",
                "What do you think about current events?",
            ],
        )
    )

    # Domain-specific skills
    if knowledge_domains:
        for domain, topics in knowledge_domains.items():
            topic_list = topics if isinstance(topics, list) else [str(topics)]
            domain_label = domain.replace("_", " ").title()

            skills.append(
                AgentSkill(
                    id=f"{persona_id}-{domain}",
                    name=f"{name} on {domain_label}",
                    description=f"Ask {name} about {domain_label}: {', '.join(topic_list[:5])}",
                    tags=[domain, persona_id, "expertise"],
                    examples=[
                        f"What's your view on {topic_list[0]}?"
                        if topic_list
                        else f"Tell me about {domain_label}",
                    ],
                )
            )

    return skills
