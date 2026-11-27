"""A2A protocol integration layer."""

from .cards import build_agent_card, build_aggregate_card, serialize_agent_card
from .executor import PersonaAgentExecutor
from .registry import A2ARegistry

__all__ = [
    "A2ARegistry",
    "PersonaAgentExecutor",
    "build_agent_card",
    "build_aggregate_card",
    "serialize_agent_card",
]
