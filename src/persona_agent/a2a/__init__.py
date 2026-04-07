"""A2A (Agent-to-Agent) protocol integration layer."""

from persona_agent.a2a.agent_card import build_agent_card
from persona_agent.a2a.executor import PersonaAgentExecutor

__all__ = ["PersonaAgentExecutor", "build_agent_card"]
