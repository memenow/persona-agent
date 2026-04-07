"""Agent factory for creating and managing A2A persona agents.

This module provides a factory class for creating and managing AI agents
powered by the A2A protocol with direct LLM and MCP tool integration.
"""

import json
import logging
import os
import time
import uuid
from typing import Any

from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.api.persona_manager import Persona
from persona_agent.llm.client import LLMClient, OpenAICompatibleClient
from persona_agent.mcp.direct_mcp import DirectMCPManager

logger = logging.getLogger("agent_factory")


class AgentSession:
    """Represents an agent session with conversation history.

    Attributes:
        id: Unique identifier for the session.
        agent_id: ID of the agent associated with this session.
        persona_id: ID of the persona used by the agent.
        executor: PersonaAgentExecutor instance for this session.
        messages: List of messages exchanged in this session.
        created_at: Timestamp when the session was created.
        last_active: Timestamp of the last activity in this session.
    """

    def __init__(self, agent_id: str, persona_id: str, executor: PersonaAgentExecutor):
        self.id: str = str(uuid.uuid4())
        self.agent_id: str = agent_id
        self.persona_id: str = persona_id
        self.executor: PersonaAgentExecutor = executor
        self.messages: list[dict[str, Any]] = []
        self.created_at: float = time.monotonic()
        self.last_active: float = self.created_at

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": time.monotonic(),
            }
        )
        self.last_active = time.monotonic()

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages


class AgentFactory:
    """Factory for creating and managing A2A persona agents.

    This class provides methods for creating, retrieving, and managing
    agents and sessions using PersonaAgentExecutor with direct LLM
    and MCP tool integration.
    """

    def __init__(
        self,
        llm_config_path: str | None = None,
        llm_client: LLMClient | None = None,
        mcp_manager: DirectMCPManager | None = None,
    ):
        self.llm_config_path = llm_config_path
        self.agents: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, AgentSession] = {}

        # LLM client — injected or created from config
        self._llm_client = llm_client or self._create_llm_client()

        # MCP manager — injected or None (lazy init)
        self._mcp_manager = mcp_manager
        self._mcp_initialized = False

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client from configuration file."""
        llm_configs: dict[str, Any] = {}

        if self.llm_config_path and os.path.exists(self.llm_config_path):
            with open(self.llm_config_path, encoding="utf-8") as f:
                llm_configs = json.load(f)
            logger.info("Loaded LLM config from %s", self.llm_config_path)

        return OpenAICompatibleClient.from_config(llm_configs)

    async def _ensure_mcp(self) -> DirectMCPManager | None:
        """Ensure MCP manager is initialized."""
        if self._mcp_initialized:
            return self._mcp_manager

        if not self.llm_config_path:
            self._mcp_initialized = True
            return None

        mcp_config_path = os.path.join(
            os.path.dirname(self.llm_config_path), "mcp_config.json"
        )
        if not os.path.exists(mcp_config_path):
            logger.warning("MCP config not found: %s", mcp_config_path)
            self._mcp_initialized = True
            return None

        if self._mcp_manager is None:
            self._mcp_manager = DirectMCPManager()

        await self._mcp_manager.load_config(mcp_config_path)
        self._mcp_initialized = True
        logger.info(
            "MCP manager initialized with %d tools",
            len(self._mcp_manager.get_all_tools()),
        )
        return self._mcp_manager

    async def create_agent(self, persona: Persona, model: str | None = None) -> str:
        """Create a new agent for a persona.

        Args:
            persona: The persona to create an agent for.
            model: Optional model name override (not used in current impl,
                   reserved for future multi-model support).

        Returns:
            The ID of the created agent.
        """
        agent_id = str(uuid.uuid4())

        # Ensure MCP is initialized
        mcp = await self._ensure_mcp()

        # Create the executor
        executor = PersonaAgentExecutor(
            persona_id=persona.id,
            persona_name=persona.name,
            system_prompt=persona.generate_system_prompt(),
            llm_client=self._llm_client,
            mcp_manager=mcp,
        )

        self.agents[agent_id] = {
            "id": agent_id,
            "persona_id": persona.id,
            "executor": executor,
            "created_at": time.monotonic(),
        }

        logger.info("Created agent %s for persona %s", agent_id, persona.name)
        return agent_id

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all available agents."""
        return [
            {
                "id": agent_id,
                "persona_id": info["persona_id"],
                "name": info["executor"].persona_name,
                "created_at": info["created_at"],
            }
            for agent_id, info in self.agents.items()
        ]

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID, including any associated sessions."""
        if agent_id not in self.agents:
            return False

        session_ids_to_delete = [
            sid
            for sid, session in self.sessions.items()
            if session.agent_id == agent_id
        ]
        for sid in session_ids_to_delete:
            del self.sessions[sid]

        del self.agents[agent_id]
        return True

    def create_session(self, agent_id: str) -> str | None:
        """Create a new conversation session with an agent."""
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return None

        session = AgentSession(
            agent_id=agent_id,
            persona_id=agent_info["persona_id"],
            executor=agent_info["executor"],
        )

        self.sessions[session.id] = session
        return session.id

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions."""
        return [
            {
                "id": session.id,
                "agent_id": session.agent_id,
                "persona_id": session.persona_id,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "last_active": session.last_active,
            }
            for session in self.sessions.values()
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        if session_id not in self.sessions:
            return False
        # Clear executor history for this session's context
        session = self.sessions[session_id]
        session.executor.clear_history(session.id)
        del self.sessions[session_id]
        return True

    async def send_message(
        self, session_id: str, message: str
    ) -> tuple[bool, str | None]:
        """Send a message to an agent and get a response.

        Args:
            session_id: ID of the session to send the message to.
            message: The message to send.

        Returns:
            Tuple of (success, response_or_error_message).
        """
        session = self.sessions.get(session_id)
        if not session:
            return False, "Session not found"

        try:
            session.add_message("user", message)

            # Use the executor's LLM loop directly (session.id as context_id)
            response = await session.executor._run_llm_loop(session.id, message)

            if not response:
                return False, "Failed to get response from agent"

            session.add_message("assistant", response)
            return True, response

        except Exception as e:
            logger.exception("Error in send_message")
            return False, f"Error: {e}"
