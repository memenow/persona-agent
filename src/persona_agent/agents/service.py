"""Runtime management for persona agent instances and conversation sessions."""

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.config import load_llm_config
from persona_agent.llm.client import LLMClient, OpenAICompatibleClient
from persona_agent.mcp.manager import MCPManager
from persona_agent.personas.model import Persona

logger = logging.getLogger(__name__)


@dataclass
class AgentRecord:
    """In-memory record binding a REST agent ID to its persona executor."""

    id: str
    persona_id: str
    executor: PersonaAgentExecutor
    created_at: float

    @property
    def name(self) -> str:
        return self.executor.persona_name


class AgentSession:
    """Metadata wrapper for a conversation session backed by an executor context."""

    def __init__(
        self,
        agent_id: str,
        persona_id: str,
        executor: PersonaAgentExecutor,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.agent_id = agent_id
        self.persona_id = persona_id
        self.executor = executor
        # Wall-clock seconds match message timestamps recorded by PersonaAgentExecutor.
        self.created_at: float = time.time()
        self.last_active: float = self.created_at

    @property
    def messages(self) -> list[dict[str, Any]]:
        history = self.executor.get_history(self.id)
        return [
            {
                "role": entry.get("role", ""),
                "content": entry.get("content", ""),
                "timestamp": entry.get("timestamp", 0.0),
            }
            for entry in history
        ]


class AgentService:
    """Creates and tracks in-memory persona agents and conversation sessions."""

    def __init__(
        self,
        llm_config_path: str | None = None,
        llm_client: LLMClient | None = None,
        mcp_manager: MCPManager | None = None,
        mcp_config_path: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.llm_config_path = llm_config_path
        self.mcp_config_path = mcp_config_path
        self.agents: dict[str, AgentRecord] = {}
        self.sessions: dict[str, AgentSession] = {}

        self._llm_client = llm_client or OpenAICompatibleClient.from_config(
            load_llm_config(llm_config_path),
            default_model=default_model,
        )
        self._mcp_manager = mcp_manager
        self._mcp_initialized = bool(
            self._mcp_manager is not None and self._mcp_manager.is_initialized
        )

    async def _ensure_mcp(self) -> MCPManager | None:
        if self._mcp_initialized:
            return self._mcp_manager

        if self._mcp_manager is not None and self._mcp_manager.is_initialized:
            self._mcp_initialized = True
            return self._mcp_manager

        mcp_path: Path | None = None
        if self.mcp_config_path:
            mcp_path = Path(self.mcp_config_path)
        elif self.llm_config_path:
            mcp_path = Path(self.llm_config_path).parent / "mcp_config.json"

        if mcp_path is None or not mcp_path.is_file():
            self._mcp_initialized = True
            return self._mcp_manager

        if self._mcp_manager is None:
            self._mcp_manager = MCPManager()

        await self._mcp_manager.load_config(mcp_path)
        self._mcp_initialized = True
        logger.info(
            "MCP manager initialized with %d tools",
            len(self._mcp_manager.get_all_tools()),
        )
        return self._mcp_manager

    async def create_agent(
        self, persona: Persona, model: str | None = None
    ) -> AgentRecord:
        """Create a new agent backed by PersonaAgentExecutor.

        The optional ``model`` argument is reserved for future per-agent model overrides.
        """
        _ = model
        mcp = await self._ensure_mcp()
        executor = PersonaAgentExecutor(
            persona_id=persona.id,
            persona_name=persona.name,
            system_prompt=persona.build_system_prompt(),
            llm_client=self._llm_client,
            mcp_manager=mcp,
        )
        record = AgentRecord(
            id=str(uuid.uuid4()),
            persona_id=persona.id,
            executor=executor,
            created_at=time.time(),
        )
        self.agents[record.id] = record
        logger.info("Created agent %s for persona %s", record.id, persona.name)
        return record

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        return self.agents.get(agent_id)

    def list_agents(self) -> list[AgentRecord]:
        return list(self.agents.values())

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all sessions bound to it."""
        if agent_id not in self.agents:
            return False

        session_ids = [
            sid
            for sid, session in self.sessions.items()
            if session.agent_id == agent_id
        ]
        for sid in session_ids:
            session = self.sessions.pop(sid)
            session.executor.clear_history(session.id)

        del self.agents[agent_id]
        return True

    def create_session(self, agent_id: str) -> AgentSession | None:
        record = self.agents.get(agent_id)
        if record is None:
            return None

        session = AgentSession(
            agent_id=agent_id,
            persona_id=record.persona_id,
            executor=record.executor,
        )
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[AgentSession]:
        return list(self.sessions.values())

    def delete_session(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.executor.clear_history(session.id)
        return True

    async def send_message(self, session_id: str, message: str) -> tuple[bool, str]:
        """Send a user message through the session's executor and return (success, reply)."""
        session = self.sessions.get(session_id)
        if session is None:
            return False, "Session not found"

        try:
            response = await session.executor.chat(session.id, message)
            if not response:
                return False, "Failed to get response from agent"

            session.last_active = time.time()
            return True, response
        except Exception:
            logger.exception("Error processing message for session %s", session_id)
            return False, "Error processing message"
