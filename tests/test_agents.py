"""Tests for AgentService and AgentSession."""

import time
from unittest.mock import AsyncMock, MagicMock

from persona_agent.agents import AgentService, AgentSession
from persona_agent.llm import ChatResponse
from persona_agent.personas import Persona


def test_agent_session_timestamps_are_wall_clock_seconds() -> None:
    before = time.time()
    session = AgentSession(agent_id="a", persona_id="p", executor=MagicMock())
    after = time.time()
    assert before <= session.created_at <= after
    assert before <= session.last_active <= after


async def test_agent_service_reuses_initialized_injected_mcp_manager(tmp_path) -> None:
    llm_config_path = tmp_path / "llm_config.json"
    llm_config_path.write_text("{}", encoding="utf-8")
    mcp_manager = MagicMock()
    mcp_manager.is_initialized = True
    mcp_manager.load_config = AsyncMock()

    service = AgentService(
        llm_config_path=str(llm_config_path),
        llm_client=MagicMock(),
        mcp_manager=mcp_manager,
    )

    assert await service._ensure_mcp() is mcp_manager  # noqa: SLF001
    mcp_manager.load_config.assert_not_awaited()


async def test_agent_service_session_lifecycle_and_cascading_delete() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=ChatResponse(content="hello back"))
    service = AgentService(llm_client=llm)

    record = await service.create_agent(Persona(id="alice", name="Alice"))
    session = service.create_session(record.id)
    assert session is not None

    ok, reply = await service.send_message(session.id, "hi")
    assert ok is True
    assert reply == "hello back"
    assert len(session.messages) == 2

    assert service.delete_agent(record.id) is True
    assert service.get_session(session.id) is None
    assert record.executor.get_history(session.id) == []
