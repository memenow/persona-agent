"""Regression tests for ``PersonaAgentExecutor``.

The primary concern is the historical bug where ``event_queue.enqueue_event``
(an ``async def`` in a2a-sdk 0.3.x) was invoked without ``await``, so status
events never reached A2A clients. These tests assert each lifecycle hook
actually awaits the queue.
"""

from unittest.mock import AsyncMock, MagicMock

from persona_agent.a2a.executor import PersonaAgentExecutor
from persona_agent.llm.client import ChatResponse, ToolCall


def _make_executor(llm_response: ChatResponse) -> PersonaAgentExecutor:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=llm_response)
    return PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=llm,
    )


def _make_context(text: str = "hello"):
    ctx = MagicMock()
    ctx.context_id = "ctx"
    ctx.task_id = "task"
    ctx.get_user_input = MagicMock(return_value=text)
    return ctx


def _make_queue() -> MagicMock:
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()
    return queue


async def test_execute_awaits_working_and_completed_events() -> None:
    executor = _make_executor(ChatResponse(content="hi", tool_calls=[]))
    queue = _make_queue()
    await executor.execute(_make_context(), queue)
    assert queue.enqueue_event.await_count == 2


async def test_cancel_awaits_canceled_event() -> None:
    executor = _make_executor(ChatResponse(content="", tool_calls=[]))
    queue = _make_queue()
    await executor.cancel(_make_context(), queue)
    assert queue.enqueue_event.await_count == 1


async def test_empty_user_input_emits_failed_event() -> None:
    executor = _make_executor(ChatResponse(content="", tool_calls=[]))
    queue = _make_queue()
    ctx = _make_context()
    ctx.get_user_input = MagicMock(return_value="")
    await executor.execute(ctx, queue)
    assert queue.enqueue_event.await_count == 1


async def test_chat_persists_user_and_assistant_to_history() -> None:
    executor = _make_executor(ChatResponse(content="answer", tool_calls=[]))
    result = await executor.chat("ctx", "question")
    assert result == "answer"
    history = executor.get_history("ctx")
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]


async def test_chat_fallback_strips_tool_messages_after_max_iterations() -> None:
    """When the tool loop exhausts ``MAX_TOOL_ITERATIONS``, the executor must
    re-issue the final completion with only ``system``/``user``/plain
    ``assistant`` turns to satisfy stricter providers."""
    llm = MagicMock()
    tool_response = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="t", arguments={})],
    )
    final_response = ChatResponse(content="forced", tool_calls=[])

    # Always return tool calls until the fallback call, which has tools=None.
    async def fake_chat(messages, tools=None, **kwargs):
        if tools is None:
            sanitized_roles = {m["role"] for m in messages}
            assert "tool" not in sanitized_roles
            assert all(
                "tool_calls" not in m for m in messages if m["role"] == "assistant"
            )
            return final_response
        return tool_response

    llm.chat = fake_chat
    mcp = MagicMock()
    mcp.is_initialized = True
    mcp.get_openai_tools = MagicMock(return_value=[{"type": "function"}])
    mcp.call_tool = AsyncMock(return_value="tool-output")

    executor = PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=llm,
        mcp_manager=mcp,
    )

    result = await executor.chat("ctx", "question")
    assert result == "forced"
