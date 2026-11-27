"""Tests for PersonaAgentExecutor task lifecycle, history, locking, and tool loop."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from a2a.types import Task, TaskState, TaskStatusUpdateEvent

from persona_agent.a2a import PersonaAgentExecutor
from persona_agent.llm import ChatResponse, ToolCall


def _make_executor(llm_response: ChatResponse) -> PersonaAgentExecutor:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=llm_response)
    return PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=llm,
    )


def _make_context(text: str = "hello", *, current_task: Task | None = None):
    ctx = MagicMock()
    ctx.context_id = "ctx"
    ctx.task_id = "task"
    ctx.current_task = current_task
    ctx.get_user_input = MagicMock(return_value=text)
    return ctx


def _make_queue() -> MagicMock:
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()
    return queue


async def test_execute_enqueues_task_then_working_and_completed_events() -> None:
    executor = _make_executor(ChatResponse(content="hi", tool_calls=[]))
    queue = _make_queue()

    await executor.execute(_make_context(), queue)

    events = [call.args[0] for call in queue.enqueue_event.await_args_list]
    assert len(events) == 3
    assert isinstance(events[0], Task)
    assert events[0].status.state == TaskState.TASK_STATE_SUBMITTED
    assert isinstance(events[1], TaskStatusUpdateEvent)
    assert events[1].status.state == TaskState.TASK_STATE_WORKING
    assert isinstance(events[2], TaskStatusUpdateEvent)
    assert events[2].status.state == TaskState.TASK_STATE_COMPLETED
    assert events[2].status.message.parts[0].text == "hi"


async def test_cancel_emits_canceled_event() -> None:
    executor = _make_executor(ChatResponse(content="", tool_calls=[]))
    queue = _make_queue()

    await executor.cancel(_make_context(), queue)

    assert queue.enqueue_event.await_count == 1
    event = queue.enqueue_event.await_args_list[0].args[0]
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.TASK_STATE_CANCELED


async def test_empty_user_input_emits_task_and_failed_event() -> None:
    executor = _make_executor(ChatResponse(content="", tool_calls=[]))
    queue = _make_queue()

    await executor.execute(_make_context(text=""), queue)

    events = [call.args[0] for call in queue.enqueue_event.await_args_list]
    assert len(events) == 2
    assert isinstance(events[0], Task)
    assert isinstance(events[1], TaskStatusUpdateEvent)
    assert events[1].status.state == TaskState.TASK_STATE_FAILED


async def test_chat_persists_user_and_assistant_with_wall_clock_timestamps() -> None:
    before = time.time()
    executor = _make_executor(ChatResponse(content="answer", tool_calls=[]))
    result = await executor.chat("ctx", "question")
    after = time.time()

    assert result == "answer"
    history = executor.get_history("ctx")
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]
    for entry in history:
        assert before <= entry["timestamp"] <= after


async def test_chat_fallback_uses_tool_choice_none_with_full_context() -> None:
    captured: dict = {}
    tool_response = ChatResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="t", arguments={})],
    )
    final_response = ChatResponse(content="forced", tool_calls=[])

    async def fake_chat(messages, tools=None, tool_choice=None, **kwargs):
        if tool_choice == "none":
            captured["messages"] = messages
            captured["tools"] = tools
            return final_response
        return tool_response

    llm = MagicMock()
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
    roles = [m["role"] for m in captured["messages"]]
    assert "tool" in roles
    assert any("tool_calls" in m for m in captured["messages"])
    assert captured["tools"] is not None


async def test_chat_serializes_concurrent_calls_on_same_context() -> None:
    async def fake_chat(messages, **kwargs):
        latest_user = next(m for m in reversed(messages) if m["role"] == "user")[
            "content"
        ]
        await asyncio.sleep(0.01)
        return ChatResponse(content=f"reply-{latest_user}", tool_calls=[])

    llm = MagicMock()
    llm.chat = fake_chat
    executor = PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=llm,
    )

    await asyncio.gather(
        executor.chat("ctx", "a"),
        executor.chat("ctx", "b"),
    )
    history = executor.get_history("ctx")
    assert len(history) == 4
    for user_entry, assistant_entry in (history[0:2], history[2:4]):
        assert user_entry["role"] == "user"
        assert assistant_entry["role"] == "assistant"
        assert assistant_entry["content"] == f"reply-{user_entry['content']}"


async def test_clear_history_drops_lock() -> None:
    executor = PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=MagicMock(),
    )
    lock = executor._get_lock("ctx")  # noqa: SLF001
    assert "ctx" in executor._locks  # noqa: SLF001
    executor._histories["ctx"] = [{"role": "user", "content": "x"}]  # noqa: SLF001

    executor.clear_history("ctx")

    assert "ctx" not in executor._locks  # noqa: SLF001
    assert "ctx" not in executor._histories  # noqa: SLF001
    del lock
