"""Regression tests for ``PersonaAgentExecutor``.

Coverage:
- Each lifecycle hook awaits ``EventQueue.enqueue_event`` (regression for the
  historical missing-await bug).
- ``chat()`` persists user/assistant turns with wall-clock timestamps.
- The tool-iteration cap forces a textual answer using ``tool_choice="none"``
  while preserving full message context.
- Concurrent ``chat()`` calls sharing a context_id are serialized so history
  pairs stay intact.
"""

import asyncio
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


async def test_chat_records_wall_clock_timestamps() -> None:
    """History entries must carry ``timestamp`` so REST callers can order
    messages chronologically."""
    import time

    before = time.time()
    executor = _make_executor(ChatResponse(content="answer", tool_calls=[]))
    await executor.chat("ctx", "question")
    after = time.time()
    history = executor.get_history("ctx")
    for entry in history:
        assert "timestamp" in entry
        assert before <= entry["timestamp"] <= after


async def test_chat_fallback_uses_tool_choice_none_with_full_context() -> None:
    """When the tool loop exhausts ``MAX_TOOL_ITERATIONS``, the executor must
    re-issue the final completion with ``tool_choice="none"`` while keeping
    ``tool_calls``/``tool`` messages intact (so the model retains context)."""
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
    # The full context (including tool/tool_calls turns) must be preserved.
    roles = [m["role"] for m in captured["messages"]]
    assert "tool" in roles
    assert any("tool_calls" in m for m in captured["messages"])
    # And tools are still passed (so the provider validates the request).
    assert captured["tools"] is not None


async def test_chat_serializes_concurrent_calls_on_same_context() -> None:
    """Two concurrent ``chat()`` calls on the same context must produce
    paired (user, assistant) entries instead of interleaved history."""

    async def fake_chat(messages, **kwargs):
        latest_user = next(m for m in reversed(messages) if m["role"] == "user")[
            "content"
        ]
        # Yield to the scheduler so the other coroutine has a chance to
        # interleave if locking is broken.
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
    # Each user turn must be immediately followed by its matching assistant.
    pair1, pair2 = history[0:2], history[2:4]
    for user_entry, assistant_entry in (pair1, pair2):
        assert user_entry["role"] == "user"
        assert assistant_entry["role"] == "assistant"
        assert assistant_entry["content"] == f"reply-{user_entry['content']}"


async def test_clear_history_drops_lock() -> None:
    """Locks should not outlive their context's history."""
    executor = PersonaAgentExecutor(
        persona_id="p",
        persona_name="P",
        system_prompt="sys",
        llm_client=MagicMock(),
    )
    # Force lock creation by entering chat() machinery indirectly.
    lock = executor._get_lock("ctx")  # noqa: SLF001 — internal helper
    assert "ctx" in executor._locks  # noqa: SLF001
    executor._histories["ctx"] = [{"role": "user", "content": "x"}]
    executor.clear_history("ctx")
    assert "ctx" not in executor._locks  # noqa: SLF001
    assert "ctx" not in executor._histories  # noqa: SLF001
    del lock
