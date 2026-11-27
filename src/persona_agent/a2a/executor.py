"""A2A v1.0 AgentExecutor implementation with per-context history and MCP tool loop."""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from persona_agent.llm.client import LLMClient
from persona_agent.mcp.manager import MCPManager

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10
MAX_CONTEXT_HISTORIES = 200


class PersonaAgentExecutor(AgentExecutor):
    """Runs LLM conversations with MCP tool execution for a single persona."""

    def __init__(
        self,
        persona_id: str,
        persona_name: str,
        system_prompt: str,
        llm_client: LLMClient,
        mcp_manager: MCPManager | None = None,
    ) -> None:
        self.persona_id = persona_id
        self.persona_name = persona_name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager

        self._histories: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        # Per-context locks serialize concurrent chat() calls on the same context_id
        # so history appends stay paired. Locks are evicted alongside their histories.
        self._locks: dict[str, asyncio.Lock] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle an A2A task execution request."""
        task = context.current_task
        if task is None:
            # A2A v1.0 requires a Task event before any TaskStatusUpdateEvent.
            task = Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        user_text = context.get_user_input()
        if not user_text:
            await updater.failed(
                message=updater.new_agent_message([Part(text="Empty message received")])
            )
            return

        logger.info(
            "Persona %s executing task %s: %s",
            self.persona_name,
            task.id,
            user_text[:80],
        )

        await updater.start_work()

        try:
            response_text = await self.chat(task.context_id, user_text)
            await updater.complete(
                message=updater.new_agent_message([Part(text=response_text)])
            )
        except Exception:
            logger.exception("Error executing persona %s", self.persona_name)
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(text="An error occurred while processing your request.")]
                )
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Emit a best-effort canceled state for a task."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

    async def chat(self, context_id: str, user_text: str) -> str:
        """Execute a turn with optional MCP tool calling, serialized per context_id."""
        async with self._get_lock(context_id):
            return await self._chat_locked(context_id, user_text)

    async def _chat_locked(self, context_id: str, user_text: str) -> str:
        history = self._get_history(context_id)
        history.append({"role": "user", "content": user_text, "timestamp": time.time()})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *(
                {"role": entry["role"], "content": entry["content"]}
                for entry in history
            ),
        ]

        tools = None
        if self.mcp_manager and self.mcp_manager.is_initialized:
            openai_tools = self.mcp_manager.get_openai_tools()
            if openai_tools:
                tools = openai_tools

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = await self.llm_client.chat(messages=messages, tools=tools)

            if not response.has_tool_calls:
                content = response.content or ""
                history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "timestamp": time.time(),
                    }
                )
                return content

            logger.info(
                "Persona %s tool calls (iteration %d): %s",
                self.persona_name,
                iteration + 1,
                [tc.name for tc in response.tool_calls],
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                if self.mcp_manager:
                    tool_result = await self.mcp_manager.call_tool(
                        tc.name, tc.arguments
                    )
                else:
                    tool_result = json.dumps({"error": "No MCP manager available"})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

        # Force a final text answer while retaining prior tool messages so strict
        # OpenAI-compatible providers accept the conversation history.
        logger.warning(
            "Persona %s exceeded max tool iterations (%d), forcing final response",
            self.persona_name,
            MAX_TOOL_ITERATIONS,
        )
        if tools:
            response = await self.llm_client.chat(
                messages=messages, tools=tools, tool_choice="none"
            )
        else:
            response = await self.llm_client.chat(messages=messages)

        content = response.content or ""
        history.append(
            {"role": "assistant", "content": content, "timestamp": time.time()}
        )
        return content

    def _get_lock(self, context_id: str) -> asyncio.Lock:
        lock = self._locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context_id] = lock
        return lock

    def _get_history(self, context_id: str) -> list[dict[str, Any]]:
        if context_id in self._histories:
            self._histories.move_to_end(context_id)
        else:
            self._histories[context_id] = []
            while len(self._histories) > MAX_CONTEXT_HISTORIES:
                evicted, _ = self._histories.popitem(last=False)
                self._locks.pop(evicted, None)
        return self._histories[context_id]

    def clear_history(self, context_id: str) -> None:
        self._histories.pop(context_id, None)
        self._locks.pop(context_id, None)

    def get_history(self, context_id: str) -> list[dict[str, Any]]:
        """Return a shallow copy of the recorded history for context_id."""
        return list(self._histories.get(context_id, []))
