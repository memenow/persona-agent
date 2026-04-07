"""PersonaAgentExecutor: A2A executor that wraps persona logic with LLM + MCP tools.

Uses a direct LLM call loop that handles tool execution internally.
"""

import json
import logging
import uuid
from collections import OrderedDict
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from persona_agent.llm.client import LLMClient
from persona_agent.mcp.direct_mcp import DirectMCPManager

logger = logging.getLogger(__name__)

# Maximum number of tool call iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10
MAX_CONTEXT_HISTORIES = 200


class PersonaAgentExecutor(AgentExecutor):
    """A2A executor that simulates a persona using LLM with MCP tool support.

    Each instance is bound to a specific persona and uses the persona's
    system prompt to guide LLM responses.
    """

    def __init__(
        self,
        persona_id: str,
        persona_name: str,
        system_prompt: str,
        llm_client: LLMClient,
        mcp_manager: DirectMCPManager | None = None,
    ):
        self.persona_id = persona_id
        self.persona_name = persona_name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager

        # Conversation history per context_id for multi-turn support.
        # Uses OrderedDict as an LRU cache to cap memory usage.
        self._histories: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a persona agent task.

        Processes the user message through the LLM with tool support,
        emitting status updates and the final artifact via the event queue.
        """
        context_id = context.context_id
        task_id = context.task_id

        # Extract user message text
        user_text = context.get_user_input()
        if not user_text:
            await self._emit_error(
                event_queue, context_id, task_id, "Empty message received"
            )
            return

        logger.info(
            "Persona %s executing task %s: %s",
            self.persona_name,
            task_id,
            user_text[:80],
        )

        # Signal working state
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            response_text = await self._run_llm_loop(context_id, user_text)

            # Store assistant response in history
            history = self._get_history(context_id)
            history.append({"role": "assistant", "content": response_text})

            # Emit completed status with the agent message
            event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=response_text))],
                        ),
                    ),
                    final=True,
                )
            )

        except Exception:
            logger.exception("Error executing persona %s", self.persona_name)
            await self._emit_error(
                event_queue,
                context_id,
                task_id,
                "An error occurred while processing your request.",
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel a running task."""
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context.context_id,
                task_id=context.task_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )

    async def _run_llm_loop(self, context_id: str, user_text: str) -> str:
        """Run the LLM with tool calling loop until a text response is produced.

        Args:
            context_id: Conversation context ID for history tracking.
            user_text: The user's message text.

        Returns:
            The final text response from the LLM.
        """
        history = self._get_history(context_id)
        history.append({"role": "user", "content": user_text})

        # Build messages with system prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
        ]

        # Get available tools in OpenAI format
        tools = None
        if self.mcp_manager and self.mcp_manager.is_initialized:
            openai_tools = self.mcp_manager.get_openai_tools()
            if openai_tools:
                tools = openai_tools

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = await self.llm_client.chat(messages=messages, tools=tools)

            if not response.has_tool_calls:
                return response.content or ""

            # Process tool calls
            logger.info(
                "Persona %s tool calls (iteration %d): %s",
                self.persona_name,
                iteration + 1,
                [tc.name for tc in response.tool_calls],
            )

            # Append assistant message with tool calls to messages
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

            # Execute each tool and add results
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

        # Exceeded max iterations — force a final response without tools
        logger.warning(
            "Persona %s exceeded max tool iterations (%d), forcing final response",
            self.persona_name,
            MAX_TOOL_ITERATIONS,
        )
        response = await self.llm_client.chat(messages=messages, tools=None)
        return response.content or ""

    def _get_history(self, context_id: str) -> list[dict[str, Any]]:
        """Get or create conversation history for a context.

        Evicts the oldest context when the cache exceeds MAX_CONTEXT_HISTORIES.
        """
        if context_id in self._histories:
            self._histories.move_to_end(context_id)
        else:
            self._histories[context_id] = []
            # Evict oldest entries when cache is full
            while len(self._histories) > MAX_CONTEXT_HISTORIES:
                self._histories.popitem(last=False)
        return self._histories[context_id]

    def clear_history(self, context_id: str) -> None:
        """Clear conversation history for a context."""
        self._histories.pop(context_id, None)

    async def _emit_error(
        self,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        error_message: str,
    ) -> None:
        """Emit a failed task status."""
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[Part(root=TextPart(text=error_message))],
                    ),
                ),
                final=True,
            )
        )
