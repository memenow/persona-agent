"""LLM client abstraction with OpenAI-compatible implementation.

Provides a framework-agnostic LLM interface using the openai SDK directly.
Supports any OpenAI-compatible provider (OpenAI, Azure, Ollama, vLLM, etc.).
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool call from the LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Structured response from an LLM chat completion."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Optional tool definitions in OpenAI function calling format.
            temperature: Sampling temperature override.
            max_tokens: Max tokens override.

        Returns:
            Structured ChatResponse with content and/or tool calls.
        """


class OpenAICompatibleClient(LLMClient):
    """LLM client using the OpenAI SDK, compatible with any OpenAI-API provider.

    Supports OpenAI, Azure OpenAI, Ollama, vLLM, and any other
    provider exposing an OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: float = 120.0,
    ):
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = base_url or os.environ.get("OPENAI_API_BASE")

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            timeout=timeout,
        )
        logger.info(
            "OpenAICompatibleClient initialized: model=%s base_url=%s",
            model,
            resolved_base,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleClient":
        """Create a client from llm_config.json format.

        Expected config structure::

            {
                "default_model": "gpt-4o-mini",
                "api_key": "...",
                "api_base": "...",
                "model_configs": [{"model": "...", "temperature": 0.7, ...}]
            }
        """
        default_model = config.get("default_model", "gpt-4o-mini")
        model_configs = config.get("model_configs", [])

        # Find the config matching default_model, fall back to first entry
        matched = {}
        for mc in model_configs:
            if mc.get("model") == default_model:
                matched = mc
                break
        if not matched and model_configs:
            matched = model_configs[0]

        return cls(
            model=matched.get("model", default_model),
            api_key=matched.get("api_key") or config.get("api_key"),
            base_url=config.get("api_base"),
            temperature=matched.get("temperature", 0.7),
            max_tokens=matched.get("max_tokens", 4000),
        )

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.default_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug(
            "LLM request: model=%s messages=%d tools=%d",
            self.model,
            len(messages),
            len(tools or []),
        )

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        parsed_tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parsed_tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage_data = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=msg.content,
            tool_calls=parsed_tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage_data,
        )
