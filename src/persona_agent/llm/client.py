"""LLM client abstraction backed by the OpenAI Python SDK."""

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
    """Single function call requested by an LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Structured chat completion response."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient(ABC):
    """Abstract interface for async chat completion clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return a parsed response."""


class OpenAICompatibleClient(LLMClient):
    """Chat completion client for OpenAI and OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        self._api_key = resolved_key
        self._base_url = resolved_base

        # openai>=3.0 rejects empty api_key eagerly in AsyncOpenAI.__init__.
        # Defer client creation when unconfigured so the server can still boot
        # for health checks, persona management, and A2A discovery.
        self._client: AsyncOpenAI | None = (
            AsyncOpenAI(
                api_key=resolved_key,
                base_url=resolved_base,
                timeout=timeout,
            )
            if resolved_key
            else None
        )
        if self._client is None:
            logger.warning(
                "OpenAICompatibleClient initialized without API key: model=%s base_url=%s",
                model,
                resolved_base,
            )
        else:
            logger.info(
                "OpenAICompatibleClient initialized: model=%s base_url=%s",
                model,
                resolved_base,
            )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        default_model: str | None = None,
    ) -> "OpenAICompatibleClient":
        """Build a client from a parsed llm_config.json mapping.

        Credential precedence:
        - api_key: matched model entry > top-level config > OPENAI_API_KEY
        - base_url: OPENAI_BASE_URL / OPENAI_API_BASE > matched model entry > top-level config
        """
        target_model = default_model or config.get("default_model") or "gpt-4o-mini"
        model_configs = config.get("model_configs", [])

        matched: dict[str, Any] = {}
        if isinstance(model_configs, list):
            for entry in model_configs:
                if not isinstance(entry, dict):
                    continue
                if (
                    entry.get("model") == target_model
                    or entry.get("name") == target_model
                ):
                    matched = entry
                    break
            if not matched and default_model is None and model_configs:
                first = model_configs[0]
                if isinstance(first, dict):
                    matched = first

        env_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "OPENAI_API_BASE"
        )
        api_settings = config.get("api_settings")
        timeout = 120.0
        if isinstance(api_settings, dict) and isinstance(
            api_settings.get("timeout"), int | float
        ):
            timeout = float(api_settings["timeout"])

        resolved_model = (
            matched.get("model") or matched.get("name") or target_model
            if matched
            else target_model
        )

        return cls(
            model=resolved_model,
            api_key=matched.get("api_key") or config.get("api_key"),
            base_url=env_base_url or matched.get("api_base") or config.get("api_base"),
            temperature=matched.get("temperature", 0.7),
            max_tokens=matched.get("max_tokens", 4000),
            timeout=timeout,
        )

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "LLM API key is not configured. Set OPENAI_API_KEY or api_key in llm_config.json."
                )
            self._api_key = key
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=self._base_url,
                timeout=self.timeout,
            )
        return self._client

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        client = self._get_client()
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
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        logger.debug(
            "LLM request: model=%s messages=%d tools=%d",
            self.model,
            len(messages),
            len(tools or []),
        )

        response = await client.chat.completions.create(**kwargs)
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

        usage_data: dict[str, int] = {}
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
