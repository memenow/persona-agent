"""MCP stdio server lifecycle, tool discovery, and tool execution."""

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client, MCPError, StdioServerParameters

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def mcp_tools_to_openai_functions(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling tool schemas."""
    functions: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or {
            "type": "object",
            "properties": {},
        }
        functions.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"MCP tool: {tool.name}",
                    "parameters": schema,
                },
            }
        )
    return functions


class MCPConnection:
    """Active MCP client connection and its discovered tools."""

    def __init__(self, name: str, client: Client, tools: list[Any]) -> None:
        self.name = name
        self.client = client
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        logger.info("Calling MCP tool %s on service %s", tool_name, self.name)
        return await self.client.call_tool(tool_name, arguments)


class MCPManager:
    """Manages stdio MCP client lifecycles, tool schemas, and tool invocation."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._connections: dict[str, MCPConnection] = {}
        self._tool_to_service: dict[str, str] = {}
        self._all_tools: list[Any] = []
        self._openai_tools: list[dict[str, Any]] = []
        self._initialized = False

    async def load_config(self, config_path: str | Path) -> bool:
        """Load MCP configuration and connect to all enabled stdio services."""
        path = Path(config_path)
        if not path.is_file():
            logger.warning("MCP config not found: %s", path)
            return False

        config = json.loads(path.read_text(encoding="utf-8"))
        servers: dict[str, Any] = {}

        if isinstance(config.get("mcpServers"), dict):
            servers.update(config["mcpServers"])
        if isinstance(config.get("services"), dict):
            for name, svc in config["services"].items():
                if svc.get("type", "stdio") == "stdio" and svc.get("enabled", True):
                    servers[name] = svc

        if not servers:
            logger.warning("No MCP servers configured in %s", path)
            return False

        connected_any = False
        for name, server_config in servers.items():
            if server_config.get("disabled", False):
                logger.info("Skipping disabled MCP service: %s", name)
                continue

            raw_command = server_config.get("command")
            if not raw_command:
                logger.warning("No command configured for MCP service: %s", name)
                continue

            command = self._substitute_env_vars(raw_command)
            args = [self._substitute_env_vars(a) for a in server_config.get("args", [])]
            raw_env = server_config.get("env", {})
            env = {
                key: self._substitute_env_vars(value)
                if isinstance(value, str)
                else str(value)
                for key, value in raw_env.items()
            }

            if await self._connect_service(name, command, args, env):
                connected_any = True

        self._initialized = True
        logger.info(
            "MCP initialized: %d services, %d tools",
            len(self._connections),
            len(self._all_tools),
        )
        return connected_any

    async def _connect_service(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> bool:
        """Connect to a stdio MCP server with bounded retries and clean rollback."""
        merged_env = dict(os.environ)
        merged_env.update(env)
        params = StdioServerParameters(command=command, args=args, env=merged_env)

        for attempt in range(1, MAX_RETRIES + 1):
            local_stack = AsyncExitStack()
            try:
                logger.info(
                    "Connecting to MCP service %s (attempt %d/%d): %s %s",
                    name,
                    attempt,
                    MAX_RETRIES,
                    command,
                    " ".join(args),
                )
                client = await local_stack.enter_async_context(Client(params))
                tools_result = await client.list_tools()
                tools = list(tools_result.tools)

                if not tools:
                    logger.warning("MCP service %s returned no tools", name)
                    await local_stack.aclose()
                    return False

                await self._exit_stack.enter_async_context(local_stack)
                conn = MCPConnection(name, client, tools)
                self._connections[name] = conn

                for tool in tools:
                    self._tool_to_service[tool.name] = name
                    self._all_tools.append(tool)

                self._openai_tools = mcp_tools_to_openai_functions(self._all_tools)
                logger.info(
                    "Loaded %d tools from %s: %s",
                    len(tools),
                    name,
                    ", ".join(conn.tool_names),
                )
                return True
            except Exception:
                await local_stack.aclose()
                logger.exception(
                    "Failed connecting to MCP service %s (attempt %d/%d)",
                    name,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt >= MAX_RETRIES:
                    return False

        return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a discovered MCP tool and return a string payload for the LLM."""
        service_name = self._tool_to_service.get(tool_name)
        if not service_name or service_name not in self._connections:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        conn = self._connections[service_name]
        try:
            result = await conn.call_tool(tool_name, arguments)
            rendered = self._extract_result_text(result)
            if getattr(result, "is_error", False):
                logger.warning("MCP tool %s reported error: %s", tool_name, rendered)
                return json.dumps({"error": rendered or f"Tool error: {tool_name}"})
            return rendered
        except (MCPError, Exception):
            logger.exception("Error calling MCP tool %s", tool_name)
            return json.dumps({"error": f"Tool execution failed: {tool_name}"})

    @staticmethod
    def _extract_result_text(result: Any) -> str:
        content = getattr(result, "content", None)
        if content:
            parts = [
                part.text if hasattr(part, "text") else str(part) for part in content
            ]
            return "\n".join(parts)
        return str(result)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return self._openai_tools

    def get_all_tools(self) -> list[Any]:
        return self._all_tools

    def get_tool_names(self) -> list[str]:
        return list(self._tool_to_service.keys())

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def close(self) -> None:
        logger.info("Closing MCP manager and active connections")
        await self._exit_stack.aclose()
        self._connections.clear()
        self._tool_to_service.clear()
        self._all_tools.clear()
        self._openai_tools.clear()
        self._initialized = False

    @staticmethod
    def _substitute_env_vars(value: str) -> str:
        return re.sub(
            r"\$\{([^}]+)}",
            lambda match: os.environ.get(match.group(1), ""),
            value,
        )
