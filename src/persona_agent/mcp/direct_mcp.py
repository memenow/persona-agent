"""MCP manager using the mcp library directly.

Handles stdio server lifecycle, tool discovery, and tool execution.
"""

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Maximum number of retries for loading tools from a service
MAX_RETRIES = 3


def mcp_tools_to_openai_functions(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI function calling schema.

    Args:
        tools: List of mcp.types.Tool objects.

    Returns:
        List of OpenAI-compatible tool definitions.
    """
    result = []
    for tool in tools:
        func_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
            },
        }
        if tool.inputSchema:
            func_def["function"]["parameters"] = tool.inputSchema
        else:
            func_def["function"]["parameters"] = {"type": "object", "properties": {}}
        result.append(func_def)
    return result


class MCPServiceConnection:
    """A live connection to a single MCP stdio server."""

    def __init__(self, name: str, session: ClientSession, tools: list[Any]):
        self.name = name
        self.session = session
        self.tools = tools
        self.tool_names = [t.name for t in tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call on this MCP service.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool execution result.
        """
        logger.info("Calling MCP tool %s on service %s", tool_name, self.name)
        result = await self.session.call_tool(tool_name, arguments)
        return result


class DirectMCPManager:
    """MCP service and tool manager using the mcp library directly.

    Manages stdio server lifecycle, tool loading, and tool execution.
    """

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._connections: dict[str, MCPServiceConnection] = {}
        self._tool_to_service: dict[str, str] = {}
        self._all_tools: list[Any] = []
        self._openai_tools: list[dict[str, Any]] = []
        self._initialized = False

    async def load_config(self, config_path: str) -> bool:
        """Load MCP configuration and connect to all services.

        Args:
            config_path: Path to mcp_config.json.

        Returns:
            True if at least one service was loaded successfully.
        """
        if not os.path.exists(config_path):
            logger.warning("MCP config not found: %s", config_path)
            return False

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        servers: dict[str, Any] = {}

        # Support both "mcpServers" (new) and "services" (old) sections
        if "mcpServers" in config:
            servers.update(config["mcpServers"])
        if "services" in config:
            for name, svc in config["services"].items():
                if svc.get("type", "stdio") == "stdio" and svc.get("enabled", True):
                    servers[name] = svc

        if not servers:
            logger.warning("No MCP servers found in config")
            return False

        success = False
        for name, server_config in servers.items():
            if server_config.get("disabled", False):
                logger.info("Skipping disabled MCP service: %s", name)
                continue

            command = server_config.get("command")
            if not command:
                logger.warning("No command for MCP service: %s", name)
                continue

            # Substitute environment variables in command and args
            command = self._substitute_env_vars(command)
            args = [self._substitute_env_vars(a) for a in server_config.get("args", [])]
            env = server_config.get("env", {})

            if await self._connect_service(name, command, args, env):
                success = True

        self._initialized = True
        logger.info(
            "MCP initialized: %d services, %d tools",
            len(self._connections),
            len(self._all_tools),
        )
        return success

    async def _connect_service(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> bool:
        """Connect to a single MCP stdio service with retries."""
        # Merge PATH from current environment if not provided
        merged_env = dict(os.environ)
        merged_env.update(env)

        params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            # Use a local exit stack per attempt to avoid leaking
            # half-initialized connections on failure
            local_stack = AsyncExitStack()
            try:
                logger.info(
                    "Connecting to MCP service %s (attempt %d): %s %s",
                    name,
                    attempt,
                    command,
                    " ".join(args),
                )

                read_stream, write_stream = await local_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await local_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                # List available tools
                tools_result = await session.list_tools()
                tools = tools_result.tools

                if not tools:
                    logger.warning("Service %s returned no tools", name)
                    await local_stack.aclose()
                    return False

                # Success — transfer ownership to the shared exit stack
                await self._exit_stack.enter_async_context(local_stack)

                conn = MCPServiceConnection(name, session, tools)
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
                # Clean up the failed attempt's resources
                await local_stack.aclose()
                logger.exception(
                    "Failed to connect to MCP service %s (attempt %d/%d)",
                    name,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt >= MAX_RETRIES:
                    return False

        return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call by name.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments dict for the tool.

        Returns:
            Stringified tool result for inclusion in LLM messages.
        """
        service_name = self._tool_to_service.get(tool_name)
        if not service_name or service_name not in self._connections:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        conn = self._connections[service_name]
        try:
            result = await conn.call_tool(tool_name, arguments)

            # Extract text content from result
            if hasattr(result, "content") and result.content:
                parts = []
                for part in result.content:
                    if hasattr(part, "text"):
                        parts.append(part.text)
                    else:
                        parts.append(str(part))
                return "\n".join(parts)

            return str(result)
        except Exception:
            logger.exception("Error calling tool %s", tool_name)
            return json.dumps({"error": f"Tool execution failed: {tool_name}"})

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return self._openai_tools

    def get_all_tools(self) -> list[Any]:
        """Get all raw MCP tool objects."""
        return self._all_tools

    def get_tool_names(self) -> list[str]:
        """Get names of all available tools."""
        return list(self._tool_to_service.keys())

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def close(self) -> None:
        """Close all MCP service connections."""
        logger.info("Closing MCP manager and all service connections")
        await self._exit_stack.aclose()
        self._connections.clear()
        self._tool_to_service.clear()
        self._all_tools.clear()
        self._openai_tools.clear()
        self._initialized = False

    @staticmethod
    def _substitute_env_vars(value: str) -> str:
        """Replace ${VAR_NAME} patterns with environment variable values."""

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return re.sub(r"\$\{([^}]+)}", replacer, value)
