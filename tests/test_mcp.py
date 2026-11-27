"""Tests for MCPManager configuration parsing, schema conversion, and tool errors."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mcp import MCPError
from mcp.types import Tool

from persona_agent.mcp import MCPConnection, MCPManager, mcp_tools_to_openai_functions


class CapturingMCPManager(MCPManager):
    def __init__(self) -> None:
        super().__init__()
        self.captured_env: dict[str, str] = {}

    async def _connect_service(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> bool:
        self.captured_env = env
        return True


async def test_mcp_config_substitutes_env_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "real-key")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brave": {
                        "command": "node",
                        "args": ["server.js"],
                        "env": {
                            "BRAVE_API_KEY": "${BRAVE_API_KEY}",
                            "PATH": "${PATH}:/opt/mcp/bin",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = CapturingMCPManager()
    assert await manager.load_config(config_path) is True
    assert manager.captured_env == {
        "BRAVE_API_KEY": "real-key",
        "PATH": "/usr/local/bin:/usr/bin:/opt/mcp/bin",
    }


def test_default_mcp_config_does_not_override_path() -> None:
    config = json.loads(Path("config/mcp_config.json").read_text(encoding="utf-8"))
    service = config["mcpServers"]["mcp-server-fetch"]
    assert "PATH" not in service.get("env", {})


def test_mcp_tools_to_openai_functions_uses_v2_input_schema() -> None:
    tool = Tool(
        name="fetch",
        description="Fetch a URL",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )
    functions = mcp_tools_to_openai_functions([tool])
    assert functions == [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Fetch a URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        }
    ]


async def test_call_tool_handles_unknown_tool_and_mcp_error() -> None:
    manager = MCPManager()
    missing = json.loads(await manager.call_tool("missing", {}))
    assert "not found" in missing["error"]

    fake_client = MagicMock()
    fake_client.call_tool = AsyncMock(side_effect=MCPError(-32000, "boom"))
    tool = Tool(name="explode", input_schema={"type": "object", "properties": {}})
    manager._connections["svc"] = MCPConnection("svc", fake_client, [tool])  # noqa: SLF001
    manager._tool_to_service["explode"] = "svc"  # noqa: SLF001

    failed = json.loads(await manager.call_tool("explode", {}))
    assert failed == {"error": "Tool execution failed: explode"}


async def test_call_tool_extracts_text_and_handles_is_error() -> None:
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text="ok-result")],
            is_error=False,
        )
    )
    tool = Tool(name="echo", input_schema={"type": "object", "properties": {}})
    manager._connections["svc"] = MCPConnection("svc", fake_client, [tool])  # noqa: SLF001
    manager._tool_to_service["echo"] = "svc"  # noqa: SLF001

    assert await manager.call_tool("echo", {"x": 1}) == "ok-result"

    fake_client.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text="bad input")],
            is_error=True,
        )
    )
    err_payload = json.loads(await manager.call_tool("echo", {"x": 1}))
    assert err_payload == {"error": "bad input"}
