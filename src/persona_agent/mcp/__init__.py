"""Model Context Protocol (MCP) integration package."""

from .direct_mcp import DirectMCPManager, mcp_tools_to_openai_functions

__all__ = ["DirectMCPManager", "mcp_tools_to_openai_functions"]
