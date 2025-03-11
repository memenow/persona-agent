"""MCP (Model Context Protocol) module.

This module provides utilities for working with the Model Context Protocol (MCP),
which enables the persona agent to interact with external tools and services.
"""

# Import MCP-related modules for easier access
from persona_agent.mcp.server_config import (
    load_mcp_config,
    save_mcp_config,
    get_available_servers,
    get_auto_approved_tools,
)
from persona_agent.mcp.tool_adapter import (
    MCPToolAdapter,
    create_tool_decorators,
)

# Version
__version__ = "0.1.0"
