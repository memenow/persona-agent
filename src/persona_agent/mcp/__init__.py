"""Model Context Protocol (MCP) integration package.

This package provides integration with Model Context Protocol (MCP) services and tools,
allowing AI agents to use external tools and services to enhance their capabilities.
"""

from typing import Dict, Any, List, Optional

# Import MCP-related classes for easier access
try:
    from .mcp_manager import McpManager, McpService, sync_register_mcp_tools_for_agent
    from .tool_adapter import MCPToolAdapter
    
    __all__ = [
        "McpManager",
        "McpService",
        "MCPToolAdapter",
        "sync_register_mcp_tools_for_agent",
    ]
except ImportError as e:
    import logging
    logging.warning(f"Failed to import MCP components: {e}")
    __all__ = []




