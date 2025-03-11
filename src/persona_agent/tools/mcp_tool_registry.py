"""MCP Tool Registry module.

This module provides a legacy compatibility layer for registering MCP tools.
New code should use the persona_agent.mcp.tool_adapter module directly.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from persona_agent.mcp.tool_adapter import MCPToolAdapter


class MCPToolRegistry:
    """Legacy registry for MCP tools used by persona agents.
    
    This class provides a backward-compatible interface for the registration
    and management of MCP tools, delegating to the improved MCPToolAdapter
    implementation.
    
    Attributes:
        tools: Dictionary of registered tools.
        function_map: Map of tool names to their execution functions.
    """
    
    def __init__(self):
        """Initialize a new MCP tool registry."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MCPToolRegistry (legacy interface)")
        
        # Create an adapter instance to delegate functionality to
        self._adapter = MCPToolAdapter()
    
    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        """Register an MCP tool for use with persona agents.
        
        Args:
            server_name: Name of the MCP server providing the tool.
            tool_name: Name of the tool.
            description: Description of the tool.
            input_schema: JSON schema for tool input parameters.
        """
        self.logger.info(f"Registering tool {tool_name} (delegating to MCPToolAdapter)")
        
        # Delegate to the adapter
        self._adapter.register_tool(
            server_name=server_name,
            tool_name=tool_name,
            description=description,
            input_schema=input_schema,
        )
    
    def register_tool_from_config(self, tool_config: Dict[str, Any]) -> None:
        """Register a tool from a configuration dictionary.
        
        Args:
            tool_config: Dictionary containing tool configuration.
        """
        self.logger.info(f"Registering tool from config (delegating to MCPToolAdapter)")
        
        # Delegate to the adapter
        self._adapter.register_tool_from_config(tool_config)
    
    @property
    def tools(self) -> Dict[str, Dict[str, Any]]:
        """Get the dictionary of registered tools.
        
        Returns:
            Dictionary mapping tool IDs to their metadata.
        """
        return self._adapter.tools
    
    def get_function_map(self) -> Dict[str, Callable]:
        """Get the function map for AutoGen agents.
        
        Returns:
            Dictionary mapping tool names to execution functions.
        """
        return self._adapter.get_function_map()
    
    def get_tool_configs(self) -> List[Dict[str, Any]]:
        """Get configurations for all registered tools.
        
        Returns:
            List of tool configuration dictionaries.
        """
        return self._adapter.get_tool_configs()
