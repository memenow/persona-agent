"""MCP Tool Adapter for AutoGen.

This module provides an adapter for Model Context Protocol (MCP) tools to work with AutoGen agents.
"""

import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class MCPToolAdapter:
    """Adapter for MCP tools to be used with AutoGen agents.
    
    This class provides methods for registering and managing MCP tools,
    making them available for use with AutoGen agents.
    
    Attributes:
        function_map: Dictionary mapping tool IDs to callable functions.
        tools_config: Dictionary of tool configurations keyed by tool ID.
        tools: Dictionary of tool instances keyed by tool ID.
    """
    
    def __init__(self):
        """Initialize an MCPToolAdapter instance."""
        self.function_map: Dict[str, Callable] = {}
        self.tools_config: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, Any] = {}
        logger.info("Initialized MCPToolAdapter")
    
    def register_tool(self, tool_id: str, tool_func: Callable, tool_config: Optional[Dict[str, Any]] = None):
        """Register a tool function with this adapter.
        
        Args:
            tool_id: Unique identifier for the tool.
            tool_func: Callable function implementing the tool.
            tool_config: Optional configuration for the tool.
        """
        self.function_map[tool_id] = tool_func
        
        if tool_config:
            self.tools_config[tool_id] = tool_config
            
        logger.info(f"Registered tool: {tool_id}")
    
    def register_tool_from_config(self, tool_config: Dict[str, Any]):
        """Register a tool from a configuration dictionary.
        
        Args:
            tool_config: Tool configuration dictionary containing
                        'id', 'function', and optional metadata.
        """
        if not tool_config:
            logger.warning("Empty tool configuration provided")
            return
            
        tool_id = tool_config.get('id')
        tool_func = tool_config.get('function')
        
        if not tool_id:
            logger.warning("Tool configuration missing 'id' field")
            return
            
        if not tool_func:
            logger.warning(f"Tool configuration for {tool_id} missing 'function' field")
            return
            
        # Register the tool
        self.register_tool(tool_id, tool_func, tool_config)
    
    def get_function_map(self) -> Dict[str, Callable]:
        """Get the function map containing all registered tools.
        
        Returns:
            Dictionary mapping tool IDs to callable functions.
        """
        return self.function_map
    
    def get_tool_configs(self) -> List[Dict[str, Any]]:
        """Get a list of all tool configurations.
        
        Returns:
            List of tool configuration dictionaries.
        """
        return list(self.tools_config.values())
    
    def get_tool_function(self, tool_id: str) -> Optional[Callable]:
        """Get a tool function by its ID.
        
        Args:
            tool_id: ID of the tool to retrieve.
            
        Returns:
            Callable function for the tool, or None if not found.
        """
        return self.function_map.get(tool_id)
    
    def unregister_tool(self, tool_id: str):
        """Unregister a tool by its ID.
        
        Args:
            tool_id: ID of the tool to unregister.
        """
        if tool_id in self.function_map:
            del self.function_map[tool_id]
            
        if tool_id in self.tools_config:
            del self.tools_config[tool_id]
            
        if tool_id in self.tools:
            del self.tools[tool_id]
            
        logger.info(f"Unregistered tool: {tool_id}") 