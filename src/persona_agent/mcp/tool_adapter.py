"""MCP tool adapter module.

This module provides adapters for integrating Model Context Protocol (MCP) tools
with persona agents, allowing LLMs to use MCP tools as native functions.
"""

import json
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapter for using MCP tools with persona agents.
    
    This class provides methods for converting MCP tools into callable functions
    that can be used by persona agents, and for executing those tools.
    
    Attributes:
        tools: Dictionary mapping tool identifiers to tool metadata.
        function_map: Dictionary mapping tool identifiers to callable functions.
        available_servers: List of available MCP server names.
        auto_approved_tools: Dictionary mapping server names to auto-approved tools.
    """
    
    def __init__(self):
        """Initialize a new MCP tool adapter."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.function_map: Dict[str, Callable] = {}
        self.available_servers: List[str] = []
        self.auto_approved_tools: Dict[str, List[str]] = {}
    
    def load_tools_from_config(self, config: Dict[str, Any]) -> None:
        """Load available MCP tools from configuration.
        
        Args:
            config: Dictionary containing MCP server configurations.
        """
        from persona_agent.mcp.server_config import (
            get_available_servers,
            get_auto_approved_tools,
        )
        
        self.available_servers = get_available_servers(config)
        self.auto_approved_tools = get_auto_approved_tools(config)
        
        logger.info(f"Available MCP servers: {self.available_servers}")
        for server, tools in self.auto_approved_tools.items():
            logger.info(f"Auto-approved tools for {server}: {tools}")
    
    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
        requires_approval: bool = True,
    ) -> bool:
        """Register an MCP tool for use with persona agents.
        
        Args:
            server_name: Name of the MCP server providing the tool.
            tool_name: Name of the tool.
            description: Description of the tool.
            input_schema: JSON schema for tool input parameters.
            requires_approval: Whether the tool requires user approval to execute.
            
        Returns:
            Boolean indicating if the registration was successful.
        """
        try:
            # Create a unique identifier for the tool
            tool_id = f"{server_name}.{tool_name}"
            
            # Check if the tool is auto-approved
            auto_approved = tool_name in self.auto_approved_tools.get(server_name, [])
            requires_approval = requires_approval and not auto_approved
            
            # Store tool metadata
            self.tools[tool_id] = {
                "server_name": server_name,
                "tool_name": tool_name,
                "description": description,
                "input_schema": input_schema,
                "requires_approval": requires_approval,
            }
            
            # Create a function for this tool
            self.function_map[tool_id] = self._create_tool_function(
                server_name, tool_name, input_schema, description, requires_approval
            )
            
            logger.info(f"Registered MCP tool: {tool_id} (requires_approval={requires_approval})")
            return True
        except Exception as e:
            logger.error(f"Error registering tool {tool_name}: {str(e)}")
            return False
    
    def register_tool_from_config(self, tool_config: Dict[str, Any]) -> bool:
        """Register an MCP tool from a configuration dictionary.
        
        Args:
            tool_config: Dictionary containing tool configuration with keys:
                - server_name: Name of the MCP server.
                - tool_name: Name of the tool.
                - description: Description of the tool.
                - input_schema: JSON schema for the tool's input parameters.
                - requires_approval (optional): Whether the tool requires approval.
                
        Returns:
            Boolean indicating if the registration was successful.
        """
        try:
            server_name = tool_config.get("server_name")
            tool_name = tool_config.get("tool_name")
            description = tool_config.get("description")
            input_schema = tool_config.get("input_schema")
            requires_approval = tool_config.get("requires_approval", True)
            
            if not all([server_name, tool_name, description, input_schema]):
                logger.error(f"Invalid tool configuration: {tool_config}")
                raise ValueError("Tool configuration missing required parameters")
            
            return self.register_tool(
                server_name=server_name,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                requires_approval=requires_approval,
            )
        except Exception as e:
            logger.error(f"Error registering tool from config: {str(e)}")
            return False
    
    def _create_tool_function(
        self,
        server_name: str,
        tool_name: str,
        input_schema: Dict[str, Any],
        description: str,
        requires_approval: bool,
    ) -> Callable:
        """Create a function for an MCP tool.
        
        Args:
            server_name: Name of the MCP server providing the tool.
            tool_name: Name of the tool.
            input_schema: JSON schema for tool input parameters.
            description: Description of the tool.
            requires_approval: Whether the tool requires user approval to execute.
            
        Returns:
            A callable function that executes the MCP tool.
        """
        # Define the function that will execute the MCP tool
        async def tool_function(**kwargs) -> Any:
            """Dynamically generated function for executing an MCP tool."""
            return await self._execute_mcp_tool(
                server_name, tool_name, kwargs, requires_approval
            )
        
        # Set function metadata for AutoGen
        tool_function.__name__ = f"{server_name}.{tool_name}"
        tool_function.__doc__ = description
        
        # Create a proper signature for the function based on the input schema
        sig_params = []
        required_params = input_schema.get("required", [])
        properties = input_schema.get("properties", {})
        
        # Map JSON schema types to Python types
        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": List[Any],
            "object": Dict[str, Any],
        }
        
        for param_name, param_schema in properties.items():
            param_desc = param_schema.get("description", "")
            param_type = param_schema.get("type", "string")
            is_required = param_name in required_params
            
            python_type = type_map.get(param_type, Any)
            
            if is_required:
                sig_params.append(
                    inspect.Parameter(
                        name=param_name,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=python_type,
                    )
                )
            else:
                sig_params.append(
                    inspect.Parameter(
                        name=param_name,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation=Optional[python_type],
                    )
                )
        
        # Create and set the function signature
        return_type = Union[Dict[str, Any], List[Any], str, None]
        tool_function.__signature__ = inspect.Signature(
            parameters=sig_params,
            return_annotation=return_type,
        )
        
        return tool_function
    
    async def _execute_mcp_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        requires_approval: bool,
    ) -> Any:
        """Execute an MCP tool using use_mcp_tool.
        
        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool.
            arguments: Dictionary of tool arguments.
            requires_approval: Whether the tool requires user approval.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            Exception: If there's an error executing the tool.
        """
        try:
            logger.info(
                f"Executing MCP tool: {tool_name} on server {server_name}"
                f" with args: {json.dumps(arguments)}"
            )
            
            # Use our main MCP module which handles all implementation details
            from persona_agent.mcp.mcp import use_mcp_tool
            
            # Execute the tool
            result = await use_mcp_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
            )
            
            logger.info(f"MCP tool execution result: {json.dumps(result) if isinstance(result, (dict, list)) else str(result)[:100]}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {str(e)}")
            raise
    
    def get_function_map(self) -> Dict[str, Callable]:
        """Get the function map for persona agents.
        
        Returns:
            Dictionary mapping tool identifiers to callable functions.
        """
        return self.function_map
    
    def get_tool_configs(self) -> List[Dict[str, Any]]:
        """Get configurations for all registered tools.
        
        Returns:
            List of tool configuration dictionaries.
        """
        return list(self.tools.values())
    
    def get_available_tools(self, server_name: Optional[str] = None) -> List[str]:
        """Get a list of available tools.
        
        Args:
            server_name: Optional server name to filter by.
            
        Returns:
            List of available tool identifiers.
        """
        if server_name:
            return [
                tool_id for tool_id in self.tools
                if tool_id.startswith(f"{server_name}.")
            ]
        else:
            return list(self.tools.keys())
