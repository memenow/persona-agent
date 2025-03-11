"""MCP module for Model Context Protocol tool usage.

This module provides a streamlined implementation of MCP tool execution
that works with the autogen-ext MCP tooling or falls back to a direct connector.
"""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

async def use_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Use an MCP tool with the specified parameters.
    
    This implementation prioritizes the autogen-ext API with fallbacks for robustness.
    
    Args:
        server_name: Name of the MCP server.
        tool_name: Name of the tool.
        arguments: Arguments for the tool.
        
    Returns:
        The result of the tool execution.
        
    Raises:
        Exception: If there's an error in tool execution.
    """
    try:
        logger.info(f"Using MCP tool {tool_name} on server {server_name}")
        
        # Try to use autogen-ext with the correct API
        try:
            from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
            
            # Load config and get server info
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'config', 'mcp_config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                server_config = config.get("mcpServers", {}).get(server_name)
                
                if server_config:
                    # Create server params
                    server_params = StdioServerParams(
                        command=server_config["command"],
                        args=server_config.get("args", []),
                        env=server_config.get("env", {})
                    )
                    
                    logger.info(f"Using autogen_ext.tools.mcp implementation")
                    
                    # Get all tools from the server
                    tools = await mcp_server_tools(server_params=server_params)
                    
                    # Find the specific tool we want
                    for tool in tools:
                        if hasattr(tool, "__name__") and tool.__name__.endswith(f".{tool_name}"):
                            # Execute the tool with our arguments
                            return await tool(**arguments)
                    
                    logger.warning(f"Tool {tool_name} not found on server {server_name}")
                else:
                    logger.warning(f"Server {server_name} not found in configuration")
            else:
                logger.warning(f"MCP configuration file not found at {config_path}")
                
        except ImportError:
            logger.warning("autogen_ext.tools.mcp module not available, trying alternative implementation")
            
        # Try our own direct connector implementation
        try:
            from persona_agent.mcp.direct_connector import direct_use_mcp_tool
            logger.info(f"Using internal direct_connector implementation")
            return await direct_use_mcp_tool(server_name=server_name, tool_name=tool_name, arguments=arguments)
        except Exception as direct_error:
            logger.warning(f"Direct connector implementation failed: {str(direct_error)}")
        
        # If we reach here, all approaches have failed
        logger.error(f"Could not execute MCP tool {tool_name}: no MCP implementation available")
        return {
            "success": False,
            "error": "No MCP implementation available. Please install autogen-ext[mcp]"
        }
        
    except Exception as e:
        logger.error(f"Error in MCP tool execution: {str(e)}")
        return {"success": False, "error": str(e)}
