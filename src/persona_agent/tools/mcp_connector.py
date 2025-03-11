"""MCP Connector module.

This module provides a legacy compatibility layer for connecting to MCP servers
and executing tools and access resources from those servers. New code should
use the persona_agent.mcp modules directly.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MCPConnector:
    """Connector for MCP servers.
    
    This class handles communication with MCP servers, providing methods
    to execute tools and access resources. It serves as a compatibility 
    layer for older code but forwards calls to the main MCP implementation.
    
    Attributes:
        connected_servers: Dictionary of connected MCP server configurations.
    """
    
    def __init__(self):
        """Initialize a new MCP connector."""
        self.connected_servers: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def connect_server(self, server_name: str, config: Dict[str, Any]) -> bool:
        """Connect to an MCP server.
        
        Args:
            server_name: Name to identify the server.
            config: Configuration for connecting to the server.
            
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            # Store the server configuration for later use
            self.connected_servers[server_name] = {
                "name": server_name,
                "config": config,
            }
            
            self.logger.info(f"Connected to MCP server: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {server_name}: {str(e)}")
            return False
    
    def list_servers(self) -> List[str]:
        """List all connected MCP servers.
        
        Returns:
            List of server names.
        """
        return list(self.connected_servers.keys())
    
    def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            server_name: Name of the server to disconnect.
            
        Returns:
            True if disconnection was successful, False otherwise.
        """
        if server_name in self.connected_servers:
            # Remove the server from our dictionary
            del self.connected_servers[server_name]
            self.logger.info(f"Disconnected from MCP server: {server_name}")
            return True
        
        self.logger.warning(f"Server {server_name} not found, cannot disconnect")
        return False
    
    def use_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to execute.
            arguments: Dictionary of tool arguments.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ValueError: If the server is not connected.
        """
        if server_name not in self.connected_servers:
            raise ValueError(f"Server {server_name} is not connected")
        
        try:
            # Use the main MCP implementation
            from persona_agent.mcp.mcp import use_mcp_tool
            
            # We need to call the async function from a sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Use the existing event loop with run_until_complete
                    result = asyncio.run_coroutine_threadsafe(
                        use_mcp_tool(server_name, tool_name, arguments),
                        loop
                    ).result()
                else:
                    # Create a new event loop
                    result = asyncio.run(use_mcp_tool(server_name, tool_name, arguments))
                
                return result
            except RuntimeError:
                # Fallback if we can't handle the event loop properly
                self.logger.warning("Error with event loop, using alternative execution")
                
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(use_mcp_tool(server_name, tool_name, arguments))
                    return result
                finally:
                    loop.close()
            
        except Exception as e:
            self.logger.error(
                f"Error executing tool {tool_name} on server {server_name}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
            }
    
    def access_resource(
        self, server_name: str, resource_uri: str
    ) -> Dict[str, Any]:
        """Access a resource from an MCP server.
        
        Args:
            server_name: Name of the MCP server.
            resource_uri: URI of the resource to access.
            
        Returns:
            The resource content.
            
        Raises:
            ValueError: If the server is not connected.
        """
        if server_name not in self.connected_servers:
            raise ValueError(f"Server {server_name} is not connected")
        
        try:
            # This is a placeholder as we don't have a direct MCP resource accessor implemented yet
            # In a complete implementation, this would call the MCP server's resource endpoint
            self.logger.warning(
                f"Resource access not fully implemented, returning placeholder for {resource_uri}"
            )
            
            return {
                "success": True,
                "uri": resource_uri,
                "content": f"Resource access not fully implemented. URI: {resource_uri}",
                "mime_type": "text/plain",
            }
        except Exception as e:
            self.logger.error(
                f"Error accessing resource {resource_uri} on server {server_name}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a connected MCP server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            Dictionary of server information, or None if not connected.
        """
        return self.connected_servers.get(server_name)
